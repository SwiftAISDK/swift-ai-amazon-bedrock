@testable import SwiftAIAmazonBedrock
@testable import SwiftAIAnthropic
@testable import SwiftAI
@testable import SwiftAIProvider
@testable import SwiftAIProviderUtilities
@testable import SwiftAITransport
import Foundation
import Testing

private func bedrockUpstreamProvider(
    transport: any AITransport
) throws -> AmazonBedrockProvider {
    try AIProviders.amazonBedrock(configuration: AmazonBedrockProviderSettings(
        region: "us-east-1",
        apiKey: "bedrock-key",
        transport: transport
    ))
}

@Test func amazonBedrockWrapsSelectedTextAndImagePartsAsGuardContent() async throws {
    let transport = RecordingTransport(response: jsonResponse("""
    {"output":{"message":{"content":[{"text":"ok"}]}},"stopReason":"end_turn","usage":{"inputTokens":2,"outputTokens":1,"totalTokens":3}}
    """))
    let model = try bedrockUpstreamProvider(transport: transport)
        .languageModel("anthropic.claude-3-haiku-20240307-v1:0")
    let png = Data([137, 80, 78, 71, 13, 10, 26, 10])

    _ = try await model.generate(LanguageModelRequest(messages: [
        AIMessage(role: .user, content: [
            .text(
                "What is the capital of Japan?",
                providerMetadata: ["bedrock": [
                    "guardContent": true,
                    "guardContentQualifiers": ["query"]
                ]]
            ),
            .text("Background"),
            .data(
                mimeType: "image/png",
                data: png,
                providerMetadata: ["amazonBedrock": ["guardContent": true]]
            )
        ])
    ]))

    let body = try decodeJSONBody(try #require((await transport.requests()).first?.body))
    let content = try #require(body["messages"]?[0]?["content"]?.arrayValue)
    #expect(content[0]["guardContent"]?["text"]?["text"]?.stringValue == "What is the capital of Japan?")
    #expect(content[0]["guardContent"]?["text"]?["qualifiers"]?[0]?.stringValue == "query")
    #expect(content[1]["text"]?.stringValue == "Background")
    #expect(content[2]["guardContent"]?["image"]?["format"]?.stringValue == "png")
    #expect(content[2]["guardContent"]?["image"]?["source"]?["bytes"]?.stringValue == png.base64EncodedString())
}

@Test func amazonBedrockRecognizesAnthropicApplicationProfilesFromReasoningBudget() async throws {
    let transport = RecordingTransport(response: jsonResponse("""
    {"output":{"message":{"content":[{"text":"ok"}]}},"stopReason":"end_turn"}
    """))
    let model = try bedrockUpstreamProvider(transport: transport).languageModel(
        "arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/profile-id"
    )

    let result = try await model.generate(LanguageModelRequest(
        messages: [.user("Think")],
        maxOutputTokens: 100,
        providerOptions: ["amazonBedrock": [
            "reasoningConfig": [
                "type": "enabled",
                "budgetTokens": 64,
                "maxReasoningEffort": "high"
            ]
        ]]
    ))

    let body = try decodeJSONBody(try #require((await transport.requests()).first?.body))
    #expect(body["inferenceConfig"]?["maxTokens"]?.intValue == 164)
    #expect(body["additionalModelRequestFields"]?["thinking"]?["budget_tokens"]?.intValue == 64)
    #expect(body["additionalModelRequestFields"]?["output_config"]?["effort"]?.stringValue == "high")
    #expect(!result.warnings.contains { $0.feature == "budgetTokens" })
}

@Test func amazonBedrockUsesNestedReasoningForPrefixedOpenAIGPTModels() async throws {
    let transport = RecordingTransport(response: jsonResponse("""
    {"output":{"message":{"content":[{"text":"ok"}]}},"stopReason":"end_turn"}
    """))
    let model = try bedrockUpstreamProvider(transport: transport)
        .languageModel("us.openai.gpt-5.2-chat-v1:0")

    _ = try await model.generate(LanguageModelRequest(
        messages: [.user("Think")],
        providerOptions: ["bedrock": [
            "reasoningConfig": ["maxReasoningEffort": "high"]
        ]]
    ))

    let body = try decodeJSONBody(try #require((await transport.requests()).first?.body))
    #expect(body["additionalModelRequestFields"]?["reasoning"]?["effort"]?.stringValue == "high")
    #expect(body["additionalModelRequestFields"]?["reasoning_effort"] == nil)
}

@Test func amazonBedrockForwardsAnthropicDisableParallelToolUseWithoutBedrockToolChoice() async throws {
    let transport = RecordingTransport(response: jsonResponse("""
    {"output":{"message":{"content":[{"text":"ok"}]}},"stopReason":"end_turn"}
    """))
    let model = try bedrockUpstreamProvider(transport: transport)
        .languageModel("anthropic.claude-3-7-sonnet-20250219-v1:0")

    _ = try await model.generate(LanguageModelRequest(
        messages: [.user("Search")],
        tools: ["lookup": ["type": "object", "properties": [:]]],
        toolChoice: ["type": "auto"],
        providerOptions: ["anthropic": ["disableParallelToolUse": true]]
    ))

    let body = try decodeJSONBody(try #require((await transport.requests()).first?.body))
    #expect(body["toolConfig"]?["tools"]?.arrayValue?.count == 1)
    #expect(body["toolConfig"]?["toolChoice"] == nil)
    #expect(body["additionalModelRequestFields"]?["tool_choice"]?["type"]?.stringValue == "auto")
    #expect(body["additionalModelRequestFields"]?["tool_choice"]?["disable_parallel_tool_use"]?.boolValue == true)
}

@Test func amazonBedrockDropsAssistantTurnsThatOnlyRetainCachePoints() async throws {
    let transport = RecordingTransport(response: jsonResponse("""
    {"output":{"message":{"content":[{"text":"ok"}]}},"stopReason":"end_turn"}
    """))
    let model = try bedrockUpstreamProvider(transport: transport)
        .languageModel("anthropic.claude-3-7-sonnet-20250219-v1:0")

    _ = try await model.generate(LanguageModelRequest(messages: [
        .user("First"),
        AIMessage(
            role: .assistant,
            content: [.reasoning("unsigned")],
            providerMetadata: ["amazonBedrock": ["cachePoint": ["type": "default"]]]
        ),
        .user("Second")
    ]))

    let body = try decodeJSONBody(try #require((await transport.requests()).first?.body))
    let messages = try #require(body["messages"]?.arrayValue)
    #expect(messages.count == 2)
    #expect(messages.allSatisfy { $0["role"]?.stringValue == "user" })
}

@Test func amazonBedrockPreservesCompleteRawUsageInProviderMetadata() async throws {
    let transport = RecordingTransport(response: jsonResponse("""
    {"output":{"message":{"content":[{"text":"ok"}]}},"stopReason":"end_turn","usage":{"inputTokens":2,"outputTokens":1,"totalTokens":3,"futureUsageField":{"value":7}}}
    """))
    let model = try bedrockUpstreamProvider(transport: transport)
        .languageModel("anthropic.claude-3-haiku-20240307-v1:0")

    let result = try await model.generate(LanguageModelRequest(messages: [.user("Hi")]))

    #expect(result.providerMetadata["amazonBedrock"]?["usage"]?["inputTokens"]?.intValue == 2)
    #expect(result.providerMetadata["amazonBedrock"]?["usage"]?["futureUsageField"]?["value"]?.intValue == 7)
}

@Test func amazonBedrockEmbeddingModelFamilySupportsOpaqueApplicationProfileARNs() async throws {
    let transport = RecordingTransport(response: jsonResponse("""
    {"embeddings":[[0.1,0.2],[0.3,0.4]]}
    """, headers: ["x-amzn-bedrock-input-token-count": "6"]))
    let provider = try bedrockUpstreamProvider(transport: transport)
    let model = try provider.embeddingModel(
        "arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/profile-id",
        settings: AmazonBedrockEmbeddingModelSettings(modelFamily: .cohere)
    )

    let result = try await model.embed(EmbeddingRequest(values: ["one", "two"]))

    #expect(result.embeddings == [[0.1, 0.2], [0.3, 0.4]])
    #expect(result.usage?.totalTokens == 6)
    let body = try decodeJSONBody(try #require((await transport.requests()).first?.body))
    #expect(body["input_type"]?.stringValue == "search_query")
    #expect(body["texts"]?.arrayValue?.compactMap(\.stringValue) == ["one", "two"])
}
