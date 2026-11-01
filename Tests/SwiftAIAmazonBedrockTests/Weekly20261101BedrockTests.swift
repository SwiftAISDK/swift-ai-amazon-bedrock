import Foundation
import Testing
@testable import SwiftAIAmazonBedrock
import SwiftAIProvider
import SwiftAITransport

private let weeklyBedrockResponse = #"{"output":{"message":{"content":[{"text":"ok"}]}},"stopReason":"end_turn","usage":{"inputTokens":1,"outputTokens":1,"totalTokens":2}}"#
private actor WeeklyBedrockTokens {
    var calls = 0
    func next() -> String { calls += 1; return " token-\(calls) " }
}

@Suite struct Weekly20261101Bedrock {
    @Test func documentNamesAreUniqueAcrossEntirePrompt() {
        var names = BedrockDocumentNames()
        let filenames: [String?] = ["invoice.pdf", "invoice.pdf", "rates.v1.pdf", "rates.v2.pdf", "/workspace/.eve/attachments/ab12/a.pdf", "invoice (2).pdf", "document-1.pdf", nil, String(repeating: "a", count: 201)+".pdf", String(repeating: "a", count: 201)+".pdf"]
        #expect(filenames.map { bedrockDocumentName($0, documentCounter: &names) } == ["invoice", "invoice (2)", "rates", "rates (2)", "workspace", "invoice (2) (2)", "document-1", "document-1 (2)", String(repeating: "a", count: 200), String(repeating: "a", count: 196)+" (2)"])
    }

    @Test func disabledThinkingHonorsModelSupport() async throws {
        for (model, expected): (String, String?) in [("anthropic.claude-3-haiku-20240307-v1:0", nil), ("anthropic.claude-3-7-sonnet", "disabled"), ("anthropic.claude-sonnet-5", "disabled"), ("anthropic.claude-sonnet-5-5", "between_tools"), ("anthropic.claude-opus-5-5", "adaptive")] {
            let transport = RecordingTransport(response: jsonResponse(weeklyBedrockResponse))
            let provider = try AmazonBedrockProvider(settings: .init(region: "us-east-1", apiKey: "key", transport: transport))
            let result = try await provider.languageModel(model).generate(.init(messages: [.user("Hi")], reasoning: "none"))
            let request = try #require(await transport.requests().first)
            let body = try decodeJSONBody(try #require(request.body))
            #expect(body["additionalModelRequestFields"]?["thinking"]?["type"]?.stringValue == expected)
            if model.contains("opus-5-5") { #expect(body["additionalModelRequestFields"]?["output_config"]?["effort"] == "low") }
            #expect(result.warnings.filter { $0.feature == "reasoning" }.count == (model.contains("5-5") ? 1 : 0))
        }
    }

    @Test func explicitDisabledRetainsRawEffortAndLowersUnsupportedEfforts() async throws {
        for (model, expected): (String, String) in [("anthropic.claude-opus-5", "disabled"), ("anthropic.claude-sonnet-5-5", "between_tools")] {
            let transport = RecordingTransport(response: jsonResponse(weeklyBedrockResponse))
            let provider = try AmazonBedrockProvider(settings: .init(region: "us-east-1", apiKey: "key", transport: transport))
            let result = try await provider.languageModel(model).generate(.init(messages: [.user("Hi")], providerOptions: ["amazonBedrock": ["reasoningConfig": ["type": "disabled", "budgetTokens": 3000, "display": "summarized"], "additionalModelRequestFields": ["custom": "preserved", "output_config": ["effort": "max", "custom": "also-preserved"]]]]))
            let request = try #require(await transport.requests().first)
            let body = try decodeJSONBody(try #require(request.body))
            #expect(body["additionalModelRequestFields"]?["thinking"] == ["type": .string(expected)])
            #expect(body["additionalModelRequestFields"]?["output_config"] == ["effort": "high", "custom": "also-preserved"])
            #expect(result.warnings.contains { $0.feature == "providerOptions.amazonBedrock.additionalModelRequestFields.output_config.effort" })
        }
    }

    @Test func supportedReasoningEffortsRoundDownAndUnknownOpenAIEffortsPassThrough() async throws {
        for (model, reasoning, expected, field) in [("anthropic.claude-opus-4-6", "xhigh", "high", "output_config"), ("openai.gpt-oss-120b-1:0", "max", "high", "reasoning_effort"), ("openai.gpt-future", "none", "none", "reasoning")] {
            let transport = RecordingTransport(response: jsonResponse(weeklyBedrockResponse))
            let provider = try AmazonBedrockProvider(settings: .init(region: "us-east-1", apiKey: "key", transport: transport))
            _ = try await provider.languageModel(model).generate(.init(messages: [.user("Hi")], reasoning: reasoning))
            let request = try #require(await transport.requests().first)
            let body = try decodeJSONBody(try #require(request.body))
            let raw = body["additionalModelRequestFields"]?[field]
            #expect((raw?["effort"] ?? raw)?.stringValue == expected)
        }
    }

    @Test func portableNonePreservesTypedEffortOverRawWhenAdaptiveIsRequired() async throws {
        let transport = RecordingTransport(response: jsonResponse(weeklyBedrockResponse))
        let provider = try AmazonBedrockProvider(settings: .init(region: "us-east-1", apiKey: "key", transport: transport))
        _ = try await provider.languageModel("anthropic.claude-opus-5-5").generate(.init(messages: [.user("Hi")], reasoning: "none", providerOptions: ["amazonBedrock": ["reasoningConfig": ["maxReasoningEffort": "high"], "additionalModelRequestFields": ["output_config": ["effort": "max"]]]]))
        let request = try #require(await transport.requests().first)
        let body = try decodeJSONBody(try #require(request.body))
        #expect(body["additionalModelRequestFields"]?["thinking"] == ["type": "adaptive"])
        #expect(body["additionalModelRequestFields"]?["output_config"]?["effort"] == "high")
    }

    @Test func bearerCallbacksAreLazyRotateAndValidateBeforeSend() async throws {
        let tokens = WeeklyBedrockTokens()
        let transport = RecordingTransport(response: jsonResponse(weeklyBedrockResponse))
        let provider = try AmazonBedrockProvider(settings: .init(region: "us-east-1", transport: transport, apiKeyProvider: { await tokens.next() }))
        let model = try provider.languageModel("anthropic.claude-sonnet-4-6")
        #expect(await tokens.calls == 0)
        for _ in 0..<2 { _ = try await model.generate(.init(messages: [.user("Hi")])) }
        #expect(await transport.requests().map { $0.headers["authorization"] } == ["Bearer token-1", "Bearer token-2"])
        let emptyTransport = RecordingTransport(response: jsonResponse(weeklyBedrockResponse))
        let empty = try AmazonBedrockProvider(settings: .init(region: "us-east-1", transport: emptyTransport, apiKeyProvider: { " \n" }))
        await #expect(throws: (any Error).self) { _ = try await empty.languageModel("anthropic.claude-sonnet-4-6").generate(.init(messages: [.user("Hi")])) }
        #expect(await emptyTransport.requests().isEmpty)
    }

    @Test func invokeUsesNativeSchemaForModelsRejectingForcedTools() async throws {
        let transport = RecordingTransport(response: jsonResponse(#"{"content":[{"type":"text","text":"{\"answer\":\"42\"}"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}"#))
        let provider = try AmazonBedrockAnthropicProvider(settings: .init(region: "us-east-1", apiKey: "key", transport: transport))
        let schema: JSONValue = ["type": "object", "properties": ["answer": ["type": "string"]], "required": ["answer"], "additionalProperties": false]
        let result = try await provider.languageModel("us.anthropic.claude-sonnet-5-5").generate(.init(messages: [.user("Hi")], responseFormat: .json(schema: schema)))
        #expect(result.text == "{\"answer\":\"42\"}")
        let request = try #require(await transport.requests().first)
        let body = try decodeJSONBody(try #require(request.body))
        #expect(body["output_config"]?["format"]?["schema"] == schema)
        #expect(body["tools"] == nil)
    }
}
