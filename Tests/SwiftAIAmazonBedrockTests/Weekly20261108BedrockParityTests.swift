import Foundation
import Testing
import SwiftAIProvider
import SwiftAITransport
@testable import SwiftAIAmazonBedrock

@Suite struct Weekly20261108BedrockParity {
    private var schema: JSONValue { ["type": "object", "properties": ["value": ["type": "string"]], "required": ["value"], "additionalProperties": false] }
    private let response = #"{"output":{"message":{"content":[{"text":"ok"}]}},"stopReason":"end_turn","usage":{"inputTokens":1,"outputTokens":1,"totalTokens":2}}"#

    @Test func nativeStructuredOutputSupportUsesExplicitClaudeModelsAndSuffixBoundaries() {
        let supported = ["anthropic.claude-opus-5", "anthropic.claude-opus-5-5", "global.anthropic.claude-sonnet-5-5", "anthropic.claude-haiku-5-5", "us.anthropic.claude-opus-5-20260701-v1:0", "anthropic.claude-opus-4-6-v1"]
        let unsupported = ["anthropic.claude-haiku-5", "anthropic.claude-haiku-5-5-preview", "anthropic.claude-haiku-5-5-9", "anthropic.claude-sonnet-5-9", "anthropic.claude-opus-5-60-v1", "us.anthropic.claude-opus-6-v1:0", "anthropic.claude-opus-4-60-v1"]
        for id in supported { #expect(bedrockSupportsNativeStructuredOutput(modelID: id)); #expect(bedrockSupportsStrictToolSpec(modelID: id)) }
        for id in unsupported { #expect(!bedrockSupportsNativeStructuredOutput(modelID: id)); #expect(!bedrockSupportsStrictToolSpec(modelID: id)) }
        #expect(!bedrockSupportsNativeStructuredOutput(modelID: "anthropic.claude-sonnet-4-6-v1"))
        #expect(bedrockSupportsStrictToolSpec(modelID: "anthropic.claude-sonnet-4-6-v1"))
    }

    @Test func namedChoicesKeepAllAnthropicFunctionToolsAndDefaultToNoParallelCalls() {
        let tools = ["weather": schema, "other": schema]
        let named: JSONValue = ["type": "tool", "toolName": "weather"]
        let anthropic = bedrockPrepareTools(from: tools, toolChoice: named, modelID: "anthropic.claude-sonnet-4-5")
        #expect(anthropic.toolConfig?["tools"]?.arrayValue?.count == 2)
        #expect(anthropic.additionalModelRequestFields?["tool_choice"] == ["type": "tool", "name": "weather", "disable_parallel_tool_use": true])
        let parallel = bedrockPrepareTools(from: tools, toolChoice: named, modelID: "anthropic.claude-sonnet-4-5", disableParallelToolUse: false)
        #expect(parallel.toolConfig?["tools"]?.arrayValue?.count == 2)
        #expect(parallel.toolConfig?["toolChoice"] == ["tool": ["name": "weather"]])
        #expect(parallel.additionalModelRequestFields == nil)
        let nonAnthropic = bedrockPrepareTools(from: tools, toolChoice: named, modelID: "mistral.mistral-large")
        #expect(nonAnthropic.toolConfig?["tools"]?.arrayValue?.count == 1)
        let profile = bedrockPrepareTools(from: tools, toolChoice: named, modelID: "arn:aws:bedrock:us-east-1:123:application-inference-profile/custom", modelFamily: .anthropic)
        #expect(profile.toolConfig?["tools"]?.arrayValue?.count == 2)
    }

    @Test func nativeOutputAndOpenSchemaFallbackRespectNamedApplicationTools() async throws {
        let transport = RecordingTransport(response: jsonResponse(response))
        let provider = try AmazonBedrockProvider(settings: .init(region: "us-east-1", apiKey: "key", transport: transport))
        let model = try provider.languageModel("anthropic.claude-sonnet-5-5")
        _ = try await model.generate(.init(messages: [.user("Hi")], responseFormat: .json(schema: schema)))
        _ = try await provider.languageModel("anthropic.claude-opus-5").generate(.init(messages: [.user("Hi")], responseFormat: .json(schema: ["type": "object", "additionalProperties": true]), tools: ["weather": schema, "other": schema], toolChoice: ["type": "tool", "toolName": "weather"]))
        let requests = await transport.requests()
        let native = try decodeJSONBody(try #require(requests[0].body))
        #expect(native["additionalModelRequestFields"]?["output_config"]?["format"]?["schema"] == schema)
        let fallback = try decodeJSONBody(try #require(requests[1].body))
        #expect(fallback["additionalModelRequestFields"]?["output_config"]?["format"] == nil)
        let names = Set(fallback["toolConfig"]?["tools"]?.arrayValue?.compactMap { $0["toolSpec"]?["name"]?.stringValue } ?? [])
        #expect(names == ["weather", "json"])
    }

    @Test(arguments: ["google", "vertex", "googleVertex"]) func foreignEmptyTextDoesNotSuppressPrefillTrim(namespace: String) async throws {
        let transport = RecordingTransport(response: jsonResponse(response))
        let model = try AmazonBedrockProvider(settings: .init(region: "us-east-1", apiKey: "key", transport: transport)).languageModel("anthropic.claude-sonnet-4-5")
        _ = try await model.generate(.init(messages: [.user("Hi"), .init(role: .assistant, content: [.text("Answer  "), .text("", providerMetadata: [namespace: ["thoughtSignature": "opaque"]])])]))
        let raw = try decodeJSONBody(try #require(await transport.requests().first?.body))
        #expect(raw["messages"]?[1]?["content"]?.arrayValue?.count == 1)
        #expect(raw["messages"]?[1]?["content"]?[0]?["text"] == "Answer")
    }

    @Test func zeroSeedAndMalformedToolInputArePreserved() async throws {
        let transport = RecordingTransport(response: jsonResponse(#"{"images":["aGVsbG8="]}"#))
        let model = try AmazonBedrockProvider(settings: .init(region: "us-east-1", apiKey: "key", transport: transport)).imageModel("amazon.nova-canvas-v1:0")
        _ = try await model.generateImage(.init(prompt: "A lake", seed: 0))
        let raw = try decodeJSONBody(try #require(await transport.requests().first?.body))
        #expect(raw["imageGenerationConfig"]?["seed"] == 0)
        #expect(bedrockToolArguments(#"{"city":"Tokyo"}"#) == ["city": "Tokyo"])
        #expect(bedrockToolArguments("{") == ["rawInvalidInput": "{"])
    }
}
