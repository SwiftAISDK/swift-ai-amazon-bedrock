import Foundation
import Testing
import SwiftAIProvider
import SwiftAITransport
@testable import SwiftAIAmazonBedrock

@Suite struct Weekly20261114FableTests {
    @Test(arguments: ["anthropic.claude-fable-5-2", "us.anthropic.claude-fable-5-2", "global.anthropic.claude-fable-5-2"]) func fableFiveTwoKeepsBedrockStrictAndNativeOutputDisabled(modelID: String) {
        let prepared = bedrockPrepareTools(from: ["testFunction": ["type": "object", "properties": [:], "strict": true]], toolChoice: nil, modelID: modelID)
        #expect(prepared.toolConfig?["tools"]?[0]?["toolSpec"]?["strict"] == nil)
        #expect(prepared.warnings.contains { $0.feature == "strict" && $0.message?.contains("strict mode is not supported") == true })
        #expect(!bedrockSupportsStrictToolSpec(modelID: modelID))
        #expect(!bedrockSupportsNativeStructuredOutput(modelID: modelID))
        #expect(bedrockSupportsStrictToolSpec(modelID: "us.anthropic.claude-sonnet-5-5"))
        #expect(bedrockSupportsNativeStructuredOutput(modelID: "us.anthropic.claude-sonnet-5-5"))
    }

    @Test(arguments: ["anthropic.claude-fable-5", "anthropic.claude-fable-5-20260601-v1:0", "us.anthropic.claude-fable-5-20260601-v1:0"])
    func baseFableInvokeModelKeepsJSONToolFallbackIncludingDatedAliases(modelID: String) async throws {
        let transport = RecordingTransport(response: jsonResponse(#"{"content":[{"type":"tool_use","id":"json-1","name":"json","input":{"answer":"fable"}}],"stop_reason":"tool_use","usage":{"input_tokens":3,"output_tokens":2}}"#))
        let provider = try AmazonBedrockAnthropicProvider(settings: .init(region: "us-east-1", apiKey: "key", transport: transport))
        let model = try provider.messages(modelID)
        let schema: JSONValue = ["type": "object", "properties": ["answer": ["type": "string"]], "required": ["answer"], "additionalProperties": false]
        let result = try await model.generate(.init(messages: [.user("Return JSON")], responseFormat: .json(schema: schema), tools: ["weather": ["type": "object", "strict": true, "properties": ["city": ["type": "string"]]]]))
        let request = try #require(await transport.requests().first)
        let body = try decodeJSONBody(#require(request.body))
        #expect(request.url.path.hasSuffix("/invoke"))
        #expect(body["output_config"]?["format"] == nil)
        let tools = try #require(body["tools"]?.arrayValue)
        #expect(tools.contains { $0["name"] == "json" })
        #expect(tools.first { $0["name"] == "weather" }?["strict"] == nil)
        #expect(body["tool_choice"]?["type"] == "any")
        #expect(try decodeJSONBody(Data(result.text.utf8))["answer"] == "fable")
        #expect(result.warnings.contains { $0.feature == "strict" })
    }

    @Test(arguments: ["anthropic.claude-fable-5-1", "anthropic.claude-fable-5-2", "us.anthropic.claude-fable-5-1-20261101-v1:0", "us.anthropic.claude-fable-5-2-20261110-v1:0"])
    func genuineMinorAliasesRetainNativeFallbackForRejectedForcedTools(modelID: String) async throws {
        let transport = RecordingTransport(response: jsonResponse(#"{"content":[{"type":"text","text":"{\"answer\":\"native\"}"}],"stop_reason":"end_turn","usage":{"input_tokens":3,"output_tokens":2}}"#))
        let provider = try AmazonBedrockAnthropicProvider(settings: .init(region: "us-east-1", apiKey: "key", transport: transport))
        let schema: JSONValue = ["type": "object", "properties": ["answer": ["type": "string"]], "required": ["answer"], "additionalProperties": false]
        let result = try await provider.messages(modelID).generate(.init(messages: [.user("Return JSON")], responseFormat: .json(schema: schema), providerOptions: ["anthropic": ["structuredOutputMode": "jsonTool"]]))
        let body = try decodeJSONBody(#require(await transport.requests().first?.body))
        #expect(body["output_config"]?["format"]?["type"] == "json_schema")
        #expect(body["tools"] == nil)
        #expect(body["tool_choice"] == nil)
        #expect(try decodeJSONBody(Data(result.text.utf8))["answer"] == "native")
    }
}
