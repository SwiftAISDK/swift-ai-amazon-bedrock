import Foundation
import Testing
import SwiftAIProvider
import SwiftAITransport
@testable import SwiftAIAmazonBedrock

// Published 5.0.142 removes the old prefix_mismatch_behavior rename. Preserve
// the upstream adaptive/block_binding and no-binding scenarios at the HTTP boundary.
@Suite struct Weekly20261115ThinkingBindingTests {
    @Test(arguments: [true, false])
    func preservesThinkingFieldsWithoutBedrockRename(withBinding: Bool) async throws {
        let transport = RecordingTransport(response: jsonResponse(#"{"content":[{"type":"text","text":"Hello"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}"#))
        let provider = try AmazonBedrockAnthropicProvider(settings: .init(region: "us-east-1", apiKey: "key", transport: transport))
        let thinking: JSONValue = withBinding
            ? ["type": "adaptive", "display": "summarized", "block_binding": ["prefix_mismatch_behavior": "drop_block"]]
            : ["type": "enabled", "budget_tokens": 2000]
        _ = try await provider.messages("test-model-id").generate(.init(
            messages: [.user("Hello")], maxOutputTokens: 1024,
            providerOptions: ["anthropic": ["thinking": thinking]]
        ))
        let requests = await transport.requests()
        #expect(requests.count == 1)
        let request = try #require(requests.first)
        #expect(request.url.path.hasSuffix("/invoke"))
        let body = try decodeJSONBody(#require(request.body))
        #expect(body["thinking"] == thinking)
        #expect(body["thinking"]?["block_binding"]?["mismatch_behavior"] == nil)
    }
}
