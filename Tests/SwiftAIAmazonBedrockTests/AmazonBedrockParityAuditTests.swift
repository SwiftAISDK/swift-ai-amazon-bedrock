@testable import SwiftAIAmazonBedrock
@testable import SwiftAIProvider
@testable import SwiftAIProviderUtilities
@testable import SwiftAITransport
import Foundation
import Testing

// Translations of @ai-sdk/amazon-bedrock 5.0.142 tests that cover the
// behavior aligned in the parity audit. Each test names its upstream case.

private let parityModelID = "anthropic.claude-3-haiku-20240307-v1:0"
private let parityAnthropicModelID = "anthropic.claude-3-5-sonnet-20240620-v1:0"
private let parityTextResponse = #"{"output":{"message":{"role":"assistant","content":[{"text":"Hello"}]}},"stopReason":"end_turn","usage":{"inputTokens":4,"outputTokens":34,"totalTokens":38}}"#
private let parityPrompt: [AIMessage] = [.system("System Prompt"), .user("Hello")]

private func parityProvider(_ transport: any AITransport) throws -> AmazonBedrockProvider {
    try AIProviders.amazonBedrock(configuration: AmazonBedrockProviderSettings(
        region: "us-east-1",
        apiKey: "bedrock-key",
        transport: transport
    ))
}

private func parityBody(_ transport: RecordingTransport) async throws -> JSONValue {
    try decodeJSONBody(try #require((await transport.requests()).first?.body))
}

private func parityStream(
    _ events: [(eventType: String, payload: String)],
    modelID: String = parityModelID,
    request: LanguageModelRequest = LanguageModelRequest(messages: parityPrompt),
    headers: [String: String] = [:]
) async throws -> [LanguageStreamPart] {
    var response = amazonEventStreamResponse(events)
    response.headers.merge(headers) { _, new in new }
    let transport = RecordingTransport(response: response)
    let model = try parityProvider(transport).languageModel(modelID)
    var parts: [LanguageStreamPart] = []
    for try await part in model.stream(request) {
        switch part {
        case .requestMetadata, .toolCallDelta: continue
        default: parts.append(part)
        }
    }
    return parts
}

// MARK: - doStream

// "should stream text deltas with metadata and usage": no text-end parts for
// blocks without contentBlockStop and no provider metadata on finish.
@Test func bedrockParityStreamsTextDeltasWithMetadataAndUsage() async throws {
    let parts = try await parityStream([
        ("contentBlockDelta", #"{"contentBlockIndex":0,"delta":{"text":"Hello"}}"#),
        ("contentBlockDelta", #"{"contentBlockIndex":1,"delta":{"text":", "}}"#),
        ("contentBlockDelta", #"{"contentBlockIndex":2,"delta":{"text":"World!"}}"#),
        ("metadata", #"{"usage":{"inputTokens":4,"outputTokens":34,"totalTokens":38},"metrics":{"latencyMs":10}}"#),
        ("messageStop", #"{"stopReason":"stop_sequence"}"#)
    ])

    let usage = TokenUsage(
        inputTokens: 4,
        outputTokens: 34,
        totalTokens: 38,
        inputTokensNoCache: 4,
        inputTokensCacheRead: 0,
        inputTokensCacheWrite: 0,
        outputTextTokens: 34,
        rawValue: ["inputTokens": 4, "outputTokens": 34, "totalTokens": 38]
    )
    #expect(parts == [
        .streamStart(warnings: []),
        .responseMetadata(AIResponseMetadata(modelID: parityModelID, headers: ["content-type": "application/vnd.amazon.eventstream"])),
        .textStart(id: "0"),
        .textDeltaPart(id: "0", delta: "Hello"),
        .textStart(id: "1"),
        .textDeltaPart(id: "1", delta: ", "),
        .textStart(id: "2"),
        .textDeltaPart(id: "2", delta: "World!"),
        .finishMetadata(reason: "stop", usage: usage, providerMetadata: [:])
    ])
}

// "should stream tool deltas"
@Test func bedrockParityStreamsToolDeltas() async throws {
    let parts = try await parityStream([
        ("contentBlockStart", #"{"contentBlockIndex":0,"start":{"toolUse":{"toolUseId":"tool-use-id","name":"test-tool"}}}"#),
        ("contentBlockDelta", #"{"contentBlockIndex":0,"delta":{"toolUse":{"input":"{\"value\":"}}}"#),
        ("contentBlockDelta", #"{"contentBlockIndex":0,"delta":{"toolUse":{"input":"\"Sparkle Day\"}"}}}"#),
        ("contentBlockStop", #"{"contentBlockIndex":0}"#),
        ("messageStop", #"{"stopReason":"tool_use"}"#)
    ], request: LanguageModelRequest(
        messages: parityPrompt,
        tools: ["test-tool": [
            "type": "object",
            "properties": ["value": ["type": "string"]],
            "required": ["value"],
            "additionalProperties": false
        ]],
        toolChoice: ["type": "tool", "toolName": "test-tool"]
    ))

    #expect(Array(parts.dropFirst(2)) == [
        .toolInputStart(id: "tool-use-id", name: "test-tool"),
        .toolInputDelta(id: "tool-use-id", delta: #"{"value":"#),
        .toolInputDelta(id: "tool-use-id", delta: #""Sparkle Day"}"#),
        .toolInputEnd(id: "tool-use-id"),
        .toolCall(AIToolCall(
            id: "tool-use-id",
            name: "test-tool",
            arguments: #"{"value":"Sparkle Day"}"#,
            rawValue: ["toolUseId": "tool-use-id", "name": "test-tool"]
        )),
        .finishMetadata(reason: "tool-calls", usage: nil, providerMetadata: [:])
    ])
}

// "should include stop_sequence in provider metadata" (stream)
@Test func bedrockParityStreamIncludesStopSequenceInProviderMetadata() async throws {
    let parts = try await parityStream([
        ("contentBlockDelta", #"{"contentBlockIndex":0,"delta":{"text":"Hello"}}"#),
        ("metadata", #"{"usage":{"inputTokens":4,"outputTokens":34,"totalTokens":38}}"#),
        ("messageStop", #"{"stopReason":"stop_sequence","additionalModelResponseFields":{"delta":{"stop_sequence":"STOP"},"otherField":"value"}}"#)
    ], request: LanguageModelRequest(messages: parityPrompt, stopSequences: ["STOP"]))

    guard case let .finishMetadata(reason, _, metadata)? = parts.last else {
        Issue.record("missing finish part")
        return
    }
    #expect(reason == "stop")
    #expect(metadata == [
        "amazonBedrock": ["stopSequence": "STOP"],
        "bedrock": ["stopSequence": "STOP"]
    ])
}

// "should include cache token usage in providerMetadata" (stream)
@Test func bedrockParityStreamIncludesCacheTokenUsage() async throws {
    let parts = try await parityStream([
        ("contentBlockDelta", #"{"contentBlockIndex":0,"delta":{"text":"Hello"}}"#),
        ("metadata", #"{"usage":{"inputTokens":4,"outputTokens":34,"totalTokens":38,"cacheReadInputTokens":2,"cacheWriteInputTokens":3}}"#),
        ("messageStop", #"{"stopReason":"stop_sequence"}"#)
    ])

    guard case let .finishMetadata(_, usage, metadata)? = parts.last else {
        Issue.record("missing finish part")
        return
    }
    #expect(usage?.inputTokens == 9)
    #expect(usage?.inputTokensNoCache == 4)
    #expect(usage?.inputTokensCacheRead == 2)
    #expect(usage?.inputTokensCacheWrite == 3)
    #expect(usage?.outputTokens == 34)
    #expect(metadata == [
        "amazonBedrock": ["usage": ["cacheWriteInputTokens": 3]],
        "bedrock": ["usage": ["cacheWriteInputTokens": 3]]
    ])
    #expect(!parts.contains { if case .metadata = $0 { return true } else { return false } })
}

// "should include performanceConfig/serviceTier in providerMetadata" (stream)
@Test func bedrockParityStreamIncludesPerformanceConfigAndServiceTier() async throws {
    let parts = try await parityStream([
        ("messageStop", #"{"stopReason":"end_turn"}"#),
        ("metadata", #"{"usage":{"inputTokens":1,"outputTokens":1,"totalTokens":2},"performanceConfig":{"latency":"optimized"},"serviceTier":{"type":"priority"}}"#)
    ])
    guard case let .finishMetadata(_, _, metadata)? = parts.last else {
        Issue.record("missing finish part")
        return
    }
    #expect(metadata["amazonBedrock"] == [
        "performanceConfig": ["latency": "optimized"],
        "serviceTier": ["type": "priority"]
    ])
}

// "should extract response metadata" (stream)
@Test func bedrockParityStreamExtractsResponseMetadata() async throws {
    let parts = try await parityStream([
        ("contentBlockDelta", #"{"contentBlockIndex":0,"delta":{"text":"Hello"}}"#),
        ("messageStop", #"{"stopReason":"end_turn"}"#)
    ], headers: ["x-amzn-requestid": "test-request-id", "date": "Wed, 01 Jan 2025 00:00:00 GMT"])

    guard case let .responseMetadata(metadata)? = parts.first(where: {
        if case .responseMetadata = $0 { return true } else { return false }
    }) else {
        Issue.record("missing response metadata")
        return
    }
    #expect(metadata.id == "test-request-id")
    #expect(metadata.modelID == parityModelID)
    #expect(metadata.timestamp == Date(timeIntervalSince1970: 1_735_689_600))
}

// "should include raw chunks when includeRawChunks is true": raw chunks are the
// event-type-wrapped payload without the `p` padding field.
@Test func bedrockParityRawChunksDropPaddingField() async throws {
    let transport = RecordingTransport(response: amazonEventStreamResponse([
        ("contentBlockDelta", #"{"contentBlockIndex":0,"delta":{"text":"Hi"},"p":"abcdef"}"#),
        ("messageStop", #"{"stopReason":"end_turn","p":"abc"}"#)
    ]))
    let model = try parityProvider(transport).languageModel(parityModelID)
    var raw: [JSONValue] = []
    for try await part in model.stream(LanguageModelRequest(messages: parityPrompt, includeRawChunks: true)) {
        if case let .raw(value) = part { raw.append(value) }
    }
    #expect(raw == [
        ["contentBlockDelta": ["contentBlockIndex": 0, "delta": ["text": "Hi"]]],
        ["messageStop": ["stopReason": "end_turn"]]
    ])
}

// "should handle JSON response format in streaming": the JSON tool emits text
// at block stop and finish metadata carries isJsonResponseFromTool.
@Test func bedrockParityStreamTextBeforeJSONToolIsKept() async throws {
    let parts = try await parityStream([
        ("contentBlockDelta", #"{"contentBlockIndex":0,"delta":{"text":"Here it is"}}"#),
        ("contentBlockStop", #"{"contentBlockIndex":0}"#),
        ("contentBlockStart", #"{"contentBlockIndex":1,"start":{"toolUse":{"toolUseId":"json-1","name":"json"}}}"#),
        ("contentBlockDelta", #"{"contentBlockIndex":1,"delta":{"toolUse":{"input":"{\"a\":1}"}}}"#),
        ("contentBlockStop", #"{"contentBlockIndex":1}"#),
        ("messageStop", #"{"stopReason":"tool_use"}"#)
    ], modelID: "mistral.mistral-large-2402-v1:0", request: LanguageModelRequest(
        messages: [.user("Hi")],
        responseFormat: .json(schema: ["type": "object", "properties": ["a": ["type": "number"]]])
    ))
    #expect(Array(parts.dropFirst(2)) == [
        .textStart(id: "0"),
        .textDeltaPart(id: "0", delta: "Here it is"),
        .textEnd(id: "0"),
        .textStart(id: "1"),
        .textDeltaPart(id: "1", delta: #"{"a":1}"#),
        .textEnd(id: "1"),
        .finishMetadata(reason: "stop", usage: nil, providerMetadata: [
            "amazonBedrock": ["isJsonResponseFromTool": true, "stopSequence": .null],
            "bedrock": ["isJsonResponseFromTool": true, "stopSequence": .null]
        ])
    ])
}

// MARK: - doGenerate

// "should extract usage" / "should send additional response information" /
// "should extract response metadata"
@Test func bedrockParityGenerateExtractsUsageAndResponseMetadata() async throws {
    var response = jsonResponse(#"{"output":{"message":{"role":"assistant","content":[{"text":"Hi"}]}},"stopReason":"end_turn","usage":{"inputTokens":22,"outputTokens":57,"totalTokens":79,"cacheReadInputTokens":0,"cacheWriteInputTokens":0,"cacheReadInputTokenCount":0,"cacheWriteInputTokenCount":0,"serverToolUsage":{}}}"#)
    response.headers["x-amzn-requestid"] = "test-request-id"
    response.headers["date"] = "Wed, 01 Jan 2025 00:00:00 GMT"
    let transport = RecordingTransport(response: response)
    let result = try await parityProvider(transport).languageModel(parityModelID)
        .generate(LanguageModelRequest(messages: parityPrompt))

    #expect(result.usage?.inputTokens == 22)
    #expect(result.usage?.inputTokensNoCache == 22)
    #expect(result.usage?.inputTokensCacheRead == 0)
    #expect(result.usage?.inputTokensCacheWrite == 0)
    #expect(result.usage?.outputTextTokens == 57)
    #expect(result.usage?.rawValue?["serverToolUsage"] == [:])
    #expect(result.responseMetadata.id == "test-request-id")
    #expect(result.responseMetadata.modelID == parityModelID)
    #expect(result.responseMetadata.timestamp == Date(timeIntervalSince1970: 1_735_689_600))
    // A zero cacheWriteInputTokens is non-null, so upstream still reports it.
    #expect(result.providerMetadata == [
        "amazonBedrock": ["usage": ["cacheWriteInputTokens": 0], "stopSequence": .null],
        "bedrock": ["usage": ["cacheWriteInputTokens": 0], "stopSequence": .null]
    ])
}

// "should handle stop_sequence: null when stopReason is tool_use"
@Test func bedrockParityGenerateHandlesNullStopSequence() async throws {
    let transport = RecordingTransport(response: jsonResponse(#"{"output":{"message":{"role":"assistant","content":[{"text":"I'll query your Salesforce..."},{"toolUse":{"toolUseId":"tool-use-id","name":"querySalesforce","input":{"query":"SELECT Id FROM Account"}}}]}},"stopReason":"tool_use","additionalModelResponseFields":{"delta":{"stop_sequence":null}},"usage":{"inputTokens":10,"outputTokens":20,"totalTokens":30}}"#))
    let result = try await parityProvider(transport).languageModel(parityModelID).generate(LanguageModelRequest(
        messages: parityPrompt,
        tools: ["querySalesforce": [
            "description": "Query Salesforce",
            "type": "object",
            "properties": ["query": ["type": "string"]],
            "required": ["query"],
            "additionalProperties": false
        ]]
    ))
    #expect(result.finishReason == "tool-calls")
    #expect(result.toolCalls.first?.arguments == #"{"query":"SELECT Id FROM Account"}"#)
    #expect(result.providerMetadata == [
        "amazonBedrock": ["stopSequence": .null],
        "bedrock": ["stopSequence": .null]
    ])
    let body = try await parityBody(transport)
    // The tool description is sent beside the schema, not inside it.
    #expect(body["toolConfig"]?["tools"]?[0]?["toolSpec"] == [
        "name": "querySalesforce",
        "description": "Query Salesforce",
        "inputSchema": ["json": [
            "type": "object",
            "properties": ["query": ["type": "string"]],
            "required": ["query"],
            "additionalProperties": false
        ]]
    ])
}

// "should include cache token usage in providerMetadata" / "cacheDetails" (generate)
@Test func bedrockParityGenerateIncludesCacheUsageInProviderMetadata() async throws {
    let transport = RecordingTransport(response: jsonResponse(#"{"output":{"message":{"role":"assistant","content":[{"text":"Hi"}]}},"stopReason":"end_turn","usage":{"inputTokens":4,"outputTokens":34,"totalTokens":38,"cacheReadInputTokens":2,"cacheWriteInputTokens":3,"cacheDetails":[{"inputTokens":3,"ttl":"5m"}]}}"#))
    let result = try await parityProvider(transport).languageModel(parityModelID)
        .generate(LanguageModelRequest(messages: parityPrompt))
    #expect(result.providerMetadata["amazonBedrock"] == [
        "usage": ["cacheWriteInputTokens": 3, "cacheDetails": [["inputTokens": 3, "ttl": "5m"]]],
        "stopSequence": .null
    ])
    #expect(result.usage?.inputTokens == 9)
}

// "should return the request body": Anthropic models request the stop
// sequence response field.
@Test func bedrockParityRequestsStopSequenceFieldForAnthropicModels() async throws {
    let transport = RecordingTransport(response: jsonResponse(parityTextResponse))
    _ = try await parityProvider(transport).languageModel(parityModelID)
        .generate(LanguageModelRequest(messages: parityPrompt))
    #expect(try await parityBody(transport) == [
        "additionalModelResponseFieldPaths": ["/delta/stop_sequence"],
        "messages": [["role": "user", "content": [["text": "Hello"]]]],
        "system": [["text": "System Prompt"]]
    ])

    let novaTransport = RecordingTransport(response: jsonResponse(parityTextResponse))
    _ = try await parityProvider(novaTransport).languageModel("us.amazon.nova-pro-v1:0")
        .generate(LanguageModelRequest(messages: parityPrompt))
    #expect(try await parityBody(novaTransport)["additionalModelResponseFieldPaths"] == nil)
}

// "should handle Anthropic provider-defined tools"
@Test func bedrockParityAnthropicProviderToolsSendBetaAndFactorySchema() async throws {
    let transport = RecordingTransport(response: jsonResponse(#"{"output":{"message":{"role":"assistant","content":[{"toolUse":{"toolUseId":"tool-use-id","name":"bash","input":{"command":"ls -l"}}}]}},"usage":{"inputTokens":10,"outputTokens":20,"totalTokens":30},"stopReason":"tool_use"}"#))
    let result = try await parityProvider(transport).languageModel(parityAnthropicModelID).generate(LanguageModelRequest(
        messages: parityPrompt,
        tools: ["bash": ["type": "provider", "id": "anthropic.bash_20241022", "name": "bash", "args": [:]]],
        toolChoice: ["type": "auto"]
    ))

    let body = try await parityBody(transport)
    #expect(body["additionalModelRequestFields"] == [
        "tool_choice": ["type": "auto"],
        "anthropic_beta": ["computer-use-2024-10-22"]
    ])
    #expect(body["toolConfig"]?["tools"]?.arrayValue?.count == 1)
    #expect(body["toolConfig"]?["tools"]?[0]?["toolSpec"]?["name"] == "bash")
    #expect(body["toolConfig"]?["tools"]?[0]?["toolSpec"]?["inputSchema"]?["json"] == [
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": ["command": ["type": "string"], "restart": ["type": "boolean"]],
        "required": ["command"],
        "additionalProperties": false
    ])
    #expect(result.warnings.isEmpty)
    #expect(result.toolCalls == [AIToolCall(
        id: "tool-use-id",
        name: "bash",
        arguments: #"{"command":"ls -l"}"#,
        rawValue: ["toolUse": ["toolUseId": "tool-use-id", "name": "bash", "input": ["command": "ls -l"]]]
    )])
    // "should not include anthropic-beta in HTTP headers"
    #expect(await transport.requests().first?.headers["anthropic-beta"] == nil)
}

// "should include anthropic_beta in additionalModelRequestFields when using
// extended context" and "should combine user-provided and tool-generated betas".
@Test func bedrockParityAnthropicBetaProviderOption() async throws {
    let transport = RecordingTransport(response: jsonResponse(parityTextResponse))
    let model = try parityProvider(transport).languageModel(parityAnthropicModelID)
    _ = try await model.generate(LanguageModelRequest(
        messages: parityPrompt,
        providerOptions: ["bedrock": ["anthropicBeta": ["context-1m-2025-08-07"]]]
    ))
    #expect(try await parityBody(transport)["additionalModelRequestFields"] == [
        "anthropic_beta": ["context-1m-2025-08-07"]
    ])

    let combined = RecordingTransport(response: jsonResponse(parityTextResponse))
    _ = try await parityProvider(combined).languageModel(parityAnthropicModelID).generate(LanguageModelRequest(
        messages: parityPrompt,
        tools: ["bash": ["type": "provider", "id": "anthropic.bash_20241022", "name": "bash", "args": [:]]],
        providerOptions: ["bedrock": ["anthropicBeta": ["context-1m-2025-08-07"]]]
    ))
    #expect(try await parityBody(combined)["additionalModelRequestFields"]?["anthropic_beta"] == [
        "context-1m-2025-08-07",
        "computer-use-2024-10-22"
    ])
}

// "should prefer `amazonBedrock` providerOptions over the legacy `bedrock` key"
// and "should support guardrails": the first namespace wins and its remaining
// keys are spread into the body.
@Test func bedrockParityPrefersAmazonBedrockNamespaceAndSpreadsRemainingKeys() async throws {
    let transport = RecordingTransport(response: jsonResponse(parityTextResponse))
    _ = try await parityProvider(transport).languageModel(parityModelID).generate(LanguageModelRequest(
        messages: parityPrompt,
        providerOptions: [
            "bedrock": [
                "guardrailConfig": ["guardrailIdentifier": "legacy", "guardrailVersion": "1"],
                "performanceConfig": ["latency": "standard"]
            ],
            "amazonBedrock": [
                "guardrailConfig": ["guardrailIdentifier": "preferred", "guardrailVersion": "2"],
                "promptVariables": ["topic": ["text": "x"]]
            ]
        ]
    ))
    let body = try await parityBody(transport)
    #expect(body["guardrailConfig"] == ["guardrailIdentifier": "preferred", "guardrailVersion": "2"])
    #expect(body["promptVariables"] == ["topic": ["text": "x"]])
    #expect(body["performanceConfig"] == nil)
}

// "should omit toolConfig and filter tool content when conversation has tool
// calls but no active tools"
@Test func bedrockParityFiltersToolContentWithoutActiveTools() async throws {
    let transport = RecordingTransport(response: jsonResponse(parityTextResponse))
    let result = try await parityProvider(transport).languageModel(parityModelID).generate(LanguageModelRequest(messages: [
        .user("What is the weather in Toronto?"),
        AIMessage(role: .assistant, content: [
            .toolCall(AIToolCall(id: "tool-call-1", name: "weather", arguments: #"{"city":"Toronto"}"#))
        ]),
        AIMessage(role: .tool, content: [
            .toolResult(AIToolResult(
                toolCallID: "tool-call-1",
                toolName: "weather",
                result: "The weather in Toronto is 20°C.",
                modelOutput: ["type": "text", "value": "The weather in Toronto is 20°C."]
            ))
        ]),
        .user("Now give me a summary.")
    ]))

    let body = try await parityBody(transport)
    #expect(body["toolConfig"] == nil)
    #expect(body["messages"] == [[
        "role": "user",
        "content": [["text": "What is the weather in Toronto?"], ["text": "Now give me a summary."]]
    ]])
    #expect(result.warnings == [AIWarning(
        type: "unsupported",
        feature: "toolContent",
        message: "Tool calls and results removed from conversation because Bedrock does not support tool content without active tools."
    )])
}

// "should omit toolConfig when conversation has tool calls but toolChoice is none"
@Test func bedrockParityToolChoiceNoneOmitsToolConfigAndFiltersToolContent() async throws {
    let transport = RecordingTransport(response: jsonResponse(parityTextResponse))
    _ = try await parityProvider(transport).languageModel("us.amazon.nova-pro-v1:0").generate(LanguageModelRequest(
        messages: [
            .user("Weather?"),
            AIMessage(role: .assistant, content: [
                .toolCall(AIToolCall(id: "call-1", name: "weather", arguments: "{}"))
            ]),
            AIMessage(role: .tool, content: [
                .toolResult(AIToolResult(toolCallID: "call-1", toolName: "weather", result: "sunny"))
            ])
        ],
        tools: ["weather": ["type": "object", "properties": [:]]],
        toolChoice: ["type": "none"]
    ))
    let body = try await parityBody(transport)
    #expect(body["toolConfig"] == nil)
    #expect(body["messages"] == [["role": "user", "content": [["text": "Weather?"]]]])
}

// "should clamp temperature above 1 to 1 and add warning": warnings follow
// the upstream order (unsupported settings before clamping).
@Test func bedrockParityWarningOrderMatchesUpstream() async throws {
    let transport = RecordingTransport(response: jsonResponse(parityTextResponse))
    let result = try await parityProvider(transport).languageModel(parityModelID).generate(LanguageModelRequest(
        messages: parityPrompt,
        temperature: 1.5,
        frequencyPenalty: 0.5,
        seed: 1
    ))
    #expect(result.warnings == [
        AIWarning(type: "unsupported", feature: "frequencyPenalty"),
        AIWarning(type: "unsupported", feature: "seed"),
        AIWarning(type: "unsupported", feature: "temperature", message: "1.5 exceeds bedrock maximum of 1.0. clamped to 1.0")
    ])
    #expect(try await parityBody(transport)["inferenceConfig"] == ["temperature": 1])
}

// "should handle JSON response format with schema": the JSON tool uses the
// upstream description and the response schema unchanged.
@Test func bedrockParityJSONResponseToolDefinition() async throws {
    let transport = RecordingTransport(response: jsonResponse(#"{"output":{"message":{"role":"assistant","content":[{"toolUse":{"toolUseId":"t","name":"json","input":{"a":1}}}]}},"stopReason":"tool_use","usage":{"inputTokens":1,"outputTokens":1,"totalTokens":2}}"#))
    let schema: JSONValue = ["type": "object", "description": "Answer", "properties": ["a": ["type": "number"]]]
    let result = try await parityProvider(transport).languageModel("us.amazon.nova-pro-v1:0").generate(LanguageModelRequest(
        messages: parityPrompt,
        responseFormat: .json(schema: schema)
    ))
    let body = try await parityBody(transport)
    #expect(body["toolConfig"] == [
        "tools": [["toolSpec": [
            "name": "json",
            "description": "Respond with a JSON object.",
            "inputSchema": ["json": schema]
        ]]],
        "toolChoice": ["any": [:]]
    ])
    #expect(result.text == #"{"a":1}"#)
    #expect(result.finishReason == "stop")
    #expect(result.providerMetadata["amazonBedrock"] == ["isJsonResponseFromTool": true, "stopSequence": .null])
}

// MARK: - convertToAmazonBedrockChatMessages

// "should combine multiple leading system messages into a single system message"
@Test func bedrockParityCombinesLeadingSystemMessages() throws {
    let result = try bedrockConvertMessages([.system("Hello"), .system("World")], modelID: parityModelID)
    #expect(result.system == [["text": "Hello"], ["text": "World"]])
    #expect(result.messages.isEmpty)
}

// "should throw an error if a system message is provided after a non-system message"
@Test func bedrockParityRejectsSystemMessageAfterConversation() {
    #expect(throws: AIError.self) {
        try bedrockConvertMessages([.user("Hello"), .system("World")], modelID: parityModelID)
    }
}

// "should remove trailing whitespace from last assistant message when there is
// no further user message" (also multi-part) and "should keep trailing
// whitespace from assistant message when there is a further user message"
@Test func bedrockParityTrimsOnlyTheFinalAssistantPrefill() throws {
    let trimmed = try bedrockConvertMessages([
        .user("user content"),
        AIMessage(role: .assistant, content: [.text("assistant "), .text("content  ")])
    ], modelID: parityModelID)
    #expect(trimmed.messages == [
        ["role": "user", "content": [["text": "user content"]]],
        ["role": "assistant", "content": [["text": "assistant "], ["text": "content"]]]
    ])

    let kept = try bedrockConvertMessages([
        .user("user content"),
        .assistant("assistant content  "),
        .user("user content 2")
    ], modelID: parityModelID)
    #expect(kept.messages == [
        ["role": "user", "content": [["text": "user content"]]],
        ["role": "assistant", "content": [["text": "assistant content  "]]],
        ["role": "user", "content": [["text": "user content 2"]]]
    ])
}

// "should combine multiple sequential assistant messages into a single message"
@Test func bedrockParityCombinesSequentialAssistantMessages() throws {
    let result = try bedrockConvertMessages([
        .user("Hi!"),
        .assistant("Hello"),
        .assistant("World"),
        .assistant("!")
    ], modelID: parityModelID)
    #expect(result.messages == [
        ["role": "user", "content": [["text": "Hi!"]]],
        ["role": "assistant", "content": [["text": "Hello"], ["text": "World"], ["text": "!"]]]
    ])
}

// Upstream groups consecutive user and tool messages into one user message.
@Test func bedrockParityCombinesSequentialUserAndToolMessages() throws {
    let result = try bedrockConvertMessages([
        .user("first"),
        .user("second"),
        AIMessage(role: .tool, content: [
            .toolResult(AIToolResult(toolCallID: "call-1", toolName: "t", result: "ok"))
        ])
    ], modelID: parityModelID)
    #expect(result.messages == [[
        "role": "user",
        "content": [
            ["text": "first"],
            ["text": "second"],
            ["toolResult": ["toolUseId": "call-1", "content": [["text": "ok"]]]]
        ]
    ]])
}

// "should filter out empty text blocks in assistant messages"
@Test func bedrockParityFiltersWhitespaceOnlyAssistantText() throws {
    let result = try bedrockConvertMessages([
        .user("Hello"),
        AIMessage(role: .assistant, content: [
            .text("\n\n"),
            .toolCall(AIToolCall(id: "call-123", name: "test", arguments: "{}")),
            .text("  "),
            .text("actual content")
        ])
    ], modelID: parityModelID)
    #expect(result.messages == [
        ["role": "user", "content": [["text": "Hello"]]],
        ["role": "assistant", "content": [
            ["toolUse": ["toolUseId": "call-123", "name": "test", "input": [:]]],
            ["text": "actual content"]
        ]]
    ])
}

// "should fallback to stringified result when content is undefined": JSON
// tool output is sent as text, and tool results carry no status field.
@Test func bedrockParityStringifiesJSONToolResults() throws {
    let typed = try bedrockConvertMessages([
        AIMessage(role: .tool, content: [
            .toolResult(AIToolResult(
                toolCallID: "call-123",
                toolName: "calculator",
                result: ["value": 42],
                modelOutput: ["type": "json", "value": ["value": 42]]
            ))
        ])
    ], modelID: parityModelID)
    #expect(typed.messages[0] == [
        "role": "user",
        "content": [["toolResult": ["toolUseId": "call-123", "content": [["text": #"{"value":42}"#]]]]]
    ])

    let untyped = try bedrockConvertMessages([
        AIMessage(role: .tool, content: [
            .toolResult(AIToolResult(toolCallID: "call-123", toolName: "calculator", result: ["value": 42], isError: true))
        ])
    ], modelID: parityModelID)
    #expect(untyped.messages[0]["content"]?[0]?["toolResult"] == [
        "toolUseId": "call-123",
        "content": [["text": #"{"value":42}"#]]
    ])
}

// "should throw for file parts with provider references"
@Test func bedrockParityRejectsProviderReferenceFileParts() {
    #expect(throws: AIError.self) {
        try bedrockConvertMessages([
            AIMessage(role: .user, content: [
                .providerReference(mimeType: "application/pdf", reference: ["amazonBedrock": "file-1"])
            ])
        ], modelID: parityModelID)
    }
}

// Parts upstream ignores in assistant messages are not replaced by blank text.
@Test func bedrockParityIgnoresUnsupportedAssistantParts() throws {
    let result = try bedrockConvertMessages([
        .user("Hi"),
        AIMessage(role: .assistant, content: [
            .custom(["kind": "x"]),
            .text("answer")
        ]),
        .user("next")
    ], modelID: parityModelID)
    #expect(result.messages[1] == ["role": "assistant", "content": [["text": "answer"]]])
}
