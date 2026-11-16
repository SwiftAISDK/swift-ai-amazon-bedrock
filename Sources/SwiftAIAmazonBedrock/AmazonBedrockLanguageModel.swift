import SwiftAIOpenAICompatible
import SwiftAIOpenAI
import SwiftAIAnthropic
import Foundation
import SwiftAIProvider
import SwiftAITransport
import SwiftAIProviderUtilities


public final class AmazonBedrockLanguageModel: LanguageModel, @unchecked Sendable {
    public let providerID = "amazon-bedrock"
    public let modelID: String
    public let supportedURLs: [String: [AISupportedURLPattern]] = [
        "image/*": [AISupportedURLPattern(bedrockIsS3URL)]
    ]
    private let config: BedrockRuntimeConfig
    private let settings: AmazonBedrockChatModelSettings

    init(
        modelID: String,
        config: BedrockRuntimeConfig,
        settings: AmazonBedrockChatModelSettings = AmazonBedrockChatModelSettings()
    ) {
        self.modelID = modelID
        self.config = config
        self.settings = settings
    }

    public func generate(_ request: LanguageModelRequest) async throws -> TextGenerationResult {
        let prepared = try converseBody(for: request)
        let (raw, httpResponse) = try await config.sendJSONResponse(path: "/model/\(encodedModelID)/converse", body: prepared.body, headers: request.headers, abortSignal: request.abortSignal, errorIncludesType: false)
        let responseBlocks = raw["output"]?["message"]?["content"]?.arrayValue ?? []
        let jsonTextExtractor = prepared.usesJsonInstruction ? BedrockJSONObjectTextExtractor() : nil
        var content: [AIResultContentPart] = []
        var text = ""
        var reasoning = ""
        var toolCalls: [AIToolCall] = []
        var isJsonResponseFromTool = false

        for (index, block) in responseBlocks.enumerated() {
            let textParts: [String]
            if let rawText = block["text"]?.stringValue {
                textParts = [rawText]
            } else {
                textParts = block["citationsContent"]?["content"]?.arrayValue?
                    .compactMap { $0["text"]?.stringValue } ?? []
            }
            if !textParts.isEmpty {
                for rawText in textParts {
                    let parsedText = jsonTextExtractor?.process(rawText) ?? rawText
                    text += parsedText
                    content.append(.text(parsedText))
                }
                continue
            }

            if let reasoningContent = block["reasoningContent"] {
                if let reasoningText = reasoningContent["reasoningText"] {
                    let value = reasoningText["text"]?.stringValue ?? ""
                    reasoning += value
                    let signature = reasoningText["signature"]?.stringValue
                    let metadata = bedrockReasoningProviderMetadata(
                        key: "signature",
                        value: signature?.isEmpty == false ? signature.map(JSONValue.string) : nil
                    )
                    content.append(.reasoning(value, providerMetadata: metadata))
                    continue
                }
                if let redactedReasoning = bedrockNonNull(reasoningContent["redactedReasoning"]) {
                    content.append(.reasoning(
                        "",
                        providerMetadata: bedrockReasoningProviderMetadata(
                            key: "redactedData",
                            value: bedrockNonNull(redactedReasoning["data"]) ?? .string("")
                        )
                    ))
                    continue
                }
                if let redactedContent = reasoningContent["redactedContent"] {
                    content.append(.reasoning(
                        "",
                        providerMetadata: bedrockReasoningProviderMetadata(key: "redactedContent", value: redactedContent)
                    ))
                    continue
                }
            }

            if let toolCall = bedrockToolCall(from: block, index: index, modelID: modelID) {
                if prepared.usesJsonResponseTool, toolCall.name == "json" {
                    text += toolCall.arguments
                    content.append(.text(toolCall.arguments))
                    isJsonResponseFromTool = true
                } else {
                    toolCalls.append(toolCall)
                    content.append(.toolCall(toolCall))
                }
            }
        }

        return TextGenerationResult(
            text: text,
            content: content,
            reasoning: reasoning,
            finishReason: bedrockFinishReason(raw["stopReason"]?.stringValue, isJsonResponseFromTool: isJsonResponseFromTool),
            usage: bedrockUsage(from: raw["usage"]),
            toolCalls: toolCalls,
            providerMetadata: bedrockProviderMetadata(from: raw, isJsonResponseFromTool: isJsonResponseFromTool),
            rawValue: raw,
            warnings: prepared.warnings,
            requestMetadata: AIRequestMetadata(body: prepared.body, headers: request.headers, url: try config.request(path: "/model/\(encodedModelID)/converse", body: prepared.body, headers: request.headers, abortSignal: request.abortSignal).url.absoluteString),
            responseMetadata: bedrockResponseMetadata(headers: httpResponse.headers, modelID: modelID, body: raw)
        )
    }

    public func stream(_ request: LanguageModelRequest) -> AsyncThrowingStream<LanguageStreamPart, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    let prepared = try converseBody(for: request)
                    let httpRequest = try config.request(
                        path: "/model/\(encodedModelID)/converse-stream",
                        body: prepared.body,
                        headers: request.headers.mergingHeaders(["accept": "application/vnd.amazon.eventstream"]),
                        abortSignal: request.abortSignal
                    )
                    let response = try await config.streamRequest(httpRequest)
                    continuation.yield(.requestMetadata(AIRequestMetadata(body: prepared.body, headers: request.headers, url: httpRequest.url.absoluteString)))
                    try await streamFromBedrockResponse(
                        providerID: providerID,
                        response: response,
                        requestURL: httpRequest.url,
                        maxResponseBytes: httpRequest.maxResponseBytes,
                        includeRawChunks: request.includeRawChunks,
                        warnings: prepared.warnings,
                        jsonResponseToolName: prepared.usesJsonResponseTool ? "json" : nil,
                        modelID: modelID,
                        extractJSONObjectText: prepared.usesJsonInstruction,
                        emit: { part in
                            continuation.yield(part)
                        }
                    )
                    if !Task.isCancelled {
                        continuation.finish()
                    }
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in
                task.cancel()
            }
        }
    }

    private var encodedModelID: String {
        bedrockEncodeModelID(modelID)
    }

    private func bedrockReasoningProviderMetadata(key: String, value: JSONValue?) -> [String: JSONValue] {
        guard let value else { return [:] }
        let payload: JSONValue = .object([key: value])
        return [
            "amazonBedrock": payload,
            "bedrock": payload
        ]
    }

    private func converseBody(for request: LanguageModelRequest) throws -> BedrockPreparedConverseCall {
        var providerOptions = try bedrockRequestProviderOptions(providerOptions: request.providerOptions, extraBody: request.extraBody)
        // Upstream keeps the raw, unparsed namespace and spreads its remaining
        // keys into the Converse body (for example guardrailConfig).
        let rawNamespaceOptions = providerOptions
        var warnings: [AIWarning] = []
        if request.frequencyPenalty != nil {
            warnings.append(AIWarning(type: "unsupported", feature: "frequencyPenalty"))
        }
        if request.presencePenalty != nil {
            warnings.append(AIWarning(type: "unsupported", feature: "presencePenalty"))
        }
        if request.seed != nil {
            warnings.append(AIWarning(type: "unsupported", feature: "seed"))
        }
        let capabilities = anthropicModelCapabilities(modelID)
        let openAIModelID = bedrockOpenAIModelID(modelID)
        let isOpenAIModel = openAIModelID != nil
        let isOpenAIGptOssModel = openAIModelID?.hasPrefix("openai.gpt-oss-") == true
        var temperature = request.temperature
        var topP = request.topP
        var topK = request.topK
        if capabilities.rejectsSamplingParameters {
            if temperature != nil {
                warnings.append(AIWarning(type: "unsupported", feature: "temperature", message: "temperature is not supported by \(modelID) and will be ignored"))
                temperature = nil
            }
            if topK != nil {
                warnings.append(AIWarning(type: "unsupported", feature: "topK", message: "topK is not supported by \(modelID) and will be ignored"))
                topK = nil
            }
            if topP != nil {
                warnings.append(AIWarning(type: "unsupported", feature: "topP", message: "topP is not supported by \(modelID) and will be ignored"))
                topP = nil
            }
        }
        if !isOpenAIModel || isOpenAIGptOssModel, let value = temperature {
            if value > 1 {
                warnings.append(AIWarning(type: "unsupported", feature: "temperature", message: "\(bedrockJSNumber(value)) exceeds bedrock maximum of 1.0. clamped to 1.0"))
                temperature = 1
            } else if value < 0 {
                warnings.append(AIWarning(type: "unsupported", feature: "temperature", message: "\(bedrockJSNumber(value)) is below bedrock minimum of 0. clamped to 0"))
                temperature = 0
            }
        }

        let explicitReasoningEffort = providerOptions["reasoningConfig"]?["maxReasoningEffort"]?.stringValue
        let explicitlyDisabled = providerOptions["reasoningConfig"]?["type"]?.stringValue == "disabled"
        bedrockApplyTopLevelReasoning(
            request.reasoning,
            modelID: modelID,
            modelFamily: settings.modelFamily,
            maxOutputTokens: request.maxOutputTokens,
            providerOptions: &providerOptions,
            warnings: &warnings
        )
        bedrockNormalizeDisabledThinking(
            modelID: modelID,
            modelFamily: settings.modelFamily,
            explicitReasoningEffort: explicitReasoningEffort,
            explicitlyDisabled: explicitlyDisabled,
            providerOptions: &providerOptions,
            warnings: &warnings
        )
        let enableDocumentCitations = bedrockDocumentCitationsEnabled(providerOptions)
        var effectiveTools = request.tools
        var effectiveToolChoice = request.toolChoice ?? providerOptions["toolChoice"] ?? request.extraBody["toolChoice"]
        let responseJSONSchema = bedrockResponseJSONSchema(from: request.responseFormat)
        let isAnthropicModel = bedrockIsAnthropicModel(
            modelID: modelID,
            modelFamily: settings.modelFamily,
            reasoningConfig: providerOptions["reasoningConfig"]
        )
        let structuredOutputMode = providerOptions["structuredOutputMode"]?.stringValue
            ?? request.providerOptions["anthropic"]?["structuredOutputMode"]?.stringValue
            ?? request.extraBody["anthropic"]?["structuredOutputMode"]?.stringValue
            ?? "auto"
        providerOptions.removeValue(forKey: "structuredOutputMode")

        if structuredOutputMode == "jsonTool",
           var additionalModelRequestFields = providerOptions["additionalModelRequestFields"]?.objectValue,
           var outputConfig = additionalModelRequestFields["output_config"]?.objectValue {
            outputConfig.removeValue(forKey: "format")
            if outputConfig.isEmpty {
                additionalModelRequestFields.removeValue(forKey: "output_config")
            } else {
                additionalModelRequestFields["output_config"] = .object(outputConfig)
            }
            providerOptions["additionalModelRequestFields"] = .object(additionalModelRequestFields)
        }

        let modelSupportsNativeStructuredOutput = bedrockSupportsNativeStructuredOutput(modelID: modelID)
            && (capabilities.supportsStructuredOutput
                || bedrockReasoningConfigEnabled(providerOptions["reasoningConfig"])
                || settings.modelFamily == .anthropic)
        let useNativeStructuredOutput = responseJSONSchema != nil
            && isAnthropicModel
            && (structuredOutputMode == "outputFormat"
                || (structuredOutputMode == "auto" && modelSupportsNativeStructuredOutput && responseJSONSchema.map(anthropicSupportsNativeStructuredOutputSchema) == true))
        let usesJsonInstruction = responseJSONSchema != nil
            && !useNativeStructuredOutput
            && isAnthropicModel
            && (capabilities.rejectsForcedToolUse
                || (modelID.contains("claude-")
                    && (!capabilities.isKnownModel || !capabilities.supportsStructuredOutput)
                    && !bedrockSupportsNativeStructuredOutput(modelID: modelID)
                    && structuredOutputMode != "jsonTool")
                || (structuredOutputMode != "jsonTool"
                    && !bedrockSupportsStrictToolSpec(modelID: modelID)
                    && !request.tools.isEmpty))
        let usesJsonResponseTool = responseJSONSchema != nil && !useNativeStructuredOutput && !usesJsonInstruction
        if let responseJSONSchema, usesJsonResponseTool {
            let choice = effectiveToolChoice?["type"]?.stringValue ?? effectiveToolChoice?.stringValue
            let selected = effectiveToolChoice?["toolName"]?.stringValue ?? effectiveToolChoice?["name"]?.stringValue
            effectiveTools = effectiveTools.filter { name, _ in choice != "none" && (choice != "tool" || name == selected) }
            effectiveTools["json"] = bedrockJSONResponseTool(schema: responseJSONSchema)
            effectiveToolChoice = .object(["type": .string("required")])
        }
        let preparedTools = bedrockPrepareTools(
            from: effectiveTools,
            toolChoice: effectiveToolChoice,
            modelID: modelID,
            modelFamily: settings.modelFamily,
            reasoningConfig: providerOptions["reasoningConfig"],
            disableParallelToolUse: request.providerOptions["anthropic"]?["disableParallelToolUse"]?.boolValue
                ?? request.extraBody["anthropic"]?["disableParallelToolUse"]?.boolValue
        )
        warnings.append(contentsOf: preparedTools.warnings)
        if let additionalModelRequestFields = preparedTools.additionalModelRequestFields {
            var existing = providerOptions["additionalModelRequestFields"]?.objectValue ?? [:]
            existing.merge(additionalModelRequestFields) { _, new in new }
            providerOptions["additionalModelRequestFields"] = .object(existing)
        }
        let userBetas = providerOptions["anthropicBeta"]?.arrayValue
        if !preparedTools.betas.isEmpty || userBetas != nil {
            var existing = providerOptions["additionalModelRequestFields"]?.objectValue ?? [:]
            existing["anthropic_beta"] = .array((userBetas ?? []) + preparedTools.betas.map(JSONValue.string))
            providerOptions["additionalModelRequestFields"] = .object(existing)
        }

        var inferenceConfig: [String: JSONValue] = [:]
        if let maxOutputTokens = request.maxOutputTokens { inferenceConfig["maxTokens"] = .number(Double(maxOutputTokens)) }
        if let temperature { inferenceConfig["temperature"] = .number(temperature) }
        if let topP { inferenceConfig["topP"] = .number(topP) }
        if let topK { inferenceConfig["topK"] = .number(Double(topK)) }
        if !request.stopSequences.isEmpty { inferenceConfig["stopSequences"] = .array(request.stopSequences.map(JSONValue.string)) }
        let reasoningConfig = providerOptions.removeValue(forKey: "reasoningConfig")
        bedrockApplyReasoningConfig(
            reasoningConfig,
            modelID: modelID,
            modelFamily: settings.modelFamily,
            inferenceConfig: &inferenceConfig,
            providerOptions: &providerOptions,
            warnings: &warnings,
            nativeStructuredOutputSchema: useNativeStructuredOutput ? responseJSONSchema : nil
        )
        if isOpenAIModel {
            let unsupportedFeatures = isOpenAIGptOssModel ? ["stopSequences"] : ["temperature", "topP", "stopSequences"]
            for feature in unsupportedFeatures where inferenceConfig.removeValue(forKey: feature) != nil {
                warnings.append(AIWarning(type: "unsupported", feature: feature, message: "\(feature) is not supported by this OpenAI model on the Converse API"))
            }
        }

        var promptMessages = request.messages
        let hasAnyTools = preparedTools.toolConfig != nil || preparedTools.additionalModelRequestFields != nil
        if !hasAnyTools {
            let hasToolContent = promptMessages.contains { message in
                message.content.contains { part in
                    switch part {
                    case .toolCall, .toolResult: return true
                    default: return false
                    }
                }
            }
            if hasToolContent {
                promptMessages = promptMessages.compactMap { message in
                    guard message.role != .system else { return message }
                    var filtered = message
                    filtered.content = message.content.filter { part in
                        switch part {
                        case .toolCall, .toolResult: return false
                        default: return true
                        }
                    }
                    return filtered.content.isEmpty ? nil : filtered
                }
                warnings.append(AIWarning(
                    type: "unsupported",
                    feature: "toolContent",
                    message: "Tool calls and results removed from conversation because Bedrock does not support tool content without active tools."
                ))
            }
        }

        if usesJsonInstruction {
            promptMessages = injectJSONInstruction(
                into: promptMessages,
                schema: responseJSONSchema,
                instruction: AIJSONInstruction(
                    schemaSuffix: "You MUST answer with only a JSON object that matches the JSON schema above. Do not wrap it in markdown fences or include any other text."
                )
            )
        }

        let converted = try bedrockConvertMessages(
            promptMessages,
            modelID: modelID,
            enableDocumentCitations: enableDocumentCitations
        )

        var body: [String: JSONValue] = ["messages": .array(converted.messages)]
        if !converted.system.isEmpty { body["system"] = .array(converted.system) }
        if let additionalModelRequestFields = providerOptions["additionalModelRequestFields"] {
            body["additionalModelRequestFields"] = additionalModelRequestFields
        }
        if isAnthropicModel {
            body["additionalModelResponseFieldPaths"] = .array([.string("/delta/stop_sequence")])
        }
        if !inferenceConfig.isEmpty { body["inferenceConfig"] = .object(inferenceConfig) }
        if let serviceTier = bedrockNonNull(providerOptions["serviceTier"]) {
            body["serviceTier"] = serviceTier.stringValue.map { .object(["type": .string($0)]) } ?? serviceTier
        }
        if let requestMetadata = bedrockNonNull(providerOptions["requestMetadata"]) {
            body["requestMetadata"] = requestMetadata
        }
        for (key, value) in rawNamespaceOptions where !bedrockParsedLanguageOptionKeys.contains(key) {
            body[key] = value
        }
        if let toolConfig = preparedTools.toolConfig {
            body["toolConfig"] = toolConfig
        }
        body.merge(bedrockPassthroughExtraBody(request.extraBody)) { _, new in new }
        return BedrockPreparedConverseCall(
            body: .object(body),
            warnings: warnings,
            usesJsonResponseTool: usesJsonResponseTool,
            usesJsonInstruction: usesJsonInstruction
        )
    }
}

/// Keys that upstream removes from the raw `amazonBedrock`/`bedrock`
/// namespace before spreading the remainder into the Converse body. The
/// Swift-only call-level `citations` and `toolChoice` options are consumed here too.
let bedrockParsedLanguageOptionKeys: Set<String> = [
    "reasoningConfig",
    "additionalModelRequestFields",
    "serviceTier",
    "structuredOutputMode",
    "requestMetadata",
    "citations",
    "toolChoice"
]

/// JavaScript number formatting for warning text (`1.5`, `2`, `-0.5`).
func bedrockJSNumber(_ value: Double) -> String {
    if value.rounded() == value, abs(value) < 1e15 {
        return String(Int(value))
    }
    return String(value)
}

/// Upstream JSON response tool: `{ name: 'json', description: 'Respond with a
/// JSON object.', inputSchema: responseFormat.schema }` with the schema kept intact.
func bedrockJSONResponseTool(schema: JSONValue) -> JSONValue {
    .object([
        "description": .string("Respond with a JSON object."),
        "inputSchema": schema
    ])
}

struct BedrockConvertedMessages {
    var system: [JSONValue]
    var messages: [JSONValue]
}

/// Port of upstream `convertToAmazonBedrockChatMessages`: consecutive user and
/// tool messages share one Bedrock user message, consecutive assistant
/// messages share one assistant message, and system messages must lead.
func bedrockConvertMessages(
    _ prompt: [AIMessage],
    modelID: String,
    enableDocumentCitations: Bool = false
) throws -> BedrockConvertedMessages {
    enum BlockType { case system, user, assistant }
    var blocks: [(type: BlockType, messages: [AIMessage])] = []
    for message in prompt {
        let type: BlockType
        switch message.role {
        case .system: type = .system
        case .assistant: type = .assistant
        case .user, .tool: type = .user
        }
        if blocks.last?.type == type {
            blocks[blocks.count - 1].messages.append(message)
        } else {
            blocks.append((type, [message]))
        }
    }

    var system: [JSONValue] = []
    var messages: [JSONValue] = []
    var documentNames = BedrockDocumentNames()

    for (blockIndex, block) in blocks.enumerated() {
        let isLastBlock = blockIndex == blocks.count - 1
        switch block.type {
        case .system:
            if !messages.isEmpty {
                throw AIError.invalidArgument(
                    argument: "messages",
                    message: "'Multiple system messages that are separated by user/assistant messages' functionality not supported."
                )
            }
            for message in block.messages {
                for part in message.content {
                    if let text = part.text {
                        system.append(.object(["text": .string(text)]))
                    }
                }
                if let cachePoint = bedrockCachePoint(from: message.providerMetadata) {
                    system.append(cachePoint)
                }
            }

        case .user:
            var content: [JSONValue] = []
            for message in block.messages {
                for part in message.content {
                    switch part {
                    case let .text(text, providerMetadata):
                        content.append(try bedrockGuardedTextBlock(text: text, providerMetadata: providerMetadata))
                    case let .imageURL(url, providerMetadata):
                        content.append(bedrockGuardedImageBlock(try bedrockS3ImageBlock(url), providerMetadata: providerMetadata))
                    case let .data(mimeType, data, providerMetadata):
                        content.append(try bedrockFileBlock(
                            mimeType: mimeType,
                            data: data,
                            filename: nil,
                            providerMetadata: providerMetadata,
                            enableDocumentCitations: enableDocumentCitations,
                            documentNames: &documentNames,
                            argument: "messages.content.data.mimeType"
                        ))
                    case let .file(mimeType, data, filename, providerMetadata):
                        content.append(try bedrockFileBlock(
                            mimeType: mimeType,
                            data: data,
                            filename: filename,
                            providerMetadata: providerMetadata,
                            enableDocumentCitations: enableDocumentCitations,
                            documentNames: &documentNames,
                            argument: "messages.content.file.mimeType"
                        ))
                    case .providerReference:
                        throw AIError.invalidArgument(
                            argument: "messages.content.providerReference",
                            message: "'file parts with provider references' functionality not supported."
                        )
                    case let .toolResult(result):
                        content.append(.object([
                            "toolResult": .object([
                                "toolUseId": .string(bedrockNormalizeToolCallID(result.toolCallID, modelID: modelID)),
                                "content": .array(try bedrockToolResultContent(result, documentCounter: &documentNames))
                            ])
                        ]))
                    case let .reasoning(text, providerMetadata):
                        if let reasoning = bedrockReasoningContentBlock(text: text, providerMetadata: providerMetadata) {
                            content.append(reasoning)
                        }
                    case let .toolCall(call):
                        content.append(.object([
                            "toolUse": .object([
                                "toolUseId": .string(bedrockNormalizeToolCallID(call.id, modelID: modelID)),
                                "name": .string(bedrockSanitizeToolName(call.name)),
                                "input": bedrockToolArguments(call.arguments)
                            ])
                        ]))
                    case .reasoningFile, .custom, .toolApprovalRequest, .toolApprovalResponse:
                        break
                    }
                    if case .toolApprovalResponse = part { continue }
                    if let cachePoint = bedrockCachePoint(from: part.providerMetadata) {
                        content.append(cachePoint)
                    }
                }
                if let cachePoint = bedrockCachePoint(from: message.providerMetadata) {
                    content.append(cachePoint)
                }
            }
            bedrockAppendToUserMessage(&messages, content: content)

        case .assistant:
            var assistantContent: [JSONValue] = []
            var toolResultContent: [JSONValue] = []

            func flushAssistantContent() {
                if assistantContent.contains(where: { $0["cachePoint"] == nil }) {
                    messages.append(.object([
                        "role": .string("assistant"),
                        "content": .array(assistantContent)
                    ]))
                }
                assistantContent = []
            }
            func flushToolResultContent() {
                if !toolResultContent.isEmpty {
                    bedrockAppendToUserMessage(&messages, content: toolResultContent)
                    toolResultContent = []
                }
            }

            for (messageIndex, message) in block.messages.enumerated() {
                let isLastMessage = messageIndex == block.messages.count - 1
                // Foreign empty text is metadata only. Remove it before finding
                // the last part so assistant prefill trimming still applies.
                let content = message.content.filter { part in
                    guard case let .text(text, providerMetadata) = part else { return true }
                    return !text.isEmpty
                        || providerMetadata.isEmpty
                        || providerMetadata["amazonBedrock"] != nil
                        || providerMetadata["bedrock"] != nil
                }
                let hasReasoningBlocks = content.contains { part in
                    if case .reasoning = part { return true }
                    return false
                }

                for (partIndex, part) in content.enumerated() {
                    let isLastContentPart = partIndex == content.count - 1
                    var isToolResult = false
                    if case .toolResult = part { isToolResult = true }
                    if !isToolResult {
                        flushToolResultContent()
                    }

                    switch part {
                    case let .text(text, _):
                        if text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && !hasReasoningBlocks {
                            break
                        }
                        // Bedrock rejects trailing whitespace in a prefilled
                        // assistant response, so trim the final text part.
                        let value = isLastBlock && isLastMessage && isLastContentPart
                            ? text.trimmingCharacters(in: .whitespacesAndNewlines)
                            : text
                        assistantContent.append(.object(["text": .string(value)]))
                    case let .reasoning(text, providerMetadata):
                        // Unsigned reasoning is intentionally not replayed.
                        if let reasoning = bedrockReasoningContentBlock(text: text, providerMetadata: providerMetadata) {
                            assistantContent.append(reasoning)
                        }
                    case let .toolCall(call):
                        assistantContent.append(.object([
                            "toolUse": .object([
                                "toolUseId": .string(bedrockNormalizeToolCallID(call.id, modelID: modelID)),
                                "name": .string(bedrockSanitizeToolName(call.name)),
                                "input": bedrockToolArguments(call.arguments)
                            ])
                        ]))
                    case let .toolResult(result):
                        flushAssistantContent()
                        toolResultContent.append(.object([
                            "toolResult": .object([
                                "toolUseId": .string(bedrockNormalizeToolCallID(result.toolCallID, modelID: modelID)),
                                "content": .array(try bedrockToolResultContent(result, documentCounter: &documentNames))
                            ])
                        ]))
                    default:
                        break
                    }

                    if let cachePoint = bedrockCachePoint(from: part.providerMetadata) {
                        if isToolResult {
                            toolResultContent.append(cachePoint)
                        } else {
                            assistantContent.append(cachePoint)
                        }
                    }
                }

                if let cachePoint = bedrockCachePoint(from: message.providerMetadata) {
                    if case .toolResult = content.last {
                        toolResultContent.append(cachePoint)
                    } else {
                        assistantContent.append(cachePoint)
                    }
                }
            }

            flushToolResultContent()
            flushAssistantContent()
        }
    }

    return BedrockConvertedMessages(system: system, messages: messages)
}

/// A trailing assistant tool result and the following user content must share
/// a single Bedrock user message so that conversation roles keep alternating.
private func bedrockAppendToUserMessage(_ messages: inout [JSONValue], content: [JSONValue]) {
    if let lastIndex = messages.indices.last,
       messages[lastIndex]["role"]?.stringValue == "user",
       messages[lastIndex]["content"]?.arrayValue?.contains(where: { $0["toolResult"] != nil }) == true,
       var previous = messages[lastIndex].objectValue {
        previous["content"] = .array((previous["content"]?.arrayValue ?? []) + content)
        messages[lastIndex] = .object(previous)
    } else {
        messages.append(.object([
            "role": .string("user"),
            "content": .array(content)
        ]))
    }
}

private func bedrockS3ImageBlock(_ url: String) throws -> JSONValue {
    guard bedrockIsS3URL(url), let imageFormat = bedrockImageFormat(forURL: url) else {
        throw AIError.invalidArgument(
            argument: "messages.content.imageURL",
            message: "Amazon Bedrock Converse supports only s3:// image URLs ending in .jpg, .jpeg, .png, .gif, or .webp."
        )
    }
    return .object([
        "image": .object([
            "format": .string(imageFormat),
            "source": .object([
                "s3Location": .object(["uri": .string(url)])
            ])
        ])
    ])
}

private func bedrockFileBlock(
    mimeType: String,
    data: Data,
    filename: String?,
    providerMetadata: [String: JSONValue],
    enableDocumentCitations: Bool,
    documentNames: inout BedrockDocumentNames,
    argument: String
) throws -> JSONValue {
    let resolvedMimeType = try resolveFullMediaType(mediaType: mimeType, data: data)
    if let imageFormat = bedrockImageFormat(for: resolvedMimeType) {
        return bedrockGuardedImageBlock(.object([
            "image": .object([
                "format": .string(imageFormat),
                "source": .object(["bytes": .string(data.base64EncodedString())])
            ])
        ]), providerMetadata: providerMetadata)
    }
    if let videoFormat = bedrockVideoFormat(for: resolvedMimeType) {
        return .object([
            "video": .object([
                "format": .string(videoFormat),
                "source": .object(["bytes": .string(data.base64EncodedString())])
            ])
        ])
    }
    guard let documentFormat = bedrockDocumentFormat(for: resolvedMimeType) else {
        throw AIError.invalidArgument(
            argument: argument,
            message: "Amazon Bedrock Converse supports image MIME types \(bedrockSupportedImageMimeTypes.joined(separator: ", ")), video MIME types \(bedrockSupportedVideoMimeTypes.joined(separator: ", ")), or document MIME types \(bedrockSupportedDocumentMimeTypes.joined(separator: ", ")); got \(mimeType)."
        )
    }
    var document: [String: JSONValue] = [
        "format": .string(documentFormat),
        "name": .string(bedrockDocumentName(filename, documentCounter: &documentNames)),
        "source": .object(["bytes": .string(data.base64EncodedString())])
    ]
    if enableDocumentCitations || bedrockDocumentCitationsEnabled(providerMetadata) {
        document["citations"] = .object(["enabled": .bool(true)])
    }
    return .object(["document": .object(document)])
}

struct BedrockPreparedConverseCall {
    var body: JSONValue
    var warnings: [AIWarning]
    var usesJsonResponseTool: Bool
    var usesJsonInstruction: Bool
}

func bedrockToolArguments(_ arguments: String) -> JSONValue {
    let decoded = (try? decodeJSONBody(Data(arguments.utf8))) ?? .string(arguments)
    guard case .object = decoded else {
        return .object(["rawInvalidInput": decoded])
    }
    return decoded
}