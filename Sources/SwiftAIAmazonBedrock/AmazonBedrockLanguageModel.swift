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
        let raw = try await config.sendJSON(path: "/model/\(encodedModelID)/converse", body: prepared.body, headers: request.headers, abortSignal: request.abortSignal)
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
                    let metadata = bedrockReasoningProviderMetadata(
                        key: "signature",
                        value: reasoningText["signature"]
                    )
                    content.append(.reasoning(value, providerMetadata: metadata))
                    continue
                }
                if let redactedData = reasoningContent["redactedReasoning"]?["data"] {
                    content.append(.reasoning(
                        "",
                        providerMetadata: bedrockReasoningProviderMetadata(key: "redactedData", value: redactedData)
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
            warnings: prepared.warnings
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
        var warnings: [AIWarning] = []
        bedrockApplyTopLevelReasoning(
            request.reasoning,
            modelID: modelID,
            modelFamily: settings.modelFamily,
            maxOutputTokens: request.maxOutputTokens,
            providerOptions: &providerOptions,
            warnings: &warnings
        )
        let enableDocumentCitations = bedrockDocumentCitationsEnabled(providerOptions)
        var documentCounter = 0
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
        var stopSequences = request.stopSequences
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
                warnings.append(AIWarning(type: "unsupported", feature: "temperature", message: "\(value) exceeds bedrock maximum of 1.0. clamped to 1.0"))
                temperature = 1
            } else if value < 0 {
                warnings.append(AIWarning(type: "unsupported", feature: "temperature", message: "\(value) is below bedrock minimum of 0. clamped to 0"))
                temperature = 0
            }
        }
        if isOpenAIModel {
            let unsupported: [(String, Bool)] = [
                ("temperature", !isOpenAIGptOssModel && temperature != nil),
                ("topP", !isOpenAIGptOssModel && topP != nil),
                ("stopSequences", !stopSequences.isEmpty)
            ]
            for (feature, isPresent) in unsupported where isPresent {
                warnings.append(AIWarning(type: "unsupported", feature: feature, message: "\(feature) is not supported by this OpenAI model on the Converse API"))
            }
            if !isOpenAIGptOssModel {
                temperature = nil
                topP = nil
            }
            stopSequences = []
        }
        var effectiveTools = request.tools
        var effectiveToolChoice = request.toolChoice ?? providerOptions["toolChoice"] ?? request.extraBody["toolChoice"]
        let responseJSONSchema = bedrockResponseJSONSchema(from: request.responseFormat)
        let isAnthropicModel = bedrockIsAnthropicModel(
            modelID: modelID,
            modelFamily: settings.modelFamily,
            reasoningConfig: providerOptions["reasoningConfig"]
        )
        let modelSupportsStructuredOutput = settings.modelFamily == .anthropic || capabilities.supportsStructuredOutput
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
            && (modelSupportsStructuredOutput || bedrockReasoningConfigEnabled(providerOptions["reasoningConfig"]))
        let useNativeStructuredOutput = responseJSONSchema != nil
            && isAnthropicModel
            && (structuredOutputMode == "outputFormat"
                || (structuredOutputMode == "auto" && modelSupportsNativeStructuredOutput))
        let usesJsonInstruction = responseJSONSchema != nil
            && !useNativeStructuredOutput
            && isAnthropicModel
            && (capabilities.rejectsForcedToolUse
                || (modelID.contains("claude-")
                    && (!capabilities.isKnownModel || !modelSupportsStructuredOutput)
                    && !bedrockSupportsNativeStructuredOutput(modelID: modelID)
                    && structuredOutputMode != "jsonTool")
                || (structuredOutputMode != "jsonTool"
                    && !bedrockSupportsStrictToolSpec(modelID: modelID)
                    && !request.tools.isEmpty))
        let usesJsonResponseTool = responseJSONSchema != nil && !useNativeStructuredOutput && !usesJsonInstruction
        if let responseJSONSchema, useNativeStructuredOutput {
            bedrockMergeAdditionalModelRequestFields([
                "output_config": .object([
                    "format": .object([
                        "type": .string("json_schema"),
                        "schema": anthropicSanitizeJSONSchema(responseJSONSchema)
                    ])
                ])
            ], into: &providerOptions)
        } else if let responseJSONSchema, usesJsonResponseTool {
            effectiveTools["json"] = responseJSONSchema
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
                ?? false
        )
        if let additionalModelRequestFields = preparedTools.additionalModelRequestFields {
            bedrockMergeAdditionalModelRequestFields(additionalModelRequestFields, into: &providerOptions)
        }
        warnings.append(contentsOf: preparedTools.warnings)

        let messagesWithJSONInstruction = usesJsonInstruction
            ? injectJSONInstruction(
                into: request.messages,
                schema: responseJSONSchema,
                instruction: AIJSONInstruction(
                    schemaSuffix: "You MUST answer with only a JSON object that matches the JSON schema above. Do not wrap it in markdown fences or include any other text."
                )
            )
            : request.messages

        let system = messagesWithJSONInstruction
            .filter { $0.role == .system }
            .flatMap { message -> [JSONValue] in
                var blocks = message.content.compactMap(\.text).map { JSONValue.object(["text": .string($0)]) }
                if let cachePoint = bedrockCachePoint(from: message.providerMetadata) {
                    blocks.append(cachePoint)
                }
                return blocks
            }

        let convertedMessages = try messagesWithJSONInstruction
            .filter { $0.role != .system }
            .compactMap { message -> JSONValue? in
                var content: [JSONValue] = []
                for part in message.content {
                    switch part {
                    case let .text(text, providerMetadata):
                        content.append(try bedrockGuardedTextBlock(text: text, providerMetadata: providerMetadata))
                        if let cachePoint = bedrockCachePoint(from: providerMetadata) {
                            content.append(cachePoint)
                        }
                    case let .reasoning(text, providerMetadata):
                        if let reasoning = bedrockReasoningContentBlock(text: text, providerMetadata: providerMetadata) {
                            content.append(reasoning)
                        }
                        if let cachePoint = bedrockCachePoint(from: providerMetadata) {
                            content.append(cachePoint)
                        }
                    case .reasoningFile, .custom:
                        content.append(.object(["text": .string("")]))
                    case let .imageURL(url, providerMetadata):
                        guard bedrockIsS3URL(url), let imageFormat = bedrockImageFormat(forURL: url) else {
                            throw AIError.invalidArgument(
                                argument: "messages.content.imageURL",
                                message: "Amazon Bedrock Converse supports only s3:// image URLs ending in .jpg, .jpeg, .png, .gif, or .webp."
                            )
                        }
                        let imageBlock: JSONValue = .object([
                            "image": .object([
                                "format": .string(imageFormat),
                                "source": .object([
                                    "s3Location": .object(["uri": .string(url)])
                                ])
                            ])
                        ])
                        content.append(bedrockGuardedImageBlock(imageBlock, providerMetadata: providerMetadata))
                        if let cachePoint = bedrockCachePoint(from: providerMetadata) {
                            content.append(cachePoint)
                        }
                    case let .data(mimeType, data, providerMetadata):
                        let resolvedMimeType = try resolveFullMediaType(mediaType: mimeType, data: data)
                        if let imageFormat = bedrockImageFormat(for: resolvedMimeType) {
                            let imageBlock: JSONValue = .object([
                                "image": .object([
                                    "format": .string(imageFormat),
                                    "source": .object(["bytes": .string(data.base64EncodedString())])
                                ])
                            ])
                            content.append(bedrockGuardedImageBlock(imageBlock, providerMetadata: providerMetadata))
                            if let cachePoint = bedrockCachePoint(from: providerMetadata) {
                                content.append(cachePoint)
                            }
                            continue
                        }

                        if let videoFormat = bedrockVideoFormat(for: resolvedMimeType) {
                            content.append(.object([
                                "video": .object([
                                    "format": .string(videoFormat),
                                    "source": .object(["bytes": .string(data.base64EncodedString())])
                                ])
                            ]))
                            if let cachePoint = bedrockCachePoint(from: providerMetadata) {
                                content.append(cachePoint)
                            }
                            continue
                        }

                        guard let documentFormat = bedrockDocumentFormat(for: resolvedMimeType) else {
                            throw AIError.invalidArgument(
                                argument: "messages.content.data.mimeType",
                                message: "Amazon Bedrock Converse supports image MIME types \(bedrockSupportedImageMimeTypes.joined(separator: ", ")), video MIME types \(bedrockSupportedVideoMimeTypes.joined(separator: ", ")), or document MIME types \(bedrockSupportedDocumentMimeTypes.joined(separator: ", ")); got \(mimeType)."
                            )
                        }

                        var document: [String: JSONValue] = [
                            "format": .string(documentFormat),
                            "name": .string(bedrockDocumentName(nil, documentCounter: &documentCounter)),
                            "source": .object(["bytes": .string(data.base64EncodedString())])
                        ]
                        if enableDocumentCitations || bedrockDocumentCitationsEnabled(providerMetadata) {
                            document["citations"] = .object(["enabled": .bool(true)])
                        }
                        content.append(.object(["document": .object(document)]))
                        if let cachePoint = bedrockCachePoint(from: providerMetadata) {
                            content.append(cachePoint)
                        }
                    case let .file(mimeType, data, filename, providerMetadata):
                        let resolvedMimeType = try resolveFullMediaType(mediaType: mimeType, data: data)
                        if let imageFormat = bedrockImageFormat(for: resolvedMimeType) {
                            let imageBlock: JSONValue = .object([
                                "image": .object([
                                    "format": .string(imageFormat),
                                    "source": .object(["bytes": .string(data.base64EncodedString())])
                                ])
                            ])
                            content.append(bedrockGuardedImageBlock(imageBlock, providerMetadata: providerMetadata))
                            if let cachePoint = bedrockCachePoint(from: providerMetadata) {
                                content.append(cachePoint)
                            }
                            continue
                        }

                        if let videoFormat = bedrockVideoFormat(for: resolvedMimeType) {
                            content.append(.object([
                                "video": .object([
                                    "format": .string(videoFormat),
                                    "source": .object(["bytes": .string(data.base64EncodedString())])
                                ])
                            ]))
                            if let cachePoint = bedrockCachePoint(from: providerMetadata) {
                                content.append(cachePoint)
                            }
                            continue
                        }

                        guard let documentFormat = bedrockDocumentFormat(for: resolvedMimeType) else {
                            throw AIError.invalidArgument(
                                argument: "messages.content.file.mimeType",
                                message: "Amazon Bedrock Converse supports image MIME types \(bedrockSupportedImageMimeTypes.joined(separator: ", ")), video MIME types \(bedrockSupportedVideoMimeTypes.joined(separator: ", ")), or document MIME types \(bedrockSupportedDocumentMimeTypes.joined(separator: ", ")); got \(mimeType)."
                            )
                        }

                        var document: [String: JSONValue] = [
                            "format": .string(documentFormat),
                            "name": .string(bedrockDocumentName(filename, documentCounter: &documentCounter)),
                            "source": .object(["bytes": .string(data.base64EncodedString())])
                        ]
                        if enableDocumentCitations || bedrockDocumentCitationsEnabled(providerMetadata) {
                            document["citations"] = .object(["enabled": .bool(true)])
                        }
                        content.append(.object(["document": .object(document)]))
                        if let cachePoint = bedrockCachePoint(from: providerMetadata) {
                            content.append(cachePoint)
                        }
                    case let .toolCall(call):
                        if preparedTools.toolConfig == nil {
                            warnings.append(AIWarning(
                                type: "unsupported",
                                feature: "toolContent",
                                message: "Tool calls and results removed from conversation because Bedrock does not support tool content without active tools."
                            ))
                            content.append(.object(["text": .string("")]))
                            continue
                        }
                        content.append(.object([
                            "toolUse": .object([
                                "toolUseId": .string(bedrockNormalizeToolCallID(call.id, modelID: modelID)),
                                "name": .string(bedrockSanitizeToolName(call.name)),
                                "input": bedrockToolArguments(call.arguments)
                            ])
                        ]))
                    case let .toolResult(result):
                        if preparedTools.toolConfig == nil {
                            warnings.append(AIWarning(
                                type: "unsupported",
                                feature: "toolContent",
                                message: "Tool calls and results removed from conversation because Bedrock does not support tool content without active tools."
                            ))
                            content.append(.object(["text": .string("")]))
                            continue
                        }
                        content.append(.object([
                            "toolResult": .object([
                                "toolUseId": .string(bedrockNormalizeToolCallID(result.toolCallID, modelID: modelID)),
                                "content": .array(try bedrockToolResultContent(result, documentCounter: &documentCounter)),
                                "status": .string(result.isError ? "error" : "success")
                            ])
                        ]))
                        if let cachePoint = bedrockCachePoint(from: result.providerMetadata) {
                            content.append(cachePoint)
                        }
                    case .providerReference, .toolApprovalRequest, .toolApprovalResponse:
                        content.append(.object(["text": .string("")]))
                    }
                }
                if let cachePoint = bedrockCachePoint(from: message.providerMetadata) {
                    content.append(cachePoint)
                }

                // Bedrock rejects empty Converse messages. Unsigned reasoning is
                // intentionally filtered above, so drop an assistant turn when
                // it has no replayable content left.
                if message.role == .assistant,
                   content.allSatisfy({ $0["cachePoint"] != nil }) {
                    return nil
                }

                return JSONValue.object([
                    "role": .string(message.role == .assistant ? "assistant" : "user"),
                    "content": .array(content)
                ])
            }

        var messages: [JSONValue] = []
        for message in convertedMessages {
            bedrockAppendConvertedMessage(message, to: &messages)
        }

        var body: [String: JSONValue] = ["messages": .array(messages)]
        if !system.isEmpty { body["system"] = .array(system) }

        var inferenceConfig: [String: JSONValue] = [:]
        if let maxOutputTokens = request.maxOutputTokens { inferenceConfig["maxTokens"] = .number(Double(maxOutputTokens)) }
        if let temperature { inferenceConfig["temperature"] = .number(temperature) }
        if let topP { inferenceConfig["topP"] = .number(topP) }
        if let topK { inferenceConfig["topK"] = .number(Double(topK)) }
        if !stopSequences.isEmpty { inferenceConfig["stopSequences"] = .array(stopSequences) }
        let reasoningConfig = providerOptions.removeValue(forKey: "reasoningConfig")
        bedrockApplyReasoningConfig(
            reasoningConfig,
            modelID: modelID,
            modelFamily: settings.modelFamily,
            inferenceConfig: &inferenceConfig,
            providerOptions: &providerOptions,
            warnings: &warnings
        )
        if modelID.contains("amazon.nova-2-lite-v1:0"),
           reasoningConfig?["type"]?.stringValue == "enabled",
           ["high", "xhigh", "max"].contains(reasoningConfig?["maxReasoningEffort"]?.stringValue ?? ""),
           inferenceConfig.removeValue(forKey: "maxTokens") != nil {
            warnings.append(AIWarning(type: "unsupported", feature: "maxOutputTokens", message: "maxOutputTokens is not supported by \(modelID) when high reasoning is enabled and will be ignored"))
        }
        if !inferenceConfig.isEmpty { body["inferenceConfig"] = .object(inferenceConfig) }
        if let toolConfig = preparedTools.toolConfig {
            body["toolConfig"] = toolConfig
        }
        bedrockApplyRequestProviderOptions(providerOptions, to: &body)
        body.merge(bedrockPassthroughExtraBody(request.extraBody)) { _, new in new }
        return BedrockPreparedConverseCall(
            body: .object(body),
            warnings: bedrockDeduplicatedWarnings(warnings),
            usesJsonResponseTool: usesJsonResponseTool,
            usesJsonInstruction: usesJsonInstruction
        )
    }
}

private func bedrockAppendConvertedMessage(
    _ message: JSONValue,
    to messages: inout [JSONValue]
) {
    guard message["role"]?.stringValue == "assistant",
          let content = message["content"]?.arrayValue else {
        bedrockAppendUserMessage(message, to: &messages)
        return
    }

    var segmentRole: String?
    var segment: [JSONValue] = []

    for block in content {
        let nextRole: String
        if block["toolResult"] != nil {
            nextRole = "user"
        } else if block["cachePoint"] != nil {
            nextRole = segmentRole ?? "assistant"
        } else {
            nextRole = "assistant"
        }

        if let currentRole = segmentRole, currentRole != nextRole {
            bedrockAppendConvertedSegment(
                role: currentRole,
                content: segment,
                to: &messages
            )
            segment.removeAll(keepingCapacity: true)
        }
        segmentRole = nextRole
        segment.append(block)
    }
    if let segmentRole {
        bedrockAppendConvertedSegment(
            role: segmentRole,
            content: segment,
            to: &messages
        )
    }
}

private func bedrockAppendConvertedSegment(
    role: String,
    content: [JSONValue],
    to messages: inout [JSONValue]
) {
    guard !content.isEmpty else { return }
    let converted: JSONValue = .object([
        "role": .string(role),
        "content": .array(content)
    ])
    if role == "user" {
        bedrockAppendUserMessage(converted, to: &messages)
    } else if content.contains(where: { $0["cachePoint"] == nil }) {
        messages.append(converted)
    }
}

private func bedrockAppendUserMessage(
    _ message: JSONValue,
    to messages: inout [JSONValue]
) {
    guard message["role"]?.stringValue == "user",
          let content = message["content"]?.arrayValue else {
        messages.append(message)
        return
    }

    if let lastIndex = messages.indices.last,
       messages[lastIndex]["role"]?.stringValue == "user",
       messages[lastIndex]["content"]?.arrayValue?.contains(where: { $0["toolResult"] != nil }) == true,
       var previous = messages[lastIndex].objectValue {
        previous["content"] = .array((previous["content"]?.arrayValue ?? []) + content)
        messages[lastIndex] = .object(previous)
    } else {
        messages.append(message)
    }
}

struct BedrockPreparedConverseCall {
    var body: JSONValue
    var warnings: [AIWarning]
    var usesJsonResponseTool: Bool
    var usesJsonInstruction: Bool
}

func bedrockToolArguments(_ arguments: String) -> JSONValue {
    let decoded = (try? decodeJSONBody(Data(arguments.utf8))) ?? .object([:])
    guard case .object = decoded else {
        return .object(["rawInvalidInput": decoded])
    }
    return decoded
}