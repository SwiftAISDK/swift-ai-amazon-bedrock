import Foundation
import SwiftAIProvider
import SwiftAITransport
import SwiftAIProviderUtilities
import SwiftAIAnthropic
public final class AmazonBedrockAnthropicLanguageModel: LanguageModel, @unchecked Sendable {
    public let providerID = "bedrock.anthropic.messages"
    public let modelID: String
    private let config: BedrockRuntimeConfig

    init(modelID: String, config: BedrockRuntimeConfig) {
        self.modelID = modelID
        self.config = config
    }

    public func generate(_ request: LanguageModelRequest) async throws -> TextGenerationResult {
        let prepared = try await preparedCall(for: request)
        let body = amazonBedrockThinkingBindingBody(amazonBedrockAnthropicBody(prepared.body, betas: prepared.betas))
        let response = try await config.sendJSONResponse(path: "/model/\(bedrockEncodeModelID(modelID))/invoke", body: .object(body), headers: request.headers, abortSignal: request.abortSignal)
        let raw = response.json
        try anthropicValidateFallbackContent(raw["content"], providerID: providerID)
        let generatedContent = anthropicGeneratedContent(
            from: raw["content"],
            providerID: providerID,
            citationDocuments: anthropicCitationDocuments(from: request.messages),
            usesJSONToolResponseFormat: prepared.usesJSONToolResponseFormat,
            toolNameMapping: prepared.toolNameMapping,
            markCodeExecutionDynamic: prepared.markCodeExecutionDynamic
        )
        guard let text = generatedContent.text else {
            throw AIError.invalidResponse(provider: providerID, message: "No text block found in Bedrock Anthropic response.")
        }
        return TextGenerationResult(
            text: text,
            content: generatedContent.content,
            reasoning: generatedContent.reasoning,
            finishReason: anthropicFinishReason(raw["stop_reason"]?.stringValue, toolCalls: generatedContent.toolCalls),
            usage: anthropicTokenUsage(from: raw["usage"]),
            toolCalls: generatedContent.toolCalls,
            toolResults: generatedContent.toolResults,
            sources: generatedContent.sources,
            providerMetadata: anthropicProviderMetadata(from: raw, providerID: providerID),
            rawValue: raw,
            warnings: prepared.warnings,
            responseMetadata: aiResponseMetadata(from: raw, response: response.response, modelID: modelID)
        )
    }

    public func stream(_ request: LanguageModelRequest) -> AsyncThrowingStream<LanguageStreamPart, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    let prepared = try await preparedCall(for: request, stream: true)
                    let body = amazonBedrockThinkingBindingBody(amazonBedrockAnthropicBody(prepared.body, betas: prepared.betas))
                    let httpRequest = try config.request(
                        path: "/model/\(bedrockEncodeModelID(modelID))/invoke-with-response-stream",
                        body: .object(body),
                        headers: request.headers.mergingHeaders(["accept": "application/vnd.amazon.eventstream"]),
                        abortSignal: request.abortSignal
                    )
                    let response = try await config.streamRequest(httpRequest)
                    guard (200..<300).contains(response.statusCode) else {
                        let responseBody = try await readResponseWithSizeLimit(
                            response: response,
                            url: httpRequest.url.absoluteString,
                            maxBytes: httpRequest.maxResponseBytes ?? AIDefaultMaxDownloadSize
                        )
                        throw bedrockHTTPStatusError(provider: providerID, response: AIHTTPResponse(
                            statusCode: response.statusCode,
                            headers: response.headers,
                            body: responseBody,
                            url: httpRequest.url
                        ))
                    }
                    let metadataResponse = AIHTTPResponse(
                        statusCode: response.statusCode,
                        headers: response.headers,
                        url: httpRequest.url
                    )
                    continuation.yield(.responseMetadata(aiResponseMetadata(response: metadataResponse, modelID: modelID)))
                    continuation.yield(.streamStart(warnings: prepared.warnings))

                    var contentBlocks = AnthropicStreamingContentBlocks(
                        providerID: providerID,
                        ignoresTextBlocks: prepared.usesJSONToolResponseFormat
                    )
                    var jsonToolText = AnthropicStreamingJSONToolText()
                    var providerToolResults = AnthropicStreamingProviderToolResults(
                        providerID: providerID,
                        toolNameMapping: prepared.toolNameMapping
                    )
                    var toolCalls = AnthropicStreamingToolCalls(
                        providerID: providerID,
                        toolNameMapping: prepared.toolNameMapping,
                        markCodeExecutionDynamic: prepared.markCodeExecutionDynamic
                    )
                    var realToolCallCount = 0
                    var streamingSources = AnthropicStreamingSources(citationDocuments: anthropicCitationDocuments(from: request.messages))
                    var parser = BedrockRawStreamParser(
                        providerID: providerID,
                        contentType: response.headerValue("content-type")
                    )
                    var finishReason: String?
                    var finishUsage: TokenUsage?
                    var rawUsage: JSONValue?
                    var stopSequence: JSONValue = .null
                    var stopDetails: JSONValue = .null
                    var container: JSONValue = .null
                    var contextManagement: JSONValue = .null
                    var inputTransformations: JSONValue = .null
                    var safeguardResults: JSONValue = .null
                    var didEmitTerminal = false
                    var hasOutput = false

                    func emitTerminal() {
                        guard !didEmitTerminal else { return }
                        for source in streamingSources.finishSources() {
                            continuation.yield(.source(source))
                        }
                        for part in contentBlocks.finishParts(citationDocuments: streamingSources.citationDocuments) {
                            continuation.yield(part)
                        }
                        for part in jsonToolText.finishParts() {
                            continuation.yield(part)
                        }
                        continuation.yield(.finishMetadata(
                            reason: finishReason ?? "other",
                            usage: finishUsage,
                            providerMetadata: anthropicProviderMetadata(
                                usage: rawUsage,
                                stopSequence: stopSequence,
                                stopDetails: stopDetails,
                                container: container,
                                contextManagement: contextManagement,
                                inputTransformations: inputTransformations,
                                safeguardResults: safeguardResults,
                                providerID: providerID,
                                requestProviderOptions: request.providerOptions
                            )
                        ))
                        didEmitTerminal = true
                    }

                    func process(_ items: [BedrockRawStreamItem]) throws -> Bool {
                        for outerItem in items {
                            let adaptedItem = try amazonBedrockAnthropicStreamItem(from: outerItem, providerID: providerID)
                            let item: BedrockRawStreamItem
                            switch adaptedItem {
                            case .messageStop:
                                emitTerminal()
                                return true
                            case let .event(event):
                                item = event
                            }
                            let raw = item.rawValue
                            if raw["type"] == "content_block_delta" {
                                let delta = raw["delta"]
                                hasOutput = hasOutput || !(delta?["text"]?.stringValue ?? delta?["thinking"]?.stringValue ?? delta?["partial_json"]?.stringValue ?? delta?["content"]?.stringValue ?? "").isEmpty
                            }
                            if raw["message"]?["content"]?.arrayValue?.contains(where: { $0["type"] == "tool_use" }) == true || raw["type"] == "content_block_stop" && realToolCallCount > 0 { hasOutput = true }
                            try anthropicValidateFallbackEvent(raw, providerID: providerID)
                            if request.includeRawChunks {
                                continuation.yield(.raw(raw))
                            }
                            if let providerError = amazonBedrockStreamProviderError(from: item) {
                                finishReason = "error"
                                continuation.yield(.providerError(providerError))
                                emitTerminal()
                                return true
                            }
                            if let providerError = anthropicStreamProviderError(from: raw) {
                                finishReason = "error"
                                continuation.yield(.providerError(providerError))
                                emitTerminal()
                                return true
                            }
                            switch raw["type"]?.stringValue {
                            case "message_start":
                                if let usage = raw["message"]?["usage"] {
                                    rawUsage = usage
                                    finishUsage = anthropicTokenUsage(from: usage)
                                }
                                continuation.yield(.custom(
                                    ["kind": "anthropic.message_start"],
                                    providerMetadata: ["anthropic": .object([
                                        "id": raw["message"]?["id"] ?? .null,
                                        "model": raw["message"]?["model"] ?? .null,
                                        "usage": rawUsage ?? .null
                                    ])]
                                ))
                                if let value = anthropicContainerMetadata(from: raw["message"]?["container"]) {
                                    container = value
                                }
                                if let value = raw["message"]?["input_transformations"], value != .null {
                                    inputTransformations = value
                                }
                                if let reason = raw["message"]?["stop_reason"]?.stringValue {
                                    finishReason = anthropicFinishReason(reason, toolCallCount: realToolCallCount)
                                }
                            case "message_delta":
                                finishReason = anthropicFinishReason(raw["delta"]?["stop_reason"]?.stringValue, toolCallCount: realToolCallCount)
                                stopSequence = raw["delta"]?["stop_sequence"] ?? stopSequence
                                stopDetails = anthropicStopDetailsMetadata(from: raw["delta"]?["stop_details"]) ?? stopDetails
                                if let usage = raw["usage"] {
                                    rawUsage = rawUsage.map { anthropicMergedUsage($0, usage) } ?? usage
                                    finishUsage = anthropicTokenUsage(from: rawUsage)
                                }
                                if let value = anthropicContainerMetadata(from: raw["delta"]?["container"]) {
                                    container = value
                                }
                                if let value = anthropicContextManagementMetadata(from: raw["context_management"]) {
                                    contextManagement = value
                                }
                                if let value = raw["input_transformations"], value != .null {
                                    inputTransformations = value
                                }
                                if let value = raw["delta"]?["safeguard_results"], value != .null {
                                    safeguardResults = value
                                }
                            default:
                                break
                            }
                            for part in toolCalls.apply(messageStartContent: raw["message"]?["content"]) {
                                if case .toolCall = part {
                                    realToolCallCount += 1
                                }
                                continuation.yield(part)
                            }
                            for part in contentBlocks.apply(event: raw, citationDocuments: streamingSources.citationDocuments) {
                                continuation.yield(part)
                            }
                            for part in jsonToolText.apply(event: raw) {
                                continuation.yield(part)
                            }
                            for part in providerToolResults.apply(event: raw) {
                                continuation.yield(part)
                            }
                            for source in streamingSources.apply(event: raw) {
                                continuation.yield(.source(source))
                            }
                            for part in toolCalls.apply(event: raw) {
                                if case .toolCall = part {
                                    realToolCallCount += 1
                                }
                                continuation.yield(part)
                            }
                            if raw["type"]?.stringValue == "message_stop" {
                                emitTerminal()
                                return true
                            }
                        }
                        return false
                    }

                    var reachedMessageStop = false
                    streamLoop: for try await data in response.body {
                        reachedMessageStop = try process(parser.append(data))
                        if reachedMessageStop || parser.isDone {
                            break streamLoop
                        }
                    }
                    if !reachedMessageStop {
                        reachedMessageStop = try process(parser.finish())
                    }
                    if !reachedMessageStop, !didEmitTerminal, hasOutput {
                        finishReason = "interrupted"
                        finishUsage = nil
                        rawUsage = nil
                        emitTerminal()
                    }
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

    private func preparedCall(for request: LanguageModelRequest, stream: Bool = false) async throws -> AnthropicPreparedCall {
        var resolvedRequest = request
        resolvedRequest.messages = try await amazonBedrockAnthropicResolveURLContent(
            in: request.messages,
            transport: config.transport,
            abortSignal: request.abortSignal,
            providerID: providerID
        )
        return try AnthropicLanguageModel.body(
            for: resolvedRequest,
            modelID: modelID,
            providerID: providerID,
            stream: stream,
            supportsNativeStructuredOutput: bedrockSupportsNativeStructuredOutput(modelID: modelID),
            supportsStrictTools: bedrockSupportsStrictToolSpec(modelID: modelID)
        )
    }
}
