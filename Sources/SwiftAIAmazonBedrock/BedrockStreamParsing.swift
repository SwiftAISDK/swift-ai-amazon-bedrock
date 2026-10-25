import SwiftAIOpenAICompatible
import SwiftAIOpenAI
import SwiftAIAnthropic
import Foundation
import SwiftAIProvider
import SwiftAITransport
import SwiftAIProviderUtilities


func bedrockStreamParts(from raw: JSONValue) -> [LanguageStreamPart] {
    var parts: [LanguageStreamPart] = []
    let index = raw["contentBlockDelta"]?["contentBlockIndex"]?.intValue ?? 0
    if let text = raw["contentBlockDelta"]?["delta"]?["text"]?.stringValue {
        let id = String(index)
        parts.append(.textStart(id: id))
        parts.append(.textDeltaPart(id: id, delta: text))
        parts.append(.textEnd(id: id))
    }
    if let reasoning = raw["contentBlockDelta"]?["delta"]?["reasoningContent"]?["text"]?.stringValue {
        let id = String(index)
        parts.append(.reasoningStart(id: id))
        parts.append(.reasoningDeltaPart(id: id, delta: reasoning))
        parts.append(.reasoningEnd(id: id))
    }
    if raw["messageStop"] != nil || raw["metadata"]?["usage"] != nil {
        parts.append(.finishMetadata(
            reason: bedrockFinishReason(raw["messageStop"]?["stopReason"]?.stringValue),
            usage: TokenUsage(
                inputTokens: raw["metadata"]?["usage"]?["inputTokens"]?.intValue,
                outputTokens: raw["metadata"]?["usage"]?["outputTokens"]?.intValue,
                totalTokens: raw["metadata"]?["usage"]?["totalTokens"]?.intValue
            ),
            providerMetadata: bedrockProviderMetadata(fromStreamMetadata: raw["metadata"])
        ))
    }
    return parts
}

struct BedrockStreamingToolCall {
    var id: String
    var name: String
    var arguments: String = ""
    var rawValue: JSONValue?
}

struct BedrockStreamState {
    private enum ContentBlock: Equatable {
        case text
        case reasoning
        case tool
    }

    var toolCalls: [Int: BedrockStreamingToolCall] = [:]
    var latestFinishReason: String? = "interrupted"
    var latestUsage: TokenUsage?
    var jsonResponseToolName: String?
    var modelID: String
    var jsonObjectTextExtractor: BedrockJSONObjectTextExtractor?
    var isJsonResponseFromTool = false
    private var contentBlocks: [Int: ContentBlock] = [:]
    private var redactedReasoningContent: [Int: String] = [:]
    private var latestProviderMetadata: [String: JSONValue] = [:]
    private var sawTerminalEvent = false

    init(
        jsonResponseToolName: String? = nil,
        modelID: String = "",
        jsonObjectTextExtractor: BedrockJSONObjectTextExtractor? = nil
    ) {
        self.jsonResponseToolName = jsonResponseToolName
        self.modelID = modelID
        self.jsonObjectTextExtractor = jsonObjectTextExtractor
    }

    mutating func parts(from raw: JSONValue) -> [LanguageStreamPart] {
        var parts: [LanguageStreamPart] = []
        let contentBlockIndex = raw["contentBlockDelta"]?["contentBlockIndex"]?.intValue ?? 0
        if let text = raw["contentBlockDelta"]?["delta"]?["text"]?.stringValue {
            let textDelta = jsonObjectTextExtractor?.process(text) ?? text
            if !textDelta.isEmpty {
                let id = String(contentBlockIndex)
                if contentBlocks[contentBlockIndex] == nil {
                    contentBlocks[contentBlockIndex] = .text
                    parts.append(.textStart(id: id))
                }
                parts.append(.textDeltaPart(id: id, delta: textDelta))
            }
        }
        if let reasoning = raw["contentBlockDelta"]?["delta"]?["reasoningContent"]?["text"]?.stringValue {
            let id = String(contentBlockIndex)
            if contentBlocks[contentBlockIndex] == nil {
                contentBlocks[contentBlockIndex] = .reasoning
                parts.append(.reasoningStart(id: id))
            }
            parts.append(.reasoningDeltaPart(id: id, delta: reasoning))
        }
        if let signature = raw["contentBlockDelta"]?["delta"]?["reasoningContent"]?["signature"] {
            let payload: [String: JSONValue] = ["signature": signature]
            let id = String(contentBlockIndex)
            if contentBlocks[contentBlockIndex] == nil {
                contentBlocks[contentBlockIndex] = .reasoning
                parts.append(.reasoningStart(id: id))
            }
            parts.append(.reasoningDeltaPart(
                id: id,
                delta: "",
                providerMetadata: ["amazonBedrock": .object(payload), "bedrock": .object(payload)]
            ))
        }
        if let redactedData = raw["contentBlockDelta"]?["delta"]?["reasoningContent"]?["data"] {
            let payload: [String: JSONValue] = ["redactedData": redactedData]
            let id = String(contentBlockIndex)
            if contentBlocks[contentBlockIndex] == nil {
                contentBlocks[contentBlockIndex] = .reasoning
                parts.append(.reasoningStart(id: id))
            }
            parts.append(.reasoningDeltaPart(
                id: id,
                delta: "",
                providerMetadata: ["amazonBedrock": .object(payload), "bedrock": .object(payload)]
            ))
        }
        if let redactedContent = raw["contentBlockDelta"]?["delta"]?["reasoningContent"]?["redactedContent"]?.stringValue,
           !redactedContent.isEmpty {
            let id = String(contentBlockIndex)
            if contentBlocks[contentBlockIndex] == nil {
                contentBlocks[contentBlockIndex] = .reasoning
                parts.append(.reasoningStart(id: id))
            }
            redactedReasoningContent[contentBlockIndex, default: ""] += redactedContent
        }
        if let start = raw["contentBlockStart"],
           let toolUse = start["start"]?["toolUse"] {
            let index = start["contentBlockIndex"]?.intValue ?? 0
            let rawID = toolUse["toolUseId"]?.stringValue ?? "tool-call-\(index)"
            let id = bedrockNormalizeToolCallID(rawID, modelID: modelID)
            let name = toolUse["name"]?.stringValue ?? "tool-\(index)"
            contentBlocks[index] = .tool
            toolCalls[index] = BedrockStreamingToolCall(id: id, name: name, rawValue: toolUse)
            if name == jsonResponseToolName {
                isJsonResponseFromTool = true
                parts.append(.textStart(id: String(index)))
                return parts
            }
            parts.append(.toolInputStart(id: id, name: name))
        }
        if let delta = raw["contentBlockDelta"],
           let toolUse = delta["delta"]?["toolUse"] {
            let index = delta["contentBlockIndex"]?.intValue ?? 0
            var toolCall = toolCalls[index] ?? BedrockStreamingToolCall(id: "tool-call-\(index)", name: "tool-\(index)")
            let argumentsDelta = toolUse["input"]?.stringValue ?? ""
            toolCall.arguments += argumentsDelta
            toolCall.rawValue = toolUse
            toolCalls[index] = toolCall
            if toolCall.name == jsonResponseToolName {
                if !argumentsDelta.isEmpty {
                    parts.append(.textDeltaPart(id: String(index), delta: argumentsDelta))
                }
                return parts
            }
            parts.append(.toolCallDelta(id: toolCall.id, name: toolCall.name, argumentsDelta: argumentsDelta, index: index))
            if !argumentsDelta.isEmpty {
                parts.append(.toolInputDelta(id: toolCall.id, delta: argumentsDelta))
            }
        }
        if let stop = raw["contentBlockStop"],
           let index = stop["contentBlockIndex"]?.intValue {
            let block = contentBlocks.removeValue(forKey: index)
            let redactedContent = redactedReasoningContent.removeValue(forKey: index)
            if let toolCall = toolCalls.removeValue(forKey: index) {
                if toolCall.name == jsonResponseToolName {
                    isJsonResponseFromTool = true
                    parts.append(.textEnd(id: String(index)))
                    return parts
                }
                parts.append(.toolInputEnd(id: toolCall.id))
                parts.append(.toolCall(AIToolCall(
                    id: toolCall.id,
                    name: toolCall.name,
                    arguments: toolCall.arguments.isEmpty ? "{}" : toolCall.arguments,
                    rawValue: toolCall.rawValue
                )))
            } else if block == .reasoning {
                parts.append(.reasoningEnd(
                    id: String(index),
                    providerMetadata: bedrockRedactedReasoningProviderMetadata(redactedContent)
                ))
            } else if block == .text {
                parts.append(.textEnd(id: String(index)))
            }
        }
        if let stopReason = raw["messageStop"]?["stopReason"]?.stringValue {
            latestFinishReason = bedrockFinishReason(stopReason, isJsonResponseFromTool: isJsonResponseFromTool)
        }
        if raw["messageStop"] != nil {
            sawTerminalEvent = true
        }
        if let usage = bedrockUsage(from: raw["metadata"]?["usage"]) {
            latestUsage = usage
        }
        let metadata = bedrockProviderMetadata(fromStreamMetadata: raw["metadata"])
        if !metadata.isEmpty {
            latestProviderMetadata = bedrockMergeStreamProviderMetadata(latestProviderMetadata, metadata)
            parts.append(.metadata(metadata))
        }
        if raw["metadata"] != nil {
            sawTerminalEvent = true
        }
        return parts
    }

    mutating func finishParts() -> [LanguageStreamPart] {
        var parts: [LanguageStreamPart] = []
        for index in contentBlocks.keys.sorted() {
            switch contentBlocks[index] {
            case .reasoning:
                parts.append(.reasoningEnd(
                    id: String(index),
                    providerMetadata: bedrockRedactedReasoningProviderMetadata(redactedReasoningContent[index])
                ))
            case .text:
                parts.append(.textEnd(id: String(index)))
            case .tool, nil:
                break
            }
        }
        contentBlocks.removeAll()
        redactedReasoningContent.removeAll()
        parts.append(.finishMetadata(
            reason: latestFinishReason,
            usage: latestUsage,
            providerMetadata: latestProviderMetadata
        ))
        sawTerminalEvent = false
        return parts
    }

    mutating func markFailed() {
        latestFinishReason = "error"
        sawTerminalEvent = true
    }
}

private func bedrockRedactedReasoningProviderMetadata(_ redactedContent: String?) -> [String: JSONValue] {
    guard let redactedContent else { return [:] }
    let payload: JSONValue = .object(["redactedContent": .string(redactedContent)])
    return [
        "amazonBedrock": payload,
        "bedrock": payload
    ]
}

private func bedrockMergeStreamProviderMetadata(
    _ base: [String: JSONValue],
    _ update: [String: JSONValue]
) -> [String: JSONValue] {
    var result = base
    for (key, value) in update {
        if let baseObject = result[key]?.objectValue,
           let updateObject = value.objectValue {
            result[key] = .object(bedrockMergeStreamProviderMetadata(baseObject, updateObject))
        } else {
            result[key] = value
        }
    }
    return result
}

struct BedrockRawStreamItem {
    var rawValue: JSONValue
    var errorMessage: String?
    var eventType: String? = nil
}

func amazonBedrockStreamErrorMetadata(
    for type: String
) -> (statusCode: Int?, isRetryable: Bool?) {
    switch type {
    case "internalServerException", "InternalServerException":
        return (500, true)
    case "modelStreamErrorException", "ModelStreamErrorException":
        return (424, true)
    case "serviceUnavailableException", "ServiceUnavailableException":
        return (503, true)
    case "throttlingException", "ThrottlingException":
        return (429, true)
    case "validationException", "ValidationException":
        return (400, false)
    default:
        return (nil, nil)
    }
}

func amazonBedrockStreamProviderError(
    from item: BedrockRawStreamItem
) -> AIStreamProviderError? {
    let raw = item.rawValue
    guard item.errorMessage != nil || raw["type"]?.stringValue == "error" else {
        return nil
    }
    let type = item.eventType
        ?? raw["error"]?["type"]?.stringValue
        ?? raw["type"]?.stringValue
        ?? "error"
    let payload = raw[type] ?? raw["error"] ?? raw
    let message = payload["message"]?.stringValue
        ?? payload["Message"]?.stringValue
        ?? raw["message"]?.stringValue
        ?? item.errorMessage
        ?? "Amazon Bedrock stream failed with \(type)"
    let metadata = amazonBedrockStreamErrorMetadata(for: type)
    return AIStreamProviderError(
        message: message,
        type: type,
        code: payload["code"],
        statusCode: metadata.statusCode,
        isRetryable: metadata.isRetryable,
        data: raw
    )
}

struct BedrockRawStreamParser {
    private enum Format {
        case amazon(AmazonBedrockEventStreamParser)
        case serverSentEvents(ServerSentEventParser)
    }

    private var format: Format
    private let providerID: String
    private(set) var isDone = false

    init(providerID: String, contentType: String?) {
        self.providerID = providerID
        if contentType?.localizedCaseInsensitiveContains("application/vnd.amazon.eventstream") == true {
            format = .amazon(AmazonBedrockEventStreamParser(providerID: providerID))
        } else {
            format = .serverSentEvents(ServerSentEventParser())
        }
    }

    mutating func append(_ data: Data) throws -> [BedrockRawStreamItem] {
        guard !isDone else { return [] }
        switch format {
        case var .amazon(parser):
            let events = try parser.append(data)
            format = .amazon(parser)
            return events.map { event in
                BedrockRawStreamItem(
                    rawValue: event.rawValue,
                    errorMessage: event.errorMessage,
                    eventType: event.eventType
                )
            }
        case var .serverSentEvents(parser):
            let events = try parser.append(data)
            format = .serverSentEvents(parser)
            return try decodeServerSentEvents(events)
        }
    }

    mutating func finish() throws -> [BedrockRawStreamItem] {
        switch format {
        case var .amazon(parser):
            try parser.finish()
            format = .amazon(parser)
            return []
        case var .serverSentEvents(parser):
            let events = try parser.finish()
            format = .serverSentEvents(parser)
            return try decodeServerSentEvents(events)
        }
    }

    private mutating func decodeServerSentEvents(_ events: [ServerSentEvent]) throws -> [BedrockRawStreamItem] {
        var items: [BedrockRawStreamItem] = []
        for event in events {
            if event.data == "[DONE]" {
                isDone = true
                break
            }
            let raw: JSONValue
            do {
                raw = try decodeJSONBody(Data(event.data.utf8))
            } catch {
                throw AIError.invalidResponse(
                    provider: providerID,
                    message: "Invalid server-sent event JSON in Bedrock stream: \(error.localizedDescription)"
                )
            }
            let errorMessage: String?
            if raw["type"]?.stringValue == "error" {
                errorMessage = raw["error"]?["message"]?.stringValue
                    ?? raw["message"]?.stringValue
                    ?? "Amazon Bedrock stream returned an error event."
            } else {
                errorMessage = nil
            }
            items.append(BedrockRawStreamItem(rawValue: raw, errorMessage: errorMessage))
        }
        return items
    }
}

func streamFromBedrockResponse(
    providerID: String,
    response: AIHTTPStreamResponse,
    requestURL: URL,
    maxResponseBytes: Int? = nil,
    includeRawChunks: Bool = false,
    warnings: [AIWarning] = [],
    jsonResponseToolName: String? = nil,
    modelID: String = "",
    extractJSONObjectText: Bool = false,
    emit: @Sendable (LanguageStreamPart) -> Void
) async throws {
    guard (200..<300).contains(response.statusCode) else {
        let body = try await readResponseWithSizeLimit(
            response: response,
            url: requestURL.absoluteString,
            maxBytes: maxResponseBytes ?? AIDefaultMaxDownloadSize
        )
        throw bedrockHTTPStatusError(provider: providerID, response: AIHTTPResponse(
            statusCode: response.statusCode,
            headers: response.headers,
            body: body,
            url: requestURL
        ))
    }

    emit(.streamStart(warnings: warnings))
    var state = BedrockStreamState(
        jsonResponseToolName: jsonResponseToolName,
        modelID: modelID,
        jsonObjectTextExtractor: extractJSONObjectText ? BedrockJSONObjectTextExtractor() : nil
    )
    var parser = BedrockRawStreamParser(
        providerID: providerID,
        contentType: response.headerValue("content-type")
    )

    func emitItems(_ items: [BedrockRawStreamItem]) throws {
        for item in items {
            if includeRawChunks {
                emit(.raw(item.rawValue))
            }
            if let providerError = amazonBedrockStreamProviderError(from: item) {
                state.markFailed()
                emit(.providerError(providerError))
                continue
            }
            for part in state.parts(from: item.rawValue) {
                emit(part)
            }
        }
    }

    streamLoop: for try await data in response.body {
        try emitItems(parser.append(data))
        if parser.isDone {
            break streamLoop
        }
    }
    try emitItems(parser.finish())
    for part in state.finishParts() {
        emit(part)
    }
}

final class BedrockJSONObjectTextExtractor {
    private var started = false
    private var completed = false
    private var depth = 0
    private var isInsideString = false
    private var isEscaped = false

    func process(_ text: String) -> String {
        var result = ""

        for character in text {
            if completed {
                break
            }

            if !started {
                guard character == "{" else { continue }
                started = true
                depth = 1
                result.append(character)
                continue
            }

            result.append(character)

            if isEscaped {
                isEscaped = false
                continue
            }

            if character == "\\", isInsideString {
                isEscaped = true
                continue
            }

            if character == "\"" {
                isInsideString.toggle()
                continue
            }

            guard !isInsideString else { continue }
            if character == "{" {
                depth += 1
            } else if character == "}" {
                depth -= 1
                if depth == 0 {
                    completed = true
                }
            }
        }

        return result
    }
}

enum AmazonEventStreamHeaderValue: Equatable {
    case boolean(Bool)
    case byte(Int8)
    case short(Int16)
    case integer(Int32)
    case long(Int64)
    case byteArray(Data)
    case string(String)
    case timestamp(Int64)
    case uuid(Data)

    var stringValue: String? {
        guard case let .string(value) = self else { return nil }
        return value
    }
}

struct AmazonBedrockEventStreamEvent: Equatable {
    var messageType: String
    var eventType: String
    var payload: JSONValue
    var headers: [String: AmazonEventStreamHeaderValue]

    var rawValue: JSONValue {
        .object([eventType: payload])
    }

    var errorMessage: String? {
        let lowercasedEventType = eventType.lowercased()
        let isError = messageType == "exception"
            || messageType == "error"
            || lowercasedEventType.hasSuffix("exception")
            || lowercasedEventType.hasSuffix("error")
        guard isError else { return nil }
        let message = payload["message"]?.stringValue
            ?? payload["Message"]?.stringValue
            ?? headers[":error-message"]?.stringValue
        return message.map { "\(eventType): \($0)" } ?? eventType
    }
}

struct AmazonBedrockEventStreamParser {
    private var buffer = Data()
    private let providerID: String

    init(providerID: String) {
        self.providerID = providerID
    }

    mutating func append(_ data: Data) throws -> [AmazonBedrockEventStreamEvent] {
        buffer.append(data)
        var events: [AmazonBedrockEventStreamEvent] = []

        while buffer.count >= 4 {
            let totalLength = Int(readUInt32(buffer, at: 0))
            guard totalLength >= 16 else {
                throw invalidResponse("frame length \(totalLength) is smaller than the 16-byte EventStream overhead.")
            }
            guard buffer.count >= 8 else { break }

            let headersLength = Int(readUInt32(buffer, at: 4))
            guard headersLength <= totalLength - 16 else {
                throw invalidResponse("headers length \(headersLength) exceeds frame length \(totalLength).")
            }
            guard buffer.count >= 12 else { break }

            let expectedPreludeCRC = readUInt32(buffer, at: 8)
            let actualPreludeCRC = amazonEventStreamCRC32(buffer.subdata(in: 0..<8))
            guard expectedPreludeCRC == actualPreludeCRC else {
                throw invalidResponse("prelude CRC32 \(expectedPreludeCRC) does not match \(actualPreludeCRC).")
            }
            guard buffer.count >= totalLength else { break }

            let frame = buffer.subdata(in: 0..<totalLength)
            let expectedMessageCRC = readUInt32(frame, at: totalLength - 4)
            let actualMessageCRC = amazonEventStreamCRC32(frame.subdata(in: 0..<(totalLength - 4)))
            guard expectedMessageCRC == actualMessageCRC else {
                throw invalidResponse("message CRC32 \(expectedMessageCRC) does not match \(actualMessageCRC).")
            }

            let headers = try parseAmazonEventStreamHeaders(
                frame,
                start: 12,
                length: headersLength,
                providerID: providerID
            )
            let payloadStart = 12 + headersLength
            let payloadEnd = totalLength - 4
            let payloadData = frame.subdata(in: payloadStart..<payloadEnd)
            let payload: JSONValue
            if payloadData.isEmpty {
                payload = .object([:])
            } else {
                do {
                    payload = try decodeJSONBody(payloadData)
                } catch {
                    throw invalidResponse("event payload is not valid JSON: \(error.localizedDescription)")
                }
            }

            guard let messageType = headers[":message-type"]?.stringValue else {
                throw invalidResponse("frame is missing the string :message-type header.")
            }
            let eventType: String?
            switch messageType {
            case "event":
                eventType = headers[":event-type"]?.stringValue
            case "exception":
                eventType = headers[":exception-type"]?.stringValue ?? headers[":event-type"]?.stringValue
            case "error":
                eventType = headers[":error-code"]?.stringValue ?? headers[":event-type"]?.stringValue
            default:
                throw invalidResponse("unsupported :message-type value \(messageType).")
            }
            guard let eventType, !eventType.isEmpty else {
                throw invalidResponse("frame with message type \(messageType) is missing its event type header.")
            }

            events.append(AmazonBedrockEventStreamEvent(
                messageType: messageType,
                eventType: eventType,
                payload: payload,
                headers: headers
            ))
            buffer.removeSubrange(0..<totalLength)
        }

        return events
    }

    mutating func finish() throws {
        guard !buffer.isEmpty else { return }
        let expectedLength = buffer.count >= 4 ? Int(readUInt32(buffer, at: 0)) : nil
        let suffix = expectedLength.map { "; expected \($0) bytes" } ?? ""
        throw invalidResponse("truncated EventStream frame at EOF (received \(buffer.count) bytes\(suffix)).")
    }

    private func invalidResponse(_ message: String) -> AIError {
        .invalidResponse(provider: providerID, message: "Invalid Amazon EventStream frame: \(message)")
    }
}

func parseAmazonBedrockEventStream(_ data: Data, providerID: String = "amazon-bedrock") throws -> [JSONValue] {
    var parser = AmazonBedrockEventStreamParser(providerID: providerID)
    let events = try parser.append(data)
    try parser.finish()
    return events.map(\.rawValue)
}

private func parseAmazonEventStreamHeaders(
    _ data: Data,
    start: Int,
    length: Int,
    providerID: String
) throws -> [String: AmazonEventStreamHeaderValue] {
    let end = start + length
    guard start >= 0, length >= 0, end <= data.count else {
        throw AIError.invalidResponse(provider: providerID, message: "Invalid Amazon EventStream frame: header section is out of bounds.")
    }

    var headers: [String: AmazonEventStreamHeaderValue] = [:]
    var offset = start

    func invalidHeader(_ message: String) -> AIError {
        .invalidResponse(provider: providerID, message: "Invalid Amazon EventStream frame: \(message)")
    }

    func requireBytes(_ count: Int, context: String) throws {
        guard count >= 0, offset <= end - count else {
            throw invalidHeader("truncated \(context) in headers.")
        }
    }

    while offset < end {
        try requireBytes(1, context: "header name length")
        let nameLength = Int(data[offset])
        offset += 1
        guard nameLength > 0 else {
            throw invalidHeader("header name must not be empty.")
        }
        try requireBytes(nameLength, context: "header name")
        let nameData = data.subdata(in: offset..<(offset + nameLength))
        guard let name = String(data: nameData, encoding: .utf8) else {
            throw invalidHeader("header name is not valid UTF-8.")
        }
        offset += nameLength

        try requireBytes(1, context: "header type")
        let type = data[offset]
        offset += 1

        let value: AmazonEventStreamHeaderValue
        switch type {
        case 0:
            value = .boolean(true)
        case 1:
            value = .boolean(false)
        case 2:
            try requireBytes(1, context: "byte header")
            value = .byte(Int8(bitPattern: data[offset]))
            offset += 1
        case 3:
            try requireBytes(2, context: "short header")
            value = .short(Int16(bitPattern: readUInt16(data, at: offset)))
            offset += 2
        case 4:
            try requireBytes(4, context: "integer header")
            value = .integer(Int32(bitPattern: readUInt32(data, at: offset)))
            offset += 4
        case 5:
            try requireBytes(8, context: "long header")
            value = .long(Int64(bitPattern: readUInt64(data, at: offset)))
            offset += 8
        case 6, 7:
            try requireBytes(2, context: "variable-length header size")
            let valueLength = Int(readUInt16(data, at: offset))
            offset += 2
            try requireBytes(valueLength, context: "variable-length header value")
            let valueData = data.subdata(in: offset..<(offset + valueLength))
            offset += valueLength
            if type == 6 {
                value = .byteArray(valueData)
            } else {
                guard let string = String(data: valueData, encoding: .utf8) else {
                    throw invalidHeader("string header value is not valid UTF-8.")
                }
                value = .string(string)
            }
        case 8:
            try requireBytes(8, context: "timestamp header")
            value = .timestamp(Int64(bitPattern: readUInt64(data, at: offset)))
            offset += 8
        case 9:
            try requireBytes(16, context: "UUID header")
            value = .uuid(data.subdata(in: offset..<(offset + 16)))
            offset += 16
        default:
            throw invalidHeader("unrecognized header type tag \(type).")
        }
        guard headers[name] == nil else {
            throw invalidHeader("duplicate header \(name).")
        }
        headers[name] = value
    }

    return headers
}

func amazonEventStreamCRC32(_ data: Data) -> UInt32 {
    var crc = UInt32.max
    for byte in data {
        crc = (crc >> 8) ^ amazonEventStreamCRC32Table[Int((crc ^ UInt32(byte)) & 0xff)]
    }
    return crc ^ UInt32.max
}

private let amazonEventStreamCRC32Table: [UInt32] = (0..<256).map { index in
    var value = UInt32(index)
    for _ in 0..<8 {
        value = (value & 1) == 1 ? (value >> 1) ^ 0xedb8_8320 : value >> 1
    }
    return value
}

func readUInt16(_ data: Data, at offset: Int) -> UInt16 {
    (UInt16(data[offset]) << 8) | UInt16(data[offset + 1])
}

func readUInt32(_ data: Data, at offset: Int) -> UInt32 {
    (UInt32(data[offset]) << 24)
        | (UInt32(data[offset + 1]) << 16)
        | (UInt32(data[offset + 2]) << 8)
        | UInt32(data[offset + 3])
}

private func readUInt64(_ data: Data, at offset: Int) -> UInt64 {
    (UInt64(readUInt32(data, at: offset)) << 32) | UInt64(readUInt32(data, at: offset + 4))
}