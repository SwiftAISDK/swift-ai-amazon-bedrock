import SwiftAIOpenAICompatible
import SwiftAIOpenAI
import SwiftAIAnthropic
import Foundation
import SwiftAIProvider
import SwiftAITransport
import SwiftAIProviderUtilities


struct BedrockStreamingToolCall {
    var id: String
    var name: String
    var arguments: String = ""
    var rawValue: JSONValue?
    var isJsonResponseTool = false
}

/// Mirrors the upstream `doStream` transform of `@ai-sdk/amazon-bedrock`:
/// one Converse stream event in, zero or more canonical stream parts out.
struct BedrockStreamState {
    private enum ContentBlock {
        case text
        case reasoning(redactedContent: String?)
        case tool(BedrockStreamingToolCall)
    }

    var latestFinishReason: String? = "interrupted"
    var latestUsage: TokenUsage?
    var jsonResponseToolName: String?
    var modelID: String
    var jsonObjectTextExtractor: BedrockJSONObjectTextExtractor?
    var isJsonResponseFromTool = false
    private var contentBlocks: [Int: ContentBlock] = [:]
    private var latestProviderMetadata: [String: JSONValue]?
    private var stopSequence: JSONValue = .null

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

        if let messageStop = raw["messageStop"], messageStop != .null {
            latestFinishReason = bedrockFinishReason(
                messageStop["stopReason"]?.stringValue ?? "",
                isJsonResponseFromTool: isJsonResponseFromTool
            )
            let stop = messageStop["additionalModelResponseFields"]?["delta"]?["stop_sequence"]
            stopSequence = stop?.stringValue.map(JSONValue.string) ?? .null
        }

        if let metadata = raw["metadata"], metadata != .null {
            if let usage = bedrockUsage(from: metadata["usage"]) {
                latestUsage = usage
            }
            let payload = bedrockMetadataPayload(
                trace: metadata["trace"],
                performanceConfig: metadata["performanceConfig"],
                serviceTier: metadata["serviceTier"],
                usage: metadata["usage"]
            )
            if !payload.isEmpty {
                latestProviderMetadata = [
                    "amazonBedrock": .object(payload),
                    "bedrock": .object(payload)
                ]
            }
        }

        let blockStart = raw["contentBlockStart"]
        if let index = blockStart?["contentBlockIndex"]?.intValue,
           bedrockNonNull(blockStart?["start"]?["toolUse"]) == nil {
            contentBlocks[index] = .text
            parts.append(.textStart(id: String(index)))
        }

        let blockDelta = raw["contentBlockDelta"]
        let deltaIndex = blockDelta?["contentBlockIndex"]?.intValue ?? 0
        if let text = blockDelta?["delta"]?["text"]?.stringValue, !text.isEmpty {
            if contentBlocks[deltaIndex] == nil {
                contentBlocks[deltaIndex] = .text
                parts.append(.textStart(id: String(deltaIndex)))
            }
            let textDelta = jsonObjectTextExtractor?.process(text) ?? text
            if !textDelta.isEmpty {
                parts.append(.textDeltaPart(id: String(deltaIndex), delta: textDelta))
            }
        }

        if let index = raw["contentBlockStop"]?["contentBlockIndex"]?.intValue,
           let block = contentBlocks.removeValue(forKey: index) {
            let id = String(index)
            switch block {
            case let .reasoning(redactedContent):
                parts.append(.reasoningEnd(
                    id: id,
                    providerMetadata: bedrockRedactedReasoningProviderMetadata(redactedContent)
                ))
            case .text:
                parts.append(.textEnd(id: id))
            case let .tool(toolCall):
                if toolCall.isJsonResponseTool {
                    isJsonResponseFromTool = true
                    parts.append(.textStart(id: id))
                    parts.append(.textDeltaPart(id: id, delta: toolCall.arguments))
                    parts.append(.textEnd(id: id))
                } else {
                    parts.append(.toolInputEnd(id: toolCall.id))
                    parts.append(.toolCall(AIToolCall(
                        id: toolCall.id,
                        name: toolCall.name,
                        arguments: toolCall.arguments.isEmpty ? "{}" : toolCall.arguments,
                        rawValue: toolCall.rawValue
                    )))
                }
            }
        }

        if let reasoningContent = bedrockNonNull(blockDelta?["delta"]?["reasoningContent"]) {
            let id = String(deltaIndex)
            func startReasoningIfNeeded() {
                if contentBlocks[deltaIndex] == nil {
                    contentBlocks[deltaIndex] = .reasoning(redactedContent: nil)
                    parts.append(.reasoningStart(id: id))
                }
            }
            if let text = reasoningContent["text"]?.stringValue, !text.isEmpty {
                startReasoningIfNeeded()
                parts.append(.reasoningDeltaPart(id: id, delta: text))
            } else if let signature = reasoningContent["signature"]?.stringValue, !signature.isEmpty {
                startReasoningIfNeeded()
                let payload: JSONValue = .object(["signature": .string(signature)])
                parts.append(.reasoningDeltaPart(
                    id: id,
                    delta: "",
                    providerMetadata: ["amazonBedrock": payload, "bedrock": payload]
                ))
            } else if let data = reasoningContent["data"]?.stringValue, !data.isEmpty {
                startReasoningIfNeeded()
                let payload: JSONValue = .object(["redactedData": .string(data)])
                parts.append(.reasoningDeltaPart(
                    id: id,
                    delta: "",
                    providerMetadata: ["amazonBedrock": payload, "bedrock": payload]
                ))
            } else if let redactedContent = reasoningContent["redactedContent"]?.stringValue, !redactedContent.isEmpty {
                startReasoningIfNeeded()
                // Accumulate and attach once through reasoning-end: per-delta
                // metadata would be last-write-wins and drop earlier chunks.
                if case let .reasoning(existing) = contentBlocks[deltaIndex] {
                    contentBlocks[deltaIndex] = .reasoning(redactedContent: (existing ?? "") + redactedContent)
                }
            }
        }

        if let toolUse = bedrockNonNull(blockStart?["start"]?["toolUse"]) {
            let index = blockStart?["contentBlockIndex"]?.intValue ?? 0
            let name = toolUse["name"]?.stringValue ?? ""
            let id = bedrockNormalizeToolCallID(toolUse["toolUseId"]?.stringValue ?? "", modelID: modelID)
            let isJsonResponseTool = jsonResponseToolName != nil && name == jsonResponseToolName
            contentBlocks[index] = .tool(BedrockStreamingToolCall(
                id: id,
                name: name,
                rawValue: toolUse,
                isJsonResponseTool: isJsonResponseTool
            ))
            if !isJsonResponseTool {
                parts.append(.toolInputStart(id: id, name: name))
            }
        }

        if let toolUse = bedrockNonNull(blockDelta?["delta"]?["toolUse"]),
           case var .tool(toolCall) = contentBlocks[deltaIndex] {
            let delta = toolUse["input"]?.stringValue ?? ""
            if !toolCall.isJsonResponseTool {
                parts.append(.toolCallDelta(id: toolCall.id, name: toolCall.name, argumentsDelta: delta, index: deltaIndex))
                parts.append(.toolInputDelta(id: toolCall.id, delta: delta))
            }
            toolCall.arguments += delta
            contentBlocks[deltaIndex] = .tool(toolCall)
        }

        return parts
    }

    mutating func finishParts() -> [LanguageStreamPart] {
        var providerMetadata = latestProviderMetadata
        if isJsonResponseFromTool || stopSequence != .null {
            var payload = (providerMetadata?["amazonBedrock"] ?? providerMetadata?["bedrock"])?.objectValue ?? [:]
            if isJsonResponseFromTool {
                payload["isJsonResponseFromTool"] = .bool(true)
            }
            payload["stopSequence"] = stopSequence
            var updated = providerMetadata ?? [:]
            updated["amazonBedrock"] = .object(payload)
            updated["bedrock"] = .object(payload)
            providerMetadata = updated
        }
        return [.finishMetadata(
            reason: latestFinishReason,
            usage: latestUsage,
            providerMetadata: providerMetadata ?? [:]
        )]
    }

    mutating func markFailed() {
        latestFinishReason = "error"
    }
}

func bedrockNonNull(_ value: JSONValue?) -> JSONValue? {
    guard let value, value != .null else { return nil }
    return value
}

private func bedrockRedactedReasoningProviderMetadata(_ redactedContent: String?) -> [String: JSONValue] {
    guard let redactedContent else { return [:] }
    let payload: JSONValue = .object(["redactedContent": .string(redactedContent)])
    return [
        "amazonBedrock": payload,
        "bedrock": payload
    ]
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
    emit(.responseMetadata(bedrockResponseMetadata(headers: response.headers, modelID: modelID)))
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

    /// Upstream wraps the decoded payload under its event type after deleting
    /// the Bedrock `p` padding field.
    var rawValue: JSONValue {
        var value = payload
        if var object = payload.objectValue {
            object.removeValue(forKey: "p")
            value = .object(object)
        }
        return .object([eventType: value])
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
/// Upstream response metadata: the id comes from `x-amzn-requestid` and the
/// timestamp from the `date` response header; both are absent otherwise.
func bedrockResponseMetadata(headers: [String: String], modelID: String, body: JSONValue? = nil) -> AIResponseMetadata {
    func header(_ name: String) -> String? {
        headers.first { $0.key.caseInsensitiveCompare(name) == .orderedSame }?.value
    }
    return AIResponseMetadata(
        id: header("x-amzn-requestid"),
        timestamp: header("date").flatMap(bedrockParseHTTPDate),
        modelID: modelID,
        headers: headers,
        body: body
    )
}

func bedrockParseHTTPDate(_ value: String) -> Date? {
    let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
    for format in ["EEE, dd MMM yyyy HH:mm:ss zzz", "EEEE, dd-MMM-yy HH:mm:ss zzz", "EEE MMM d HH:mm:ss yyyy"] {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.timeZone = TimeZone(secondsFromGMT: 0)
        formatter.dateFormat = format
        if let date = formatter.date(from: trimmed) { return date }
    }
    let iso = ISO8601DateFormatter()
    iso.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
    if let date = iso.date(from: trimmed) { return date }
    iso.formatOptions = [.withInternetDateTime]
    return iso.date(from: trimmed)
}
