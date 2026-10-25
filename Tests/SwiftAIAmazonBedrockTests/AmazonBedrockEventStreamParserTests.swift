@testable import SwiftAIAmazonBedrock
@testable import SwiftAIAnthropic
@testable import SwiftAI
@testable import SwiftAIProvider
@testable import SwiftAIProviderUtilities
@testable import SwiftAITransport
import Foundation
import Testing

@Test func amazonBedrockEventStreamCRC32MatchesStandardVector() {
    #expect(amazonEventStreamCRC32(Data("123456789".utf8)) == 0xcbf4_3926)
}

@Test func amazonBedrockEventStreamParserEmitsFramesAcrossEveryByteBoundary() throws {
    let response = amazonEventStreamResponse([
        ("contentBlockDelta", #"{"contentBlockIndex":0,"delta":{"text":"日🙂"}}"#),
        ("messageStop", #"{"stopReason":"end_turn"}"#)
    ])
    var parser = AmazonBedrockEventStreamParser(providerID: "amazon-bedrock")
    var events: [AmazonBedrockEventStreamEvent] = []

    for byte in response.body {
        events.append(contentsOf: try parser.append(Data([byte])))
    }
    try parser.finish()

    #expect(events.count == 2)
    #expect(events[0].messageType == "event")
    #expect(events[0].eventType == "contentBlockDelta")
    #expect(events[0].payload["delta"]?["text"]?.stringValue == "日🙂")
    #expect(events[0].headers[":message-type"] == .string("event"))
    #expect(events[1].eventType == "messageStop")
}

@Test func amazonBedrockEventStreamParserEmitsMultipleFramesFromOneChunk() throws {
    let response = amazonEventStreamResponse([
        ("contentBlockDelta", #"{"delta":{"text":"bed"}}"#),
        ("contentBlockDelta", #"{"delta":{"text":"rock"}}"#)
    ])
    var parser = AmazonBedrockEventStreamParser(providerID: "amazon-bedrock")

    let events = try parser.append(response.body)
    try parser.finish()

    #expect(events.map { $0.payload["delta"]?["text"]?.stringValue } == ["bed", "rock"])
}

@Test func amazonBedrockEventStreamParserRejectsInvalidLengthsAndChecksums() throws {
    let valid = amazonEventStreamResponse([("chunk", #"{"value":1}"#)]).body

    var tooShort = valid
    tooShort[3] = 15
    expectInvalidAmazonEventStream(tooShort, containing: "frame length 15")

    var invalidHeaderLength = valid
    invalidHeaderLength[4] = 0xff
    expectInvalidAmazonEventStream(invalidHeaderLength, containing: "headers length")

    var invalidPreludeCRC = valid
    invalidPreludeCRC[8] ^= 0xff
    expectInvalidAmazonEventStream(invalidPreludeCRC, containing: "prelude CRC32")

    var invalidMessageCRC = valid
    invalidMessageCRC[invalidMessageCRC.count - 1] ^= 0xff
    expectInvalidAmazonEventStream(invalidMessageCRC, containing: "message CRC32")
}

@Test func amazonBedrockEventStreamParserRejectsMalformedHeaders() throws {
    var frame = amazonEventStreamResponse([("chunk", #"{"value":1}"#)]).body
    let firstHeaderNameLength = Int(frame[12])
    let firstHeaderTypeOffset = 12 + 1 + firstHeaderNameLength
    frame[firstHeaderTypeOffset] = 0xff
    rewriteAmazonEventStreamMessageCRC(&frame)

    expectInvalidAmazonEventStream(frame, containing: "unrecognized header type tag 255")

    let duplicateHeaderFrame = amazonEventStreamFrame(
        messageType: "event",
        eventTypeHeader: ":event-type",
        eventType: "chunk",
        payload: Data(#"{"value":1}"#.utf8),
        additionalStringHeaders: [(":message-type", "event")]
    )
    expectInvalidAmazonEventStream(duplicateHeaderFrame, containing: "duplicate header :message-type")

    let unknownMessageTypeFrame = amazonEventStreamFrame(
        messageType: "mystery",
        eventTypeHeader: ":event-type",
        eventType: "chunk",
        payload: Data(#"{"value":1}"#.utf8)
    )
    expectInvalidAmazonEventStream(unknownMessageTypeFrame, containing: "unsupported :message-type value mystery")
}

@Test func amazonBedrockEventStreamParserRejectsTruncatedFrameAtEOF() throws {
    let frame = amazonEventStreamResponse([("chunk", #"{"value":1}"#)]).body
    var parser = AmazonBedrockEventStreamParser(providerID: "amazon-bedrock")

    let events = try parser.append(Data(frame.dropLast()))
    #expect(events.isEmpty)
    do {
        try parser.finish()
        Issue.record("Expected a truncated EventStream error.")
    } catch let AIError.invalidResponse(provider, message) {
        #expect(provider == "amazon-bedrock")
        #expect(message.contains("truncated EventStream frame at EOF"))
    }
}

@Test func amazonBedrockEventStreamParserSurfacesSmithyExceptionAndErrorFrames() throws {
    let exceptionFrame = amazonEventStreamFrame(
        messageType: "exception",
        eventTypeHeader: ":exception-type",
        eventType: "throttlingException",
        payload: Data(#"{"message":"slow down"}"#.utf8)
    )
    let errorFrame = amazonEventStreamFrame(
        messageType: "error",
        eventTypeHeader: ":error-code",
        eventType: "InternalFailure",
        payload: Data("{}".utf8),
        additionalStringHeaders: [(":error-message", "service failed")]
    )
    var parser = AmazonBedrockEventStreamParser(providerID: "amazon-bedrock")

    let events = try parser.append(exceptionFrame + errorFrame)
    try parser.finish()

    #expect(events.count == 2)
    #expect(events[0].messageType == "exception")
    #expect(events[0].eventType == "throttlingException")
    #expect(events[0].errorMessage == "throttlingException: slow down")
    #expect(events[0].rawValue["throttlingException"]?["message"]?.stringValue == "slow down")
    #expect(events[1].messageType == "error")
    #expect(events[1].eventType == "InternalFailure")
    #expect(events[1].errorMessage == "InternalFailure: service failed")
}

@Test func amazonBedrockConverseEmitsSmithyExceptionAsTypedError() async throws {
    let frame = amazonEventStreamFrame(
        messageType: "exception",
        eventTypeHeader: ":exception-type",
        eventType: "validationException",
        payload: Data(#"{"message":"bad request"}"#.utf8)
    )
    let transport = RecordingTransport(response: AIHTTPResponse(
        statusCode: 200,
        headers: ["content-type": "application/vnd.amazon.eventstream"],
        body: frame
    ))
    let provider = try AIProviders.amazonBedrock(configuration: AmazonBedrockProviderSettings(
        region: "us-east-1",
        apiKey: "bedrock-key",
        transport: transport
    ))
    let model = try provider.languageModel("anthropic.claude-3-haiku-20240307-v1:0")

    var parts: [LanguageStreamPart] = []
    for try await part in model.stream(LanguageModelRequest(messages: [.user("Hi")])) {
        parts.append(part)
    }
    let error = try #require(parts.compactMap(\.streamProviderError).first)
    #expect(error.message == "bad request")
    #expect(error.type == "validationException")
    #expect(error.statusCode == 400)
    #expect(!error.isRetryable)
    #expect(parts.contains { if case let .finishMetadata(reason, _, _) = $0 { reason == "error" } else { false } })
}

@Test func amazonBedrockAnthropicEmitsSmithyErrorAsTypedError() async throws {
    let frame = amazonEventStreamFrame(
        messageType: "error",
        eventTypeHeader: ":error-code",
        eventType: "InternalFailure",
        payload: Data("{}".utf8),
        additionalStringHeaders: [(":error-message", "service failed")]
    )
    let transport = RecordingTransport(response: AIHTTPResponse(
        statusCode: 200,
        headers: ["content-type": "application/vnd.amazon.eventstream"],
        body: frame
    ))
    let provider = try AIProviders.amazonBedrockAnthropic(configuration: AmazonBedrockProviderSettings(
        region: "us-east-1",
        apiKey: "bedrock-key",
        transport: transport
    ))
    let model = try provider.languageModel("anthropic.claude-3-haiku-20240307-v1:0")

    var parts: [LanguageStreamPart] = []
    for try await part in model.stream(LanguageModelRequest(messages: [.user("Hi")])) {
        parts.append(part)
    }
    let error = try #require(parts.compactMap(\.streamProviderError).first)
    #expect(error.message == "InternalFailure: service failed")
    #expect(error.type == "InternalFailure")
    #expect(error.statusCode == nil)
    #expect(!error.isRetryable)
    #expect(parts.contains { if case let .finishMetadata(reason, _, _) = $0 { reason == "error" } else { false } })
}

@Test func amazonBedrockConverseYieldsDeltaBeforeEventStreamEOF() async throws {
    let firstFrame = amazonEventStreamResponse([
        ("contentBlockDelta", #"{"contentBlockIndex":0,"delta":{"text":"bed"}}"#)
    ]).body
    let finalFrames = amazonEventStreamResponse([
        ("messageStop", #"{"stopReason":"end_turn"}"#),
        ("metadata", #"{"usage":{"inputTokens":1,"outputTokens":1,"totalTokens":2}}"#)
    ]).body
    let transport = BedrockGatedStreamingTransport(firstChunk: firstFrame, finalChunk: finalFrames)
    let provider = try AIProviders.amazonBedrock(configuration: AmazonBedrockProviderSettings(
        region: "us-east-1",
        apiKey: "bedrock-key",
        transport: transport
    ))
    let model = try provider.languageModel("anthropic.claude-3-haiku-20240307-v1:0")
    let receivedDelta = AIDelayedPromise<String>()
    let consumer = Task {
        var parts: [LanguageStreamPart] = []
        for try await part in model.stream(LanguageModelRequest(messages: [.user("Hi")])) {
            parts.append(part)
            if case let .textDeltaPart(_, delta, _) = part, receivedDelta.isPending {
                receivedDelta.resolve(delta)
            }
        }
        return parts
    }
    defer { transport.releaseBody.resolve(()) }

    #expect(try await awaitBedrockTestSignal(receivedDelta) == "bed")
    #expect(transport.releaseBody.isPending)
    #expect(transport.sendCallCount == 0)
    #expect(transport.streamCallCount == 1)
    #expect(transport.requests.first?.headers["Authorization"] == "Bearer bedrock-key")

    transport.releaseBody.resolve(())
    let parts = try await consumer.value
    #expect(parts.contains { part in
        if case let .finishMetadata(_, usage, _) = part {
            return usage?.totalTokens == 2
        }
        return false
    })
}

@Test func amazonBedrockJSONResponseToolYieldsCanonicalDeltaBeforeBlockStop() async throws {
    let firstFrames = amazonEventStreamResponse([
        ("contentBlockStart", #"{"contentBlockIndex":0,"start":{"toolUse":{"toolUseId":"json-1","name":"json"}}}"#),
        ("contentBlockDelta", #"{"contentBlockIndex":0,"delta":{"toolUse":{"input":"{\"name\":"}}}"#)
    ]).body
    let finalFrames = amazonEventStreamResponse([
        ("contentBlockDelta", #"{"contentBlockIndex":0,"delta":{"toolUse":{"input":"\"Test\"}"}}}"#),
        ("contentBlockStop", #"{"contentBlockIndex":0}"#),
        ("messageStop", #"{"stopReason":"tool_use"}"#),
        ("metadata", #"{"usage":{"inputTokens":1,"outputTokens":1,"totalTokens":2}}"#)
    ]).body
    let transport = BedrockGatedStreamingTransport(firstChunk: firstFrames, finalChunk: finalFrames)
    let provider = try AIProviders.amazonBedrock(configuration: AmazonBedrockProviderSettings(
        region: "us-east-1",
        apiKey: "bedrock-key",
        transport: transport
    ))
    let model = try provider.languageModel("mistral.large")
    let receivedDelta = AIDelayedPromise<String>()
    let consumer = Task {
        var parts: [LanguageStreamPart] = []
        for try await part in model.stream(LanguageModelRequest(
            messages: [.user("Return JSON")],
            responseFormat: .json(schema: ["type": "object", "properties": ["name": ["type": "string"]]])
        )) {
            parts.append(part)
            if case let .textDeltaPart(_, delta, _) = part, receivedDelta.isPending {
                receivedDelta.resolve(delta)
            }
        }
        return parts
    }
    defer { transport.releaseBody.resolve(()) }

    #expect(try await awaitBedrockTestSignal(receivedDelta) == #"{"name":"#)
    #expect(transport.releaseBody.isPending)
    transport.releaseBody.resolve(())
    let parts = try await consumer.value
    #expect(parts.contains(.textStart(id: "0")))
    #expect(parts.contains(.textEnd(id: "0")))
    #expect(parts.filter {
        if case .finishMetadata = $0 { return true }
        return false
    }.count == 1)
    #expect(parts.contains {
        if case let .finishMetadata(reason, usage, _) = $0 {
            return reason == "stop" && usage?.totalTokens == 2
        }
        return false
    })
}

@Test func amazonBedrockAnthropicYieldsDeltaBeforeEventStreamEOF() async throws {
    let anthropicDelta = #"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"rock"}}"#
    let anthropicFinish = #"{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"input_tokens":1,"output_tokens":1}}"#
    let firstFrame = amazonEventStreamResponse([
        ("chunk", #"{"bytes":"\#(Data(anthropicDelta.utf8).base64EncodedString())"}"#)
    ]).body
    let finalFrames = amazonEventStreamResponse([
        ("chunk", #"{"bytes":"\#(Data(anthropicFinish.utf8).base64EncodedString())"}"#),
        ("messageStop", "{}")
    ]).body
    let transport = BedrockGatedStreamingTransport(firstChunk: firstFrame, finalChunk: finalFrames)
    let provider = try AIProviders.amazonBedrockAnthropic(configuration: AmazonBedrockProviderSettings(
        region: "us-east-1",
        apiKey: "bedrock-key",
        transport: transport
    ))
    let model = try provider.languageModel("anthropic.claude-3-haiku-20240307-v1:0")
    let receivedDelta = AIDelayedPromise<String>()
    let consumer = Task {
        var parts: [LanguageStreamPart] = []
        for try await part in model.stream(LanguageModelRequest(messages: [.user("Hi")])) {
            parts.append(part)
            if case let .textDeltaPart(_, delta, _) = part, receivedDelta.isPending {
                receivedDelta.resolve(delta)
            }
        }
        return parts
    }
    defer { transport.releaseBody.resolve(()) }

    #expect(try await awaitBedrockTestSignal(receivedDelta) == "rock")
    #expect(transport.releaseBody.isPending)
    #expect(transport.sendCallCount == 0)
    #expect(transport.streamCallCount == 1)
    #expect(transport.requests.first?.headers["Authorization"] == "Bearer bedrock-key")

    transport.releaseBody.resolve(())
    let parts = try await consumer.value
    #expect(parts.contains { part in
        if case let .finishMetadata(reason, _, _) = part {
            return reason == "stop"
        }
        return false
    })
}

@Test func amazonBedrockAnthropicMessageStopTerminatesAndCancelsOpenBody() async throws {
    let anthropicDelta = #"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"done"}}"#
    let anthropicFinish = #"{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"input_tokens":1,"output_tokens":1}}"#
    let anthropicStop = #"{"type":"message_stop"}"#
    let openBodyFrames = amazonEventStreamResponse([
        ("chunk", #"{"bytes":"\#(Data(anthropicDelta.utf8).base64EncodedString())"}"#),
        ("chunk", #"{"bytes":"\#(Data(anthropicFinish.utf8).base64EncodedString())"}"#),
        ("chunk", #"{"bytes":"\#(Data(anthropicStop.utf8).base64EncodedString())"}"#),
        ("messageStop", "{}")
    ]).body
    let transport = BedrockGatedStreamingTransport(
        firstChunk: openBodyFrames,
        finalChunk: amazonEventStreamResponse([
            ("internalServerException", #"{"message":"must not be observed"}"#)
        ]).body
    )
    let provider = try AIProviders.amazonBedrockAnthropic(configuration: AmazonBedrockProviderSettings(
        region: "us-east-1",
        apiKey: "bedrock-key",
        transport: transport
    ))
    let model = try provider.languageModel("anthropic.claude-3-haiku-20240307-v1:0")
    let completed = AIDelayedPromise<Void>()
    let consumer = Task {
        do {
            var parts: [LanguageStreamPart] = []
            for try await part in model.stream(LanguageModelRequest(messages: [.user("Hi")])) {
                parts.append(part)
            }
            completed.resolve(())
            return parts
        } catch {
            completed.reject(error)
            throw error
        }
    }
    defer { transport.releaseBody.resolve(()) }

    _ = try await awaitBedrockTestSignal(completed)
    let parts = try await consumer.value
    _ = try await awaitBedrockTestSignal(transport.bodyTerminated)
    _ = try await awaitBedrockTestSignal(transport.producerCancelled)

    #expect(transport.releaseBody.isPending)
    #expect(parts.contains(.textDeltaPart(id: "0", delta: "done")))
    #expect(parts.contains { part in
        if case let .finishMetadata(reason, usage, _) = part {
            return reason == "stop"
                && usage?.inputTokens == 1
                && usage?.outputTokens == 1
        }
        return false
    })
}

@Test func amazonBedrockAnthropicSignsStreamingRequestWithSigV4() async throws {
    let fixedDate = DateComponents(
        calendar: Calendar(identifier: .gregorian),
        timeZone: TimeZone(secondsFromGMT: 0),
        year: 2024,
        month: 3,
        day: 15,
        hour: 0,
        minute: 0,
        second: 0
    ).date!
    let transport = RecordingTransport(response: amazonEventStreamResponse([
        ("messageStop", "{}")
    ]))
    let provider = try AIProviders.amazonBedrockAnthropic(configuration: AmazonBedrockProviderSettings(
        region: "us-east-1",
        accessKeyID: "AKIDEXAMPLE",
        secretAccessKey: "secret",
        transport: transport,
        date: { fixedDate }
    ))
    let model = try provider.languageModel("anthropic.claude-3-haiku-20240307-v1:0")

    for try await _ in model.stream(LanguageModelRequest(messages: [.user("Hi")])) {}

    #expect(await transport.sendRequests().isEmpty)
    let request = try #require(await transport.streamRequests().first)
    #expect(request.headers["x-amz-date"] == "20240315T000000Z")
    #expect(request.headers["authorization"]?.contains("Credential=AKIDEXAMPLE/20240315/us-east-1/bedrock/aws4_request") == true)
}

@Test func amazonBedrockConverseCancelsBodyWhenConsumerStopsAfterFirstDelta() async throws {
    let firstFrame = amazonEventStreamResponse([
        ("contentBlockDelta", #"{"contentBlockIndex":0,"delta":{"text":"first"}}"#)
    ]).body
    let transport = BedrockGatedStreamingTransport(
        firstChunk: firstFrame,
        finalChunk: amazonEventStreamResponse([("messageStop", #"{"stopReason":"end_turn"}"#)]).body
    )
    let provider = try AIProviders.amazonBedrock(configuration: AmazonBedrockProviderSettings(
        region: "us-east-1",
        apiKey: "bedrock-key",
        transport: transport
    ))
    let model = try provider.languageModel("anthropic.claude-3-haiku-20240307-v1:0")
    let receivedDelta = AIDelayedPromise<Void>()
    let consumer = Task {
        for try await part in model.stream(LanguageModelRequest(messages: [.user("Hi")])) {
            if case .textDeltaPart = part {
                receivedDelta.resolve(())
                return
            }
        }
    }
    defer { transport.releaseBody.resolve(()) }

    _ = try await awaitBedrockTestSignal(receivedDelta)
    try await consumer.value
    _ = try await awaitBedrockTestSignal(transport.bodyTerminated)
    #expect(transport.releaseBody.isPending)
}

private func expectInvalidAmazonEventStream(_ data: Data, containing expectedMessage: String) {
    do {
        _ = try parseAmazonBedrockEventStream(data)
        Issue.record("Expected an invalid Amazon EventStream error.")
    } catch let AIError.invalidResponse(provider, message) {
        #expect(provider == "amazon-bedrock")
        #expect(message.contains(expectedMessage))
    } catch {
        Issue.record("Expected AIError.invalidResponse, got \(error).")
    }
}

private func rewriteAmazonEventStreamMessageCRC(_ frame: inout Data) {
    let crc = testBedrockCRC32(frame.subdata(in: 0..<(frame.count - 4)))
    frame[frame.count - 4] = UInt8((crc >> 24) & 0xff)
    frame[frame.count - 3] = UInt8((crc >> 16) & 0xff)
    frame[frame.count - 2] = UInt8((crc >> 8) & 0xff)
    frame[frame.count - 1] = UInt8(crc & 0xff)
}

private func testBedrockCRC32(_ data: Data) -> UInt32 {
    var crc = UInt32.max
    for byte in data {
        crc ^= UInt32(byte)
        for _ in 0..<8 {
            crc = (crc & 1) == 1 ? (crc >> 1) ^ 0xedb8_8320 : crc >> 1
        }
    }
    return crc ^ UInt32.max
}

private struct BedrockStreamingTestTimeout: Error {}

private func awaitBedrockTestSignal<Value: Sendable>(
    _ signal: AIDelayedPromise<Value>,
    timeoutNanoseconds: UInt64 = 2_000_000_000
) async throws -> Value {
    let race = AIDelayedPromise<Value>()
    let valueTask = Task {
        do {
            race.resolve(try await signal.value())
        } catch {
            race.reject(error)
        }
    }
    let timeoutTask = Task {
        do {
            try await Task.sleep(nanoseconds: timeoutNanoseconds)
            race.reject(BedrockStreamingTestTimeout())
        } catch {}
    }
    defer {
        valueTask.cancel()
        timeoutTask.cancel()
    }
    return try await race.value()
}

private final class BedrockGatedStreamingTransport: AIStreamingTransport, @unchecked Sendable {
    let releaseBody = AIDelayedPromise<Void>()
    let bodyTerminated = AIDelayedPromise<Void>()
    let producerCancelled = AIDelayedPromise<Void>()

    private let firstChunk: Data
    private let finalChunk: Data
    private let lock = NSLock()
    private var recordedRequests: [AIHTTPRequest] = []
    private var sendCalls = 0
    private var streamCalls = 0

    init(firstChunk: Data, finalChunk: Data) {
        self.firstChunk = firstChunk
        self.finalChunk = finalChunk
    }

    var requests: [AIHTTPRequest] {
        withLock { recordedRequests }
    }

    var sendCallCount: Int {
        withLock { sendCalls }
    }

    var streamCallCount: Int {
        withLock { streamCalls }
    }

    func send(_ request: AIHTTPRequest) async throws -> AIHTTPResponse {
        withLock {
            sendCalls += 1
            recordedRequests.append(request)
        }
        throw AIError.invalidResponse(provider: "bedrock-test", message: "send() must not be called for a stream request.")
    }

    func stream(_ request: AIHTTPRequest) async throws -> AIHTTPStreamResponse {
        withLock {
            streamCalls += 1
            recordedRequests.append(request)
        }
        let firstChunk = self.firstChunk
        let finalChunk = self.finalChunk
        let releaseBody = self.releaseBody
        let bodyTerminated = self.bodyTerminated
        let producerCancelled = self.producerCancelled
        return AIHTTPStreamResponse(
            statusCode: 200,
            headers: ["content-type": "application/vnd.amazon.eventstream"],
            body: AsyncThrowingStream { continuation in
                let task = Task {
                    await withTaskCancellationHandler {
                        continuation.yield(firstChunk)
                        do {
                            try await releaseBody.value()
                            try Task.checkCancellation()
                            continuation.yield(finalChunk)
                            continuation.finish()
                        } catch {
                            continuation.finish(throwing: error)
                        }
                    } onCancel: {
                        producerCancelled.resolve(())
                    }
                }
                continuation.onTermination = { _ in
                    bodyTerminated.resolve(())
                    task.cancel()
                }
            }
        )
    }

    private func withLock<Value>(_ operation: () -> Value) -> Value {
        lock.lock()
        defer { lock.unlock() }
        return operation()
    }
}
