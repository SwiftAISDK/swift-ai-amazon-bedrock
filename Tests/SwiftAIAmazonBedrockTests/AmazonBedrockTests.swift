@testable import SwiftAIAmazonBedrock
@testable import SwiftAIAnthropic
@testable import SwiftAI
@testable import SwiftAIProvider
@testable import SwiftAIProviderUtilities
@testable import SwiftAITransport
import Foundation
import Testing

@Test func amazonBedrockConverseUsesSigV4AndConverseShape() async throws {
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
    let transport = RecordingTransport(response: jsonResponse("""
    {"output":{"message":{"content":[{"text":"bedrock"}]}},"stopReason":"end_turn","usage":{"inputTokens":2,"outputTokens":1,"totalTokens":3}}
    """))
    let provider = try AIProviders.amazonBedrock(configuration: AmazonBedrockProviderSettings(
        region: "us-east-1",
        accessKeyID: "AKIDEXAMPLE",
        secretAccessKey: "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY",
        headers: ["custom-header": "value"],
        transport: transport,
        date: { fixedDate }
    ))
    let model = try provider.languageModel("anthropic.claude-3-haiku-20240307-v1:0")

    let result = try await model.generate(LanguageModelRequest(messages: [.system("Brief."), .user("Hi")], temperature: 1.4, maxOutputTokens: 12))

    #expect(result.text == "bedrock")
    #expect(result.usage?.totalTokens == 3)
    let request = try #require(await transport.requests().first)
    #expect(request.url.absoluteString == "https://bedrock-runtime.us-east-1.amazonaws.com/model/anthropic.claude-3-haiku-20240307-v1%3A0/converse")
    #expect(request.headers["x-amz-date"] == "20240315T000000Z")
    #expect(request.headers["x-amz-content-sha256"] != nil)
    #expect(request.headers["authorization"]?.contains("Credential=AKIDEXAMPLE/20240315/us-east-1/bedrock/aws4_request") == true)
    #expect(request.headers["authorization"]?.contains("SignedHeaders=") == true)
    #expect(request.headers["custom-header"] == "value")
    #expect(request.headers["user-agent"] == "ai-sdk-amazon-bedrock/5.0.124")
    let body = try decodeJSONBody(try #require(request.body))
    #expect(body["system"]?[0]?["text"]?.stringValue == "Brief.")
    #expect(body["messages"]?[0]?["content"]?[0]?["text"]?.stringValue == "Hi")
    #expect(body["inferenceConfig"]?["temperature"]?.doubleValue == 1)
}

@Test func amazonBedrockConversePercentEncodesSlashesInARNModelIDsLikeUpstream() async throws {
    let transport = RecordingTransport(response: jsonResponse("""
    {"output":{"message":{"content":[{"text":"arn accepted"}]}},"stopReason":"end_turn","usage":{"inputTokens":2,"outputTokens":1,"totalTokens":3}}
    """))
    let provider = try AIProviders.amazonBedrock(configuration: AmazonBedrockProviderSettings(
        region: "us-east-1",
        apiKey: "bearer-key",
        transport: transport
    ))
    let model = try provider.languageModel(
        "arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/profile-id"
    )

    _ = try await model.generate(LanguageModelRequest(messages: [.user("Hi")]))

    let request = try #require(await transport.requests().first)
    #expect(
        request.url.absoluteString ==
            "https://bedrock-runtime.us-east-1.amazonaws.com/model/arn%3Aaws%3Abedrock%3Aus-east-1%3A123456789012%3Aapplication-inference-profile%2Fprofile-id/converse"
    )
}
@Test func amazonBedrockAppendsVersionedUserAgentToCustomHeaders() async throws {
    let converseTransport = RecordingTransport(response: jsonResponse("""
    {"output":{"message":{"content":[{"text":"bedrock"}]}},"stopReason":"end_turn","usage":{"inputTokens":1,"outputTokens":1,"totalTokens":2}}
    """))
    let converseProvider = try AIProviders.amazonBedrock(configuration: AmazonBedrockProviderSettings(
        region: "us-east-1",
        apiKey: "bedrock-key",
        headers: ["User-Agent": "CustomApp/1.0"],
        transport: converseTransport
    ))
    _ = try await converseProvider.languageModel("anthropic.claude-3-haiku-20240307-v1:0")
        .generate(LanguageModelRequest(messages: [.user("Hi")]))

    let anthropicTransport = RecordingTransport(response: jsonResponse("""
    {"content":[{"type":"text","text":"anthropic"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}
    """))
    let anthropicProvider = try AIProviders.amazonBedrockAnthropic(configuration: AmazonBedrockProviderSettings(
        region: "us-east-1",
        apiKey: "bedrock-key",
        headers: ["User-Agent": "CustomApp/1.0"],
        transport: anthropicTransport
    ))
    _ = try await anthropicProvider.languageModel("anthropic.claude-3-haiku-20240307-v1:0")
        .generate(LanguageModelRequest(messages: [.user("Hi")]))

    let mantleTransport = RecordingTransport(response: jsonResponse("""
    {"choices":[{"message":{"content":"mantle"},"finish_reason":"stop"}],"usage":{"total_tokens":2}}
    """))
    let mantleProvider = try AIProviders.bedrockMantle(configuration: AmazonBedrockProviderSettings(
        region: "us-east-1",
        apiKey: "mantle-key",
        headers: ["User-Agent": "CustomApp/1.0"],
        transport: mantleTransport
    ))
    _ = try await mantleProvider.languageModel("openai.gpt-oss-20b")
        .generate(LanguageModelRequest(messages: [.user("Hi")]))

    let converseRequest = try #require(await converseTransport.requests().first)
    let anthropicRequest = try #require(await anthropicTransport.requests().first)
    let mantleRequest = try #require(await mantleTransport.requests().first)
    #expect(converseRequest.headers["user-agent"] == "CustomApp/1.0 ai-sdk-amazon-bedrock/5.0.124")
    #expect(anthropicRequest.headers["user-agent"] == "CustomApp/1.0 ai-sdk-amazon-bedrock/5.0.124")
    #expect(mantleRequest.headers["user-agent"] == "CustomApp/1.0 ai-sdk-amazon-bedrock/5.0.124")
}
@Test func amazonBedrockCredentialProviderSignsAllProviderSurfaces() async throws {
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

    let converseTransport = RecordingTransport(response: jsonResponse("""
    {"output":{"message":{"content":[{"text":"bedrock"}]}},"stopReason":"end_turn","usage":{"inputTokens":1,"outputTokens":1,"totalTokens":2}}
    """))
    let converseProvider = try AIProviders.amazonBedrock(configuration: AmazonBedrockProviderSettings(
        region: "us-east-1",
        credentialProvider: {
            AmazonBedrockCredentials(accessKeyID: "DYNAMIC-CONVERSE", secretAccessKey: "secret", sessionToken: "session-converse")
        },
        transport: converseTransport,
        date: { fixedDate }
    ))
    _ = try await converseProvider.languageModel("anthropic.claude-3-haiku-20240307-v1:0")
        .generate(LanguageModelRequest(messages: [.user("Hi")]))

    let anthropicTransport = RecordingTransport(response: jsonResponse("""
    {"content":[{"type":"text","text":"anthropic"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}
    """))
    let anthropicProvider = try AIProviders.amazonBedrockAnthropic(configuration: AmazonBedrockProviderSettings(
        region: "us-east-1",
        credentialProvider: {
            AmazonBedrockCredentials(accessKeyID: "DYNAMIC-ANTHROPIC", secretAccessKey: "secret", sessionToken: "session-anthropic")
        },
        transport: anthropicTransport,
        date: { fixedDate }
    ))
    _ = try await anthropicProvider.languageModel("anthropic.claude-3-haiku-20240307-v1:0")
        .generate(LanguageModelRequest(messages: [.user("Hi")]))

    let mantleTransport = RecordingTransport(response: jsonResponse("""
    {"choices":[{"message":{"content":"mantle"},"finish_reason":"stop"}],"usage":{"total_tokens":2}}
    """))
    let mantleProvider = try AIProviders.bedrockMantle(configuration: AmazonBedrockProviderSettings(
        region: "us-east-1",
        credentialProvider: {
            AmazonBedrockCredentials(accessKeyID: "DYNAMIC-MANTLE", secretAccessKey: "secret", sessionToken: "session-mantle")
        },
        transport: mantleTransport,
        date: { fixedDate }
    ))
    _ = try await mantleProvider.languageModel("openai.gpt-oss-20b")
        .generate(LanguageModelRequest(messages: [.user("Hi")]))

    let converseRequest = try #require(await converseTransport.requests().first)
    let anthropicRequest = try #require(await anthropicTransport.requests().first)
    let mantleRequest = try #require(await mantleTransport.requests().first)
    #expect(converseRequest.headers["authorization"]?.contains("Credential=DYNAMIC-CONVERSE/20240315/us-east-1/bedrock/aws4_request") == true)
    #expect(converseRequest.headers["x-amz-security-token"] == "session-converse")
    #expect(anthropicRequest.headers["authorization"]?.contains("Credential=DYNAMIC-ANTHROPIC/20240315/us-east-1/bedrock/aws4_request") == true)
    #expect(anthropicRequest.headers["x-amz-security-token"] == "session-anthropic")
    #expect(mantleRequest.headers["authorization"]?.contains("Credential=DYNAMIC-MANTLE/20240315/us-east-1/bedrock-mantle/aws4_request") == true)
    #expect(mantleRequest.headers["x-amz-security-token"] == "session-mantle")
}
@Test func amazonBedrockConverseMapsDocumentDataAndProviderOptions() async throws {
    let transport = RecordingTransport(response: jsonResponse("""
    {"output":{"message":{"content":[{"text":"ok"}]}},"stopReason":"end_turn","usage":{"inputTokens":2,"outputTokens":1,"totalTokens":3}}
    """))
    let provider = try AIProviders.amazonBedrock(configuration: AmazonBedrockProviderSettings(
        region: "us-east-1",
        accessKeyID: "AKIDEXAMPLE",
        secretAccessKey: "secret",
        transport: transport
    ))
    let model = try provider.languageModel("anthropic.claude-3-haiku-20240307-v1:0")

    _ = try await model.generate(LanguageModelRequest(
        messages: [
            AIMessage(role: .user, content: [
                .text("Use these inputs."),
                .data(mimeType: "application/pdf", data: Data("pdf bytes".utf8)),
                .data(mimeType: "image/png", data: Data([0x89, 0x50, 0x4E, 0x47]))
            ])
        ],
        extraBody: [
            "amazonBedrock": .object([
                "citations": .object(["enabled": .bool(true)]),
                "guardrailConfig": .object([
                    "guardrailIdentifier": .string("gr-1"),
                    "guardrailVersion": .string("1")
                ]),
                "serviceTier": .string("priority"),
                "additionalModelRequestFields": .object(["trace": .string("enabled")])
            ]),
            "bedrock": .object([
                "serviceTier": .string("legacy")
            ])
        ]
    ))

    let request = try #require(await transport.requests().first)
    let body = try decodeJSONBody(try #require(request.body))
    let content = try #require(body["messages"]?[0]?["content"]?.arrayValue)
    #expect(content[0]["text"]?.stringValue == "Use these inputs.")
    #expect(content[1]["document"]?["format"]?.stringValue == "pdf")
    #expect(content[1]["document"]?["name"]?.stringValue == "document-1")
    #expect(content[1]["document"]?["source"]?["bytes"]?.stringValue == Data("pdf bytes".utf8).base64EncodedString())
    #expect(content[1]["document"]?["citations"]?["enabled"]?.boolValue == true)
    #expect(content[2]["image"]?["format"]?.stringValue == "png")
    #expect(content[2]["image"]?["source"]?["bytes"]?.stringValue == Data([0x89, 0x50, 0x4E, 0x47]).base64EncodedString())
    #expect(body["guardrailConfig"]?["guardrailIdentifier"]?.stringValue == "gr-1")
    #expect(body["serviceTier"]?["type"]?.stringValue == "priority")
    #expect(body["additionalModelRequestFields"]?["trace"]?.stringValue == "enabled")
    #expect(body["amazonBedrock"] == nil)
    #expect(body["bedrock"] == nil)
}
@Test func amazonBedrockConverseMapsNativeToolsToolChoiceAndProviderOptions() async throws {
    let transport = RecordingTransport(response: jsonResponse("""
    {"output":{"message":{"content":[{"text":"tool ready"}]}},"stopReason":"end_turn","usage":{"inputTokens":2,"outputTokens":1,"totalTokens":3}}
    """))
    let provider = try AIProviders.amazonBedrock(configuration: AmazonBedrockProviderSettings(
        region: "us-east-1",
        accessKeyID: "AKIDEXAMPLE",
        secretAccessKey: "secret",
        transport: transport
    ))
    let model = try provider.languageModel("anthropic.claude-3-haiku-20240307-v1:0")

    let result = try await model.generate(LanguageModelRequest(
        messages: [.user("Use weather.")],
        topK: 12,
        tools: [
            "weather": [
                "type": "object",
                "description": "Look up weather.",
                "properties": ["city": ["type": "string"]],
                "required": ["city"]
            ],
            "unused": [
                "type": "object",
                "properties": [:]
            ]
        ],
        toolChoice: ["type": "tool", "toolName": "weather"],
        providerOptions: [
            "amazonBedrock": [
                "serviceTier": "priority",
                "additionalModelRequestFields": ["custom": "value"]
            ]
        ]
    ))

    #expect(result.text == "tool ready")
    #expect(result.warnings.isEmpty)
    let body = try decodeJSONBody(try #require((await transport.requests()).first?.body))
    #expect(body["inferenceConfig"]?["topK"]?.intValue == 12)
    #expect(body["serviceTier"]?["type"]?.stringValue == "priority")
    #expect(body["additionalModelRequestFields"]?["custom"]?.stringValue == "value")
    let tools = try #require(body["toolConfig"]?["tools"]?.arrayValue)
    #expect(tools.count == 1)
    #expect(tools[0]["toolSpec"]?["name"]?.stringValue == "weather")
    #expect(tools[0]["toolSpec"]?["description"]?.stringValue == "Look up weather.")
    #expect(tools[0]["toolSpec"]?["inputSchema"]?["json"]?["properties"]?["city"]?["type"]?.stringValue == "string")
    #expect(body["toolConfig"]?["toolChoice"]?["tool"]?["name"]?.stringValue == "weather")
    #expect(body["amazonBedrock"] == nil)
}
@Test func amazonBedrockConverseMapsRichToolResultContentAndCachePoints() async throws {
    let transport = RecordingTransport(response: jsonResponse("""
    {"output":{"message":{"content":[{"text":"tool consumed"}]}},"stopReason":"end_turn","usage":{"inputTokens":2,"outputTokens":1,"totalTokens":3}}
    """))
    let provider = try AIProviders.amazonBedrock(configuration: AmazonBedrockProviderSettings(
        region: "us-east-1",
        apiKey: "bearer-key",
        transport: transport
    ))
    let model = try provider.languageModel("anthropic.claude-3-haiku-20240307-v1:0")
    let imageData = Data([0x89, 0x50, 0x4E, 0x47]).base64EncodedString()
    let videoData = Data([0, 1, 2, 3]).base64EncodedString()
    let pdfData = Data("pdf bytes".utf8).base64EncodedString()

    _ = try await model.generate(LanguageModelRequest(
        messages: [
            AIMessage(role: .tool, content: [
                .toolResult(AIToolResult(
                    toolCallID: "tool-1",
                    toolName: "lookup",
                    result: ["fallback": "ignored"],
                    modelOutput: [
                        "type": "content",
                        "value": [
                            ["type": "text", "text": "first result"],
                            ["type": "image-data", "mediaType": "image/png", "data": .string(imageData)],
                            ["type": "file-data", "mediaType": "video/mp4", "data": .string(videoData)],
                            [
                                "type": "file",
                                "mediaType": "video/webm",
                                "data": [
                                    "type": "url",
                                    "url": "s3://my-test-bucket/path/to/generated.webm"
                                ]
                            ],
                            [
                                "type": "file-data",
                                "mediaType": "application/pdf",
                                "filename": "report.pdf",
                                "data": .string(pdfData),
                                "providerMetadata": ["amazonBedrock": ["citations": ["enabled": true]]]
                            ],
                            [
                                "type": "file",
                                "mediaType": "image/png",
                                "data": [
                                    "type": "url",
                                    "url": "s3://my-test-bucket/path/to/generated.png"
                                ]
                            ]
                        ]
                    ],
                    providerMetadata: ["amazonBedrock": ["cachePoint": ["type": "default"]]]
                ))
            ])
        ],
        tools: [
            "lookup": [
                "type": "object",
                "properties": ["query": ["type": "string"]]
            ]
        ]
    ))

    let body = try decodeJSONBody(try #require((await transport.requests()).first?.body))
    let content = try #require(body["messages"]?[0]?["content"]?.arrayValue)
    let toolResult = try #require(content[0]["toolResult"])
    #expect(toolResult["toolUseId"]?.stringValue == "tool-1")
    #expect(toolResult["status"]?.stringValue == "success")
    #expect(toolResult["content"]?[0]?["text"]?.stringValue == "first result")
    #expect(toolResult["content"]?[1]?["image"]?["format"]?.stringValue == "png")
    #expect(toolResult["content"]?[1]?["image"]?["source"]?["bytes"]?.stringValue == imageData)
    #expect(toolResult["content"]?[2]?["video"]?["format"]?.stringValue == "mp4")
    #expect(toolResult["content"]?[2]?["video"]?["source"]?["bytes"]?.stringValue == videoData)
    #expect(toolResult["content"]?[3]?["video"]?["format"]?.stringValue == "webm")
    #expect(toolResult["content"]?[3]?["video"]?["source"]?["s3Location"]?["uri"]?.stringValue == "s3://my-test-bucket/path/to/generated.webm")
    #expect(toolResult["content"]?[4]?["document"]?["format"]?.stringValue == "pdf")
    #expect(toolResult["content"]?[4]?["document"]?["name"]?.stringValue == "report")
    #expect(toolResult["content"]?[4]?["document"]?["source"]?["bytes"]?.stringValue == pdfData)
    #expect(toolResult["content"]?[4]?["document"]?["citations"]?["enabled"]?.boolValue == true)
    #expect(toolResult["content"]?[5]?["image"]?["format"]?.stringValue == "png")
    #expect(toolResult["content"]?[5]?["image"]?["source"]?["s3Location"]?["uri"]?.stringValue == "s3://my-test-bucket/path/to/generated.png")
    #expect(content[1]["cachePoint"]?["type"]?.stringValue == "default")
}

@Test func amazonBedrockConversePassesThroughS3ImagesAndSanitizesReplayedToolNames() async throws {
    let transport = RecordingTransport(response: jsonResponse("""
    {"output":{"message":{"content":[{"text":"history accepted"}]}},"stopReason":"end_turn","usage":{"inputTokens":2,"outputTokens":1,"totalTokens":3}}
    """))
    let provider = try AIProviders.amazonBedrock(configuration: AmazonBedrockProviderSettings(
        region: "us-east-1",
        apiKey: "bearer-key",
        transport: transport
    ))
    let model = try provider.languageModel("anthropic.claude-3-haiku-20240307-v1:0")
    let s3URL = "s3://my-test-bucket/path/to/image.png"

    #expect(isURLSupported(mediaType: "image/png", url: s3URL, supportedURLs: model.supportedURLs))
    #expect(!isURLSupported(mediaType: "image/png", url: "https://example.com/image.png", supportedURLs: model.supportedURLs))
    let unsupportedS3URLs = [
        "s3:my-test-bucket/path/to/image.png",
        "s3:/my-test-bucket/path/to/image.png",
        "S3://my-test-bucket/path/to/image.png"
    ]
    for url in unsupportedS3URLs {
        #expect(!bedrockIsS3URL(url))
    }
    for url in unsupportedS3URLs.prefix(2) {
        #expect(!isURLSupported(mediaType: "image/png", url: url, supportedURLs: model.supportedURLs))
    }

    _ = try await model.generate(LanguageModelRequest(
        messages: [
            AIMessage(role: .user, content: [
                .imageURL(
                    s3URL,
                    providerMetadata: ["amazonBedrock": ["cachePoint": ["type": "default"]]]
                )
            ]),
            AIMessage(role: .assistant, content: [
                .toolCall(AIToolCall(id: "tool-1", name: "$READFILE", arguments: "{}")),
                .toolCall(AIToolCall(id: "tool-2", name: "exchange_delivered_order_items<|channel|>", arguments: "{}")),
                .toolCall(AIToolCall(id: "tool-3", name: "$", arguments: "{}"))
            ])
        ],
        tools: [
            "active": [
                "type": "object",
                "properties": [:]
            ]
        ]
    ))

    let body = try decodeJSONBody(try #require((await transport.requests()).first?.body))
    #expect(body["messages"]?[0]?["content"]?[0]?["image"]?["format"]?.stringValue == "png")
    #expect(body["messages"]?[0]?["content"]?[0]?["image"]?["source"]?["s3Location"]?["uri"]?.stringValue == s3URL)
    #expect(body["messages"]?[0]?["content"]?[1]?["cachePoint"]?["type"]?.stringValue == "default")
    #expect(body["messages"]?[1]?["content"]?[0]?["toolUse"]?["name"]?.stringValue == "READFILE")
    #expect(body["messages"]?[1]?["content"]?[1]?["toolUse"]?["name"]?.stringValue == "exchange_delivered_order_itemschannel")
    #expect(body["messages"]?[1]?["content"]?[2]?["toolUse"]?["name"]?.stringValue == "_")

    await #expect(throws: AIError.invalidArgument(
        argument: "messages.content.imageURL",
        message: "Amazon Bedrock Converse supports only s3:// image URLs ending in .jpg, .jpeg, .png, .gif, or .webp."
    )) {
        _ = try await model.generate(LanguageModelRequest(messages: [
            AIMessage(role: .user, content: [.imageURL("https://example.com/image.png")])
        ]))
    }
    for url in unsupportedS3URLs {
        await #expect(throws: AIError.invalidArgument(
            argument: "messages.content.imageURL",
            message: "Amazon Bedrock Converse supports only s3:// image URLs ending in .jpg, .jpeg, .png, .gif, or .webp."
        )) {
            _ = try await model.generate(LanguageModelRequest(messages: [
                AIMessage(role: .user, content: [.imageURL(url)])
            ]))
        }
    }
}

@Test func amazonBedrockConverseMapsInlineVideoPartsLikeUpstream() async throws {
    let transport = RecordingTransport(response: jsonResponse("""
    {"output":{"message":{"content":[{"text":"video accepted"}]}},"stopReason":"end_turn"}
    """))
    let provider = try AIProviders.amazonBedrock(configuration: AmazonBedrockProviderSettings(
        region: "us-east-1",
        apiKey: "bearer-key",
        transport: transport
    ))
    let model = try provider.languageModel("anthropic.claude-3-haiku-20240307-v1:0")
    let inlineVideo = Data([0, 1, 2, 3])
    let detectedMP4 = Data([0x00, 0x00, 0x00, 0x18, 0x66, 0x74, 0x79, 0x70])

    _ = try await model.generate(LanguageModelRequest(messages: [
        AIMessage(role: .user, content: [
            .data(mimeType: "video/mp4", data: inlineVideo),
            .file(mimeType: "video", data: detectedMP4, filename: "clip")
        ])
    ]))

    let body = try decodeJSONBody(try #require((await transport.requests()).first?.body))
    let content = try #require(body["messages"]?[0]?["content"]?.arrayValue)
    #expect(content[0]["video"]?["format"]?.stringValue == "mp4")
    #expect(content[0]["video"]?["source"]?["bytes"]?.stringValue == "AAECAw==")
    #expect(content[1]["video"]?["format"]?.stringValue == "mp4")
    #expect(content[1]["video"]?["source"]?["bytes"]?.stringValue == "AAAAGGZ0eXA=")

    let expectedFormats = [
        "video/x-matroska": "mkv",
        "video/quicktime": "mov",
        "video/mp4": "mp4",
        "video/webm": "webm",
        "video/x-flv": "flv",
        "video/mpeg": "mpeg",
        "video/mpg": "mpg",
        "video/wmv": "wmv",
        "video/x-ms-wmv": "wmv",
        "video/3gpp": "three_gp"
    ]
    for (mimeType, format) in expectedFormats {
        #expect(bedrockVideoFormat(for: mimeType) == format)
    }
    #expect(bedrockVideoFormat(for: "video/unsupported") == nil)
}

@Test func amazonBedrockOmitsStrictToolSpecForAnthropicFamiliesThatRejectNewerSchemaFields() async throws {
    let transport = RecordingTransport(response: jsonResponse("""
    {"output":{"message":{"content":[{"text":"tool ready"}]}},"stopReason":"end_turn","usage":{"inputTokens":2,"outputTokens":1,"totalTokens":3}}
    """))
    let provider = try AIProviders.amazonBedrock(configuration: AmazonBedrockProviderSettings(
        region: "us-east-1",
        accessKeyID: "AKIDEXAMPLE",
        secretAccessKey: "secret",
        transport: transport
    ))
    for modelID in [
        "anthropic.claude-opus-4-7-20260219-v1:0",
        "anthropic.claude-opus-4-8-20260401-v1:0",
        "anthropic.claude-opus-5-20260701-v1:0",
        "anthropic.claude-fable-5-20260601-v1:0",
        "anthropic.claude-sonnet-5-20260701-v1:0"
    ] {
        #expect(!bedrockSupportsStrictToolSpec(modelID: modelID))
        #expect(!bedrockSupportsNativeStructuredOutput(modelID: modelID))
        let model = try provider.languageModel(modelID)
        let result = try await model.generate(LanguageModelRequest(
            messages: [.user("Use weather.")],
            tools: [
                "weather": [
                    "type": "object",
                    "strict": true,
                    "properties": ["city": ["type": "string"]]
                ]
            ]
        ))
        #expect(result.warnings.contains(AIWarning(
            type: "unsupported",
            feature: "strict",
            message: "Tool 'weather' has strict: true, but strict mode is not supported by this model on Amazon Bedrock. The strict property will be ignored."
        )))
    }

    for request in await transport.requests() {
        let body = try decodeJSONBody(try #require(request.body))
        let toolSpec = try #require(body["toolConfig"]?["tools"]?[0]?["toolSpec"])
        #expect(toolSpec["name"]?.stringValue == "weather")
        #expect(toolSpec["strict"] == nil)
    }
}

@Test func amazonBedrockConverseMapsReasoningConfigLikeUpstream() async throws {
    let transport = RecordingTransport(response: jsonResponse("""
    {"output":{"message":{"content":[{"text":"thinking done"}]}},"stopReason":"end_turn","usage":{"inputTokens":2,"outputTokens":1,"totalTokens":3}}
    """))
    let provider = try AIProviders.amazonBedrock(configuration: AmazonBedrockProviderSettings(
        region: "us-east-1",
        accessKeyID: "AKIDEXAMPLE",
        secretAccessKey: "secret",
        transport: transport
    ))
    let model = try provider.languageModel("anthropic.claude-3-7-sonnet-20250219-v1:0")

    let result = try await model.generate(LanguageModelRequest(
        messages: [.user("Think.")],
        temperature: 0.5,
        topP: 0.9,
        topK: 50,
        maxOutputTokens: 1000,
        providerOptions: [
            "amazonBedrock": [
                "reasoningConfig": [
                    "type": "enabled",
                    "budgetTokens": 200,
                    "maxReasoningEffort": "high"
                ]
            ]
        ]
    ))

    let body = try decodeJSONBody(try #require((await transport.requests()).first?.body))
    #expect(body["inferenceConfig"]?["maxTokens"]?.intValue == 1200)
    #expect(body["inferenceConfig"]?["temperature"] == nil)
    #expect(body["inferenceConfig"]?["topP"] == nil)
    #expect(body["inferenceConfig"]?["topK"] == nil)
    #expect(body["additionalModelRequestFields"]?["thinking"]?["type"]?.stringValue == "enabled")
    #expect(body["additionalModelRequestFields"]?["thinking"]?["budget_tokens"]?.intValue == 200)
    #expect(body["additionalModelRequestFields"]?["output_config"]?["effort"]?.stringValue == "high")
    #expect(body["reasoningConfig"] == nil)
    #expect(result.warnings.contains(AIWarning(type: "unsupported", feature: "temperature", message: "temperature is not supported when thinking is enabled")))
    #expect(result.warnings.contains(AIWarning(type: "unsupported", feature: "topP", message: "topP is not supported when thinking is enabled")))
    #expect(result.warnings.contains(AIWarning(type: "unsupported", feature: "topK", message: "topK is not supported when thinking is enabled")))
}
@Test func amazonBedrockConverseMapsJSONResponseFormatThroughToolLikeUpstream() async throws {
    let transport = RecordingTransport(response: jsonResponse("""
    {"output":{"message":{"content":[{"toolUse":{"toolUseId":"json-tool","name":"json","input":{"answer":"ok"}}}]}},"stopReason":"tool_use","usage":{"inputTokens":2,"outputTokens":1,"totalTokens":3}}
    """))
    let provider = try AIProviders.amazonBedrock(configuration: AmazonBedrockProviderSettings(
        region: "us-east-1",
        apiKey: "bearer-key",
        transport: transport
    ))
    let schema: JSONValue = [
        "type": "object",
        "properties": ["answer": ["type": "string"]],
        "required": ["answer"]
    ]
    let model = try provider.languageModel("cohere.command-r-v1:0")

    let result = try await model.generate(LanguageModelRequest(
        messages: [.user("Reply as JSON")],
        presencePenalty: 0.5,
        frequencyPenalty: 0.5,
        seed: 42,
        responseFormat: .json(schema: schema)
    ))

    #expect(try decodeJSONBody(Data(result.text.utf8))["answer"]?.stringValue == "ok")
    #expect(result.finishReason == "stop")
    #expect(result.toolCalls.isEmpty)
    #expect(result.providerMetadata["amazonBedrock"]?["isJsonResponseFromTool"]?.boolValue == true)
    #expect(result.warnings.contains(AIWarning(type: "unsupported", feature: "frequencyPenalty")))
    #expect(result.warnings.contains(AIWarning(type: "unsupported", feature: "presencePenalty")))
    #expect(result.warnings.contains(AIWarning(type: "unsupported", feature: "seed")))
    let request = try #require(await transport.requests().first)
    let body = try decodeJSONBody(try #require(request.body))
    #expect(body["toolConfig"]?["tools"]?[0]?["toolSpec"]?["name"]?.stringValue == "json")
    #expect(body["toolConfig"]?["tools"]?[0]?["toolSpec"]?["inputSchema"]?["json"] == schema)
    #expect(body["toolConfig"]?["toolChoice"]?["any"] != nil)
}
@Test func amazonBedrockConverseUsesNativeStructuredOutputForAnthropicThinking() async throws {
    let transport = RecordingTransport(response: jsonResponse("""
    {"output":{"message":{"content":[{"text":"ok"}]}},"stopReason":"end_turn","usage":{"inputTokens":2,"outputTokens":1,"totalTokens":3}}
    """))
    let provider = try AIProviders.amazonBedrock(configuration: AmazonBedrockProviderSettings(
        region: "us-east-1",
        apiKey: "bearer-key",
        transport: transport
    ))
    let schema: JSONValue = [
        "type": "object",
        "properties": [
            "answer": ["type": "string"],
            "labels": [
                "type": "array",
                "maxItems": 3,
                "items": ["type": "string"]
            ]
        ]
    ]
    let model = try provider.languageModel("anthropic.claude-3-7-sonnet-20250219-v1:0")

    _ = try await model.generate(LanguageModelRequest(
        messages: [.user("Reply as JSON")],
        maxOutputTokens: 10,
        responseFormat: .json(schema: schema),
        providerOptions: ["bedrock": ["reasoningConfig": ["type": "enabled", "budgetTokens": 8]]]
    ))

    let request = try #require(await transport.requests().first)
    let body = try decodeJSONBody(try #require(request.body))
    #expect(body["toolConfig"] == nil)
    let outputConfig = body["additionalModelRequestFields"]?["output_config"]
    #expect(outputConfig?["format"]?["type"]?.stringValue == "json_schema")
    let sentSchema = try #require(outputConfig?["format"]?["schema"])
    #expect(sentSchema["properties"]?["labels"]?["maxItems"] == nil)
    #expect(sentSchema["properties"]?["labels"]?["description"]?.stringValue == "max items: 3.")
    #expect(sentSchema["properties"]?["labels"]?["items"]?["type"]?.stringValue == "string")
    #expect(sentSchema["additionalProperties"]?.boolValue == false)
    #expect(body["additionalModelRequestFields"]?["thinking"]?["type"]?.stringValue == "enabled")
}
@Test func amazonBedrockConverseUsesJSONToolForAnthropicFamiliesWithoutNativeStructuredOutput() async throws {
    let transport = RecordingTransport(response: jsonResponse("""
    {"output":{"message":{"content":[{"toolUse":{"toolUseId":"json-tool","name":"json","input":{"answer":"ok"}}}]}},"stopReason":"tool_use","usage":{"inputTokens":2,"outputTokens":1,"totalTokens":3}}
    """))
    let provider = try AIProviders.amazonBedrock(configuration: AmazonBedrockProviderSettings(
        region: "us-east-1",
        apiKey: "bearer-key",
        transport: transport
    ))
    let model = try provider.languageModel("anthropic.claude-opus-5-20260701-v1:0")

    let result = try await model.generate(LanguageModelRequest(
        messages: [.user("Reply as JSON")],
        responseFormat: .json(schema: [
            "type": "object",
            "properties": ["answer": ["type": "string"]]
        ]),
        providerOptions: [
            "amazonBedrock": [
                "reasoningConfig": ["type": "enabled", "budgetTokens": 8]
            ]
        ]
    ))

    let request = try #require(await transport.requests().first)
    let body = try decodeJSONBody(try #require(request.body))
    #expect(body["additionalModelRequestFields"]?["output_config"]?["format"] == nil)
    #expect(body["toolConfig"]?["tools"]?[0]?["toolSpec"]?["name"]?.stringValue == "json")
    #expect(body["toolConfig"]?["toolChoice"]?["any"] != nil)
    #expect(try decodeJSONBody(Data(result.text.utf8))["answer"]?.stringValue == "ok")
}

@Test func amazonBedrockConverseUsesJSONInstructionWhenStructuredOutputKeepsTools() async throws {
    let transport = RecordingTransport(response: jsonResponse(#"""
    {"output":{"message":{"content":[{"text":"```json\n{\"name\":\"Test\"}\n```."}]}},"stopReason":"end_turn","usage":{"inputTokens":4,"outputTokens":10,"totalTokens":14}}
    """#))
    let provider = try AIProviders.amazonBedrock(configuration: AmazonBedrockProviderSettings(
        region: "us-east-1",
        apiKey: "bearer-key",
        transport: transport
    ))
    let model = try provider.languageModel("anthropic.claude-sonnet-5-20260701-v1:0")
    let schema: JSONValue = [
        "type": "object",
        "properties": ["name": ["type": "string"]],
        "required": ["name"]
    ]

    let result = try await model.generate(LanguageModelRequest(
        messages: [.user("Look up and generate a name")],
        responseFormat: .json(schema: schema),
        tools: ["lookupName": ["type": "object", "properties": [:]]]
    ))

    #expect(result.text == #"{"name":"Test"}"#)
    let body = try decodeJSONBody(try #require((await transport.requests()).first?.body))
    let tools = try #require(body["toolConfig"]?["tools"]?.arrayValue)
    #expect(tools.count == 1)
    #expect(tools[0]["toolSpec"]?["name"]?.stringValue == "lookupName")
    #expect(body["toolConfig"]?["toolChoice"] == nil)
    #expect(body["additionalModelRequestFields"]?["output_config"]?["format"] == nil)
    let systemText = try #require(body["system"]?[0]?["text"]?.stringValue)
    #expect(systemText.contains("JSON schema:"))
    #expect(systemText.contains(#""required":["name"]"#))
    #expect(systemText.hasSuffix("You MUST answer with only a JSON object that matches the JSON schema above. Do not wrap it in markdown fences or include any other text."))
}

@Test func amazonBedrockConverseStreamsOnlyJSONObjectWhenStructuredOutputKeepsTools() async throws {
    let transport = RecordingTransport(response: amazonEventStreamResponse([
        ("contentBlockDelta", #"{"contentBlockIndex":0,"delta":{"text":"```json\nleading "}}"#),
        ("contentBlockDelta", #"{"contentBlockIndex":0,"delta":{"text":"{\"name\":\"Te"}}"#),
        ("contentBlockDelta", #"{"contentBlockIndex":0,"delta":{"text":"st\",\"nested\":{\"ok\":true}}"}}"#),
        ("contentBlockDelta", #"{"contentBlockIndex":0,"delta":{"text":"\n``` trailing"}}"#),
        ("messageStop", #"{"stopReason":"end_turn"}"#)
    ]))
    let provider = try AIProviders.amazonBedrock(configuration: AmazonBedrockProviderSettings(
        region: "us-east-1",
        apiKey: "bearer-key",
        transport: transport
    ))
    let model = try provider.languageModel("anthropic.claude-opus-4-8-20260401-v1:0")
    var text = ""

    for try await part in model.stream(LanguageModelRequest(
        messages: [.user("Generate a name")],
        responseFormat: .json(schema: ["type": "object", "properties": ["name": ["type": "string"]]]),
        tools: ["lookupName": ["type": "object", "properties": [:]]]
    )) {
        if case let .textDeltaPart(_, delta, _) = part {
            text += delta
        }
    }

    #expect(text == #"{"name":"Test","nested":{"ok":true}}"#)
    let body = try decodeJSONBody(try #require((await transport.requests()).first?.body))
    #expect(body["toolConfig"]?["tools"]?[0]?["toolSpec"]?["name"]?.stringValue == "lookupName")
    #expect(body["toolConfig"]?["tools"]?.arrayValue?.count == 1)
}
@Test func amazonBedrockConverseMapsReasoningEffortForOpenAIAndNovaModels() async throws {
    let openAITransport = RecordingTransport(response: jsonResponse("""
    {"output":{"message":{"content":[{"text":"openai"}]}},"stopReason":"end_turn"}
    """))
    let openAIProvider = try AIProviders.amazonBedrock(configuration: AmazonBedrockProviderSettings(
        region: "us-east-1",
        accessKeyID: "AKIDEXAMPLE",
        secretAccessKey: "secret",
        transport: openAITransport
    ))
    let openAIModel = try openAIProvider.languageModel("openai.gpt-oss-120b-1:0")
    _ = try await openAIModel.generate(LanguageModelRequest(
        messages: [.user("Think.")],
        providerOptions: ["bedrock": ["reasoningConfig": ["maxReasoningEffort": "medium"]]]
    ))
    let openAIBody = try decodeJSONBody(try #require((await openAITransport.requests()).first?.body))
    #expect(openAIBody["additionalModelRequestFields"]?["reasoning_effort"]?.stringValue == "medium")

    let novaTransport = RecordingTransport(response: jsonResponse("""
    {"output":{"message":{"content":[{"text":"nova"}]}},"stopReason":"end_turn"}
    """))
    let novaProvider = try AIProviders.amazonBedrock(configuration: AmazonBedrockProviderSettings(
        region: "us-east-1",
        accessKeyID: "AKIDEXAMPLE",
        secretAccessKey: "secret",
        transport: novaTransport
    ))
    let novaModel = try novaProvider.languageModel("us.amazon.nova-pro-v1:0")
    let novaResult = try await novaModel.generate(LanguageModelRequest(
        messages: [.user("Think.")],
        providerOptions: ["amazonBedrock": ["reasoningConfig": ["type": "enabled", "budgetTokens": 64, "maxReasoningEffort": "low"]]]
    ))
    let novaBody = try decodeJSONBody(try #require((await novaTransport.requests()).first?.body))
    #expect(novaBody["additionalModelRequestFields"]?["reasoningConfig"]?["type"]?.stringValue == "enabled")
    #expect(novaBody["additionalModelRequestFields"]?["reasoningConfig"]?["budgetTokens"]?.intValue == 64)
    #expect(novaBody["additionalModelRequestFields"]?["reasoningConfig"]?["maxReasoningEffort"]?.stringValue == "low")
    #expect(novaResult.warnings.contains(AIWarning(type: "unsupported", feature: "budgetTokens", message: "budgetTokens applies only to Anthropic models on Bedrock and will be ignored for this model.")))
}
@Test func amazonBedrockLanguageStreamStartCarriesRequestWarnings() async throws {
    let transport = RecordingTransport(response: amazonEventStreamResponse([
        ("contentBlockDelta", #"{"contentBlockIndex":0,"delta":{"text":"ok"}}"#),
        ("messageStop", #"{"stopReason":"end_turn"}"#)
    ]))
    let provider = try AIProviders.amazonBedrock(configuration: AmazonBedrockProviderSettings(
        region: "us-east-1",
        accessKeyID: "AKIDEXAMPLE",
        secretAccessKey: "secret",
        transport: transport
    ))
    let model = try provider.languageModel("anthropic.claude-3-7-sonnet-20250219-v1:0")

    var startWarnings: [AIWarning] = []
    for try await part in model.stream(LanguageModelRequest(
        messages: [.user("Think.")],
        temperature: 0.5,
        providerOptions: ["amazonBedrock": ["reasoningConfig": ["type": "adaptive", "display": "summarized"]]]
    )) {
        if case let .streamStart(warnings) = part {
            startWarnings = warnings
        }
    }

    let body = try decodeJSONBody(try #require((await transport.requests()).first?.body))
    #expect(body["additionalModelRequestFields"]?["thinking"]?["type"]?.stringValue == "adaptive")
    #expect(body["additionalModelRequestFields"]?["thinking"]?["display"]?.stringValue == "summarized")
    #expect(startWarnings.contains(AIWarning(type: "unsupported", feature: "temperature", message: "temperature is not supported when thinking is enabled")))
}

@Test func amazonBedrockOmitsAssistantMessagesWhoseUnsignedReasoningIsFilteredLikeUpstream() async throws {
    let transport = RecordingTransport(response: jsonResponse("""
    {"output":{"message":{"content":[{"text":"ok"}]}},"stopReason":"end_turn"}
    """))
    let provider = try AIProviders.amazonBedrock(configuration: AmazonBedrockProviderSettings(
        region: "us-east-1",
        accessKeyID: "AKIDEXAMPLE",
        secretAccessKey: "secret",
        transport: transport
    ))
    let model = try provider.languageModel("anthropic.claude-3-7-sonnet-20250219-v1:0")

    _ = try await model.generate(LanguageModelRequest(messages: [
        .user("First"),
        AIMessage(role: .assistant, content: [.reasoning("unsigned")]),
        .user("Second")
    ]))

    let body = try decodeJSONBody(try #require((await transport.requests()).first?.body))
    let messages = try #require(body["messages"]?.arrayValue)
    #expect(messages.count == 2)
    #expect(messages.allSatisfy { $0["role"]?.stringValue == "user" })
}
