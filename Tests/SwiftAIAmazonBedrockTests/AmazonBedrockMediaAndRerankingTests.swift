@testable import SwiftAIAmazonBedrock
@testable import SwiftAIAnthropic
@testable import SwiftAI
@testable import SwiftAIProvider
@testable import SwiftAIProviderUtilities
@testable import SwiftAITransport
import Foundation
import Testing

@Test func amazonBedrockEmbeddingUsesInvokeEndpoint() async throws {
    let transport = RecordingTransport(response: jsonResponse("""
    {"embedding":[0.1,0.2,0.3]}
    """))
    let provider = try AIProviders.amazonBedrock(configuration: AmazonBedrockProviderSettings(
        region: "us-west-2",
        apiKey: "bearer-key",
        transport: transport
    ))
    let model = try provider.embeddingModel("amazon.titan-embed-text-v2:0")
    #expect(model.maxEmbeddingsPerCall == 1)

    let result = try await model.embed(EmbeddingRequest(values: ["hello"], dimensions: 256))

    #expect(result.embeddings == [[0.1, 0.2, 0.3]])
    let request = try #require(await transport.requests().first)
    #expect(request.url.absoluteString == "https://bedrock-runtime.us-west-2.amazonaws.com/model/amazon.titan-embed-text-v2%3A0/invoke")
    #expect(request.headers["Authorization"] == "Bearer bearer-key")
    let body = try decodeJSONBody(try #require(request.body))
    #expect(body["inputText"]?.stringValue == "hello")
    #expect(body["dimensions"]?.intValue == 256)
}
@Test func amazonBedrockEmbeddingMapsProviderOptionsAndResponseShapesLikeUpstream() async throws {
    let titanTransport = RecordingTransport(response: jsonResponse("""
    {"embedding":[0.1,0.2],"inputTextTokenCount":4}
    """))
    let titanProvider = try AIProviders.amazonBedrock(configuration: AmazonBedrockProviderSettings(region: "us-west-2", apiKey: "bearer-key", transport: titanTransport))
    let titanResult = try await titanProvider.embeddingModel("amazon.titan-embed-text-v2:0").embed(EmbeddingRequest(
        values: ["hello"],
        providerOptions: ["bedrock": ["dimensions": 512, "normalize": false]]
    ))

    #expect(titanResult.embeddings == [[0.1, 0.2]])
    #expect(titanResult.usage?.totalTokens == 4)
    let titanBody = try decodeJSONBody(try #require((await titanTransport.requests()).first?.body))
    #expect(titanBody["inputText"]?.stringValue == "hello")
    #expect(titanBody["dimensions"]?.intValue == 512)
    #expect(titanBody["normalize"]?.boolValue == false)
    #expect(titanBody["bedrock"] == nil)

    let cohereTransport = RecordingTransport(response: jsonResponse("""
    {"embeddings":[[0.3,0.4]]}
    """))
    let cohereProvider = try AIProviders.amazonBedrock(configuration: AmazonBedrockProviderSettings(region: "us-west-2", apiKey: "bearer-key", transport: cohereTransport))
    let cohereResult = try await cohereProvider.embeddingModel("cohere.embed-english-v3").embed(EmbeddingRequest(
        values: ["hello"],
        providerOptions: ["bedrock": ["inputType": "search_document", "truncate": "START", "outputDimension": 1024]]
    ))

    #expect(cohereResult.embeddings == [[0.3, 0.4]])
    let cohereBody = try decodeJSONBody(try #require((await cohereTransport.requests()).first?.body))
    #expect(cohereBody["input_type"]?.stringValue == "search_document")
    #expect(cohereBody["texts"]?[0]?.stringValue == "hello")
    #expect(cohereBody["truncate"]?.stringValue == "START")
    #expect(cohereBody["output_dimension"]?.intValue == 1024)

    let novaTransport = RecordingTransport(response: jsonResponse("""
    {"embeddings":[{"embeddingType":"float","embedding":[0.5,0.6]}],"inputTokenCount":5}
    """))
    let novaProvider = try AIProviders.amazonBedrock(configuration: AmazonBedrockProviderSettings(region: "us-west-2", apiKey: "bearer-key", transport: novaTransport))
    let novaResult = try await novaProvider.embeddingModel("amazon.nova-embed-text-v1:0").embed(EmbeddingRequest(
        values: ["hello"],
        providerOptions: ["bedrock": ["embeddingPurpose": "TEXT_RETRIEVAL", "embeddingDimension": 3072, "truncate": "NONE"]]
    ))

    #expect(novaResult.embeddings == [[0.5, 0.6]])
    #expect(novaResult.usage?.totalTokens == 5)
    let novaBody = try decodeJSONBody(try #require((await novaTransport.requests()).first?.body))
    #expect(novaBody["taskType"]?.stringValue == "SINGLE_EMBEDDING")
    let novaParams = novaBody["singleEmbeddingParams"]
    #expect(novaParams?["embeddingPurpose"]?.stringValue == "TEXT_RETRIEVAL")
    #expect(novaParams?["embeddingDimension"]?.intValue == 3072)
    #expect(novaParams?["text"]?["truncationMode"]?.stringValue == "NONE")

    let cohereV4Transport = RecordingTransport(response: jsonResponse("""
    {"embeddings":{"float":[[0.7,0.8]]}}
    """, headers: ["x-amzn-bedrock-input-token-count": "6"]))
    let cohereV4Provider = try AIProviders.amazonBedrock(configuration: AmazonBedrockProviderSettings(region: "us-west-2", apiKey: "bearer-key", transport: cohereV4Transport))
    let cohereV4Result = try await cohereV4Provider.embeddingModel("us.cohere.embed-v4:0").embed(EmbeddingRequest(values: ["hello"]))
    #expect(cohereV4Result.embeddings == [[0.7, 0.8]])
    #expect(cohereV4Result.usage?.totalTokens == 6)
    let cohereV4Body = try decodeJSONBody(try #require((await cohereV4Transport.requests()).first?.body))
    #expect(cohereV4Body["texts"]?[0]?.stringValue == "hello")
}
@Test func amazonBedrockCohereEmbeddingBatchesMultipleInputsLikeUpstream() async throws {
    let transport = RecordingTransport(response: jsonResponse("""
    {"embeddings":[[0.1,0.2],[0.3,0.4]]}
    """))
    let provider = try AIProviders.amazonBedrock(configuration: AmazonBedrockProviderSettings(
        region: "us-west-2",
        apiKey: "bearer-key",
        transport: transport
    ))
    let model = try provider.embeddingModel("cohere.embed-english-v3")

    let result = try await model.embed(EmbeddingRequest(values: ["hello", "world"]))

    #expect(result.embeddings == [[0.1, 0.2], [0.3, 0.4]])
    let body = try decodeJSONBody(try #require((await transport.requests()).first?.body))
    #expect(body["texts"]?[0]?.stringValue == "hello")
    #expect(body["texts"]?[1]?.stringValue == "world")
    #expect(body["input_type"]?.stringValue == "search_query")
}
@Test func amazonBedrockEmbeddingRejectsTooManyValuesLikeUpstream() async throws {
    let provider = try AIProviders.amazonBedrock(configuration: AmazonBedrockProviderSettings(
        region: "us-west-2",
        apiKey: "bearer-key",
        transport: RecordingTransport(response: jsonResponse("{}"))
    ))
    let model = try provider.embeddingModel("amazon.titan-embed-text-v2:0")

    await #expect(throws: AITooManyEmbeddingValuesForCallError(
        provider: "amazon-bedrock",
        modelID: "amazon.titan-embed-text-v2:0",
        maxEmbeddingsPerCall: 1,
        values: ["hello", "world"]
    )) {
        _ = try await model.embed(EmbeddingRequest(values: ["hello", "world"]))
    }

    let cohere = try provider.embeddingModel("cohere.embed-english-v3")
    #expect(cohere.maxEmbeddingsPerCall == 96)
    let values = (0..<97).map { "value-\($0)" }
    await #expect(throws: AITooManyEmbeddingValuesForCallError(
        provider: "amazon-bedrock",
        modelID: "cohere.embed-english-v3",
        maxEmbeddingsPerCall: 96,
        values: values
    )) {
        _ = try await cohere.embed(EmbeddingRequest(values: values))
    }
}
@Test func amazonBedrockRerankingUsesAgentRuntimeShapeAndNestedOptions() async throws {
    let transport = RecordingTransport(response: jsonResponse("""
    {"results":[{"index":1,"relevanceScore":0.81},{"index":0,"relevanceScore":0.42}]}
    """))
    let provider = try AIProviders.amazonBedrock(configuration: AmazonBedrockProviderSettings(
        region: "us-west-2",
        apiKey: "bearer-key",
        transport: transport
    ))
    let model = try provider.rerankingModel("cohere.rerank-v3-5:0")

    let result = try await model.rerank(RerankingRequest(
        query: "rainy day",
        documents: ["sunny beach", "rainy city"],
        topK: 2,
        extraBody: [
            "amazonBedrock": .object([
                "nextToken": .string("token-1"),
                "additionalModelRequestFields": .object(["truncate": .string("END")])
            ])
        ]
    ))

    #expect(result.results.map(\.index) == [1, 0])
    #expect(result.results.map(\.score) == [0.81, 0.42])
    let request = try #require(await transport.requests().first)
    #expect(request.url.absoluteString == "https://bedrock-agent-runtime.us-west-2.amazonaws.com/rerank")
    #expect(request.headers["Authorization"] == "Bearer bearer-key")
    let body = try decodeJSONBody(try #require(request.body))
    #expect(body["nextToken"]?.stringValue == "token-1")
    #expect(body["queries"]?[0]?["textQuery"]?["text"]?.stringValue == "rainy day")
    #expect(body["sources"]?[0]?["inlineDocumentSource"]?["textDocument"]?["text"]?.stringValue == "sunny beach")
    #expect(body["sources"]?[1]?["inlineDocumentSource"]?["textDocument"]?["text"]?.stringValue == "rainy city")
    let rerankingConfig = body["rerankingConfiguration"]
    #expect(rerankingConfig?["type"]?.stringValue == "BEDROCK_RERANKING_MODEL")
    let bedrockConfig = rerankingConfig?["bedrockRerankingConfiguration"]
    #expect(bedrockConfig?["numberOfResults"]?.intValue == 2)
    #expect(bedrockConfig?["modelConfiguration"]?["modelArn"]?.stringValue == "arn:aws:bedrock:us-west-2::foundation-model/cohere.rerank-v3-5:0")
    #expect(bedrockConfig?["modelConfiguration"]?["additionalModelRequestFields"]?["truncate"]?.stringValue == "END")
    #expect(rerankingConfig?["amazonBedrockRerankingConfiguration"] == nil)
    #expect(body["amazonBedrock"] == nil)
}
@Test func amazonBedrockImageMapsTextOptions() async throws {
    let transport = RecordingTransport(response: jsonResponse(#"{"images":["image-1"]}"#))
    let provider = try AIProviders.amazonBedrock(configuration: AmazonBedrockProviderSettings(
        region: "us-east-1",
        apiKey: "bearer-key",
        transport: transport
    ))
    let model = try provider.imageModel("amazon.nova-canvas-v1:0")

    let result = try await model.generateImage(ImageGenerationRequest(
        prompt: "A studio portrait",
        size: "1024x768",
        aspectRatio: "4:3",
        count: 2,
        extraBody: [
            "bedrock": .object([
                "negativeText": .string("blur"),
                "quality": .string("premium"),
                "cfgScale": .number(7),
                "style": .string("PHOTOREALISM"),
                "seed": .number(1234)
            ])
        ]
    ))

    #expect(result.base64Images == ["image-1"])
    #expect(result.warnings == [AIWarning(type: "unsupported", feature: "aspectRatio", message: "This model does not support aspect ratio. Use size instead.")])
    let request = try #require(await transport.requests().first)
    #expect(request.url.absoluteString == "https://bedrock-runtime.us-east-1.amazonaws.com/model/amazon.nova-canvas-v1%3A0/invoke")
    let body = try decodeJSONBody(try #require(request.body))
    #expect(body["taskType"]?.stringValue == "TEXT_IMAGE")
    #expect(body["textToImageParams"]?["text"]?.stringValue == "A studio portrait")
    #expect(body["textToImageParams"]?["negativeText"]?.stringValue == "blur")
    #expect(body["textToImageParams"]?["style"]?.stringValue == "PHOTOREALISM")
    #expect(body["imageGenerationConfig"]?["width"]?.intValue == 1024)
    #expect(body["imageGenerationConfig"]?["height"]?.intValue == 768)
    #expect(body["imageGenerationConfig"]?["numberOfImages"]?.intValue == 2)
    #expect(body["imageGenerationConfig"]?["quality"]?.stringValue == "premium")
    #expect(body["imageGenerationConfig"]?["cfgScale"]?.intValue == 7)
    #expect(body["imageGenerationConfig"]?["seed"]?.intValue == 1234)
    #expect(body["bedrock"] == nil)
}
@Test func amazonBedrockImageValidatesModerationEmptyImagesAndCountLikeUpstream() async throws {
    let moderatedProvider = try AIProviders.amazonBedrock(configuration: AmazonBedrockProviderSettings(
        region: "us-east-1",
        apiKey: "bearer-key",
        transport: RecordingTransport(response: jsonResponse(#"{"status":"Request Moderated","details":{"Moderation Reasons":["SAFETY"]}}"#))
    ))
    let moderatedModel = try moderatedProvider.imageModel("amazon.nova-canvas-v1:0")
    await #expect(throws: AIError.invalidResponse(provider: "amazon-bedrock", message: "Amazon Bedrock request was moderated: SAFETY.")) {
        _ = try await moderatedModel.generateImage(ImageGenerationRequest(prompt: "blocked"))
    }

    let emptyProvider = try AIProviders.amazonBedrock(configuration: AmazonBedrockProviderSettings(
        region: "us-east-1",
        apiKey: "bearer-key",
        transport: RecordingTransport(response: jsonResponse(#"{"status":"Completed","images":[]}"#))
    ))
    let emptyModel = try emptyProvider.imageModel("amazon.nova-canvas-v1:0")
    await #expect(throws: AIError.invalidResponse(provider: "amazon-bedrock", message: "Amazon Bedrock returned no images. Status: Completed")) {
        _ = try await emptyModel.generateImage(ImageGenerationRequest(prompt: "empty"))
    }

    await #expect(throws: AIError.invalidArgument(argument: "count", message: "Amazon Bedrock image model amazon.nova-canvas-v1:0 supports at most 5 image(s) per call.")) {
        _ = try await emptyModel.generateImage(ImageGenerationRequest(prompt: "too many", count: 6))
    }
}
@Test func amazonBedrockImageMapsEditModes() async throws {
    let transport = RecordingTransport(responses: [
        jsonResponse(#"{"images":["inpainted"]}"#),
        jsonResponse(#"{"images":["outpainted"]}"#),
        jsonResponse(#"{"images":["background-removed"]}"#),
        jsonResponse(#"{"images":["variation"]}"#)
    ])
    let provider = try AIProviders.amazonBedrock(configuration: AmazonBedrockProviderSettings(
        region: "us-east-1",
        apiKey: "bearer-key",
        transport: transport
    ))
    let model = try provider.imageModel("amazon.nova-canvas-v1:0")

    _ = try await model.generateImage(ImageGenerationRequest(
        prompt: "Replace the sky",
        count: 1,
        files: [ImageInputFile(data: Data([137, 80, 78, 71]), mediaType: "image/png")],
        extraBody: [
            "amazonBedrock": .object([
                "maskPrompt": .string("sky"),
                "negativeText": .string("rain"),
                "quality": .string("standard")
            ])
        ]
    ))
    _ = try await model.generateImage(ImageGenerationRequest(
        prompt: "Extend the scene",
        files: [ImageInputFile(data: Data([137, 80, 78, 71]), mediaType: "image/png")],
        mask: ImageInputFile(data: Data([255, 255, 255, 0]), mediaType: "image/png"),
        extraBody: [
            "amazonBedrock": .object([
                "taskType": .string("OUTPAINTING"),
                "outPaintingMode": .string("DEFAULT")
            ])
        ]
    ))
    _ = try await model.generateImage(ImageGenerationRequest(
        prompt: "",
        files: [ImageInputFile(data: Data([137, 80, 78, 71]), mediaType: "image/png")],
        extraBody: ["amazonBedrock": .object(["taskType": .string("BACKGROUND_REMOVAL")])]
    ))
    _ = try await model.generateImage(ImageGenerationRequest(
        prompt: "Create a variation",
        size: "512x512",
        count: 3,
        files: [
            ImageInputFile(data: Data([137, 80, 78, 71]), mediaType: "image/png"),
            ImageInputFile(data: Data([255, 216, 255, 224]), mediaType: "image/jpeg")
        ],
        extraBody: [
            "amazonBedrock": .object([
                "taskType": .string("IMAGE_VARIATION"),
                "similarityStrength": .number(0.7),
                "negativeText": .string("low quality")
            ])
        ]
    ))

    let requests = await transport.requests()
    let inpainting = try decodeJSONBody(try #require(requests[0].body))
    #expect(inpainting["taskType"]?.stringValue == "INPAINTING")
    #expect(inpainting["inPaintingParams"]?["image"]?.stringValue == Data([137, 80, 78, 71]).base64EncodedString())
    #expect(inpainting["inPaintingParams"]?["maskPrompt"]?.stringValue == "sky")
    #expect(inpainting["inPaintingParams"]?["negativeText"]?.stringValue == "rain")
    #expect(inpainting["imageGenerationConfig"]?["quality"]?.stringValue == "standard")

    let outpainting = try decodeJSONBody(try #require(requests[1].body))
    #expect(outpainting["taskType"]?.stringValue == "OUTPAINTING")
    #expect(outpainting["outPaintingParams"]?["maskImage"]?.stringValue == Data([255, 255, 255, 0]).base64EncodedString())
    #expect(outpainting["outPaintingParams"]?["outPaintingMode"]?.stringValue == "DEFAULT")

    let backgroundRemoval = try decodeJSONBody(try #require(requests[2].body))
    #expect(backgroundRemoval["taskType"]?.stringValue == "BACKGROUND_REMOVAL")
    #expect(backgroundRemoval["backgroundRemovalParams"]?["image"]?.stringValue == Data([137, 80, 78, 71]).base64EncodedString())
    #expect(backgroundRemoval["imageGenerationConfig"] == nil)

    let variation = try decodeJSONBody(try #require(requests[3].body))
    #expect(variation["taskType"]?.stringValue == "IMAGE_VARIATION")
    #expect(variation["imageVariationParams"]?["images"]?[0]?.stringValue == Data([137, 80, 78, 71]).base64EncodedString())
    #expect(variation["imageVariationParams"]?["images"]?[1]?.stringValue == Data([255, 216, 255, 224]).base64EncodedString())
    #expect(variation["imageVariationParams"]?["similarityStrength"]?.doubleValue == 0.7)
    #expect(variation["imageGenerationConfig"]?["width"]?.intValue == 512)
    #expect(variation["imageGenerationConfig"]?["height"]?.intValue == 512)
    #expect(variation["imageGenerationConfig"]?["numberOfImages"]?.intValue == 3)
}
