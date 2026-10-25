import SwiftAIOpenAICompatible
import SwiftAIOpenAI
import SwiftAIAnthropic
import Foundation
import SwiftAIProvider
import SwiftAITransport
import SwiftAIProviderUtilities


public final class AmazonBedrockImageModel: ImageModel, @unchecked Sendable {
    public let providerID = "amazon-bedrock"
    public let modelID: String
    private let config: BedrockRuntimeConfig

    init(modelID: String, config: BedrockRuntimeConfig) {
        self.modelID = modelID
        self.config = config
    }

    public func generateImage(_ request: ImageGenerationRequest) async throws -> ImageGenerationResult {
        if let count = request.count, count > maxImagesPerCall {
            throw AIError.invalidArgument(argument: "count", message: "Amazon Bedrock image model \(modelID) supports at most \(maxImagesPerCall) image(s) per call.")
        }

        let providerOptions = try bedrockImageProviderOptions(extraBody: request.extraBody, providerOptions: request.providerOptions)
        var warnings: [AIWarning] = []
        if request.aspectRatio != nil {
            warnings.append(AIWarning(
                type: "unsupported",
                feature: "aspectRatio",
                message: "This model does not support aspect ratio. Use size instead."
            ))
        }
        let size = request.size?.split(separator: "x").compactMap { Int($0) } ?? []
        var imageGenerationConfig: [String: JSONValue] = [:]
        if size.count == 2 {
            imageGenerationConfig["width"] = .number(Double(size[0]))
            imageGenerationConfig["height"] = .number(Double(size[1]))
        }
        if let count = request.count { imageGenerationConfig["numberOfImages"] = .number(Double(count)) }
        if let seed = providerOptions["seed"] {
            imageGenerationConfig["seed"] = seed
        } else if let seed = request.extraBody["seed"] {
            imageGenerationConfig["seed"] = seed
        }
        if let quality = providerOptions["quality"] {
            imageGenerationConfig["quality"] = quality
        }
        if let cfgScale = providerOptions["cfgScale"] ?? providerOptions["cfg_scale"] {
            imageGenerationConfig["cfgScale"] = cfgScale
        }

        let body = try bedrockImageBody(
            request: request,
            providerOptions: providerOptions,
            imageGenerationConfig: imageGenerationConfig
        )
        let response = try await config.sendJSONResponse(path: "/model/\(encodedModelID)/invoke", body: .object(body), headers: request.headers, abortSignal: request.abortSignal)
        let raw = response.json
        if raw["status"]?.stringValue == "Request Moderated" {
            let reasons = raw["details"]?["Moderation Reasons"]?.arrayValue?.compactMap(\.stringValue)
            let message = (reasons?.isEmpty == false ? reasons ?? [] : ["Unknown"]).joined(separator: ", ")
            throw AIError.invalidResponse(provider: providerID, message: "Amazon Bedrock request was moderated: \(message).")
        }
        let base64Images = raw["images"]?.arrayValue?.compactMap(\.stringValue)
            ?? raw["artifacts"]?.arrayValue?.compactMap { $0["base64"]?.stringValue }
            ?? []
        guard !base64Images.isEmpty else {
            let statusSuffix = raw["status"]?.stringValue.map { " Status: \($0)" } ?? ""
            throw AIError.invalidResponse(provider: providerID, message: "Amazon Bedrock returned no images.\(statusSuffix)")
        }
        return ImageGenerationResult(
            urls: [],
            base64Images: base64Images,
            rawValue: raw,
            warnings: warnings,
            requestMetadata: imageGenerationRequestMetadata(request, body: .object(body)),
            responseMetadata: aiResponseMetadata(from: raw, response: response.response, modelID: modelID)
        )
    }

    private var maxImagesPerCall: Int {
        modelID == "amazon.nova-canvas-v1:0" ? 5 : 1
    }

    private var encodedModelID: String {
        bedrockEncodeModelID(modelID)
    }

    private func bedrockImageBody(
        request: ImageGenerationRequest,
        providerOptions: [String: JSONValue],
        imageGenerationConfig: [String: JSONValue]
    ) throws -> [String: JSONValue] {
        var body: [String: JSONValue]
        if request.files.isEmpty {
            var textToImageParams: [String: JSONValue] = ["text": .string(request.prompt)]
            if let negativeText = providerOptions["negativeText"] ?? providerOptions["negative_text"] {
                textToImageParams["negativeText"] = negativeText
            }
            if let style = providerOptions["style"] {
                textToImageParams["style"] = style
            }
            body = [
                "taskType": .string("TEXT_IMAGE"),
                "textToImageParams": .object(textToImageParams)
            ]
            if !imageGenerationConfig.isEmpty {
                body["imageGenerationConfig"] = .object(imageGenerationConfig)
            }
            body.merge(bedrockImagePassthroughExtraBody(request.extraBody)) { _, new in new }
            return body
        }

        let taskType = providerOptions["taskType"]?.stringValue
            ?? providerOptions["task_type"]?.stringValue
            ?? ((request.mask != nil || providerOptions["maskPrompt"] != nil || providerOptions["mask_prompt"] != nil) ? "INPAINTING" : "IMAGE_VARIATION")
        let sourceImage = try bedrockImageBase64(request.files[0])

        switch taskType {
        case "INPAINTING":
            var params: [String: JSONValue] = ["image": .string(sourceImage)]
            if !request.prompt.isEmpty { params["text"] = .string(request.prompt) }
            if let negativeText = providerOptions["negativeText"] ?? providerOptions["negative_text"] {
                params["negativeText"] = negativeText
            }
            if let mask = request.mask {
                params["maskImage"] = .string(try bedrockImageBase64(mask))
            } else if let maskPrompt = providerOptions["maskPrompt"] ?? providerOptions["mask_prompt"] {
                params["maskPrompt"] = maskPrompt
            }
            body = [
                "taskType": .string("INPAINTING"),
                "inPaintingParams": .object(params)
            ]
        case "OUTPAINTING":
            var params: [String: JSONValue] = ["image": .string(sourceImage)]
            if !request.prompt.isEmpty { params["text"] = .string(request.prompt) }
            if let negativeText = providerOptions["negativeText"] ?? providerOptions["negative_text"] {
                params["negativeText"] = negativeText
            }
            if let outPaintingMode = providerOptions["outPaintingMode"] ?? providerOptions["out_painting_mode"] {
                params["outPaintingMode"] = outPaintingMode
            }
            if let mask = request.mask {
                params["maskImage"] = .string(try bedrockImageBase64(mask))
            } else if let maskPrompt = providerOptions["maskPrompt"] ?? providerOptions["mask_prompt"] {
                params["maskPrompt"] = maskPrompt
            }
            body = [
                "taskType": .string("OUTPAINTING"),
                "outPaintingParams": .object(params)
            ]
        case "BACKGROUND_REMOVAL":
            body = [
                "taskType": .string("BACKGROUND_REMOVAL"),
                "backgroundRemovalParams": .object(["image": .string(sourceImage)])
            ]
        case "IMAGE_VARIATION":
            var params: [String: JSONValue] = [
                "images": .array(try request.files.map { .string(try bedrockImageBase64($0)) })
            ]
            if !request.prompt.isEmpty { params["text"] = .string(request.prompt) }
            if let negativeText = providerOptions["negativeText"] ?? providerOptions["negative_text"] {
                params["negativeText"] = negativeText
            }
            if let similarityStrength = providerOptions["similarityStrength"] ?? providerOptions["similarity_strength"] {
                params["similarityStrength"] = similarityStrength
            }
            body = [
                "taskType": .string("IMAGE_VARIATION"),
                "imageVariationParams": .object(params)
            ]
        default:
            throw AIError.invalidArgument(argument: "extraBody.amazonBedrock.taskType", message: "Unsupported Amazon Bedrock image task type: \(taskType).")
        }

        if taskType != "BACKGROUND_REMOVAL", !imageGenerationConfig.isEmpty {
            body["imageGenerationConfig"] = .object(imageGenerationConfig)
        }
        body.merge(bedrockImagePassthroughExtraBody(request.extraBody)) { _, new in new }
        return body
    }
}

let bedrockImageProviderOptionKeys: Set<String> = [
    "negativeText",
    "negative_text",
    "quality",
    "cfgScale",
    "cfg_scale",
    "style",
    "taskType",
    "task_type",
    "maskPrompt",
    "mask_prompt",
    "outPaintingMode",
    "out_painting_mode",
    "similarityStrength",
    "similarity_strength",
    "seed"
]

func bedrockImageProviderOptions(extraBody: [String: JSONValue], providerOptions: [String: JSONValue]) throws -> [String: JSONValue] {
    var output = extraBody.filter { key, _ in bedrockImageProviderOptionKeys.contains(key) }
    if let bedrock = extraBody["bedrock"]?.objectValue {
        output.merge(bedrock) { _, new in new }
    }
    if let amazonBedrock = extraBody["amazonBedrock"]?.objectValue {
        output.merge(amazonBedrock) { _, new in new }
    }
    if let bedrock = providerOptions["bedrock"] {
        guard let object = bedrock.objectValue else {
            throw AIError.invalidArgument(argument: "providerOptions.bedrock", message: "Bedrock provider options must be an object.")
        }
        output.merge(object) { _, new in new }
    }
    if let amazonBedrock = providerOptions["amazonBedrock"] {
        guard let object = amazonBedrock.objectValue else {
            throw AIError.invalidArgument(argument: "providerOptions.amazonBedrock", message: "Amazon Bedrock provider options must be an object.")
        }
        output.merge(object) { _, new in new }
    }
    return output
}

func bedrockImagePassthroughExtraBody(_ extraBody: [String: JSONValue]) -> [String: JSONValue] {
    extraBody.filter { key, _ in
        key != "amazonBedrock" && key != "bedrock" && !bedrockImageProviderOptionKeys.contains(key)
    }
}

func bedrockImageBase64(_ file: ImageInputFile) throws -> String {
    guard let data = file.data else {
        throw AIError.invalidArgument(
            argument: "files",
            message: "URL-based images are not supported for Amazon Bedrock image editing."
        )
    }
    return data.base64EncodedString()
}