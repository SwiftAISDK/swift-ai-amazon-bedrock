import SwiftAIOpenAICompatible
import SwiftAIOpenAI
import SwiftAIAnthropic
import Foundation
import SwiftAIProvider
import SwiftAITransport
import SwiftAIProviderUtilities


func bedrockEncodeModelID(_ modelID: String) -> String {
    var allowed = CharacterSet.alphanumerics
    allowed.insert(charactersIn: "-._~")
    return modelID.addingPercentEncoding(withAllowedCharacters: allowed) ?? modelID
}

func bedrockIsMistralModel(_ modelID: String) -> Bool {
    modelID.contains("mistral.")
}

func bedrockNormalizeToolCallID(_ toolCallID: String, modelID: String) -> String {
    guard bedrockIsMistralModel(modelID) else { return toolCallID }
    guard toolCallID.range(of: #"^[A-Za-z0-9]{9}$"#, options: .regularExpression) == nil else {
        return toolCallID
    }

    // Match the JavaScript package exactly: FNV-1a over UTF-16 code units,
    // reduced into the full nine-character base62 identifier space.
    let base62Characters = Array("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
    let base = UInt64(base62Characters.count)
    let outputLength = 9
    var identifierSpace: UInt64 = 1
    for _ in 0..<outputLength {
        identifierSpace &*= base
    }

    var hash: UInt64 = 14_695_981_039_346_656_037
    for codeUnit in toolCallID.utf16 {
        hash ^= UInt64(codeUnit)
        hash &*= 1_099_511_628_211
    }

    var value = hash % identifierSpace
    var output = Array(repeating: Character("0"), count: outputLength)
    for index in stride(from: outputLength - 1, through: 0, by: -1) {
        output[index] = base62Characters[Int(value % base)]
        value /= base
    }
    return String(output)
}

let bedrockDocumentMimeTypes: [String: String] = [
    "application/pdf": "pdf",
    "text/csv": "csv",
    "application/msword": "doc",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "application/vnd.ms-excel": "xls",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
    "text/html": "html",
    "text/plain": "txt",
    "text/markdown": "md"
]

let bedrockImageMimeTypes: [String: String] = [
    "image/jpeg": "jpeg",
    "image/png": "png",
    "image/gif": "gif",
    "image/webp": "webp"
]

let bedrockVideoMimeTypes: [String: String] = [
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

var bedrockSupportedDocumentMimeTypes: [String] {
    bedrockDocumentMimeTypes.keys.sorted()
}

var bedrockSupportedImageMimeTypes: [String] {
    bedrockImageMimeTypes.keys.sorted()
}

let bedrockSupportedVideoMimeTypes = [
    "video/x-matroska",
    "video/quicktime",
    "video/mp4",
    "video/webm",
    "video/x-flv",
    "video/mpeg",
    "video/mpg",
    "video/wmv",
    "video/x-ms-wmv",
    "video/3gpp"
]

func bedrockDocumentFormat(for mimeType: String) -> String? {
    bedrockDocumentMimeTypes[mimeType]
}

func bedrockImageFormat(for mimeType: String) -> String? {
    bedrockImageMimeTypes[mimeType]
}

func bedrockVideoFormat(for mimeType: String) -> String? {
    bedrockVideoMimeTypes[mimeType]
}

func bedrockIsS3URL(_ value: String) -> Bool {
    value.hasPrefix("s3://")
}

func bedrockImageFormat(forURL value: String) -> String? {
    switch URL(string: value)?.pathExtension.lowercased() {
    case "jpg", "jpeg":
        return "jpeg"
    case "png":
        return "png"
    case "gif":
        return "gif"
    case "webp":
        return "webp"
    default:
        return nil
    }
}

struct BedrockPreparedTools {
    var toolConfig: JSONValue?
    var additionalModelRequestFields: [String: JSONValue]? = nil
    var warnings: [AIWarning]
}

func bedrockRequestProviderOptions(providerOptions: [String: JSONValue], extraBody: [String: JSONValue]) throws -> [String: JSONValue] {
    var output: [String: JSONValue] = [:]
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
    if let metadata = output["requestMetadata"] {
        guard let object = metadata.objectValue, object.values.allSatisfy({ $0.stringValue != nil }) else {
            throw AIError.invalidArgument(argument: "providerOptions.amazonBedrock.requestMetadata", message: "requestMetadata must contain string values.")
        }
    }
    return output
}

func bedrockPassthroughExtraBody(_ extraBody: [String: JSONValue]) -> [String: JSONValue] {
    extraBody.filter { key, _ in key != "amazonBedrock" && key != "bedrock" && key != "toolChoice" }
}

func bedrockDocumentCitationsEnabled(_ providerOptions: [String: JSONValue]) -> Bool {
    providerOptions["citations"]?["enabled"]?.boolValue
        ?? providerOptions["amazonBedrock"]?["citations"]?["enabled"]?.boolValue
        ?? providerOptions["bedrock"]?["citations"]?["enabled"]?.boolValue
        ?? false
}

func bedrockResponseJSONSchema(from responseFormat: AIResponseFormat?) -> JSONValue? {
    guard case let .json(schema, _, _) = responseFormat else { return nil }
    return schema
}

func bedrockReasoningConfigEnabled(_ value: JSONValue?) -> Bool {
    let type = value?["type"]?.stringValue
    return type == "enabled" || type == "adaptive"
}

func bedrockApplyRequestProviderOptions(_ providerOptions: [String: JSONValue], to body: inout [String: JSONValue]) {
    if let requestMetadata = providerOptions["requestMetadata"] { body["requestMetadata"] = requestMetadata }
    if let guardrailConfig = providerOptions["guardrailConfig"] {
        body["guardrailConfig"] = guardrailConfig
    }
    if let additionalModelRequestFields = providerOptions["additionalModelRequestFields"] {
        body["additionalModelRequestFields"] = additionalModelRequestFields
    }
    if let serviceTier = providerOptions["serviceTier"] {
        if let serviceTierType = serviceTier.stringValue {
            body["serviceTier"] = .object(["type": .string(serviceTierType)])
        } else {
            body["serviceTier"] = serviceTier
        }
    }
}

func bedrockCachePoint(from providerMetadata: [String: JSONValue]) -> JSONValue? {
    let cachePoint = providerMetadata["amazonBedrock"]?["cachePoint"]
        ?? providerMetadata["bedrock"]?["cachePoint"]
        ?? providerMetadata["cachePoint"]
    guard let cachePoint else { return nil }
    return .object(["cachePoint": cachePoint])
}

func bedrockGuardedTextBlock(text: String, providerMetadata: [String: JSONValue]) throws -> JSONValue {
    let options = providerMetadata["amazonBedrock"]?.objectValue
        ?? providerMetadata["bedrock"]?.objectValue
    guard let options, options["guardContent"]?.boolValue == true else {
        return .object(["text": .string(text)])
    }
    var guardedText: [String: JSONValue] = ["text": .string(text)]
    if let rawQualifiers = options["guardContentQualifiers"] {
        guard let qualifiers = rawQualifiers.arrayValue?.compactMap(\.stringValue),
              qualifiers.count == rawQualifiers.arrayValue?.count,
              qualifiers.allSatisfy({ ["grounding_source", "query", "guard_content"].contains($0) }) else {
            throw AIError.invalidArgument(
                argument: "messages.content.providerMetadata.bedrock.guardContentQualifiers",
                message: "Bedrock guardContentQualifiers must contain only grounding_source, query, or guard_content."
            )
        }
        guardedText["qualifiers"] = .array(qualifiers.map(JSONValue.string))
    }
    return .object([
        "guardContent": .object([
            "text": .object(guardedText)
        ])
    ])
}

func bedrockGuardedImageBlock(_ imageBlock: JSONValue, providerMetadata: [String: JSONValue]) -> JSONValue {
    let options = providerMetadata["amazonBedrock"]?.objectValue
        ?? providerMetadata["bedrock"]?.objectValue
    guard options?["guardContent"]?.boolValue == true else { return imageBlock }
    return .object(["guardContent": imageBlock])
}

func bedrockReasoningContentBlock(text: String, providerMetadata: [String: JSONValue]) -> JSONValue? {
    let metadata = providerMetadata["amazonBedrock"]?.objectValue
        ?? providerMetadata["bedrock"]?.objectValue
        ?? providerMetadata
    if let signature = metadata["signature"]?.stringValue {
        return .object([
            "reasoningContent": .object([
                "reasoningText": .object([
                    "text": .string(text),
                    "signature": .string(signature)
                ])
            ])
        ])
    }
    if let redactedContent = metadata["redactedContent"]?.stringValue {
        return .object([
            "reasoningContent": .object([
                "redactedContent": .string(redactedContent)
            ])
        ])
    }
    if let redactedData = metadata["redactedData"]?.stringValue {
        return .object([
            "reasoningContent": .object([
                "redactedReasoning": .object(["data": .string(redactedData)])
            ])
        ])
    }
    return nil
}

func bedrockPrepareTools(
    from tools: [String: JSONValue],
    toolChoice: JSONValue?,
    modelID: String,
    modelFamily: AmazonBedrockChatModelFamily? = nil,
    reasoningConfig: JSONValue? = nil,
    disableParallelToolUse: Bool = false
) -> BedrockPreparedTools {
    guard !tools.isEmpty else {
        return BedrockPreparedTools(toolConfig: nil, warnings: [])
    }

    let isAnthropicModel = bedrockIsAnthropicModel(
        modelID: modelID,
        modelFamily: modelFamily,
        reasoningConfig: reasoningConfig
    )
    var warnings: [AIWarning] = []
    var bedrockTools: [JSONValue] = []
    var supportedTools: [String: JSONValue] = [:]
    var preparedToolChoice = toolChoice
    let forcedToolName = bedrockForcedToolName(from: toolChoice)
    let rejectsForcedToolUse = anthropicModelCapabilities(modelID).rejectsForcedToolUse
    let forcedToolChoiceType = toolChoice?.stringValue ?? toolChoice?["type"]?.stringValue
    if rejectsForcedToolUse, forcedToolChoiceType == "tool" || forcedToolChoiceType == "required" {
        let detail = forcedToolChoiceType == "tool"
            ? "Only the '\(forcedToolName ?? "")' tool is sent with 'auto' tool choice. Instruct the model to use the tool in the prompt and verify that a tool call was made."
            : "Using 'auto' instead. Instruct the model to use a tool in the prompt and verify that a tool call was made."
        warnings.append(AIWarning(
            type: "unsupported",
            feature: "toolChoice",
            message: "toolChoice '\(forcedToolChoiceType ?? "")' is not supported by this model because it rejects forced tool use. \(detail)"
        ))
        preparedToolChoice = .object(["type": .string("auto")])
    }

    for (name, schema) in tools {
        if let forcedToolName, forcedToolName != name {
            continue
        }
        if schema["type"]?.stringValue == "provider" {
            let id = schema["id"]?.stringValue ?? name
            let unsupportedWebTools: Set<String> = [
                "anthropic.web_search_20250305",
                "anthropic.web_search_20260318",
                "anthropic.web_fetch_20260318"
            ]
            if unsupportedWebTools.contains(id) {
                let feature = String(id.dropFirst("anthropic.".count))
                warnings.append(AIWarning(
                    type: "unsupported",
                    feature: "\(feature) tool",
                    message: "The \(feature) tool is not supported on Amazon Bedrock."
                ))
                continue
            }
            if isAnthropicModel, let anthropicToolSchema = bedrockAnthropicProviderToolInputSchema(schema) {
                bedrockTools.append(.object([
                    "toolSpec": .object([
                        "name": .string(name),
                        "inputSchema": .object(["json": anthropicToolSchema])
                    ])
                ]))
                supportedTools[name] = schema
            } else {
                warnings.append(AIWarning(type: "unsupported", feature: "tool \(id)"))
            }
            continue
        }

        var toolSpec: [String: JSONValue] = [
            "name": .string(name),
            "inputSchema": .object(["json": schema])
        ]
        if let description = schema["description"]?.stringValue,
           !description.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            toolSpec["description"] = .string(description)
        }
        if let strict = schema["strict"], strict != .null {
            if !bedrockSupportsStrictToolSpec(modelID: modelID) {
                if strict.boolValue == true {
                    warnings.append(AIWarning(
                        type: "unsupported",
                        feature: "strict",
                        message: "Tool '\(name)' has strict: true, but strict mode is not supported by this model on Amazon Bedrock. The strict property will be ignored."
                    ))
                }
            } else if strict.boolValue == true, !bedrockStrictToolSchemaCompatible(schema) {
                warnings.append(AIWarning(
                    type: "unsupported",
                    feature: "strict",
                    message: "Tool '\(name)' has strict: true, but Amazon Bedrock requires every object in a strict tool schema to set additionalProperties: false. The strict property will be ignored."
                ))
            } else {
                toolSpec["strict"] = strict
            }
        }
        bedrockTools.append(.object(["toolSpec": .object(toolSpec)]))
        supportedTools[name] = schema
    }

    guard !bedrockTools.isEmpty else {
        return BedrockPreparedTools(toolConfig: nil, warnings: warnings)
    }

    var toolConfig: [String: JSONValue] = ["tools": .array(bedrockTools)]
    var additionalModelRequestFields: [String: JSONValue]?
    let usesAnthropicProviderTools = bedrockUsesAnthropicProviderTools(
        tools: supportedTools,
        modelID: modelID,
        modelFamily: modelFamily,
        reasoningConfig: reasoningConfig
    )
    if usesAnthropicProviderTools,
       let anthropicChoice = anthropicToolChoice(
           from: preparedToolChoice,
           disableParallelToolUse: disableParallelToolUse
       ).value {
        additionalModelRequestFields = ["tool_choice": anthropicChoice]
    }
    let preparedToolChoiceType = preparedToolChoice?.stringValue ?? preparedToolChoice?["type"]?.stringValue
    if isAnthropicModel,
       !usesAnthropicProviderTools,
       disableParallelToolUse,
       preparedToolChoiceType != "none" {
        var anthropicToolChoice: [String: JSONValue]
        switch preparedToolChoiceType {
        case "required":
            anthropicToolChoice = ["type": .string("any")]
        case "tool":
            anthropicToolChoice = [
                "type": .string("tool"),
                "name": .string(bedrockForcedToolName(from: preparedToolChoice) ?? "")
            ]
        default:
            anthropicToolChoice = ["type": .string("auto")]
        }
        anthropicToolChoice["disable_parallel_tool_use"] = .bool(true)
        additionalModelRequestFields = ["tool_choice": .object(anthropicToolChoice)]
    }
    if let choice = bedrockToolChoice(from: preparedToolChoice),
       !usesAnthropicProviderTools,
       additionalModelRequestFields == nil {
        if choice == .null {
            return BedrockPreparedTools(toolConfig: nil, warnings: warnings)
        }
        toolConfig["toolChoice"] = choice
    }
    return BedrockPreparedTools(
        toolConfig: .object(toolConfig),
        additionalModelRequestFields: additionalModelRequestFields,
        warnings: warnings
    )
}

func bedrockStrictToolSchemaCompatible(_ schema: JSONValue) -> Bool {
    if schema.boolValue != nil { return true }
    guard let object = schema.objectValue else { return true }

    let objectType = object["type"]?.stringValue == "object"
        || object["type"]?.arrayValue?.contains(where: { $0.stringValue == "object" }) == true
    if objectType, object["additionalProperties"]?.boolValue != false {
        return false
    }

    for key in ["properties", "patternProperties", "definitions", "$defs"] {
        if let schemas = object[key]?.objectValue,
           schemas.values.contains(where: { !bedrockStrictToolSchemaCompatible($0) }) {
            return false
        }
    }

    if let dependencies = object["dependencies"]?.objectValue {
        for dependency in dependencies.values where dependency.arrayValue == nil {
            if !bedrockStrictToolSchemaCompatible(dependency) { return false }
        }
    }

    for key in ["propertyNames", "contains", "not", "if", "then", "else"] {
        if let nested = object[key], !bedrockStrictToolSchemaCompatible(nested) {
            return false
        }
    }

    if let items = object["items"] {
        if let itemSchemas = items.arrayValue {
            if itemSchemas.contains(where: { !bedrockStrictToolSchemaCompatible($0) }) { return false }
        } else if !bedrockStrictToolSchemaCompatible(items) {
            return false
        }
    }

    for key in ["anyOf", "allOf", "oneOf"] {
        if let alternatives = object[key]?.arrayValue,
           alternatives.contains(where: { !bedrockStrictToolSchemaCompatible($0) }) {
            return false
        }
    }
    return true
}

func bedrockSupportsStrictToolSpec(modelID: String) -> Bool {
    !modelID.contains("claude-")
        || modelID.range(of: #"claude-(?:instant|v?2|3)(?=[-.:]|$)"#, options: .regularExpression) != nil
        || bedrockAnthropicStrictToolPatterns.contains { modelID.range(of: $0, options: .regularExpression) != nil }
}

func bedrockSupportsNativeStructuredOutput(modelID: String) -> Bool {
    bedrockSupportsStrictToolSpec(modelID: modelID)
        && modelID.range(of: #"claude-(?:sonnet-4-6|haiku-4-5)(?=[-.:]|$)"#, options: .regularExpression) == nil
}

private let bedrockAnthropicStrictToolPatterns = [
    #"claude-(?:opus|sonnet)-4(?:-\d{8})?(?:-v\d+)?(?=[:.]|$)"#,
    #"claude-opus-4-(?:1|5|6)(?=[-.:]|$)"#,
    #"claude-sonnet-4-(?:5|6)(?=[-.:]|$)"#,
    #"claude-haiku-4-5(?=[-.:]|$)"#
]

func bedrockSanitizeToolName(_ name: String) -> String {
    let sanitized = name.replacingOccurrences(
        of: #"[^A-Za-z0-9_-]"#,
        with: "",
        options: .regularExpression
    )
    return sanitized.isEmpty ? "_" : sanitized
}

func bedrockSanitizeDocumentName(_ filename: String) -> String {
    let collapsedWhitespace = stripFileExtension(filename).replacingOccurrences(
        of: #"\s+"#,
        with: " ",
        options: .regularExpression
    )
    let allowedCharactersOnly = collapsedWhitespace.replacingOccurrences(
        of: #"[^a-zA-Z0-9 ()\[\]-]"#,
        with: "",
        options: .regularExpression
    )
    let trimmed = allowedCharactersOnly.trimmingCharacters(in: .whitespacesAndNewlines)
    return String(trimmed.prefix(200)).trimmingCharacters(in: .whitespacesAndNewlines)
}

func bedrockDocumentName(_ filename: String?, documentCounter: inout Int) -> String {
    if let filename {
        let sanitized = bedrockSanitizeDocumentName(filename)
        if !sanitized.isEmpty {
            return sanitized
        }
    }
    documentCounter += 1
    return "document-\(documentCounter)"
}

func bedrockUsesAnthropicProviderTools(
    tools: [String: JSONValue],
    modelID: String,
    modelFamily: AmazonBedrockChatModelFamily? = nil,
    reasoningConfig: JSONValue? = nil
) -> Bool {
    bedrockIsAnthropicModel(
        modelID: modelID,
        modelFamily: modelFamily,
        reasoningConfig: reasoningConfig
    ) && tools.values.contains { $0["type"]?.stringValue == "provider" }
}

func bedrockForcedToolName(from toolChoice: JSONValue?) -> String? {
    guard toolChoice?["type"]?.stringValue == "tool" else { return nil }
    return toolChoice?["toolName"]?.stringValue ?? toolChoice?["name"]?.stringValue
}

func bedrockToolChoice(from value: JSONValue?) -> JSONValue? {
    guard let value else { return nil }
    if let string = value.stringValue {
        switch string {
        case "auto":
            return .object(["auto": .object([:])])
        case "required":
            return .object(["any": .object([:])])
        case "none":
            return .null
        default:
            return nil
        }
    }
    switch value["type"]?.stringValue {
    case "auto":
        return .object(["auto": .object([:])])
    case "required":
        return .object(["any": .object([:])])
    case "none":
        return .null
    case "tool":
        guard let toolName = value["toolName"]?.stringValue ?? value["name"]?.stringValue else { return nil }
        return .object(["tool": .object(["name": .string(toolName)])])
    default:
        return nil
    }
}

func bedrockAnthropicProviderToolInputSchema(_ tool: JSONValue) -> JSONValue? {
    if let inputSchema = tool["inputSchema"] {
        return inputSchema
    }
    if let parameters = tool["parameters"] {
        return parameters
    }
    return .object(["type": .string("object"), "properties": .object([:])])
}

func bedrockApplyReasoningConfig(
    _ value: JSONValue?,
    modelID: String,
    modelFamily: AmazonBedrockChatModelFamily? = nil,
    inferenceConfig: inout [String: JSONValue],
    providerOptions: inout [String: JSONValue],
    warnings: inout [AIWarning]
) {
    guard let value else { return }
    guard let reasoningConfig = value.objectValue else {
        warnings.append(AIWarning(type: "unsupported", feature: "reasoningConfig", message: "Bedrock reasoningConfig must be an object."))
        return
    }

    let type = reasoningConfig["type"]?.stringValue
    let budgetTokens = reasoningConfig["budgetTokens"]?.intValue
    let maxReasoningEffort = reasoningConfig["maxReasoningEffort"]?.stringValue
    let display = reasoningConfig["display"]?.stringValue
    let isAnthropicModel = bedrockIsAnthropicModel(
        modelID: modelID,
        modelFamily: modelFamily,
        reasoningConfig: value
    )
    let openAIModelID = bedrockOpenAIModelID(modelID)
    let isOpenAIModel = openAIModelID != nil
    let isOpenAIGptOssModel = openAIModelID?.hasPrefix("openai.gpt-oss-") == true
    let isAnthropicThinkingEnabled = isAnthropicModel && (type == "enabled" || type == "adaptive")

    if isAnthropicModel && type == "disabled" {
        bedrockMergeAdditionalModelRequestFields(["thinking": .object(["type": .string("disabled")])], into: &providerOptions)
    } else if isAnthropicThinkingEnabled {
        if let budgetTokens, type == "enabled" {
            let existingMaxTokens = inferenceConfig["maxTokens"]?.intValue
            inferenceConfig["maxTokens"] = .number(Double((existingMaxTokens ?? 4096) + budgetTokens))
            bedrockMergeAdditionalModelRequestFields([
                "thinking": .object([
                    "type": .string("enabled"),
                    "budget_tokens": .number(Double(budgetTokens))
                ])
            ], into: &providerOptions)
        } else if type == "adaptive" {
            var thinking: [String: JSONValue] = ["type": .string("adaptive")]
            if let display {
                thinking["display"] = .string(display)
            }
            bedrockMergeAdditionalModelRequestFields(["thinking": .object(thinking)], into: &providerOptions)
        }
        if inferenceConfig.removeValue(forKey: "temperature") != nil {
            warnings.append(AIWarning(type: "unsupported", feature: "temperature", message: "temperature is not supported when thinking is enabled"))
        }
        if inferenceConfig.removeValue(forKey: "topP") != nil {
            warnings.append(AIWarning(type: "unsupported", feature: "topP", message: "topP is not supported when thinking is enabled"))
        }
        if inferenceConfig.removeValue(forKey: "topK") != nil {
            warnings.append(AIWarning(type: "unsupported", feature: "topK", message: "topK is not supported when thinking is enabled"))
        }
    } else if !isAnthropicModel {
        if budgetTokens != nil {
            warnings.append(AIWarning(type: "unsupported", feature: "budgetTokens", message: "budgetTokens applies only to Anthropic models on Bedrock and will be ignored for this model."))
        }
        if type == "adaptive" {
            warnings.append(AIWarning(type: "unsupported", feature: "adaptive thinking", message: "adaptive thinking type applies only to Anthropic models on Bedrock."))
        }
    }

    guard let maxReasoningEffort else { return }
    if isAnthropicModel {
        let existing = providerOptions["additionalModelRequestFields"]?["output_config"]?.objectValue ?? [:]
        bedrockMergeAdditionalModelRequestFields([
            "output_config": .object(existing.merging(["effort": .string(maxReasoningEffort)]) { _, new in new })
        ], into: &providerOptions)
    } else if isOpenAIModel {
        if isOpenAIGptOssModel {
            bedrockMergeAdditionalModelRequestFields(["reasoning_effort": .string(maxReasoningEffort)], into: &providerOptions)
        } else {
            bedrockMergeAdditionalModelRequestFields([
                "reasoning": .object(["effort": .string(maxReasoningEffort)])
            ], into: &providerOptions)
        }
    } else {
        var nested: [String: JSONValue] = [:]
        if let type, type != "adaptive" {
            nested["type"] = .string(type)
        }
        if let budgetTokens {
            nested["budgetTokens"] = .number(Double(budgetTokens))
        }
        nested["maxReasoningEffort"] = .string(maxReasoningEffort)
        bedrockMergeAdditionalModelRequestFields(["reasoningConfig": .object(nested)], into: &providerOptions)
    }
}

func bedrockApplyTopLevelReasoning(
    _ reasoning: String?,
    modelID: String,
    modelFamily: AmazonBedrockChatModelFamily? = nil,
    maxOutputTokens: Int?,
    providerOptions: inout [String: JSONValue],
    warnings: inout [AIWarning]
) {
    guard isCustomReasoning(reasoning), let reasoning else { return }
    let explicit = providerOptions["reasoningConfig"]?.objectValue
    if explicit?["type"]?.stringValue == "disabled" {
        var disabled = explicit ?? [:]
        disabled.removeValue(forKey: "budgetTokens")
        disabled.removeValue(forKey: "maxReasoningEffort")
        providerOptions["reasoningConfig"] = .object(disabled)
        return
    }
    let isAnthropicModel = bedrockIsAnthropicModel(
        modelID: modelID,
        modelFamily: modelFamily,
        reasoningConfig: providerOptions["reasoningConfig"]
    )
    let capabilities = anthropicModelCapabilities(modelID)
    let isOpenAIModel = bedrockOpenAIModelID(modelID) != nil
    let isNovaReasoningModel = modelID.contains("amazon.nova-2-lite-v1:0")
    let effortMap = [
        "minimal": "low", "low": "low", "medium": "medium",
        "high": "high", "xhigh": "max", "max": "max"
    ]
    var reasoningConfig = explicit ?? [:]

    if isAnthropicModel {
        if reasoning == "none" {
            reasoningConfig = ["type": "disabled"]
        } else if capabilities.supportsAdaptiveThinking {
            if reasoningConfig["type"] == nil { reasoningConfig["type"] = "adaptive" }
            if reasoningConfig["maxReasoningEffort"] == nil,
               let effort = mapReasoningToProviderEffort(reasoning: reasoning, effortMap: effortMap, warnings: &warnings) {
                reasoningConfig["maxReasoningEffort"] = .string(effort)
            }
        } else {
            if reasoningConfig["type"] == nil { reasoningConfig["type"] = "enabled" }
            if reasoningConfig["budgetTokens"] == nil {
                let maximum = capabilities.isKnownModel ? capabilities.maxOutputTokens : (maxOutputTokens ?? 4_096)
                if let budget = mapReasoningToProviderBudget(
                    reasoning: reasoning,
                    maxOutputTokens: maximum,
                    maxReasoningBudget: maximum,
                    warnings: &warnings
                ) {
                    reasoningConfig["budgetTokens"] = .number(Double(budget))
                }
            }
        }
    } else if reasoning != "none", isOpenAIModel || isNovaReasoningModel || explicit != nil {
        if isNovaReasoningModel, reasoningConfig["type"] == nil { reasoningConfig["type"] = "enabled" }
        if reasoningConfig["maxReasoningEffort"] == nil,
           let effort = mapReasoningToProviderEffort(
               reasoning: reasoning,
               effortMap: isNovaReasoningModel
                   ? ["minimal": "low", "low": "low", "medium": "medium", "high": "high", "xhigh": "high", "max": "high"]
                   : effortMap,
               warnings: &warnings
           ) {
            reasoningConfig["maxReasoningEffort"] = .string(effort)
        }
    } else if reasoning != "none" {
        warnings.append(AIWarning(
            type: "unsupported",
            feature: "reasoning",
            message: "Portable reasoning is not supported for this model and will be ignored. If the model supports a provider-specific reasoning configuration, use providerOptions.amazonBedrock.reasoningConfig."
        ))
    }
    if !reasoningConfig.isEmpty { providerOptions["reasoningConfig"] = .object(reasoningConfig) }
}

func bedrockIsAnthropicModel(
    modelID: String,
    modelFamily: AmazonBedrockChatModelFamily? = nil,
    reasoningConfig: JSONValue?
) -> Bool {
    modelFamily == .anthropic
        || modelID.contains("anthropic")
        || (modelID.contains(":application-inference-profile/")
            && reasoningConfig?["budgetTokens"]?.intValue != nil)
}

func bedrockOpenAIModelID(_ modelID: String) -> String? {
    if modelID.hasPrefix("openai.") { return modelID }
    guard let dot = modelID.firstIndex(of: ".") else { return nil }
    let suffix = String(modelID[modelID.index(after: dot)...])
    return suffix.hasPrefix("openai.") ? suffix : nil
}

func bedrockMergeAdditionalModelRequestFields(_ fields: [String: JSONValue], into providerOptions: inout [String: JSONValue]) {
    var existing = providerOptions["additionalModelRequestFields"]?.objectValue ?? [:]
    existing.merge(fields) { old, new in
        if var oldObject = old.objectValue,
           let newObject = new.objectValue {
            oldObject.merge(newObject) { _, nestedNew in nestedNew }
            return .object(oldObject)
        }
        return new
    }
    providerOptions["additionalModelRequestFields"] = .object(existing)
}

func bedrockDeduplicatedWarnings(_ warnings: [AIWarning]) -> [AIWarning] {
    var seen: Set<String> = []
    var output: [AIWarning] = []
    for warning in warnings {
        let key = "\(warning.type)|\(warning.feature ?? "")|\(warning.setting ?? "")|\(warning.message ?? "")"
        if seen.insert(key).inserted {
            output.append(warning)
        }
    }
    return output
}

func bedrockToolCalls(from value: JSONValue?, modelID: String = "") -> [AIToolCall] {
    value?.arrayValue?.enumerated().compactMap { index, part in
        bedrockToolCall(from: part, index: index, modelID: modelID)
    } ?? []
}

func bedrockToolCall(from part: JSONValue, index: Int, modelID: String = "") -> AIToolCall? {
    guard let toolUse = part["toolUse"] else { return nil }
    let name = toolUse["name"]?.stringValue ?? "tool-\(index)"
    let rawID = resolvedToolCallID(
        toolUse["toolUseId"]?.stringValue,
        whenMissing: "tool-call-\(index)"
    )
    return AIToolCall(
        id: bedrockNormalizeToolCallID(rawID, modelID: modelID),
        name: name,
        arguments: bedrockToolArguments(toolUse["input"]),
        rawValue: part
    )
}

func bedrockToolArguments(_ value: JSONValue?) -> String {
    guard let value else { return "{}" }
    let input: JSONValue
    if case .object = value {
        input = value
    } else {
        input = .object(["rawInvalidInput": value])
    }
    guard let data = try? encodeJSONBody(input),
          let text = String(data: data, encoding: .utf8) else {
        return "{}"
    }
    return text
}

func bedrockJSONString(_ value: JSONValue) -> String? {
    guard let data = try? encodeJSONBody(value) else { return nil }
    return String(data: data, encoding: .utf8)
}

func bedrockToolResultContent(_ result: AIToolResult, documentCounter: inout Int) throws -> [JSONValue] {
    let output = result.modelOutput ?? result.result
    if let text = output.stringValue {
        return [.object(["text": .string(text)])]
    }
    guard let object = output.objectValue, let type = object["type"]?.stringValue else {
        return [.object(["json": output])]
    }
    switch type {
    case "content":
        return try (object["value"]?.arrayValue ?? []).map { item in
            try bedrockToolResultContentPart(item, documentCounter: &documentCounter)
        }
    case "text", "error-text":
        return [.object(["text": object["value"] ?? .string("")])]
    case "execution-denied":
        return [.object(["text": object["reason"] ?? .string("Tool call execution denied.")])]
    case "json", "error-json":
        return [.object(["text": .string(bedrockJSONString(object["value"] ?? .object([:])) ?? "")])]
    default:
        return [.object(["text": .string(bedrockJSONString(object["value"] ?? output) ?? "")])]
    }
}

func bedrockToolResultContentPart(_ item: JSONValue, documentCounter: inout Int) throws -> JSONValue {
    switch item["type"]?.stringValue {
    case "text":
        return .object(["text": item["text"] ?? item["value"] ?? .string("")])
    case "image-data", "file-data", "file":
        let mediaType = item["mediaType"]?.stringValue ?? item["mimeType"]?.stringValue ?? "application/octet-stream"
        if item["type"]?.stringValue == "file",
           item["data"]?["type"]?.stringValue == "url" {
            guard let url = item["data"]?["url"]?.stringValue,
                  bedrockIsS3URL(url) else {
                throw AIError.invalidArgument(
                    argument: "toolResult.content.data",
                    message: "Amazon Bedrock tool result URL content supports only s3:// image or video URLs with a supported MIME type."
                )
            }
            if let imageFormat = bedrockImageFormat(for: mediaType) {
                return .object([
                    "image": .object([
                        "format": .string(imageFormat),
                        "source": .object([
                            "s3Location": .object(["uri": .string(url)])
                        ])
                    ])
                ])
            }
            if let videoFormat = bedrockVideoFormat(for: mediaType) {
                return .object([
                    "video": .object([
                        "format": .string(videoFormat),
                        "source": .object([
                            "s3Location": .object(["uri": .string(url)])
                        ])
                    ])
                ])
            }
            throw AIError.invalidArgument(
                argument: "toolResult.content.mediaType",
                message: "Amazon Bedrock tool result URL content supports image MIME types \(bedrockSupportedImageMimeTypes.joined(separator: ", ")) or video MIME types \(bedrockSupportedVideoMimeTypes.joined(separator: ", ")); got \(mediaType)."
            )
        }
        let data = item["data"]?["type"]?.stringValue == "data"
            ? item["data"]?["data"]?.stringValue ?? ""
            : item["data"]?.stringValue ?? ""
        let resolvedMediaType: String
        if !mediaType.contains("/"), let decoded = Data(base64Encoded: data) {
            resolvedMediaType = try resolveFullMediaType(mediaType: mediaType, data: decoded)
        } else {
            resolvedMediaType = mediaType
        }
        if let imageFormat = bedrockImageFormat(for: resolvedMediaType) {
            return .object([
                "image": .object([
                    "format": .string(imageFormat),
                    "source": .object(["bytes": .string(data)])
                ])
            ])
        }
        if let videoFormat = bedrockVideoFormat(for: resolvedMediaType) {
            return .object([
                "video": .object([
                    "format": .string(videoFormat),
                    "source": .object(["bytes": .string(data)])
                ])
            ])
        }
        guard let documentFormat = bedrockDocumentFormat(for: resolvedMediaType) else {
            throw AIError.invalidArgument(
                argument: "toolResult.content.mediaType",
                message: "Amazon Bedrock tool result content supports image MIME types \(bedrockSupportedImageMimeTypes.joined(separator: ", ")), video MIME types \(bedrockSupportedVideoMimeTypes.joined(separator: ", ")), or document MIME types \(bedrockSupportedDocumentMimeTypes.joined(separator: ", ")); got \(mediaType)."
            )
        }
        var document: [String: JSONValue] = [
            "format": .string(documentFormat),
            "name": .string(bedrockDocumentName(
                item["filename"]?.stringValue,
                documentCounter: &documentCounter
            )),
            "source": .object(["bytes": .string(data)])
        ]
        if bedrockDocumentCitationsEnabled(item["providerMetadata"]?.objectValue ?? item.objectValue ?? [:]) {
            document["citations"] = .object(["enabled": .bool(true)])
        }
        return .object(["document": .object(document)])
    default:
        throw AIError.invalidArgument(
            argument: "toolResult.content.type",
            message: "Amazon Bedrock does not support tool result content part type \(item["type"]?.stringValue ?? "unknown")."
        )
    }
}

func bedrockReasoningText(from value: JSONValue?) -> String {
    value?.arrayValue?.compactMap { part in
        part["reasoningContent"]?["reasoningText"]?["text"]?.stringValue
    }.joined() ?? ""
}

func bedrockProviderMetadata(from raw: JSONValue, isJsonResponseFromTool: Bool = false) -> [String: JSONValue] {
    var payload: [String: JSONValue] = [:]
    if let trace = raw["trace"] {
        payload["trace"] = trace
    }
    if let performanceConfig = raw["performanceConfig"] {
        payload["performanceConfig"] = performanceConfig
    }
    if let serviceTier = raw["serviceTier"] {
        payload["serviceTier"] = serviceTier
    }
    if let stopSequence = raw["additionalModelResponseFields"]?["delta"]?["stop_sequence"] {
        payload["stopSequence"] = stopSequence
    }
    if let usage = raw["usage"] {
        payload["usage"] = usage
    }
    if isJsonResponseFromTool {
        payload["isJsonResponseFromTool"] = .bool(true)
    }
    guard !payload.isEmpty else { return [:] }
    return [
        "amazonBedrock": .object(payload),
        "bedrock": .object(payload)
    ]
}

func bedrockProviderMetadata(fromStreamMetadata raw: JSONValue?) -> [String: JSONValue] {
    guard let raw else { return [:] }
    var payload: [String: JSONValue] = [:]
    if let trace = raw["trace"] {
        payload["trace"] = trace
    }
    if let performanceConfig = raw["performanceConfig"] {
        payload["performanceConfig"] = performanceConfig
    }
    if let serviceTier = raw["serviceTier"] {
        payload["serviceTier"] = serviceTier
    }
    if let usage = raw["usage"] {
        payload["usage"] = usage
    }
    guard !payload.isEmpty else { return [:] }
    return [
        "amazonBedrock": .object(payload),
        "bedrock": .object(payload)
    ]
}

func bedrockFinishReason(_ reason: String?, isJsonResponseFromTool: Bool = false) -> String? {
    switch reason {
    case "stop_sequence", "end_turn":
        return "stop"
    case "max_tokens":
        return "length"
    case "content_filtered", "guardrail_intervened":
        return "content-filter"
    case "tool_use":
        return isJsonResponseFromTool ? "stop" : "tool-calls"
    case nil:
        return nil
    default:
        return "other"
    }
}

func bedrockUsage(from raw: JSONValue?) -> TokenUsage? {
    guard let raw else { return nil }
    return TokenUsage(
        inputTokens: raw["inputTokens"]?.intValue,
        outputTokens: raw["outputTokens"]?.intValue,
        totalTokens: raw["totalTokens"]?.intValue
    )
}