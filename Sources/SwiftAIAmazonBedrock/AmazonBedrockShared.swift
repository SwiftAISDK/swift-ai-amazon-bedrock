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
    var betas: [String] = []
}

/// Betas that upstream `prepareTools` from `@ai-sdk/anthropic` adds for
/// Anthropic provider tools; Bedrock sends them as `anthropic_beta`.
let bedrockAnthropicProviderToolBetas: [String: String] = [
    "anthropic.code_execution_20250522": "code-execution-2025-05-22",
    "anthropic.code_execution_20250825": "code-execution-2025-08-25",
    "anthropic.computer_20250124": "computer-use-2025-01-24",
    "anthropic.computer_20251124": "computer-use-2025-11-24",
    "anthropic.computer_20241022": "computer-use-2024-10-22",
    "anthropic.text_editor_20250124": "computer-use-2025-01-24",
    "anthropic.text_editor_20241022": "computer-use-2024-10-22",
    "anthropic.text_editor_20250429": "computer-use-2025-01-24",
    "anthropic.bash_20250124": "computer-use-2025-01-24",
    "anthropic.bash_20241022": "computer-use-2024-10-22",
    "anthropic.memory_20250818": "context-management-2025-06-27",
    "anthropic.web_fetch_20250910": "web-fetch-2025-09-10",
    "anthropic.web_fetch_20260209": "code-execution-web-tools-2026-02-09",
    "anthropic.web_search_20260209": "code-execution-web-tools-2026-02-09",
    "anthropic.advisor_20260301": "advisor-tool-2026-03-01"
]

/// Input schemas of upstream Anthropic tool factories, as produced by
/// `asSchema(tool.inputSchema).jsonSchema` for the Bedrock `toolSpec`.
private let bedrockBashToolInputSchema: JSONValue = .object([
    "$schema": .string("http://json-schema.org/draft-07/schema#"),
    "type": .string("object"),
    "properties": .object([
        "command": .object(["type": .string("string")]),
        "restart": .object(["type": .string("boolean")])
    ]),
    "required": .array([.string("command")]),
    "additionalProperties": .bool(false)
])

let bedrockAnthropicProviderToolInputSchemas: [String: JSONValue] = [
    "anthropic.bash_20241022": bedrockBashToolInputSchema,
    "anthropic.bash_20250124": bedrockBashToolInputSchema
]

func bedrockRequestProviderOptions(providerOptions: [String: JSONValue], extraBody: [String: JSONValue]) throws -> [String: JSONValue] {
    var output: [String: JSONValue] = [:]
    if let bedrock = extraBody["bedrock"]?.objectValue {
        output.merge(bedrock) { _, new in new }
    }
    if let amazonBedrock = extraBody["amazonBedrock"]?.objectValue {
        output.merge(amazonBedrock) { _, new in new }
    }
    // Upstream resolves the first present namespace (`amazonBedrock`, then the
    // legacy `bedrock`) instead of merging both.
    if let amazonBedrock = providerOptions["amazonBedrock"] {
        guard let object = amazonBedrock.objectValue else {
            throw AIError.invalidArgument(argument: "providerOptions.amazonBedrock", message: "Amazon Bedrock provider options must be an object.")
        }
        output.merge(object) { _, new in new }
    } else if let bedrock = providerOptions["bedrock"] {
        guard let object = bedrock.objectValue else {
            throw AIError.invalidArgument(argument: "providerOptions.bedrock", message: "Bedrock provider options must be an object.")
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
    return type == "enabled" || type == "adaptive" || type == "between_tools"
}

func bedrockCachePoint(from providerMetadata: [String: JSONValue]) -> JSONValue? {
    let cachePoint = bedrockNonNull(providerMetadata["amazonBedrock"]?["cachePoint"])
        ?? bedrockNonNull(providerMetadata["bedrock"]?["cachePoint"])
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
    guard let metadata = providerMetadata["amazonBedrock"]?.objectValue
        ?? providerMetadata["bedrock"]?.objectValue else { return nil }
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

let bedrockKnownAnthropicProviderToolIDs: Set<String> = [
    "anthropic.advisor_20260301",
    "anthropic.bash_20241022",
    "anthropic.bash_20250124",
    "anthropic.code_execution_20250522",
    "anthropic.code_execution_20250825",
    "anthropic.code_execution_20260120",
    "anthropic.computer_20241022",
    "anthropic.computer_20250124",
    "anthropic.computer_20251124",
    "anthropic.computer_toolset_20260801",
    "anthropic.memory_20250818",
    "anthropic.text_editor_20241022",
    "anthropic.text_editor_20250124",
    "anthropic.text_editor_20250429",
    "anthropic.text_editor_20250728",
    "anthropic.tool_search_bm25_20251119",
    "anthropic.tool_search_regex_20251119",
    "anthropic.web_fetch_20250910",
    "anthropic.web_fetch_20260209",
    "anthropic.web_fetch_20260318",
    "anthropic.web_search_20250305",
    "anthropic.web_search_20260209",
    "anthropic.web_search_20260318"
]

/// Port of upstream `prepareTools` in `amazon-bedrock-prepare-tools.ts`.
func bedrockPrepareTools(
    from tools: [String: JSONValue],
    toolChoice: JSONValue?,
    modelID: String,
    modelFamily: AmazonBedrockChatModelFamily? = nil,
    reasoningConfig: JSONValue? = nil,
    disableParallelToolUse: Bool? = nil
) -> BedrockPreparedTools {
    guard !tools.isEmpty else {
        return BedrockPreparedTools(toolConfig: nil, warnings: [])
    }

    var warnings: [AIWarning] = []
    var betas: [String] = []
    // Request tools are a dictionary in Swift; sort for a deterministic order.
    let supportedTools = tools.sorted { $0.key < $1.key }.filter { name, schema in
        let id = schema["id"]?.stringValue ?? ""
        let unsupportedWebTools: Set<String> = [
            "anthropic.web_search_20250305",
            "anthropic.web_search_20260318",
            "anthropic.web_fetch_20260318"
        ]
        guard schema["type"]?.stringValue == "provider", unsupportedWebTools.contains(id) else { return true }
        let feature = String(id.dropFirst("anthropic.".count))
        warnings.append(AIWarning(
            type: "unsupported",
            feature: "\(feature) tool",
            message: "The \(feature) tool is not supported on Amazon Bedrock."
        ))
        return false
    }
    guard !supportedTools.isEmpty else {
        return BedrockPreparedTools(toolConfig: nil, warnings: warnings)
    }

    let isAnthropicModel = bedrockIsAnthropicModel(
        modelID: modelID,
        modelFamily: modelFamily,
        reasoningConfig: reasoningConfig
    )
    let providerTools = supportedTools.filter { $0.value["type"]?.stringValue == "provider" }
    let functionTools = supportedTools.filter { $0.value["type"]?.stringValue != "provider" }
    let rejectsForcedToolUse = anthropicModelCapabilities(modelID).rejectsForcedToolUse
    let choiceType = toolChoice?.stringValue ?? toolChoice?["type"]?.stringValue
    let forcedToolName = bedrockForcedToolName(from: toolChoice)
    let rejectsForcedToolChoice = isAnthropicModel && rejectsForcedToolUse
        && (choiceType == "required" || choiceType == "tool")
    let providerToolsForRequest = rejectsForcedToolChoice && choiceType == "tool"
        ? providerTools.filter { $0.key == forcedToolName }
        : providerTools
    // A named tool choice keeps every function tool for Anthropic models, so
    // parallel tool use is disabled by default to keep the response to it.
    let disableParallelToolUseForRequest = disableParallelToolUse
        ?? (isAnthropicModel && choiceType == "tool" && !rejectsForcedToolChoice ? true : nil)

    var additionalModelRequestFields: [String: JSONValue]?
    var bedrockTools: [JSONValue] = []
    let usingAnthropicTools = isAnthropicModel && !providerToolsForRequest.isEmpty

    if usingAnthropicTools {
        for (_, tool) in providerTools {
            let id = tool["id"]?.stringValue ?? ""
            if let beta = bedrockAnthropicProviderToolBetas[id], !betas.contains(beta) {
                betas.append(beta)
            }
            if !bedrockKnownAnthropicProviderToolIDs.contains(id) {
                warnings.append(AIWarning(type: "unsupported", feature: "provider-defined tool \(id)"))
            }
        }
        if let anthropicChoice = bedrockAnthropicToolChoice(
            choiceType: choiceType,
            toolName: forcedToolName,
            disableParallelToolUse: disableParallelToolUseForRequest,
            rejectsForcedToolUse: rejectsForcedToolUse,
            warnings: &warnings
        ) {
            additionalModelRequestFields = ["tool_choice": anthropicChoice]
        }
        for (name, tool) in providerToolsForRequest {
            let id = tool["id"]?.stringValue ?? ""
            if bedrockKnownAnthropicProviderToolIDs.contains(id),
               let schema = bedrockAnthropicProviderToolInputSchema(tool) {
                bedrockTools.append(.object([
                    "toolSpec": .object([
                        "name": .string(name),
                        "inputSchema": .object(["json": schema])
                    ])
                ]))
            } else {
                warnings.append(AIWarning(type: "unsupported", feature: "tool \(id)"))
            }
        }
    } else {
        for (_, tool) in providerToolsForRequest {
            warnings.append(AIWarning(type: "unsupported", feature: "tool \(tool["id"]?.stringValue ?? "")"))
        }
    }

    var preparedChoiceType = choiceType
    if !usingAnthropicTools && rejectsForcedToolChoice {
        let detail = choiceType == "tool"
            ? "toolChoice 'tool' is not supported by this model because it rejects forced tool use. Only the '\(forcedToolName ?? "")' tool is sent with 'auto' tool choice. Instruct the model to use the tool in the prompt and verify that a tool call was made."
            : "toolChoice 'required' is not supported by this model because it rejects forced tool use. Using 'auto' instead. Instruct the model to use a tool in the prompt and verify that a tool call was made."
        warnings.append(AIWarning(type: "unsupported", feature: "toolChoice", message: detail))
        preparedChoiceType = "auto"
    }

    let filteredFunctionTools = choiceType == "tool" && (!isAnthropicModel || rejectsForcedToolChoice)
        ? functionTools.filter { $0.key == forcedToolName }
        : functionTools
    let supportsStrictOnTools = bedrockSupportsStrictToolSpec(modelID: modelID)
    for (name, schema) in filteredFunctionTools {
        let strict = bedrockNonNull(schema["strict"])
        let supportsStrictForTool = supportsStrictOnTools
            && (strict?.boolValue != true || bedrockStrictToolSchemaCompatible(bedrockToolInputSchema(schema)))
        if !supportsStrictOnTools, strict?.boolValue == true {
            warnings.append(AIWarning(
                type: "unsupported",
                feature: "strict",
                message: "Tool '\(name)' has strict: true, but strict mode is not supported by this model on Amazon Bedrock. The strict property will be ignored."
            ))
        } else if strict?.boolValue == true, !supportsStrictForTool {
            warnings.append(AIWarning(
                type: "unsupported",
                feature: "strict",
                message: "Tool '\(name)' has strict: true, but Amazon Bedrock requires every object in a strict tool schema to set additionalProperties: false. The strict property will be ignored."
            ))
        }
        var toolSpec: [String: JSONValue] = [
            "name": .string(name),
            "inputSchema": .object(["json": bedrockToolInputSchema(schema)])
        ]
        // Upstream omits only descriptions that trim to an empty string.
        if let description = schema["description"]?.stringValue,
           !description.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            toolSpec["description"] = .string(description)
        }
        if let strict, supportsStrictForTool {
            toolSpec["strict"] = strict
        }
        bedrockTools.append(.object(["toolSpec": .object(toolSpec)]))
    }

    if isAnthropicModel,
       !usingAnthropicTools,
       disableParallelToolUseForRequest == true,
       !bedrockTools.isEmpty,
       preparedChoiceType != "none" {
        let anthropicChoice: [String: JSONValue]
        switch preparedChoiceType {
        case "required":
            anthropicChoice = ["type": "any", "disable_parallel_tool_use": true]
        case "tool":
            anthropicChoice = ["type": "tool", "name": .string(forcedToolName ?? ""), "disable_parallel_tool_use": true]
        default:
            anthropicChoice = ["type": "auto", "disable_parallel_tool_use": true]
        }
        additionalModelRequestFields = ["tool_choice": .object(anthropicChoice)]
    }

    var bedrockToolChoice: JSONValue?
    if !usingAnthropicTools,
       additionalModelRequestFields?["tool_choice"] == nil,
       !bedrockTools.isEmpty,
       let preparedChoiceType {
        switch preparedChoiceType {
        case "auto":
            bedrockToolChoice = .object(["auto": .object([:])])
        case "required":
            bedrockToolChoice = .object(["any": .object([:])])
        case "none":
            bedrockTools.removeAll()
        case "tool":
            bedrockToolChoice = .object(["tool": .object(["name": .string(forcedToolName ?? "")])])
        default:
            break
        }
    }

    var toolConfig: JSONValue?
    if !bedrockTools.isEmpty {
        var config: [String: JSONValue] = ["tools": .array(bedrockTools)]
        if let bedrockToolChoice { config["toolChoice"] = bedrockToolChoice }
        toolConfig = .object(config)
    }
    return BedrockPreparedTools(
        toolConfig: toolConfig,
        additionalModelRequestFields: additionalModelRequestFields,
        warnings: warnings,
        betas: usingAnthropicTools ? betas : []
    )
}

/// The Swift request carries description and strict next to the JSON schema;
/// remove them before sending the schema itself.
func bedrockToolInputSchema(_ tool: JSONValue) -> JSONValue {
    if let inputSchema = bedrockNonNull(tool["inputSchema"]) { return inputSchema }
    return aiFunctionToolInputSchema(tool)
}

/// Port of the tool-choice branch of upstream `@ai-sdk/anthropic` `prepareTools`.
private func bedrockAnthropicToolChoice(
    choiceType: String?,
    toolName: String?,
    disableParallelToolUse: Bool?,
    rejectsForcedToolUse: Bool,
    warnings: inout [AIWarning]
) -> JSONValue? {
    func choice(_ type: String, name: String? = nil) -> JSONValue {
        var object: [String: JSONValue] = ["type": .string(type)]
        if let name { object["name"] = .string(name) }
        if let disableParallelToolUse { object["disable_parallel_tool_use"] = .bool(disableParallelToolUse) }
        return .object(object)
    }
    guard let choiceType else {
        return disableParallelToolUse == true ? choice("auto") : nil
    }
    switch choiceType {
    case "auto":
        return choice("auto")
    case "required":
        if rejectsForcedToolUse {
            warnings.append(AIWarning(
                type: "unsupported",
                feature: "toolChoice",
                message: "toolChoice 'required' is not supported with this model or thinking configuration. Using 'auto' instead. Instruct the model to use a tool in the prompt and verify that a tool call was made."
            ))
            return choice("auto")
        }
        return choice("any")
    case "none":
        return .object(["type": .string("none")])
    case "tool":
        if rejectsForcedToolUse {
            warnings.append(AIWarning(
                type: "unsupported",
                feature: "toolChoice",
                message: "toolChoice 'tool' is not supported with this model or thinking configuration. Only the '\(toolName ?? "")' tool is sent with 'auto' tool choice. Instruct the model to use the tool in the prompt and verify that a tool call was made."
            ))
            return choice("auto")
        }
        return choice("tool", name: toolName)
    default:
        return nil
    }
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
    bedrockStructuredOutputSupport(modelID).strict
}

func bedrockSupportsNativeStructuredOutput(modelID: String) -> Bool {
    bedrockStructuredOutputSupport(modelID).native
}

private func bedrockStructuredOutputSupport(_ modelID: String) -> (strict: Bool, native: Bool) {
    if !modelID.contains("claude-") || modelID.range(of: #"claude-(?:instant|v?2|3)(?=[-.:]|$)"#, options: .regularExpression) != nil { return (true, true) }
    let pattern = #"claude-[a-z]+-4-\d{1,2}(?=[-.:]|$)|claude-[a-z]+-\d+(?:-\d{1,2})?(?:-\d{8})?(?:-v\d+)?(?=[:.]|$)"#
    guard let range = modelID.range(of: pattern, options: .regularExpression) else { return (false, false) }
    let alias = String(modelID[range]).replacingOccurrences(of: #"-\d{8}(?:-v\d+)?$|-v\d+$"#, with: "", options: .regularExpression)
    let native: Set<String> = ["claude-opus-4", "claude-sonnet-4", "claude-opus-4-1", "claude-opus-4-5", "claude-sonnet-4-5", "claude-opus-4-6", "claude-opus-5", "claude-sonnet-5", "claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"]
    return (native.contains(alias) || ["claude-haiku-4-5", "claude-sonnet-4-6"].contains(alias), native.contains(alias))
}

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

struct BedrockDocumentNames {
    var counter = 0
    var used: Set<String> = []
}

func bedrockDocumentName(_ filename: String?, documentCounter: inout BedrockDocumentNames) -> String {
    let sanitized = filename.map(bedrockSanitizeDocumentName) ?? ""
    let base: String
    if !sanitized.isEmpty { base = sanitized }
    else {
        documentCounter.counter += 1
        base = "document-\(documentCounter.counter)"
    }
    var name = base
    var suffix = 2
    while documentCounter.used.contains(name) {
        let suffixText = " (\(suffix))"
        name = String(base.prefix(200 - suffixText.count)).trimmingCharacters(in: .whitespacesAndNewlines) + suffixText
        suffix += 1
    }
    documentCounter.used.insert(name)
    return name
}

func bedrockForcedToolName(from toolChoice: JSONValue?) -> String? {
    guard toolChoice?["type"]?.stringValue == "tool" else { return nil }
    return toolChoice?["toolName"]?.stringValue ?? toolChoice?["name"]?.stringValue
}

func bedrockAnthropicProviderToolInputSchema(_ tool: JSONValue) -> JSONValue? {
    if let id = tool["id"]?.stringValue, let schema = bedrockAnthropicProviderToolInputSchemas[id] {
        return schema
    }
    if let inputSchema = tool["inputSchema"] {
        return inputSchema
    }
    if let parameters = tool["parameters"] {
        return parameters
    }
    return .object(["type": .string("object"), "properties": .object([:])])
}

/// Applies the resolved reasoning configuration in upstream order: thinking
/// fields, effort fields, the Nova 2 maxTokens rule, native structured output
/// and finally the sampling settings that thinking rejects.
func bedrockApplyReasoningConfig(
    _ value: JSONValue?,
    modelID: String,
    modelFamily: AmazonBedrockChatModelFamily? = nil,
    inferenceConfig: inout [String: JSONValue],
    providerOptions: inout [String: JSONValue],
    warnings: inout [AIWarning],
    nativeStructuredOutputSchema: JSONValue? = nil
) {
    let reasoningConfig = bedrockNonNull(value)?.objectValue
    if bedrockNonNull(value) != nil, reasoningConfig == nil {
        warnings.append(AIWarning(type: "unsupported", feature: "reasoningConfig", message: "Bedrock reasoningConfig must be an object."))
    }

    let type = reasoningConfig?["type"]?.stringValue
    let budgetTokens = reasoningConfig?["budgetTokens"]?.intValue
    let maxReasoningEffort = reasoningConfig?["maxReasoningEffort"]?.stringValue
    let display = reasoningConfig?["display"]?.stringValue
    let isAnthropicModel = bedrockIsAnthropicModel(
        modelID: modelID,
        modelFamily: modelFamily,
        reasoningConfig: value
    )
    let openAIModelID = bedrockOpenAIModelID(modelID)
    let isOpenAIModel = openAIModelID != nil
    let isOpenAIGptOssModel = openAIModelID?.hasPrefix("openai.gpt-oss-") == true
    let isThinkingEnabled = type == "enabled" || type == "adaptive" || type == "between_tools"
    let isAnthropicThinkingEnabled = isAnthropicModel && isThinkingEnabled
    let thinkingBudget = type == "enabled" ? budgetTokens : nil

    func setField(_ key: String, _ fieldValue: JSONValue) {
        var fields = providerOptions["additionalModelRequestFields"]?.objectValue ?? [:]
        fields[key] = fieldValue
        providerOptions["additionalModelRequestFields"] = .object(fields)
    }
    func mergeField(_ key: String, _ fieldValue: [String: JSONValue]) {
        var fields = providerOptions["additionalModelRequestFields"]?.objectValue ?? [:]
        var existing = fields[key]?.objectValue ?? [:]
        existing.merge(fieldValue) { _, new in new }
        fields[key] = .object(existing)
        providerOptions["additionalModelRequestFields"] = .object(fields)
    }

    if isAnthropicModel {
        if let thinkingBudget {
            let existingMaxTokens = inferenceConfig["maxTokens"]?.intValue
            inferenceConfig["maxTokens"] = .number(Double(existingMaxTokens.map { $0 + thinkingBudget } ?? (thinkingBudget + 4096)))
            setField("thinking", .object([
                "type": .string("enabled"),
                "budget_tokens": .number(Double(thinkingBudget))
            ]))
        } else if type == "adaptive" {
            var thinking: [String: JSONValue] = ["type": .string("adaptive")]
            if let display { thinking["display"] = .string(display) }
            setField("thinking", .object(thinking))
        } else if type == "disabled" || type == "between_tools" {
            setField("thinking", .object(["type": .string(type ?? "")]))
        }
    } else {
        if budgetTokens != nil {
            warnings.append(AIWarning(type: "unsupported", feature: "budgetTokens", message: "budgetTokens applies only to Anthropic models on Bedrock and will be ignored for this model."))
        }
        if type == "adaptive" {
            warnings.append(AIWarning(type: "unsupported", feature: "adaptive thinking", message: "adaptive thinking type applies only to Anthropic models on Bedrock."))
        }
    }

    if let maxReasoningEffort {
        if isAnthropicModel {
            mergeField("output_config", ["effort": .string(maxReasoningEffort)])
        } else if isOpenAIModel {
            if isOpenAIGptOssModel {
                setField("reasoning_effort", .string(maxReasoningEffort))
            } else {
                mergeField("reasoning", ["effort": .string(maxReasoningEffort)])
            }
        } else {
            var nested: [String: JSONValue] = [:]
            if let type, type != "adaptive" {
                nested["type"] = .string(type)
            }
            if let thinkingBudget {
                nested["budgetTokens"] = .number(Double(thinkingBudget))
            }
            nested["maxReasoningEffort"] = .string(maxReasoningEffort)
            setField("reasoningConfig", .object(nested))
        }
    }

    if modelID.contains("amazon.nova-2-lite-v1:0"),
       type == "enabled",
       ["high", "xhigh", "max"].contains(maxReasoningEffort ?? ""),
       inferenceConfig.removeValue(forKey: "maxTokens") != nil {
        warnings.append(AIWarning(type: "unsupported", feature: "maxOutputTokens", message: "maxOutputTokens is not supported by \(modelID) when high reasoning is enabled and will be ignored"))
    }

    if let nativeStructuredOutputSchema {
        mergeField("output_config", [
            "format": .object([
                "type": .string("json_schema"),
                "schema": anthropicSanitizeJSONSchema(nativeStructuredOutputSchema)
            ])
        ])
    }

    if isAnthropicThinkingEnabled {
        if inferenceConfig.removeValue(forKey: "temperature") != nil {
            warnings.append(AIWarning(type: "unsupported", feature: "temperature", message: "temperature is not supported when thinking is enabled"))
        }
        if inferenceConfig.removeValue(forKey: "topP") != nil {
            warnings.append(AIWarning(type: "unsupported", feature: "topP", message: "topP is not supported when thinking is enabled"))
        }
        if inferenceConfig.removeValue(forKey: "topK") != nil {
            warnings.append(AIWarning(type: "unsupported", feature: "topK", message: "topK is not supported when thinking is enabled"))
        }
    }
}

func bedrockNormalizeDisabledThinking(
    modelID: String,
    modelFamily: AmazonBedrockChatModelFamily?,
    explicitReasoningEffort: String?,
    explicitlyDisabled: Bool,
    providerOptions: inout [String: JSONValue],
    warnings: inout [AIWarning]
) {
    guard var config = providerOptions["reasoningConfig"]?.objectValue,
          config["type"] == "disabled",
          bedrockIsAnthropicModel(modelID: modelID, modelFamily: modelFamily, reasoningConfig: .object(config)) else { return }
    let capabilities = anthropicModelCapabilities(modelID)
    var type = "disabled"
    config.removeValue(forKey: "budgetTokens")
    config.removeValue(forKey: "display")
    if modelID.range(of: #"claude-(?:instant|v?2|3-(?:5-)?(?:haiku|sonnet|opus))(?=[-.:]|$)"#, options: .regularExpression) != nil {
        providerOptions.removeValue(forKey: "reasoningConfig")
        return
    }
    if capabilities.rejectsThinkingDisabled {
        type = capabilities.supportsBetweenToolsThinking ? "between_tools" : "adaptive"
        config["type"] = .string(type)
        if !explicitlyDisabled, type == "adaptive" {
            if config["maxReasoningEffort"] == nil, let explicitReasoningEffort {
                config["maxReasoningEffort"] = .string(explicitReasoningEffort)
            }
            if config["maxReasoningEffort"] == nil,
               providerOptions["additionalModelRequestFields"]?["output_config"]?["effort"] == nil {
                config["maxReasoningEffort"] = "low"
            }
        }
        warnings.append(AIWarning(
            type: explicitlyDisabled ? "unsupported" : "compatibility",
            feature: explicitlyDisabled ? "providerOptions.amazonBedrock.reasoningConfig" : "reasoning",
            message: "thinking cannot be disabled for \(modelID). Using '\(type)' thinking instead."
        ))
    }
    let effort = config["maxReasoningEffort"]?.stringValue
        ?? providerOptions["additionalModelRequestFields"]?["output_config"]?["effort"]?.stringValue
    if ((type == "disabled" && capabilities.rejectsThinkingDisabledAboveHighEffort) || type == "between_tools"),
       let effort, effort == "xhigh" || effort == "max" {
        warnings.append(AIWarning(
            type: "unsupported",
            feature: config["maxReasoningEffort"] != nil
                ? "providerOptions.amazonBedrock.reasoningConfig.maxReasoningEffort"
                : "providerOptions.amazonBedrock.additionalModelRequestFields.output_config.effort",
            message: "effort '\(effort)' is not supported by \(modelID) with '\(type)' thinking. The effort has been lowered to 'high'."
        ))
        config["maxReasoningEffort"] = "high"
    }
    providerOptions["reasoningConfig"] = .object(config)
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
    var effortMap = ["low": "low", "medium": "medium", "high": "high", "max": "max"]
    if capabilities.supportsXhighEffort { effortMap["xhigh"] = "xhigh" }
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
    } else if reasoning == "none", isNovaReasoningModel {
        // Omit Nova 2 reasoning settings to disable reasoning.
    } else if isOpenAIModel || isNovaReasoningModel || explicit != nil {
        if isNovaReasoningModel, reasoningConfig["type"] == nil { reasoningConfig["type"] = "enabled" }
        if reasoningConfig["maxReasoningEffort"] == nil,
           let effort = mapReasoningToProviderEffort(
               reasoning: reasoning,
               effortMap: isNovaReasoningModel || bedrockOpenAIModelID(modelID)?.hasPrefix("openai.gpt-oss-") == true
                   ? ["low": "low", "medium": "medium", "high": "high"]
                   : isOpenAIModel
                       ? ["none": "none", "minimal": "minimal", "low": "low", "medium": "medium", "high": "high", "xhigh": "xhigh", "max": "max"]
                       : effortMap,
               warnings: &warnings
           ) {
            reasoningConfig["maxReasoningEffort"] = .string(effort)
        }
    } else {
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

func bedrockToolCall(from part: JSONValue, index: Int, modelID: String = "") -> AIToolCall? {
    guard let toolUse = part["toolUse"] else { return nil }
    let name = toolUse["name"]?.stringValue ?? "tool-\(generateId())"
    let rawID = resolvedToolCallID(
        toolUse["toolUseId"]?.stringValue,
        whenMissing: generateId()
    )
    return AIToolCall(
        id: bedrockNormalizeToolCallID(rawID, modelID: modelID),
        name: name,
        arguments: bedrockToolArguments(toolUse["input"]),
        rawValue: part
    )
}

/// Upstream `JSON.stringify(part.toolUse.input ?? {})`: the response input is
/// serialized as returned; only replayed inputs are wrapped in `rawInvalidInput`.
func bedrockToolArguments(_ value: JSONValue?) -> String {
    guard let input = bedrockNonNull(value) else { return "{}" }
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

func bedrockToolResultContent(_ result: AIToolResult, documentCounter: inout BedrockDocumentNames) throws -> [JSONValue] {
    let output = result.modelOutput ?? result.result
    if let text = output.stringValue {
        return [.object(["text": .string(text)])]
    }
    guard let object = output.objectValue, let type = object["type"]?.stringValue else {
        return [.object(["text": .string(bedrockJSONString(output) ?? "")])]
    }
    switch type {
    case "content":
        return try (object["value"]?.arrayValue ?? []).map { item in
            try bedrockToolResultContentPart(item, documentCounter: &documentCounter)
        }
    case "text", "error-text":
        return [.object(["text": object["value"] ?? .string("")])]
    case "execution-denied":
        return [.object(["text": bedrockNonNull(object["reason"]) ?? .string("Tool call execution denied.")])]
    case "json", "error-json":
        return [.object(["text": .string(bedrockJSONString(object["value"] ?? .object([:])) ?? "")])]
    default:
        return [.object(["text": .string(bedrockJSONString(object["value"] ?? output) ?? "")])]
    }
}

func bedrockToolResultContentPart(_ item: JSONValue, documentCounter: inout BedrockDocumentNames) throws -> JSONValue {
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

/// Upstream `doGenerate` provider metadata: present whenever the response has
/// usage (always) or any of trace/performanceConfig/serviceTier; usage only
/// carries the cache write and cache detail fields, and `stopSequence` is
/// always included (null when absent).
func bedrockProviderMetadata(from raw: JSONValue, isJsonResponseFromTool: Bool = false) -> [String: JSONValue] {
    let trace = bedrockNonNull(raw["trace"])
    let usage = bedrockNonNull(raw["usage"])
    let performanceConfig = bedrockNonNull(raw["performanceConfig"])
    let serviceTier = bedrockNonNull(raw["serviceTier"])
    let stopSequence = raw["additionalModelResponseFields"]?["delta"]?["stop_sequence"]?.stringValue
    guard trace != nil || usage != nil || performanceConfig != nil || serviceTier != nil
        || isJsonResponseFromTool || stopSequence != nil else {
        return [:]
    }
    var payload = bedrockMetadataPayload(
        trace: trace,
        performanceConfig: performanceConfig,
        serviceTier: serviceTier,
        usage: usage
    )
    if isJsonResponseFromTool {
        payload["isJsonResponseFromTool"] = .bool(true)
    }
    payload["stopSequence"] = stopSequence.map(JSONValue.string) ?? .null
    return [
        "amazonBedrock": .object(payload),
        "bedrock": .object(payload)
    ]
}

func bedrockMetadataPayload(
    trace: JSONValue?,
    performanceConfig: JSONValue?,
    serviceTier: JSONValue?,
    usage: JSONValue?
) -> [String: JSONValue] {
    var payload: [String: JSONValue] = [:]
    if let trace = bedrockNonNull(trace), trace.objectValue != nil || trace.arrayValue != nil {
        payload["trace"] = trace
    }
    if let performanceConfig = bedrockNonNull(performanceConfig) {
        payload["performanceConfig"] = performanceConfig
    }
    if let serviceTier = bedrockNonNull(serviceTier) {
        payload["serviceTier"] = serviceTier
    }
    var cacheUsage: [String: JSONValue] = [:]
    if let cacheWrite = bedrockNonNull(usage?["cacheWriteInputTokens"]) {
        cacheUsage["cacheWriteInputTokens"] = cacheWrite
    }
    if let cacheDetails = bedrockNonNull(usage?["cacheDetails"]) {
        cacheUsage["cacheDetails"] = cacheDetails
    }
    if !cacheUsage.isEmpty {
        payload["usage"] = .object(cacheUsage)
    }
    return payload
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

/// Upstream `convertAmazonBedrockUsage`: input totals include cache reads and
/// writes, `noCache` is the reported `inputTokens`, and the raw usage is kept.
func bedrockUsage(from raw: JSONValue?) -> TokenUsage? {
    guard let raw = bedrockNonNull(raw), raw.objectValue != nil else { return nil }
    let inputTokens = raw["inputTokens"]?.intValue
    let outputTokens = raw["outputTokens"]?.intValue
    let cacheReadTokens = raw["cacheReadInputTokens"]?.intValue ?? 0
    let cacheWriteTokens = raw["cacheWriteInputTokens"]?.intValue ?? 0
    let inputTotal = inputTokens.map { $0 + cacheReadTokens + cacheWriteTokens }
    return TokenUsage(
        inputTokens: inputTotal,
        outputTokens: outputTokens,
        totalTokens: inputTotal.flatMap { input in outputTokens.map { input + $0 } },
        inputTokensNoCache: inputTokens,
        inputTokensCacheRead: cacheReadTokens,
        inputTokensCacheWrite: cacheWriteTokens,
        outputTextTokens: outputTokens,
        rawValue: raw
    )
}
