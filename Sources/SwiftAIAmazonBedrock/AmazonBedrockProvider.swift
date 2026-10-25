import SwiftAIOpenAICompatible
import SwiftAIOpenAI
import SwiftAIAnthropic
import CryptoKit
import Foundation
import SwiftAIProvider
import SwiftAITransport
import SwiftAIProviderUtilities


private let amazonBedrockUserAgent = "ai-sdk-amazon-bedrock/5.0.124"

public struct AmazonBedrockCredentials: Sendable {
    public var accessKeyID: String
    public var secretAccessKey: String
    public var sessionToken: String?

    public init(accessKeyID: String, secretAccessKey: String, sessionToken: String? = nil) {
        self.accessKeyID = accessKeyID
        self.secretAccessKey = secretAccessKey
        self.sessionToken = sessionToken
    }
}

public struct AmazonBedrockProviderSettings: Sendable {
    public var region: String?
    public var apiKey: String?
    public var accessKeyID: String?
    public var secretAccessKey: String?
    public var sessionToken: String?
    public var credentialProvider: (@Sendable () async throws -> AmazonBedrockCredentials)?
    public var baseURL: String?
    public var headers: [String: String]
    public var transport: any AITransport
    public var date: @Sendable () -> Date

    public init(
        region: String? = nil,
        apiKey: String? = nil,
        accessKeyID: String? = nil,
        secretAccessKey: String? = nil,
        sessionToken: String? = nil,
        credentialProvider: (@Sendable () async throws -> AmazonBedrockCredentials)? = nil,
        baseURL: String? = nil,
        headers: [String: String] = [:],
        transport: any AITransport = URLSessionTransport.shared,
        date: @escaping @Sendable () -> Date = Date.init
    ) {
        self.region = region
        self.apiKey = apiKey
        self.accessKeyID = accessKeyID
        self.secretAccessKey = secretAccessKey
        self.sessionToken = sessionToken
        self.credentialProvider = credentialProvider
        self.baseURL = baseURL
        self.headers = headers
        self.transport = transport
        self.date = date
    }
}

public final class AmazonBedrockProvider: AIProvider, @unchecked Sendable {
    public let providerID = "amazon-bedrock"
    public let supportedCapabilities: Set<ModelCapability> = [.language, .embedding, .image, .reranking]
    private let runtimeConfig: BedrockRuntimeConfig
    private let agentRuntimeConfig: BedrockRuntimeConfig

    public init(settings: AmazonBedrockProviderSettings = AmazonBedrockProviderSettings()) throws {
        let region = settings.region ?? environmentValue(["AWS_REGION", "AWS_DEFAULT_REGION"]) ?? "us-east-1"
        let globalEndpointURL = environmentValue(["AWS_ENDPOINT_URL"])
        if settings.baseURL == nil, globalEndpointURL == nil,
           environmentValue(["AWS_ENDPOINT_URL_BEDROCK_RUNTIME"]) == nil || environmentValue(["AWS_ENDPOINT_URL_BEDROCK_AGENT_RUNTIME"]) == nil {
            try validateHostnamePart(region, argument: "region")
        }
        let runtimeBaseURL = resolveAmazonBedrockBaseURL(
            baseURL: settings.baseURL,
            serviceEndpointURL: environmentValue(["AWS_ENDPOINT_URL_BEDROCK_RUNTIME"]),
            globalEndpointURL: globalEndpointURL,
            service: "bedrock-runtime",
            getRegion: { region }
        )
        let agentBaseURL = resolveAmazonBedrockBaseURL(
            baseURL: settings.baseURL,
            serviceEndpointURL: environmentValue(["AWS_ENDPOINT_URL_BEDROCK_AGENT_RUNTIME"]),
            globalEndpointURL: globalEndpointURL,
            service: "bedrock-agent-runtime",
            getRegion: { region }
        )
        let auth = try bedrockAuth(settings: settings, providerID: providerID, fallbackToEnvironmentSessionTokenForExplicitKeys: false)

        let headers = withUserAgentSuffix(settings.headers, amazonBedrockUserAgent)
        runtimeConfig = BedrockRuntimeConfig(providerID: providerID, region: region, service: "bedrock", baseURL: runtimeBaseURL, headers: headers, auth: auth, transport: settings.transport, date: settings.date)
        agentRuntimeConfig = BedrockRuntimeConfig(providerID: providerID, region: region, service: "bedrock", baseURL: agentBaseURL, headers: headers, auth: auth, transport: settings.transport, date: settings.date)
    }

    public func languageModel(_ modelID: String) throws -> any LanguageModel {
        AmazonBedrockLanguageModel(modelID: modelID, config: runtimeConfig)
    }

    public func languageModel(
        _ modelID: String,
        settings: AmazonBedrockChatModelSettings
    ) throws -> any LanguageModel {
        AmazonBedrockLanguageModel(modelID: modelID, config: runtimeConfig, settings: settings)
    }

    public func embeddingModel(_ modelID: String) throws -> any EmbeddingModel {
        AmazonBedrockEmbeddingModel(modelID: modelID, config: runtimeConfig)
    }

    public func embeddingModel(
        _ modelID: String,
        settings: AmazonBedrockEmbeddingModelSettings
    ) throws -> any EmbeddingModel {
        AmazonBedrockEmbeddingModel(modelID: modelID, config: runtimeConfig, settings: settings)
    }

    public func imageModel(_ modelID: String) throws -> any ImageModel {
        AmazonBedrockImageModel(modelID: modelID, config: runtimeConfig)
    }

    public func rerankingModel(_ modelID: String) throws -> any RerankingModel {
        AmazonBedrockRerankingModel(modelID: modelID, config: agentRuntimeConfig)
    }

    public func transcriptionModel(_ modelID: String) throws -> any TranscriptionModel {
        throw AIError.unsupportedModel(provider: providerID, capability: .transcription, modelID: modelID)
    }

    public func speechModel(_ modelID: String) throws -> any SpeechModel {
        throw AIError.unsupportedModel(provider: providerID, capability: .speech, modelID: modelID)
    }

    public func videoModel(_ modelID: String) throws -> any VideoModel {
        throw AIError.unsupportedModel(provider: providerID, capability: .video, modelID: modelID)
    }
}

public final class AmazonBedrockAnthropicProvider: AIProvider, @unchecked Sendable {
    public let providerID = "bedrock.anthropic"
    public let supportedCapabilities: Set<ModelCapability> = [.language]
    private let runtimeConfig: BedrockRuntimeConfig

    public init(settings: AmazonBedrockProviderSettings = AmazonBedrockProviderSettings()) throws {
        let region = settings.region ?? environmentValue(["AWS_REGION", "AWS_DEFAULT_REGION"]) ?? "us-east-1"
        if settings.baseURL == nil, environmentValue(["AWS_ENDPOINT_URL", "AWS_ENDPOINT_URL_BEDROCK_RUNTIME"]) == nil {
            try validateHostnamePart(region, argument: "region")
        }
        let runtimeBaseURL = resolveAmazonBedrockBaseURL(
            baseURL: settings.baseURL,
            serviceEndpointURL: environmentValue(["AWS_ENDPOINT_URL_BEDROCK_RUNTIME"]),
            globalEndpointURL: environmentValue(["AWS_ENDPOINT_URL"]),
            service: "bedrock-runtime",
            getRegion: { region }
        )
        let auth = try bedrockAuth(settings: settings, providerID: providerID)

        let headers = withUserAgentSuffix(settings.headers, amazonBedrockUserAgent)
        runtimeConfig = BedrockRuntimeConfig(providerID: providerID, region: region, service: "bedrock", baseURL: runtimeBaseURL, headers: headers, auth: auth, transport: settings.transport, date: settings.date)
    }

    public func languageModel(_ modelID: String) throws -> any LanguageModel {
        AmazonBedrockAnthropicLanguageModel(modelID: modelID, config: runtimeConfig)
    }

    public func messages(_ modelID: String) throws -> any LanguageModel {
        try languageModel(modelID)
    }

    public func embeddingModel(_ modelID: String) throws -> any EmbeddingModel {
        throw AIError.unsupportedModel(provider: providerID, capability: .embedding, modelID: modelID)
    }

    public func imageModel(_ modelID: String) throws -> any ImageModel {
        throw AIError.unsupportedModel(provider: providerID, capability: .image, modelID: modelID)
    }

    public func transcriptionModel(_ modelID: String) throws -> any TranscriptionModel {
        throw AIError.unsupportedModel(provider: providerID, capability: .transcription, modelID: modelID)
    }

    public func speechModel(_ modelID: String) throws -> any SpeechModel {
        throw AIError.unsupportedModel(provider: providerID, capability: .speech, modelID: modelID)
    }

    public func videoModel(_ modelID: String) throws -> any VideoModel {
        throw AIError.unsupportedModel(provider: providerID, capability: .video, modelID: modelID)
    }

    public func rerankingModel(_ modelID: String) throws -> any RerankingModel {
        throw AIError.unsupportedModel(provider: providerID, capability: .reranking, modelID: modelID)
    }
}

public final class BedrockMantleProvider: AIProvider, @unchecked Sendable {
    public let providerID = "bedrock-mantle"
    public let supportedCapabilities: Set<ModelCapability> = [.language]
    private let chatConfiguration: OpenAICompatibleModelConfiguration
    private let responsesConfiguration: OpenAIModelConfiguration

    public init(settings: AmazonBedrockProviderSettings = AmazonBedrockProviderSettings()) throws {
        let region = settings.region ?? environmentValue(["AWS_REGION", "AWS_DEFAULT_REGION"]) ?? "us-east-1"
        if settings.baseURL == nil { try validateHostnamePart(region, argument: "region") }
        let customBaseURL = settings.baseURL.map(withoutTrailingSlash)
        let defaultRoot = "https://bedrock-mantle.\(region).api.aws"
        let baseURL = customBaseURL ?? "\(defaultRoot)/v1"
        let resolveURL: @Sendable (String, String) throws -> URL = { modelID, path in
            let resolvedBaseURL = customBaseURL
                ?? "\(defaultRoot)/\(bedrockMantleUsesOpenAIRoute(modelID) ? "openai/v1" : "v1")"
            return try providerURL("\(resolvedBaseURL)\(path)", queryParams: [:])
        }
        let auth = try bedrockAuth(settings: settings, providerID: providerID)

        let headers = withUserAgentSuffix(settings.headers, amazonBedrockUserAgent)
        let transport = BedrockSigningTransport(
            providerID: providerID,
            region: region,
            service: "bedrock-mantle",
            auth: auth,
            transport: settings.transport,
            date: settings.date
        )
        let chatHTTP = AIProviderHTTPClient(providerID: "bedrock-mantle.chat", headers: headers,
            transport: transport, url: resolveURL)
        let chat = OpenAIModelConfiguration(http: chatHTTP, baseURL: baseURL, openAIBackedProviderRoot: nil)
        chatConfiguration = chat.compatibleConfiguration()
        var responses = OpenAIModelConfiguration(
            http: AIProviderHTTPClient(providerID: "bedrock-mantle.responses", headers: headers,
                transport: transport, url: resolveURL),
            baseURL: baseURL, openAIBackedProviderRoot: "openai")
        responses.supportsWebSearchSourcesInclude = false
        responsesConfiguration = responses
    }

    public func languageModel(_ modelID: String) throws -> any LanguageModel {
        try chat(modelID)
    }

    public func chat(_ modelID: String) throws -> any LanguageModel {
        OpenAICompatibleChatModel(modelID: modelID, config: chatConfiguration)
    }

    public func responses(_ modelID: String) throws -> any LanguageModel {
        OpenAICompatibleResponsesModel(modelID: modelID, config: responsesConfiguration)
    }

    public func embeddingModel(_ modelID: String) throws -> any EmbeddingModel {
        throw AIError.unsupportedModel(provider: providerID, capability: .embedding, modelID: modelID)
    }

    public func imageModel(_ modelID: String) throws -> any ImageModel {
        throw AIError.unsupportedModel(provider: providerID, capability: .image, modelID: modelID)
    }

    public func transcriptionModel(_ modelID: String) throws -> any TranscriptionModel {
        throw AIError.unsupportedModel(provider: providerID, capability: .transcription, modelID: modelID)
    }

    public func speechModel(_ modelID: String) throws -> any SpeechModel {
        throw AIError.unsupportedModel(provider: providerID, capability: .speech, modelID: modelID)
    }

    public func videoModel(_ modelID: String) throws -> any VideoModel {
        throw AIError.unsupportedModel(provider: providerID, capability: .video, modelID: modelID)
    }

    public func rerankingModel(_ modelID: String) throws -> any RerankingModel {
        throw AIError.unsupportedModel(provider: providerID, capability: .reranking, modelID: modelID)
    }
}

func bedrockMantleUsesOpenAIRoute(_ modelID: String) -> Bool {
    (modelID.hasPrefix("openai.gpt-") && !modelID.hasPrefix("openai.gpt-oss-"))
        || modelID.hasPrefix("google.gemma-4")
        || modelID.hasPrefix("xai.")
}

func resolveAmazonBedrockBaseURL(
    baseURL: String?,
    serviceEndpointURL: String?,
    globalEndpointURL: String?,
    service: String,
    getRegion: () -> String
) -> String {
    if let override = baseURL ?? serviceEndpointURL ?? globalEndpointURL {
        return withoutTrailingSlash(override)
    }

    let region = getRegion()
    let dnsSuffix: String
    switch region {
    case let value where value.hasPrefix("cn-"):
        dnsSuffix = "amazonaws.com.cn"
    case let value where value.hasPrefix("us-iso-"):
        dnsSuffix = "c2s.ic.gov"
    case let value where value.hasPrefix("us-isob-"):
        dnsSuffix = "sc2s.sgov.gov"
    case let value where value.hasPrefix("eu-isoe-"):
        dnsSuffix = "cloud.adc-e.uk"
    case let value where value.hasPrefix("us-isof-"):
        dnsSuffix = "csp.hci.ic.gov"
    case let value where value.hasPrefix("eusc-"):
        dnsSuffix = "amazonaws.eu"
    default:
        dnsSuffix = "amazonaws.com"
    }
    return withoutTrailingSlash("https://\(service).\(region).\(dnsSuffix)")
}

enum BedrockAuth: Sendable {
    case bearer(String)
    case sigV4(@Sendable () async throws -> AWSSigningCredentials)
}

private func bedrockAuth(settings: AmazonBedrockProviderSettings, providerID: String, fallbackToEnvironmentSessionTokenForExplicitKeys: Bool = true) throws -> BedrockAuth {
    let rawAPIKey = settings.apiKey ?? environmentValue(["AWS_BEARER_TOKEN_BEDROCK"])
    if let apiKey = rawAPIKey?.trimmingCharacters(in: .whitespacesAndNewlines), !apiKey.isEmpty {
        return .bearer(apiKey)
    }

    if let credentialProvider = settings.credentialProvider {
        return .sigV4 {
            do {
                let credentials = try await credentialProvider()
                return AWSSigningCredentials(
                    accessKeyID: credentials.accessKeyID,
                    secretAccessKey: credentials.secretAccessKey,
                    sessionToken: credentials.sessionToken
                )
            } catch {
                throw AIError.invalidArgument(
                    argument: "credentialProvider",
                    message: "AWS credential provider failed: \(error). Please ensure it returns accessKeyID and secretAccessKey values."
                )
            }
        }
    }

    let accessKeyID = settings.accessKeyID ?? environmentValue(["AWS_ACCESS_KEY_ID"])
    let secretAccessKey = settings.secretAccessKey ?? environmentValue(["AWS_SECRET_ACCESS_KEY"])
    guard let accessKeyID, let secretAccessKey else {
        throw AIError.missingAPIKey(provider: providerID, environmentVariables: ["AWS_BEARER_TOKEN_BEDROCK", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"])
    }
    let shouldLoadEnvironmentSessionToken = fallbackToEnvironmentSessionTokenForExplicitKeys || settings.accessKeyID == nil || settings.secretAccessKey == nil
    let sessionToken = settings.sessionToken ?? (shouldLoadEnvironmentSessionToken ? environmentValue(["AWS_SESSION_TOKEN"]) : nil)
    return .sigV4 {
        AWSSigningCredentials(accessKeyID: accessKeyID, secretAccessKey: secretAccessKey, sessionToken: sessionToken)
    }
}

private final class BedrockSigningTransport: AIStreamingTransport, @unchecked Sendable {
    private let providerID: String
    private let region: String
    private let service: String
    private let auth: BedrockAuth
    private let transport: any AITransport
    private let date: @Sendable () -> Date

    init(providerID: String, region: String, service: String, auth: BedrockAuth, transport: any AITransport, date: @escaping @Sendable () -> Date) {
        self.providerID = providerID
        self.region = region
        self.service = service
        self.auth = auth
        self.transport = transport
        self.date = date
    }

    func send(_ request: AIHTTPRequest) async throws -> AIHTTPResponse {
        try await transport.send(authenticatedRequest(request))
    }

    func stream(_ request: AIHTTPRequest) async throws -> AIHTTPStreamResponse {
        let streamingTransport = try requireStreamingTransport(transport, providerID: providerID)
        return try await streamingTransport.stream(authenticatedRequest(request))
    }

    private func authenticatedRequest(_ request: AIHTTPRequest) async throws -> AIHTTPRequest {
        switch auth {
        case let .bearer(apiKey):
            var signed = request
            signed.headers["Authorization"] = signed.headers["Authorization"] ?? "Bearer \(apiKey)"
            return signed
        case let .sigV4(credentialsProvider):
            let credentials = try await credentialsProvider()
            return try AWSSigV4.sign(
                request: request,
                body: request.body ?? Data(),
                credentials: credentials,
                region: region,
                service: service,
                date: date()
            )
        }
    }
}

func bedrockHTTPStatusError(provider: String, response: AIHTTPResponse) -> AIError {
    guard let raw = try? response.jsonValue(),
          let message = raw["message"]?.stringValue else {
        return apiCallError(provider: provider, response: response)
    }
    let body = raw["type"]?.stringValue.map { "\($0): \(message)" } ?? message
    return apiCallError(
        provider: provider,
        statusCode: response.statusCode,
        body: body,
        headers: response.headers
    )
}

struct BedrockRuntimeConfig: @unchecked Sendable {
    var providerID: String
    var region: String
    var service: String
    var baseURL: String
    var headers: [String: String]
    var auth: BedrockAuth
    var transport: any AITransport
    var date: @Sendable () -> Date

    func sendJSON(path: String, body: JSONValue, headers requestHeaders: [String: String] = [:], abortSignal: AIAbortSignal? = nil) async throws -> JSONValue {
        try await sendJSONResponse(path: path, body: body, headers: requestHeaders, abortSignal: abortSignal).json
    }

    func sendJSONResponse(path: String, body: JSONValue, headers requestHeaders: [String: String] = [:], abortSignal: AIAbortSignal? = nil) async throws -> (json: JSONValue, response: AIHTTPResponse) {
        let response = try await sendRequest(try request(path: path, body: body, headers: requestHeaders, abortSignal: abortSignal))
        guard (200..<300).contains(response.statusCode) else {
            throw bedrockHTTPStatusError(provider: providerID, response: response)
        }
        return (try response.jsonValue(), response)
    }

    func sendRequest(_ request: AIHTTPRequest) async throws -> AIHTTPResponse {
        try await transport.send(authenticatedRequest(request))
    }

    func streamRequest(_ request: AIHTTPRequest) async throws -> AIHTTPStreamResponse {
        let streamingTransport = try requireStreamingTransport(transport, providerID: providerID)
        return try await streamingTransport.stream(authenticatedRequest(request))
    }

    private func authenticatedRequest(_ request: AIHTTPRequest) async throws -> AIHTTPRequest {
        switch auth {
        case let .bearer(apiKey):
            var signed = request
            signed.headers["Authorization"] = signed.headers["Authorization"] ?? "Bearer \(apiKey)"
            return signed
        case let .sigV4(credentialsProvider):
            return try AWSSigV4.sign(
                request: request,
                body: request.body ?? Data(),
                credentials: try await credentialsProvider(),
                region: region,
                service: service,
                date: date()
            )
        }
    }

    func request(path: String, body: JSONValue, headers requestHeaders: [String: String] = [:], abortSignal: AIAbortSignal? = nil) throws -> AIHTTPRequest {
        let bodyData = try encodeJSONBody(body)
        var mergedHeaders = headers.mergingHeaders(requestHeaders)
        mergedHeaders["content-type"] = mergedHeaders["content-type"] ?? "application/json"
        return AIHTTPRequest(method: "POST", url: try requireURL("\(withoutTrailingSlash(baseURL))\(path)"), headers: mergedHeaders, body: bodyData, abortSignal: abortSignal)
    }
}


public extension AIProviders {
    static func amazonBedrock(configuration: AmazonBedrockProviderSettings) throws -> AmazonBedrockProvider { try AmazonBedrockProvider(settings: configuration) }
    static func amazonBedrockAnthropic(configuration: AmazonBedrockProviderSettings) throws -> AmazonBedrockAnthropicProvider { try AmazonBedrockAnthropicProvider(settings: configuration) }
    static func bedrockMantle(configuration: AmazonBedrockProviderSettings) throws -> BedrockMantleProvider { try BedrockMantleProvider(settings: configuration) }
}
