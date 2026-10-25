import SwiftAIOpenAICompatible
import SwiftAIOpenAI
import SwiftAIAnthropic
import Foundation
import SwiftAIProvider
import SwiftAITransport
import SwiftAIProviderUtilities


public final class AmazonBedrockRerankingModel: RerankingModel, @unchecked Sendable {
    public let providerID = "amazon-bedrock"
    public let modelID: String
    private let config: BedrockRuntimeConfig

    init(modelID: String, config: BedrockRuntimeConfig) {
        self.modelID = modelID
        self.config = config
    }

    public func rerank(_ request: RerankingRequest) async throws -> RerankingResult {
        let providerOptions = try bedrockRequestProviderOptions(providerOptions: request.providerOptions, extraBody: request.extraBody)
        let modelArn = "arn:aws:bedrock:\(config.region)::foundation-model/\(modelID)"
        var modelConfiguration: [String: JSONValue] = ["modelArn": .string(modelArn)]
        if let additionalModelRequestFields = providerOptions["additionalModelRequestFields"] {
            modelConfiguration["additionalModelRequestFields"] = additionalModelRequestFields
        }

        var bedrockRerankingConfiguration: [String: JSONValue] = [
            "modelConfiguration": .object(modelConfiguration)
        ]
        if let topK = request.topK {
            bedrockRerankingConfiguration["numberOfResults"] = .number(Double(topK))
        }

        var body: [String: JSONValue] = [
            "queries": .array([.object(["type": "TEXT", "textQuery": .object(["text": .string(request.query)])])]),
            "sources": .array(request.documents.map { .object(["type": "INLINE", "inlineDocumentSource": .object(["type": "TEXT", "textDocument": .object(["text": .string($0)])])]) }),
            "rerankingConfiguration": .object([
                "type": "BEDROCK_RERANKING_MODEL",
                "bedrockRerankingConfiguration": .object(bedrockRerankingConfiguration)
            ])
        ]
        if let nextToken = providerOptions["nextToken"] {
            body["nextToken"] = nextToken
        }
        body.merge(bedrockPassthroughExtraBody(request.extraBody)) { _, new in new }
        let response = try await config.sendJSONResponse(path: "/rerank", body: .object(body), headers: request.headers, abortSignal: request.abortSignal)
        let raw = response.json
        let results = raw["results"]?.arrayValue?.compactMap { item -> RerankedDocument? in
            guard let index = item["index"]?.intValue,
                  let score = item["relevanceScore"]?.doubleValue ?? item["score"]?.doubleValue else { return nil }
            return RerankedDocument(index: index, score: score)
        } ?? []
        return RerankingResult(
            results: results,
            rawValue: raw,
            requestMetadata: AIRequestMetadata(body: .object(body), headers: request.headers),
            responseMetadata: aiResponseMetadata(from: raw, response: response.response, modelID: modelID)
        )
    }
}