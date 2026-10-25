import SwiftAIOpenAICompatible
import SwiftAIOpenAI
import SwiftAIAnthropic
import Foundation
import SwiftAIProvider
import SwiftAITransport
import SwiftAIProviderUtilities


public enum AmazonBedrockChatModelFamily: String, Sendable {
    case anthropic
}

public struct AmazonBedrockChatModelSettings: Sendable {
    public var modelFamily: AmazonBedrockChatModelFamily?

    public init(modelFamily: AmazonBedrockChatModelFamily? = nil) {
        self.modelFamily = modelFamily
    }
}