import Foundation
import SwiftAIProvider
import SwiftAITransport
import SwiftAIProviderUtilities
public extension AmazonBedrockImageModel {
    var supportsFileInputs: Bool? {
        if ["amazon.nova-canvas-v1:0"].contains(modelID) { return true }
        return nil
    }

    var supportsMaskInputs: Bool? {
        supportsFileInputs
    }
}
