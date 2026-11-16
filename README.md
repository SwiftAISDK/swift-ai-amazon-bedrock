<!-- Generated distribution; edit the canonical repository. -->
# swift-ai-amazon-bedrock 1.2.0

Independent Swift-native port of Vercel AI SDK. Source and upstream dispositions are in PROVENANCE.json and Porting/.

## Distribution manifest

Add this package to your SwiftPM dependencies:

```swift
.package(url: "https://github.com/SwiftAISDK/swift-ai-amazon-bedrock.git", from: "1.2.0")
```

Products: `SwiftAIAmazonBedrock`.
Contribute and report issues in the canonical SwiftAISDK repository.

## Authored usage guide

# Amazon Bedrock for Swift

`SwiftAIAmazonBedrock` adds Amazon Bedrock, Amazon Bedrock (Anthropic models) and Amazon Bedrock Mantle to [SwiftAISDK](https://github.com/SwiftAISDK/SwiftAISDK), an independent Swift port of the Vercel AI SDK. It ports `@ai-sdk/amazon-bedrock` and supports text generation, embeddings, image generation and reranking.

Provider guides with models, options, streaming and errors:

- [Amazon Bedrock](https://swiftaisdk.github.io/SwiftAISDK/providers/amazon-bedrock/)
- [Amazon Bedrock (Anthropic models)](https://swiftaisdk.github.io/SwiftAISDK/providers/amazon-bedrock-anthropic/)
- [Amazon Bedrock Mantle](https://swiftaisdk.github.io/SwiftAISDK/providers/bedrock-mantle/)

## Usage

Add the `SwiftAI` product from `swift-ai-core` and the `SwiftAIAmazonBedrock` product from this package to your target, then create the provider:

```swift
import SwiftAI
import SwiftAIAmazonBedrock

let provider = try AIProviders.amazonBedrock(configuration: AmazonBedrockProviderSettings())
```

By default the provider reads `AWS_BEARER_TOKEN_BEDROCK`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN`, `AWS_REGION`, `AWS_DEFAULT_REGION`, `AWS_ENDPOINT_URL_BEDROCK_RUNTIME`, `AWS_ENDPOINT_URL_BEDROCK_AGENT_RUNTIME` and `AWS_ENDPOINT_URL` from the environment; values set in `AmazonBedrockProviderSettings` take precedence.

To use every provider through one dependency, add the complete [SwiftAISDK](https://github.com/SwiftAISDK/SwiftAISDK) package instead; see [Install](https://swiftaisdk.github.io/SwiftAISDK/getting-started/install/).

## Contributing

This package is exported from `Packages/swift-ai-amazon-bedrock` in the canonical [SwiftAISDK](https://github.com/SwiftAISDK/SwiftAISDK) repository. Open issues and pull requests there.
