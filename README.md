<!-- Generated distribution; edit the canonical repository. -->
# swift-ai-amazon-bedrock 1.1.2

Independent Swift-native port of Vercel AI SDK. Source and upstream dispositions are in PROVENANCE.json and Porting/.

## Distribution manifest

Add this package to your SwiftPM dependencies:

```swift
.package(url: "https://github.com/SwiftAISDK/swift-ai-amazon-bedrock.git", from: "1.1.2")
```

Products: `SwiftAIAmazonBedrock`.
Contribute and report issues in the canonical SwiftAISDK repository.

## Authored usage guide

# SwiftAIAmazonBedrock

Swift-native port of `@ai-sdk/amazon-bedrock`. Maintained in the canonical SwiftAISDK repository and exported as an independent Swift package.
Sources are owned by the canonical repository, with shared HTTP/SigV4 machinery
in core and Anthropic wire behavior in the Anthropic package.

Existing baselines and documented port gaps remain unchanged. See the canonical
provider guide and `Porting/map.json`.

See [provider usage and capabilities](https://github.com/SwiftAISDK/SwiftAISDK/blob/main/docs-site/scripts/provider-guides/amazon-bedrock.md) for authentication, options and supported surfaces.

The reviewed Claude 5/5.5 model families support native output schemas and strict
tools where the Bedrock model catalog declares support. Native output is selected
automatically only for schemas it can preserve. Explicit `structuredOutputMode`
continues to select output-format or JSON-tool behavior. Unknown future Claude
aliases retain conservative capability handling.

A named tool choice on Anthropic models that support forced tool use keeps all
function tools and disables parallel tool use by default. Explicit
`disableParallelToolUse: false` opts out. Other model families receive only the
named tool. Top-level image `seed: 0` is preserved. Foreign empty signature-text
parts are omitted before final assistant-prefill trimming.
