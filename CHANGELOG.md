# Changelog

## 1.3.0

- Align the provider request identity with @ai-sdk/amazon-bedrock 5.0.146 after comparing source, declarations and matching-release tests against 5.0.142. Offline validation passed: node Scripts/test-packages.mjs (56 packages, 3672 owning tests and 1760 compatibility tests), generation, tooling, isolation, and version selection. See Docs/Archive/WeeklyUpstreamAudit20261122.md for retained gaps. Paid provider calls were skipped.
  Compatibility: Source-compatible request-header update. Requires SwiftAIOpenAICompatible 1.4.0 for the public image model conformance introduced in Core 1.6.0.
