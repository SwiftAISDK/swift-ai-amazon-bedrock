# Changelog

## 1.1.2

- Distinguish dated base Fable 5 from genuine 5.1/5.2 aliases, retaining the original hosted JSON-tool fallback and minor-model native-output controls. Recognize current Claude Fable 5.2 model IDs while preserving Bedrock strict/native-output limitations. Anthropic hosting uses the updated adaptive-thinking and forced-tool policy. Reviewed npm delta: @ai-sdk/amazon-bedrock 5.0.135 to 5.0.141. Validation: `node Scripts/test-packages.mjs` passed all 52 owning suites and 1731 facade tests; tooling 27/27, generation/isolation checks, standalone exports and 106 minimum/latest consumers, documentation check and 101-page build passed. Publication requires green exact-source Swift 6.3.3 CI. Paid/live verification and full-package parity are not claimed.
  Compatibility: Compatible behavior/header correction. Original public selectors are preserved. npm baselines and Swift SemVer are independent; this reviewed delta does not establish full-package or live-service parity. Required direct minimums: swift-ai-anthropic 1.3.0 to 1.4.0.
