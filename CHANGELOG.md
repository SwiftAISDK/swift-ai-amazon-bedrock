# Changelog

## 1.1.3

- Pass thinking block binding fields through the current Anthropic InvokeModel unary and streaming wire paths, matching removal of the upstream prefix_mismatch_behavior rename. Preserve the legacy directly callable transform helper for existing callers. Reviewed npm delta: @ai-sdk/amazon-bedrock 5.0.141 to 5.0.142. Validation passed: node Scripts/test-packages.mjs (55 owning packages, 2451 owning tests and 1736 compatibility tests), generation checks, 27 tooling tests, isolation/version checks, deterministic exports and minimum-dependency consumers. Documentation check/build passed (104 pages). Publication is gated by exact-source Swift 6.3.3 CI; subsequent public refs and clients are recorded in Docs/ReleaseVerification20261115.json. Paid/live-service checks were not run; this delta does not claim full-package parity.
  Compatibility: Compatible wire-behavior correction. Existing provider and helper selectors remain; no dependency minimum change.
