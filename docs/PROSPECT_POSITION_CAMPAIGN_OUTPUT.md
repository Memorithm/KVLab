# ProspectEngine-compatible position campaign output

A successful KVLab position-native real-model campaign publishes a self-contained, atomically-created evidence directory that can be consumed directly by ProspectEngine.

The directory contains:

- `campaign.json`: the exact canonical `kvlab.prospect-kv-real-model-position-campaign/v1` input specification;
- `manifest.json`: the canonical campaign result manifest whose `campaign_spec_sha256` is the SHA-256 of `campaign.json`;
- `selection-*.json`: canonical observed selection evidence records referenced by the manifest.

The campaign harness completes all backend executions and validates replayability before creating the destination directory. Publication uses a temporary sibling directory and a final rename so a failed backend or persistence step does not leave a partial observed-evidence directory.

This layout is intentionally compatible with ProspectEngine's `verify-kv-campaign` verification boundary. The presence of a complete directory proves only that its canonical files are structurally linked and replayable under their schemas. It does not by itself establish model quality, latency, throughput, physical HBM release, avoided memory traffic, or scientific generality.
