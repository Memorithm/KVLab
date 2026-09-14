# Position-native real-model KV selection protocol

KVLab keeps the earlier logical-id backend protocol intact for replay compatibility. New integrations that can address KV rows by their active-prefix position should use the position-native contracts instead:

- `kvlab.prospect-kv-selection/v2` identifies retained and evicted occurrences by zero-based sequence positions while preserving the exact model token sequence, including duplicate vocabulary ids.
- `kvlab.prospect-kv-real-model-selection/v2` binds that selection to model/runtime/trace provenance, paired output artefact hashes, logical KV-byte accounting, and observed finite metrics.
- `kvlab.prospect-kv-backend-request/v4` and `...response/v4` attest the exact canonical request SHA-256 and the exact `retained_positions` a backend claims to have applied.

The v4 request carries the full `model_input_token_ids`, the retained positions, and a non-empty evaluation continuation. The baseline uses every input position. A candidate is accepted only when its response echoes the exact request digest, mode, policy, and retained-position sequence.

This contract maps directly to runtimes such as NNIS where active KV rows can be compacted by explicit row indices. It does not by itself prove backend internals, physical memory release, HBM residency changes, memory-traffic reduction, latency/throughput improvements, or quality preservation. Those require separately observed evidence.