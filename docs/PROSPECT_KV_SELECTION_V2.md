# ProspectEngine KV selection handoff v2

Schema: `kvlab.prospect-kv-selection/v2`.

Version 2 corrects an identity limitation in v1. Vocabulary token values are not unique sequence identities: ordinary model inputs may contain the same token value many times. V2 therefore preserves the complete `input_token_ids` sequence, including duplicates, and expresses the selected partition using zero-based sequence positions.

`retained_positions` and `evicted_positions` are strictly increasing, disjoint, in range, and must exactly partition `0..len(input_token_ids)`. Their corresponding token values are derived from `input_token_ids`; equal token values at different positions remain distinct occurrences.

The policy label remains provenance only. The handoff establishes the exact logical partition and logical byte accounting; it does not prove that the named policy implementation generated the selection.

Logical retained/evicted bytes do not imply allocator release, HBM reduction, physical traffic reduction, latency improvement, throughput improvement, or preserved model quality. Those require observed runtime evidence.

V1 remains readable for previously captured evidence. New real-model integrations should use v2 position identity rather than vocabulary-value identity.
