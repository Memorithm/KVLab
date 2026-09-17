# BKV-K6 FLAT selection-binding evidence

Status: consumer contract only; no benchmark or performance conclusion.

FLAT-ATTENTION #251, merged as `5f474c51bff4b782b3a124375cb0f4b651b4b708`, introduces the canonical `flat.bikv-selection-binding.v1` envelope. The producer binds the exact `flat.boolean-kv-selection.v2` routing decision to the canonical BKV-K6 qualification envelope and rejects accounting drift between the routing decision and qualification record.

KVLab now retains that envelope through `kvlab.flat_bikv_selection_binding.FlatBikvSelectionBindingV1`. The consumer is pinned to the merge SHA above and fails closed on:

- duplicate or reordered top-level/schema-critical JSON fields;
- non-canonical compact producer encoding;
- wrong selection/binding schemas;
- non-integer or out-of-range `u64` accounting values, including JSON booleans masquerading as integers;
- invalid page order, duplicate physical pages, Hamming/XNOR conservation or Hamming-threshold violations;
- inconsistent Boolean packed-byte and logical numerical K/V accounting;
- drift between selection signature/threshold/cardinality/byte totals and BKV-K6 qualification accounting;
- inconsistent BKV-K6 correctness-gate composition;
- producer FNV-1a corruption checks for the selection, qualification and outer binding.

The exact canonical binding bytes plus the canonical nested selection and qualification encodings receive SHA-256 content identities for KVLab retention. FNV-1a remains only an accidental-corruption/reproducibility check, not cryptographic attestation.

This bridge does **not** convert `avoided_numerical_kv_bytes` into measured DRAM/HBM/cache/PCIe traffic. It does not establish overlap, residency, TTFT, TPOT, model quality, energy, bandwidth reduction, candidate recall, or speedup. BKV-K6 target-host/model qualification still requires actual retained measurements under the frozen KVLab target protocol, and BKV-K9 remains closed until its evidence gate is satisfied.
