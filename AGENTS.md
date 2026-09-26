# KVLab Agent Bootstrap

KVLab is the Memorithm evidence bench for KV-state representation, routing,
tiering, transfer and causal analysis. It is not the owner of FLAT kernels,
SLHAv2 runtime semantics, ElasticXxx adaptation semantics or NNIS runtime
semantics.

## Primary active programme — SLHA adaptive-KV campaign

Before work that may feed SLHAv2, read:

- `ROADMAP.md`
- `docs/SLHA_ADAPTIVE_KV_RESEARCH_PROGRAMME.md`
- `docs/BOOLEAN_KV_CACHE_ROADMAP.md`
- `docs/ELASTIC_WORD_WIDTH_PREREGISTRATION.md`

The September 2026 priority is to determine which KV information must be
preserved for real-model quality and which work/bytes can safely be avoided.

### Non-negotiable rules

- Dense/full-fidelity numerical attention remains the authority unless a
  protocol explicitly defines another oracle.
- Top-k recall is not sufficient by itself. Record retained softmax mass and
  omitted mass whenever exact reference scores are available for evaluation.
- Exact-score/softmax-mass oracles are evaluation tools unless a deployable
  path can compute the same signal without reading the dense state.
- Match candidate density or budget for random/positional controls.
- Logical bytes avoided are not DRAM/HBM traffic.
- Packed word width is not a hardware-vector-width claim.
- Positive, negative, equivalent and inconclusive results are retained.
- Do not alter a preregistration after confirmatory evidence is observed.
- No KVLab result changes SLHAv2 automatically. Export immutable evidence;
  SLHAv2 owns destination requalification and promotion.

### Ecosystem ownership

- SLHAv2: compressed representation and destination quality gate.
- FLAT-ATTENTION: numerical attention consumption and portable GPU timing.
- ElasticXxx: generic adaptation lifecycle and rollback.
- NNIS: native NVIDIA physical qualification.
- BooleanLab: Boolean policy/function discovery.
- Forge/ADA: bounded candidate search after evaluator freeze.
- SciRust: reusable primitives after independent evidence supports promotion.

### Validation

For Python-only additions:

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

Run any repository-specific gates present on the exact branch before PR
promotion. Never report a campaign as executed merely because its protocol or
runner exists.
