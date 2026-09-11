# KVLab C1/C11 external baseline registry — 2026-09-11

This registry records recent external methods that may serve as falsifiable baselines for KVLab C1 (recoverable information per byte) and C11 (multi-resolution KV pyramid). It does **not** modify either preregistration, import published results as Memorithm evidence, or authorize a final/confirmatory run.

## Admission rule

An external method is eligible only as a baseline/control after its mechanism can be represented under the same model, context, cache budget, query visibility, replay trace, and evaluation metric as the KVLab comparison. Published headline numbers are never copied into KVLab results.

## Baselines

### CompressKV — semantic-retrieval-guided retention

- Source: Xiaolin Lin, Jingcun Wang, Olga Kondrateva, Yiyu Shi, Bing Li, Grace Li Zhang, *CompressKV: Semantic-Retrieval-Guided KV-Cache Compression for Resource-Efficient Long-Context LLM Inference*, arXiv:2606.24467, 2026-06-23.
- Primary link: https://arxiv.org/abs/2606.24467
- Mechanism to reproduce: semantic retrieval heads select retained KV tokens; layer budgets are assigned using offline layer-wise eviction-error estimates.
- KVLab role: C1 retention baseline against age/LRU, magnitude, attention and recoverability-based policies.
- Leakage control: classify the implementation as query-aware or query-agnostic before comparison; never mix visibility regimes in one ranking.

### SeKV — hierarchical reconstructable memory

- Source: Amirhossein Abaskohi, Giuseppe Carenini, Peter West, Yuhang He, *SeKV: Resolution-Adaptive KV Cache with Hierarchical Semantic Memory for Long-Context LLM Inference*, arXiv:2606.31145, 2026-06-30.
- Primary link: https://arxiv.org/abs/2606.31145
- Mechanism to reproduce: lightweight summaries remain on GPU while compressed low-rank span state is retained on CPU and selectively reconstructed at token resolution.
- KVLab role: C11 HOT/WARM/COLD-style recoverable-tier baseline; unlike eviction, the cold representation remains recoverable.
- Required accounting: GPU bytes, CPU bytes, transfer bytes, reconstruction compute, retrieval misses, replay fidelity and end-to-end latency must be reported separately.

### SemantiCache — semantic chunking and merging

- Source: Shunlong Wu et al., *SemantiCache: Efficient KV Cache Compression via Semantic Chunking and Clustered Merging*, arXiv:2603.14303, 2026-03-15.
- Primary link: https://arxiv.org/abs/2603.14303
- Mechanism to reproduce: semantic chunks are clustered and merged into representative cores with proportional-attention correction.
- KVLab role: structured compression control for C1/C11 against token-local eviction and fixed pages.
- Required control: compare against a size-matched non-semantic chunking policy so gains cannot be attributed only to different chunk cardinality.

### SGD-KV — summarization-guided head allocation

- Source: Zeyu Liu, Woomin Song, Xuandi Fu, Sai Muralidhar Jayanthi, Vivek Govindan, Aram Galstyan, Sravan Babu Bodapati, Srikanth Ronanki, *SGD-KV: Summarization Guided KV Cache Compression*, arXiv:2609.03235, 2026-09-03.
- Primary link: https://arxiv.org/abs/2609.03235
- Mechanism to reproduce: a summarization diagnostic ranks attention heads and guides head-aware cache-budget allocation.
- KVLab role: recent head-aware C1 baseline, especially against policies that aggregate importance uniformly across heads.
- Required control: the diagnostic task used to rank heads must be disjoint from the final evaluation set and its cost must be accounted for.

## KVLab normalization contract

For any admitted baseline, record at minimum:

1. exact upstream paper/repository revision and local adapter revision;
2. model, tokenizer, context, prompt/replay trace and query-visibility regime;
3. full-cache oracle result from the same trace;
4. physical and logical cache budgets, including auxiliary metadata and summaries;
5. eviction/compression/reconstruction compute and host-device traffic;
6. paired seeds or deterministic replay identifiers;
7. the same quality, causal-recovery and recoverability-per-byte metrics used by the competing KVLab policy;
8. negative, equivalent and inconclusive outcomes without post-hoc baseline removal.

These entries are controls, not evidence for C1 or C11. Promotion requires a KVLab execution under the applicable preregistered protocol.