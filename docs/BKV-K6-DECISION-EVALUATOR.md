# BKV-K6 frozen decision evaluator

Status: preregistered decision/evidence infrastructure only. It does not authorize BKV-K9/NBKV, adaptive routing, protected-holdout access, or a performance claim outside the exact retained campaign.

The existing `kvlab.bikv-target-analysis-plan.v1` freezes H0/H1, the primary metric and direction, minimum effect, quality noninferiority margin, uncertainty family and campaign completeness before outcome inspection. It deliberately did not define the exact effect statistic, percentile convention, Student-t convention, or candidate-side correctness thresholds. Those details cannot be chosen after seeing target outcomes.

`kvlab.bikv-target-decision-plan.v2` closes that gap by wrapping the immutable v1 plan and freezing:

- paired-mean effect statistics for primary and quality gates;
- `linear_type7` quantiles for paired percentile bootstrap;
- the canonical statistics provider `scirust-research-stats-json/v1`, operation `paired_mean_percentile`, pinned to qualified SciRust merge `be7fcca3b31cedf722d71a2a56db8f6d088037cf`;
- a common `improvement_positive` orientation so lower-is-better and higher-is-better metrics use one decision inequality;
- explicit candidate-side metric guards, including at minimum candidate recall, false-negative rate, O error, LSE error, and reset/reuse correctness;
- exact guard units and thresholds.

For a lower-is-better metric, improvement is `baseline - candidate`. For a higher-is-better metric, improvement is `candidate - baseline`. The primary gate passes only when the lower confidence bound is at least the preregistered minimum effect. The quality noninferiority gate passes only when its lower confidence bound is at least the negative preregistered margin. A threshold miss is retained as a valid negative result rather than treated as missing evidence.

The evaluator reconstructs `kvlab.bikv-target-paired-summary.v1` directly from the canonical protocol, campaign manifest and retained run payloads. It therefore does not accept a caller-provided summary that could have dropped failed or inconvenient pairs. Incomplete primary/quality evidence yields `blocked_incomplete_evidence`. A complete campaign that misses an interval or guard yields `candidate_does_not_meet_preregistered_gate`. Only a complete campaign satisfying every frozen interval and guard yields `candidate_meets_preregistered_gate`.

Bootstrap resampling uses the same SplitMix64 rejection rule, compensated paired mean, type-7 quantile convention and bounded input budget qualified by SciRust #1452. KVLab's Python implementation is an evidence-adapter mirror, not a second canonical statistics library. A frozen cross-language fixture checks `[3,-1,5,2]`, 1,000 resamples, 95% confidence and seed `20260917` against the pinned SciRust worker result `(estimate=2.25, lower=-0.25, upper=4.25)`. The v1 analysis-plan option `paired_t_interval` is deliberately rejected by this v2 evaluator until an equivalent canonical SciRust primitive is qualified; KVLab does not invent that generic statistic locally. The decision record content-addresses the exact protocol, campaign, reconstructed paired summary, and decision plan.

A passing record is still campaign-local evidence. It does not prove physical DRAM/HBM traffic reduction when the protocol records logical bytes, does not establish energy without an energy measurement, does not generalize to another model/runtime/device, and does not by itself open BKV-K9. Issue #111 remains the gate for NBKV work.

Use:

```bash
python3 tools/evaluate_bikv_target_campaign.py \
  protocol.json campaign.json decision-plan-v2.json run-*.json \
  --output decision.json
```

The v2 plan itself must be created and content-addressed before any qualifying outcome is inspected. Synthetic fixtures and unit tests qualify only the evaluator semantics; they are not target-host/model results.
