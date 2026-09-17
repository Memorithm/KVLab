# BKV-K6 frozen decision evaluator

Status: preregistered decision/evidence infrastructure only. It does not authorize BKV-K9/NBKV, adaptive routing, protected-holdout access, or a performance claim outside the exact retained campaign.

The existing `kvlab.bikv-target-analysis-plan.v1` freezes H0/H1, the primary metric and direction, minimum effect, quality noninferiority margin, uncertainty family and campaign completeness before outcome inspection. It deliberately did not define the exact effect statistic, percentile convention, Student-t convention, or candidate-side correctness thresholds. Those details cannot be chosen after seeing target outcomes.

`kvlab.bikv-target-decision-plan.v2` closes that gap by wrapping the immutable v1 plan and freezing:

- primary and quality effect statistics (`paired_mean` or `paired_median`);
- `linear_type7` quantiles for paired percentile bootstrap, or an equal-tail Student-t interval for paired means;
- a common `improvement_positive` orientation so lower-is-better and higher-is-better metrics use one decision inequality;
- explicit candidate-side metric guards, including at minimum candidate recall, false-negative rate, O error, LSE error, and reset/reuse correctness;
- exact guard units and thresholds.

For a lower-is-better metric, improvement is `baseline - candidate`. For a higher-is-better metric, improvement is `candidate - baseline`. The primary gate passes only when the lower confidence bound is at least the preregistered minimum effect. The quality noninferiority gate passes only when its lower confidence bound is at least the negative preregistered margin. A threshold miss is retained as a valid negative result rather than treated as missing evidence.

The evaluator reconstructs `kvlab.bikv-target-paired-summary.v1` directly from the canonical protocol, campaign manifest and retained run payloads. It therefore does not accept a caller-provided summary that could have dropped failed or inconvenient pairs. Incomplete primary/quality evidence yields `blocked_incomplete_evidence`. A complete campaign that misses an interval or guard yields `candidate_does_not_meet_preregistered_gate`. Only a complete campaign satisfying every frozen interval and guard yields `candidate_meets_preregistered_gate`.

Bootstrap resampling uses a specified SplitMix64 generator and the frozen seed/resample count. Student-t critical values are evaluated from the regularized incomplete beta function and equal-tail inversion; the test suite checks reference critical values. The decision record content-addresses the exact protocol, campaign, reconstructed paired summary, and decision plan.

A passing record is still campaign-local evidence. It does not prove physical DRAM/HBM traffic reduction when the protocol records logical bytes, does not establish energy without an energy measurement, does not generalize to another model/runtime/device, and does not by itself open BKV-K9. Issue #111 remains the gate for NBKV work.

Use:

```bash
python3 tools/evaluate_bikv_target_campaign.py \
  protocol.json campaign.json decision-plan-v2.json run-*.json \
  --output decision.json
```

The v2 plan itself must be created and content-addressed before any qualifying outcome is inspected. Synthetic fixtures and unit tests qualify only the evaluator semantics; they are not target-host/model results.
