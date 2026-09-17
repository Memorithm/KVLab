# BKV-K6 preregistered analysis plan

Status: analysis/preregistration infrastructure only.

`kvlab.bikv-target-analysis-plan.v1` closes the gap between the frozen target-host/model protocol and the retained paired descriptive summary. It freezes the analysis choices that must exist **before outcome inspection**; validating a plan does not inspect any target measurement and does not authorize BKV-K9/NBKV work.

The plan is content-addressed and binds the exact target protocol SHA-256, campaign ID, H0/H1, paired-summary schema, primary performance metric and direction, minimum effect threshold, quality guard and noninferiority margin, uncertainty method and confidence level, missing-pair policy, multiplicity policy, and holdout policy.

The v1 contract is deliberately fail-closed:

- the plan must preserve the protocol H0/H1 verbatim;
- the quality metric and holdout policy must match the protocol;
- every preregistered seed/repetition pair is required for any later promotion decision;
- incomplete paired evidence uses the fixed `fail_closed` policy;
- tuning is never permitted through the analysis plan;
- the caller must explicitly declare that the plan was created before outcome inspection;
- the primary performance metric must be numeric and already belong to the frozen target metric surface;
- bootstrap uncertainty requires a frozen resample count and seed; a paired t interval cannot carry bootstrap parameters;
- the plan contains no campaign outcome, p-value, confidence interval, winner, BKV-K9 authorization or protected-holdout opening.

The v1 promotion expression is only a frozen future evaluation boundary:

`all_required_pairs_complete_and_primary_interval_meets_effect_and_quality_guard_passes`

No evaluator is implemented by this contract. In particular, this PR does not choose a real campaign threshold, confidence level, bootstrap seed, quality tolerance or primary metric. Those values must be preregistered for a concrete campaign before its qualifying outcomes are inspected.

Use:

```bash
python3 tools/verify_bikv_target_analysis_plan.py protocol.json analysis-plan.json
```

Both inputs must use the repositories' canonical JSON encodings. On success the verifier prints the content-derived plan SHA-256, bound protocol SHA-256 and campaign ID.

This infrastructure does not satisfy issue #111 by itself. BKV-K9 remains blocked until a concrete target-host/model campaign has retained the required measurements and a separately implemented evaluator applies the already-frozen analysis plan without tuning or outcome-dependent rule changes.
