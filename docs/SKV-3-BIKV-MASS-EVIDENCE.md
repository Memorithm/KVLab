# SKV-3 BIKV mass-aware evidence v1

Status: implementation slice; no representative-model result.

`kvlab.bikv_mass_evidence.BikvMassEvidenceV1` composes the existing canonical
Boolean-KV handoff with an explicit page-level reference score oracle.

The evidence records:

- exact BIKV handoff and admitted page IDs;
- declared page-score aggregation semantics;
- reference page scores;
- top-k recall;
- retained/omitted softmax mass;
- mass of missed top-k pages;
- candidate density.

The dense score row is permanently marked `evaluation_oracle_only=true`.
Decoding rejects a record that attempts to relabel it as deployable evidence.

This slice does not define the correct page aggregation for a real model.
Future SKV-2/SKV-3 experiments must preregister whether a page score is, for
example, max token logit, log-sum-exp over visible token logits, or another
explicit statistic, then compare it to token-level downstream quality.

The purpose is to make the existing BIKV candidate set compatible with the
SLHAv2 mass-aware evidence contract without giving the router hidden access to
dense attention.
