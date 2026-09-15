# Mandatory R2 global verification before publication

The current `kvlab.prospect_smollm2_r2_suite` launcher does not publish a staged
result solely because its three campaigns passed individual verification. It
now also invokes the independent Rust command `verify-kv-campaign-suite-r2` on
the complete staged directory, before the final rename.

## Separate immutable verifier

The historical per-campaign verifier remains ProspectEngine
`298acdc91682ef1d09914b6f964e8934828825c0`. The additional publication verifier is
ProspectEngine `ca9685cd98f3a0a23e8c4f7e368736bb3aa28d0c`, which contains the R2
whole-suite consumer and the position-native metric delta guard.

Both checkouts are independent detached worktrees. They are built with explicit
Rust 1.89 and `--locked --release`, in separate target directories. All commits
must already be available in the supplied repositories. No moving `main` ref is
substituted for a pinned revision. The source of the launcher itself should also
be recorded by the operator or execution harness.

The launcher still reads the exact R2 preregistration, uses its declared KVLab
executor and NNIS runtime, and preserves the original R1/R2 experiment JSON. The
historical per-campaign verifier field in `suite-manifest.json` is not rewritten.
The result and preflight schemas and the strict v1 suite file set are unchanged.

## Publication sequence

1. Build the pinned backend, historical per-campaign verifier and additional global verifier.
2. Preflight all three campaign inputs without model execution.
3. Execute each campaign, verify it individually and write its summary inside staging.
4. Write the unchanged v1 suite manifest.
5. Run the actual global verifier on staging. Require exit status zero and a valid,
   finite JSON summary tied to the exact staged manifest, identities and campaign order.
6. Emit a separate consistency-check receipt to stderr.
7. Recheck that the destination is absent, then rename staging to publish the suite.

The Rust consumer remains authoritative for frozen input identities, metric deltas,
matched budgets and exact baseline agreement across budgets. The Python gate does
not implement a second numerical comparison or select a winning policy.

A nonzero verifier exit, timeout, malformed/duplicate/non-finite JSON, mismatched
summary identity or interrupt prevents publication. Existing staging/lock cleanup
applies to these failures. There is no flag to skip the global check in this
launcher revision. `--preflight-only` builds all three binaries, but does not
invoke a model, run the global observed-suite check, or publish an output directory.

## Receipt and logging

The receipt schema is `kvlab.r2-publication-gate-receipt/v1`. It records
`phase=stage_verified`, `evidence_kind=consistency_check_only`, the exact global
verifier revision, the SHA-256 of the binary file read before invocation, the
suite-manifest SHA-256 and the SHA-256 of the verifier's exact stdout bytes.

The receipt is one canonical JSON line on stderr, alongside ordinary build/tool
diagnostics. stdout retains the original result manifest or preflight JSON only.
No additional file is inserted into the strict R2 suite. Preserve the launch log
outside the output directory to retain the receipt. The launcher does not claim
the log has been durably retained, or that a `stage_verified` receipt proves the
subsequent rename succeeded. It is not a GPU, hardware or model authentication.

The verifier subprocess uses an argv list, no shell, closed stdin and a 120-second
timeout. The intended manifest is bounded to 65,536 bytes. Captured verification
stdout is rejected above 1,048,576 bytes **after capture**; this is not a hard
subprocess-memory limit. The binary and directories must be trusted and remain
unmodified. These checks are not a race-free filesystem sandbox or a crash/power-loss
durability guarantee.

## Tests

```bash
python3 -m unittest discover -s tests -p 'test_prospect_r2_publication_gate.py' -v
python3 -m unittest discover -s tests -p 'test_smollm2_r2_suite.py' -v
```

The gate tests use mocked subprocess responses. Launcher regressions prove that
global rejection and interruption remove staging and prevent publication, and
that global verification occurs after the complete manifest exists but before
rename. A cross-repository test must additionally exercise the real Rust binary
against valid and coherently baseline-drifted synthetic suites. None of these
tests are observed language-model or CUDA results.
