import unittest

from kvlab.cross_model import (
    CrossModelTransferProtocol,
    KVGeometry,
    MapperBaseline,
    ModelRevision,
)
from kvlab.k9_binding import ModelRole, bind_capture_manifest
from kvlab.k9_capture import CaptureManifest, TensorCapture, digest_bytes


class K9BindingTests(unittest.TestCase):
    def protocol(self) -> CrossModelTransferProtocol:
        geometry = KVGeometry(kv_heads=2, head_dim=64)
        return CrossModelTransferProtocol(
            source=ModelRevision("family/source", "source-r1", "tok-r1", geometry),
            target=ModelRevision("family/target", "target-r2", "tok-r2", geometry),
            mapper=MapperBaseline.RIDGE,
            ridge_alpha=0.1,
            source_layers_per_target=2,
            remove_rope_from_keys=True,
            calibration_ids=("cal-a", "cal-b"),
            final_holdout_ids=("hold-a", "hold-b"),
        )

    def manifest(
        self,
        role: ModelRole,
        split: str = "calibration",
        sample_ids: tuple[str, ...] = ("cal-a", "cal-b"),
        rope_removed: bool = True,
    ) -> CaptureManifest:
        protocol = self.protocol()
        model = protocol.source if role is ModelRole.SOURCE else protocol.target
        captures = []
        for sample_id in sample_ids:
            captures.append(
                TensorCapture(
                    sample_id=sample_id,
                    layer=1,
                    kind="key",
                    shape=(1, 2, 64),
                    dtype="float16",
                    content_sha256=digest_bytes([role.value.encode(), sample_id.encode(), b"key"]),
                    rope_removed=rope_removed,
                )
            )
            captures.append(
                TensorCapture(
                    sample_id=sample_id,
                    layer=1,
                    kind="value",
                    shape=(1, 2, 64),
                    dtype="float16",
                    content_sha256=digest_bytes([role.value.encode(), sample_id.encode(), b"value"]),
                    rope_removed=False,
                )
            )
        return CaptureManifest(
            experiment_id="k9-binding-fixture",
            repository_commit="684d3d92f091bd5c42bed754353bb133b8630260",
            model_id=model.model_id,
            model_revision=model.revision,
            tokenizer_revision=model.tokenizer_revision,
            runtime="fixture-runtime",
            split=split,
            captures=tuple(captures),
        )

    def test_source_and_target_calibration_manifests_bind(self) -> None:
        protocol = self.protocol()
        bind_capture_manifest(self.manifest(ModelRole.SOURCE), protocol, ModelRole.SOURCE)
        bind_capture_manifest(self.manifest(ModelRole.TARGET), protocol, ModelRole.TARGET)

    def test_revision_mismatch_fails_closed(self) -> None:
        protocol = self.protocol()
        target_manifest = self.manifest(ModelRole.TARGET)
        with self.assertRaisesRegex(ValueError, "revision mismatch"):
            bind_capture_manifest(target_manifest, protocol, ModelRole.SOURCE)

    def test_sample_set_must_exactly_match_frozen_split(self) -> None:
        with self.assertRaisesRegex(ValueError, "sample ids do not match"):
            bind_capture_manifest(
                self.manifest(ModelRole.SOURCE, sample_ids=("cal-a",)),
                self.protocol(),
                ModelRole.SOURCE,
            )

    def test_final_holdout_uses_only_frozen_holdout_ids(self) -> None:
        manifest = self.manifest(
            ModelRole.SOURCE,
            split="final-holdout",
            sample_ids=("hold-a", "hold-b"),
        )
        bind_capture_manifest(manifest, self.protocol(), ModelRole.SOURCE)

    def test_required_rope_removal_is_enforced(self) -> None:
        with self.assertRaisesRegex(ValueError, "RoPE-removed"):
            bind_capture_manifest(
                self.manifest(ModelRole.SOURCE, rope_removed=False),
                self.protocol(),
                ModelRole.SOURCE,
            )


if __name__ == "__main__":
    unittest.main()
