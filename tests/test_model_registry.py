from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "apps" / "api"))

from app.services.model_registry import (  # noqa: E402
    ModelRegistryError,
    build_artifact_manifest,
    manifest_digest,
    resolve_artifact_path,
    verify_immutable_artifact_location,
    verify_artifact_manifest,
)
from model_registry_artifacts import create_model_registry_snapshot  # noqa: E402


class ModelRegistryTests(unittest.TestCase):
    @staticmethod
    def _write_artifacts(path: Path) -> None:
        path.mkdir(parents=True)
        (path / "training_metadata.json").write_text(
            json.dumps({"profile_name": "short_3d", "metrics": {"auc": 0.61}}) + "\n",
            encoding="utf-8",
        )
        (path / "inference_scores_latest.parquet").write_bytes(b"immutable-scores")
        (path / "feature_importance.csv").write_text("feature,importance_gain\nx,1\n", encoding="utf-8")

    def test_snapshot_is_immutable_and_self_describing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "quant_data"
            source = data_dir / "model_profiles" / "short_3d" / "models"
            self._write_artifacts(source)

            record = create_model_registry_snapshot(data_dir=data_dir, profile="short_3d")
            destination = Path(record["artifact_path"])

            self.assertEqual(record["market"], "CN")
            self.assertEqual(record["validation_status"], "pending")
            self.assertTrue((destination / "registry_record.json").is_file())
            self.assertEqual(build_artifact_manifest(destination), record["artifact_manifest"])
            self.assertNotEqual(destination, source)

    def test_manifest_verification_detects_artifact_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact_dir = root / "quant_data" / "models" / "v1"
            self._write_artifacts(artifact_dir)
            manifest = build_artifact_manifest(artifact_dir)
            settings = SimpleNamespace(project_root=root)
            model = {"artifact_path": "quant_data/models/v1", "artifact_manifest": manifest}

            self.assertEqual(verify_artifact_manifest(model, settings), artifact_dir)
            (artifact_dir / "inference_scores_latest.parquet").write_bytes(b"mutated")
            with self.assertRaisesRegex(ModelRegistryError, "checksum mismatch"):
                verify_artifact_manifest(model, settings)

    def test_artifact_path_cannot_escape_project_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = SimpleNamespace(project_root=Path(tmp))
            with self.assertRaisesRegex(ModelRegistryError, "escapes PROJECT_ROOT"):
                resolve_artifact_path("../outside", settings)

    def test_paper_artifact_must_use_versioned_registry_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            settings = SimpleNamespace(project_root=root, quant_dir=root / "quant_data")
            model = {
                "market": "CN",
                "model_version": "cn-medium-v1",
                "artifact_path": "quant_data/model_profiles/medium/models",
            }
            with self.assertRaisesRegex(ModelRegistryError, "artifact path is mutable"):
                verify_immutable_artifact_location(model, settings)

            immutable = root / "quant_data" / "model_registry" / "CN" / "cn-medium-v1"
            immutable.mkdir(parents=True)
            model["artifact_path"] = str(immutable.relative_to(root))
            self.assertEqual(verify_immutable_artifact_location(model, settings), immutable)

    def test_manifest_digest_is_order_independent(self) -> None:
        left = {"a": {"sha256": "1", "size": 1}, "b": {"sha256": "2", "size": 2}}
        right = {"b": {"size": 2, "sha256": "2"}, "a": {"size": 1, "sha256": "1"}}
        self.assertEqual(manifest_digest(left), manifest_digest(right))


if __name__ == "__main__":
    unittest.main()
