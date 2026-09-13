from pathlib import Path
from types import SimpleNamespace
import os
import random
import tempfile
import unittest

import torch

import run as run_module
from clip.datasets import missing_tables


class RunArtifactTests(unittest.TestCase):
    def test_artifact_directories_are_bound_to_logger_version(self):
        builder = getattr(run_module, "artifact_directories")
        logger = SimpleNamespace(
            version=3,
            log_dir=os.path.join("logs", "experiment", "version_3"),
        )
        checkpoint_dir, snapshot_dir = builder(
            logger, os.path.join("workspace", "project"), "experiment"
        )
        self.assertEqual(
            os.path.normpath(checkpoint_dir),
            os.path.normpath(os.path.join(logger.log_dir, "checkpoints")),
        )
        self.assertEqual(
            os.path.normpath(snapshot_dir),
            os.path.normpath(
                os.path.join(
                    "workspace", "project", "result_model_files",
                    "experiment", "version_3",
                )
            ),
        )

    def test_source_snapshot_copies_reachable_python_without_cache(self):
        snapshot = getattr(run_module, "snapshot_source_tree")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "project"
            target = Path(temporary) / "snapshot"
            (root / "clip" / "modules").mkdir(parents=True)
            (root / "clip" / "__pycache__").mkdir(parents=True)
            (root / "run.py").write_text("run", encoding="utf-8")
            (root / "requirements.txt").write_text("deps", encoding="utf-8")
            (root / "clip" / "modules" / "model.py").write_text(
                "model", encoding="utf-8"
            )
            (root / "clip" / "notes.txt").write_text("skip", encoding="utf-8")
            (root / "clip" / "__pycache__" / "model.pyc").write_bytes(b"skip")
            snapshot(str(root), str(target))
            self.assertTrue((target / "run.py").is_file())
            self.assertTrue((target / "requirements.txt").is_file())
            self.assertTrue((target / "clip" / "modules" / "model.py").is_file())
            self.assertFalse((target / "clip" / "notes.txt").exists())
            self.assertFalse((target / "clip" / "__pycache__").exists())


class MissingTableTests(unittest.TestCase):
    def test_identity_changes_with_seed_and_both_ratio(self):
        filename = getattr(missing_tables, "missing_table_filename")
        first = filename("mmimdb_train", "both", .7, .5, 0)
        changed_seed = filename("mmimdb_train", "both", .7, .5, 1)
        changed_mix = filename("mmimdb_train", "both", .7, .25, 0)
        self.assertNotEqual(first, changed_seed)
        self.assertNotEqual(first, changed_mix)

    def test_generation_is_deterministic_and_does_not_consume_global_random(self):
        loader = getattr(missing_tables, "load_or_create_missing_table")
        with tempfile.TemporaryDirectory() as first_root, tempfile.TemporaryDirectory() as second_root:
            before = random.getstate()
            first, first_metadata = loader(
                first_root, "mmimdb_train", "train", 100,
                .7, "both", .5, seed=9,
            )
            after = random.getstate()
            second, second_metadata = loader(
                second_root, "mmimdb_train", "train", 100,
                .7, "both", .5, seed=9,
            )
        self.assertEqual(before, after)
        self.assertTrue(torch.equal(first, second))
        self.assertFalse(first_metadata["legacy"])
        self.assertEqual(first_metadata["identity"], second_metadata["identity"])

    def test_legacy_seed_zero_table_is_explicitly_reused(self):
        loader = getattr(missing_tables, "load_or_create_missing_table")
        with tempfile.TemporaryDirectory() as root:
            legacy = torch.tensor([0., 1., 2., 0.])
            legacy_path = Path(root) / "mmimdb_train_missing_both_07.pt"
            torch.save(legacy, str(legacy_path))
            loaded, metadata = loader(
                root, "mmimdb_train", "train", 4,
                .7, "both", .5, seed=0,
                allow_legacy_seed0=True,
            )
            self.assertTrue(torch.equal(loaded, legacy))
            self.assertTrue(metadata["legacy"])
            self.assertEqual(Path(metadata["path"]), legacy_path)

    def test_invalid_existing_table_raises_instead_of_exiting(self):
        loader = getattr(missing_tables, "load_or_create_missing_table")
        filename = getattr(missing_tables, "missing_table_filename")
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / filename("mmimdb_train", "both", .7, .5, 2)
            torch.save(
                {
                    "table": torch.tensor([0, 4]),
                    "metadata": {
                        "identity": "invalid",
                        "dataset": "mmimdb_train",
                        "split": "train",
                        "total_num": 2,
                        "missing_ratio": .7,
                        "missing_type": "both",
                        "both_ratio": .5,
                        "seed": 2,
                    },
                },
                str(path),
            )
            with self.assertRaises(RuntimeError):
                loader(
                    root, "mmimdb_train", "train", 2,
                    .7, "both", .5, seed=2,
                )


if __name__ == "__main__":
    unittest.main()
