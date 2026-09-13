from pathlib import Path
import re
import unittest


SOURCE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = SOURCE_ROOT.parents[1]


class PortablePathTests(unittest.TestCase):
    def test_active_launch_files_do_not_hardcode_machine_paths(self):
        files = [
            SOURCE_ROOT / "clip" / "config.py",
            SOURCE_ROOT / "run_aistation.sh",
            SOURCE_ROOT / "setup_aistation_env.sh",
            SOURCE_ROOT / "aistation_preflight.sh",
            SOURCE_ROOT / "tools" / "aistation_preflight.py",
            WORKSPACE_ROOT / "reproduce_mmimdb.py",
            WORKSPACE_ROOT / "start_dcp_reproduction.ps1",
            WORKSPACE_ROOT / "verify_mmimdb_arrow.py",
            WORKSPACE_ROOT / "tools" / "run_dcp_loss_audit.py",
            WORKSPACE_ROOT / "tools" / "probe_dcp_trainer_loader.py",
            WORKSPACE_ROOT / "tools" / "inspect_dcp_loader_lengths.py",
            WORKSPACE_ROOT / "tools" / "audit_dcp_artifacts.py",
        ]
        machine_path = re.compile(r"[A-Za-z]:[\\/]|/data2/")
        for path in files:
            with self.subTest(path=path):
                text = path.read_text(encoding="utf-8")
                self.assertIsNone(machine_path.search(text))

    def test_aistation_launcher_uses_environment_or_relative_defaults(self):
        launcher = (SOURCE_ROOT / "run_aistation.sh").read_text(
            encoding="utf-8"
        )
        for name in (
            "MMIMDB_DATA_ROOT",
            "MISSING_TABLE_ROOT",
            "CLIP_CACHE_ROOT",
            "ORIGINAL_DCP_PATH",
            "TRAIN_LOG_DIR",
        ):
            with self.subTest(name=name):
                self.assertRegex(
                    launcher,
                    r"\$\{" + name + r":-\.\./\.\./",
                )
        self.assertIn('PYTHON_BIN="${PYTHON_BIN:-python}"', launcher)
        self.assertIn('cd "$SCRIPT_DIR"', launcher)

    def test_aistation_environment_bundle_is_explicit(self):
        setup = (SOURCE_ROOT / "setup_aistation_env.sh").read_text(
            encoding="utf-8"
        )
        preflight = (SOURCE_ROOT / "aistation_preflight.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("DCP_VENV_DIR", setup)
        self.assertIn("--system-site-packages", setup)
        self.assertIn("requirements_aistation.txt", setup)
        self.assertIn("tools/aistation_preflight.py", preflight)

    def test_aistation_requirements_leave_cuda_stack_to_the_image(self):
        lines = (
            SOURCE_ROOT / "requirements_aistation.txt"
        ).read_text(encoding="utf-8").splitlines()
        requirements = [
            line.strip() for line in lines
            if line.strip() and not line.lstrip().startswith("#")
        ]
        names = [
            requirement.split("==", 1)[0].lower()
            for requirement in requirements
        ]
        self.assertEqual(len(names), len(set(names)))
        self.assertIn("pillow", names)
        self.assertNotIn("torch", names)
        self.assertNotIn("torchvision", names)
        self.assertNotIn("torchaudio", names)


if __name__ == "__main__":
    unittest.main()
