"""Regression tests for repository docs and Windows launchers.

Only the standard library is used, so these tests run without torch,
pyarrow or a GPU:

    python -m unittest tests.test_docs_and_launchers -v
"""

from pathlib import Path
import re
import unittest


SOURCE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = SOURCE_ROOT.parents[1]
README = WORKSPACE_ROOT / "README.md"
SMOKE = WORKSPACE_ROOT / "run_local_smoke.ps1"
REPRODUCE = WORKSPACE_ROOT / "reproduce_mmimdb.py"


class ReadmeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = README.read_text(encoding="utf-8")
        cls.lines = cls.text.splitlines()

    def test_goal_section_is_not_duplicated(self):
        self.assertEqual(self.text.count("本仓库当前的正式目标"), 1)

    def test_fenced_code_blocks_are_balanced(self):
        fences = [line for line in self.lines if line.strip().startswith("```")]
        self.assertEqual(
            len(fences) % 2, 0, msg="README contains an unbalanced code fence"
        )

    def test_section_numbers_are_sequential(self):
        numbers = []
        for line in self.lines:
            match = re.match(r"^## (\d+)\. ", line)
            if match:
                numbers.append(int(match.group(1)))
        self.assertEqual(numbers, list(range(1, len(numbers) + 1)))

    def test_venv_python_example_expands_the_variable(self):
        self.assertNotIn("'$DCP_VENV_DIR/bin/python'", self.text)
        self.assertIn('export PYTHON_BIN="${DCP_VENV_DIR}/bin/python"', self.text)

    def test_shell_snippets_quote_empty_variable_defaults(self):
        self.assertNotIn("[ -n ${DCP_VENV_DIR:-} ]", self.text)
        self.assertNotIn("$DCP_VENV_DIR/bin/python ]", self.text)
        self.assertIn('[ -n "${DCP_VENV_DIR:-}" ]', self.text)


class LauncherTests(unittest.TestCase):
    def test_smoke_script_supports_environment_overrides(self):
        text = SMOKE.read_text(encoding="utf-8")
        for name in (
            "PYTHON_BIN",
            "DCP_CONDA_PREFIX",
            "MMIMDB_DATA_ROOT",
            "MISSING_TABLE_ROOT",
            "CLIP_CACHE_ROOT",
            "ORIGINAL_DCP_PATH",
            "DCP_SMOKE_LOG_DIR",
        ):
            with self.subTest(name=name):
                self.assertIn(name, text)
        self.assertIn("Push-Location", text)
        self.assertIn("Pop-Location", text)

    def test_smoke_script_reports_every_missing_path(self):
        text = SMOKE.read_text(encoding="utf-8")
        self.assertNotIn("'Required path not found.'", text)
        self.assertIn("Required inputs are missing:", text)

    def test_reproduce_launcher_reports_missing_inputs(self):
        text = REPRODUCE.read_text(encoding="utf-8")
        self.assertIn("Required inputs are missing:", text)
        for name in ("DCP_CODE_ROOT", "MMIMDB_DATA_ROOT", "MISSING_TABLE_ROOT"):
            with self.subTest(name=name):
                self.assertIn(name, text)


if __name__ == "__main__":
    unittest.main()
