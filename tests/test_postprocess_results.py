from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


class PostprocessScriptTests(unittest.TestCase):
    def test_postprocess_matches_snapshot(self) -> None:
        raw = PROJECT_ROOT / "tests/data/postprocess/sample_raw.jsonl"
        expected = (PROJECT_ROOT / "tests/data/postprocess/sample_filtered.jsonl").read_text()
        blacklist = PROJECT_ROOT / "tests/data/postprocess/sample_blacklist.txt"

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            cmd = [
                PYTHON,
                "scripts/postprocess_results.py",
                "--inputs",
                str(raw),
                "--dataset",
                "biggen_bench",
                "--blacklist-file",
                str(blacklist),
                "--output-dir",
                str(output_dir),
                "--skip-validator",
                "--skip-metrics",
            ]
            subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)

            produced = (output_dir / raw.name).read_text()
            self.assertEqual(produced, expected)


if __name__ == "__main__":
    unittest.main()
