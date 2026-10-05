"""TDD tests: tools/reconfigure_orca.py runs in the 0.2.16 layout (setup.py and iq_pack.py live in terminal/).

    terminal/.venv/bin/python -m unittest tools.test_reconfigure_orca_layout_tdd
"""
from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class OrcaLayoutTest(unittest.TestCase):
    def test_the_script_starts_from_the_repo_root(self):
        run = subprocess.run([sys.executable, str(ROOT / "tools" / "reconfigure_orca.py"), "--help"],
                             cwd=ROOT, capture_output=True, text=True, check=False)
        self.assertEqual(run.returncode, 0, run.stderr)


if __name__ == "__main__":
    unittest.main()
