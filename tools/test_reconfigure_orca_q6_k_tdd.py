"""TDD tests for tools/reconfigure_orca.py --quant Q6_K (no GPU, no network).

    python -m unittest tools.test_reconfigure_orca_q6_k_tdd
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import reconfigure_orca  # noqa: E402
import setup  # noqa: E402

SHARD = "Qwen3.8-Flash-Next-Uncensored-Q6_K-0000{}-of-00005.gguf"
SHARD1 = "/gguf/" + SHARD.format(1)
SHARD2 = "/gguf/" + SHARD.format(2)
MMPROJ = "/gguf/mmproj-Qwen3.8-Flash-Next-Uncensored-F16.gguf"


class OrcaQ6KConfigTest(unittest.TestCase):
    def get_config(self, *flags) -> dict:
        try:
            a = reconfigure_orca.parse_args(["--data-dir", "/data", "--gguf-dir", "/gguf", "--quant", "Q6_K", *flags])
        except SystemExit as e:
            self.fail(f"the flags were refused: {e}")
        return reconfigure_orca.get_config(a, Path("/eng"), ["/lib"])

    def test_q6_k_gives_the_ple_shard_the_ram_budget_and_auto_prefill(self):
        cfg = self.get_config("--host", "0.0.0.0", "--port", "19090", "--context", "250000", "--kv", "int8",
                              "--kv-streaming", "off", "--vision", "yes", "--resident-budget-gib", "70",
                              "--pool-workers", "13", "--api-key", "k")
        self.assertEqual(cfg["args"], [
            "--pack", "/data/packs/orca-q6_k", "--native", SHARD1, "--ple-gguf", SHARD2,
            "--expert-profile", str(setup.ROOT / "data" / "expert-profile.bin"), "--expert-cache", "auto",
            "--prefill", "auto", "--spec", "4", "--spec-min-p", "0.5", "--mtp", "/data/mtp/rt",
            "--max-context", "250000", "--kv", "int8", "--resident-budget-gib", "70", "--pool-workers", "13",
            "--vision", "--vram-reserve-mib", "700"])
        self.assertEqual(cfg["vision"], {"exe": "/eng/strata-vision", "mmproj": MMPROJ, "model": SHARD1, "gpu": True,
                                         "max_tokens": 1024})
        self.assertEqual({k: v for k, v in cfg.items() if k not in ("args", "vision")}, {
            "exe": "/eng/strata", "cwd": str(setup.ROOT), "tokenizer": "/data/packs/orca-q6_k/tokenizer",
            "model_name": "orcarouter-qwen3.8-flash-next-uncensored-q6_k",
            "log": str(setup.home() / "strata-orca-q6_k.log"), "lib_dirs": ["/lib"], "host": "0.0.0.0",
            "port": 19090, "api_key": "k", "env": {"STRATA_MMQ_KQ": "1"}})

    def test_vram_reserve_replaces_the_default_for_a_gpu_picture_reader(self):
        args = self.get_config("--vision", "yes", "--vram-reserve-mib", "3072")["args"]
        self.assertEqual(args[-3:], ["--vision", "--vram-reserve-mib", "3072"])


class OrcaQ6KMainTest(unittest.TestCase):
    """main() in a temporary install: setup.ROOT (and so setup.home()) points at it."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        patcher = mock.patch.object(setup, "ROOT", self.root)
        patcher.start()
        self.addCleanup(patcher.stop)
        names = ["engine/strata", "engine/strata-vision", "gguf/" + Path(MMPROJ).name, "data/mtp/rt/dense.bin",
                 *(f"gguf/{SHARD.format(n)}" for n in range(1, 6))]
        for name in names:
            (self.root / name).parent.mkdir(parents=True, exist_ok=True)
            (self.root / name).write_text("x")
        (self.root / "engine" / "BUILD.json").write_text(json.dumps({"lib_dirs": ["/lib"]}))
        self.pack = self.root / "data" / "packs" / "orca-q6_k"
        self.cfg_path = self.root / "strata-orca-q6_k.json"

    def run_main(self, *flags) -> str:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            result = reconfigure_orca.main(["--quant", "Q6_K", "--data-dir", str(self.root / "data"),
                                            "--gguf-dir", str(self.root / "gguf"), *flags])
        self.result = result
        return out.getvalue()

    def test_a_missing_last_shard_stops_before_packing(self):
        (self.root / "gguf" / SHARD.format(5)).unlink()
        out = io.StringIO()
        with mock.patch.object(reconfigure_orca.subprocess, "run") as run, contextlib.redirect_stdout(out), \
                self.assertRaises(SystemExit):
            reconfigure_orca.main(["--quant", "Q6_K", "--data-dir", str(self.root / "data"),
                                   "--gguf-dir", str(self.root / "gguf")])
        self.assertIn(SHARD.format(5), out.getvalue())
        self.assertNotIn(SHARD.format(4), out.getvalue())
        run.assert_not_called()

    def test_packs_shard_1_and_writes_its_own_config_and_run_script(self):
        def write_pack(*_args, **_kwargs):
            self.pack.mkdir(parents=True)
            (self.pack / "native_experts.txt").write_text("x")

        with mock.patch.object(reconfigure_orca.subprocess, "run", side_effect=write_pack) as run:
            try:
                self.run_main("--no-start", "--port", "19090", "--vision", "yes", "--no-browser")
            except SystemExit as e:
                self.fail(f"main stopped: {e}")
        self.assertEqual(run.call_args.args[0], [
            sys.executable, str(ROOT / "tools" / "iq_pack.py"), "--gguf", str(self.root / "gguf" / SHARD.format(1)),
            "--out", str(self.pack), "--compat-bf16"])
        cfg = json.loads(self.cfg_path.read_text())
        self.assertEqual(cfg["args"][:2], ["--pack", str(self.pack)])
        self.assertEqual(cfg["env"], {"STRATA_MMQ_KQ": "1"})
        self.assertIs(cfg["open_browser"], False)
        run_script = (self.root / "run-orca-q6_k.sh").read_text()
        self.assertIn(f'"--config" "{self.cfg_path}" "--port" "19090"', run_script)

    def write_pack(self):
        self.pack.mkdir(parents=True)
        (self.pack / "native_experts.txt").write_text("x")

    def test_starts_the_server_after_configuring_like_setup(self):
        self.write_pack()
        with mock.patch.object(setup, "start", return_value=7) as start:
            self.run_main("--port", "19090", "--no-browser")
        start.assert_called_once_with(self.cfg_path, 19090)
        self.assertEqual(self.result, 7)
        self.assertIs(json.loads(self.cfg_path.read_text())["open_browser"], False)


if __name__ == "__main__":
    unittest.main()
