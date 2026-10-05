"""TDD tests for tools/reconfigure_orca.py (no GPU, no network).

    python -m unittest tools.test_reconfigure_orca_tdd
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

SHARD1 = "/gguf/Qwen3.8-Flash-Next-Uncensored-IQ3_XXS-00001-of-00002.gguf"
MMPROJ = "/gguf/mmproj-Qwen3.8-Flash-Next-Uncensored-F16.gguf"


def get_test_config(*flags) -> dict:
    a = reconfigure_orca.parse_args(["--data-dir", "/data", "--gguf-dir", "/gguf", *flags])
    return reconfigure_orca.get_config(a, Path("/eng"), ["/lib"])


class OrcaConfigTest(unittest.TestCase):
    def test_user_flags_give_the_verified_config(self):
        cfg = get_test_config("--host", "0.0.0.0", "--port", "19090", "--context", "250000", "--kv", "int8",
                              "--kv-streaming", "off", "--vision", "yes", "--api-key", "k")
        self.assertEqual(cfg["args"], [
            "--pack", "/data/packs/orca-iq3_xxs", "--native", SHARD1, "--ple-gguf", SHARD1,
            "--expert-profile", str(setup.ROOT / "data" / "expert-profile.bin"), "--expert-cache", "auto",
            "--prefill", "512", "--spec", "4", "--spec-min-p", "0.5", "--mtp", "/data/mtp/rt",
            "--max-context", "250000", "--kv", "int8", "--vision", "--vram-reserve-mib", "700"])
        self.assertEqual(cfg["vision"], {"exe": "/eng/strata-vision", "mmproj": MMPROJ, "model": SHARD1, "gpu": True,
                                         "max_tokens": 1024})
        self.assertEqual({k: v for k, v in cfg.items() if k not in ("args", "vision")}, {
            "exe": "/eng/strata", "cwd": str(setup.ROOT), "tokenizer": "/data/packs/orca-iq3_xxs/tokenizer",
            "model_name": "orcarouter-qwen3.8-flash-next-uncensored-iq3_xxs",
            "log": str(setup.home() / "strata-orca-iq3_xxs.log"), "lib_dirs": ["/lib"], "host": "0.0.0.0",
            "port": 19090, "api_key": "k"})

    def test_kv_streaming_on_keeps_the_kv_cache_in_ram_from_64k(self):
        self.assertEqual(get_test_config("--context", "250000", "--kv-streaming", "on")["args"][-2:],
                         ["--kv-resident", "32768"])
        self.assertNotIn("--kv-resident", get_test_config("--context", "32768", "--kv-streaming", "on")["args"])

    def test_context_over_the_trained_262144_is_refused(self):
        self.assertIn("--max-context", get_test_config("--context", "262144")["args"])
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
            get_test_config("--context", "262145")


class OrcaMainTest(unittest.TestCase):
    """main() in a temporary install: setup.ROOT (and so setup.home()) points at it."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        patcher = mock.patch.object(setup, "ROOT", self.root)
        patcher.start()
        self.addCleanup(patcher.stop)
        for name in ("engine/strata", "engine/strata-vision", "gguf/" + Path(SHARD1).name,
                     "gguf/Qwen3.8-Flash-Next-Uncensored-IQ3_XXS-00002-of-00002.gguf", "gguf/" + Path(MMPROJ).name,
                     "data/packs/orca-iq3_xxs/native_experts.txt", "data/packs/orca-iq3_xxs/tokenizer/vocab.json",
                     "data/mtp/rt/dense.bin"):
            (self.root / name).parent.mkdir(parents=True, exist_ok=True)
            (self.root / name).write_text("x")
        (self.root / "engine" / "BUILD.json").write_text(json.dumps({"lib_dirs": ["/lib"]}))
        self.cfg_path = self.root / "strata-orca-iq3_xxs.json"

    def run_main(self, *flags):
        with contextlib.redirect_stdout(io.StringIO()):
            reconfigure_orca.main(["--data-dir", str(self.root / "data"), "--gguf-dir", str(self.root / "gguf"),
                                   "--no-start", *flags])

    def test_writes_the_config_and_run_script_and_keeps_the_users_keys(self):
        self.cfg_path.write_text(json.dumps({"args": [], "sampling": {"temperature": 0.6}}))
        self.run_main("--port", "19090", "--vision", "yes", "--no-browser")
        cfg = json.loads(self.cfg_path.read_text())
        self.assertEqual(cfg["args"][:2], ["--pack", str(self.root / "data" / "packs" / "orca-iq3_xxs")])
        self.assertEqual(cfg["vision"]["exe"], str(self.root / "engine" / "strata-vision"))
        self.assertEqual(cfg["lib_dirs"], ["/lib"])
        self.assertEqual(cfg["sampling"], {"temperature": 0.6})
        self.assertIs(cfg["open_browser"], False)
        self.assertTrue((self.root / "strata-orca-iq3_xxs.json.bak").is_file())
        run = (self.root / "run-orca-iq3_xxs.sh").read_text()
        self.assertIn(f'"--config" "{self.cfg_path}" "--port" "19090"', run)
        self.assertNotIn("--open", run)

    def test_a_missing_pack_is_made_with_iq_pack_compat_bf16(self):
        pack = self.root / "data" / "packs" / "orca-iq3_xxs"
        (pack / "native_experts.txt").unlink()
        with mock.patch.object(reconfigure_orca.subprocess, "run",
                               side_effect=lambda *a, **k: (pack / "native_experts.txt").write_text("x")) as run:
            self.run_main()
        run.assert_called_once()
        self.assertEqual(run.call_args.args[0], [
            sys.executable, str(ROOT / "tools" / "iq_pack.py"), "--gguf", str(self.root / "gguf" / Path(SHARD1).name),
            "--out", str(pack), "--compat-bf16"])
        self.assertTrue(self.cfg_path.is_file())

    def test_a_missing_mmproj_is_downloaded_pinned_and_a_wrong_hash_stops(self):
        mmproj = self.root / "gguf" / Path(MMPROJ).name
        mmproj.unlink()
        with mock.patch.object(setup, "download", side_effect=lambda url, dst, *a: dst.write_text("bad")) as download, \
                self.assertRaises(SystemExit):
            self.run_main("--vision", "yes")
        download.assert_called_once()
        self.assertEqual(download.call_args.args[:2], (
            f"{setup.hf_endpoint()}/orcarouter/Qwen3.8-Flash-Next-Uncensored-GGUF/resolve/"
            f"d27c0a15447dd92ae7c90919352c6810496945b1/{mmproj.name}", mmproj))
        self.assertFalse(mmproj.exists())
        self.assertFalse(self.cfg_path.exists())


if __name__ == "__main__":
    unittest.main()
