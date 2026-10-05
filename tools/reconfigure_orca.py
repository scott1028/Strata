"""tools/reconfigure_orca.py - setup.sh's reconfigure for OrcaRouter's Flash-Next Uncensored GGUFs (docs/ORCA.md).

setup.py has no Orca family, so this writes the same run config and run script by hand, with setup's own helpers,
then starts the model as setup does (--no-start: configure only).

    .venv/bin/python tools/reconfigure_orca.py --data-dir DIR --gguf-dir DIR [--quant Q6_K] [--context N] ...
"""
from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402

REPO = "orcarouter/Qwen3.8-Flash-Next-Uncensored-GGUF"
REVISION = "d27c0a15447dd92ae7c90919352c6810496945b1"
MMPROJ = "mmproj-Qwen3.8-Flash-Next-Uncensored-F16.gguf"
MMPROJ_SHA256 = "f0f352a97a62a057f3aecdb597cac664762cea2ca23f7b16ec92eee28c5572d9"   # REPO at REVISION
TRAINED_CONTEXT = 262144                               # qwen4exp.context_length in shard 1
QUANTS = {
    "IQ3_XXS": {"shards": 2, "ple_shard": 1, "prefill": "512"},
    # Q6_K: the expert types of Unsloth's UD-Q6_K_XL, so setup's prefill and K-quant prompt kernels for that file.
    # No STRATA_HC_Q8: this file's hc_*_down and hc_*_inject are Q6_K, not Q8_0.
    "Q6_K": {"shards": 5, "ple_shard": 2, "prefill": "auto", "env": {"STRATA_MMQ_KQ": "1"}},
}


def get_shard_name(quant: str, number: int) -> str:
    return f"Qwen3.8-Flash-Next-Uncensored-{quant}-{number:05d}-of-{QUANTS[quant]['shards']:05d}.gguf"


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, required=True, help="holds packs/ and mtp/rt (setup's --data-dir)")
    ap.add_argument("--gguf-dir", type=Path, required=True, help="the folder with all the shards of --quant")
    ap.add_argument("--quant", choices=list(QUANTS), default="IQ3_XXS")
    ap.add_argument("--resident-budget-gib", type=int,
                    help="GiB of experts kept in RAM; needed when the experts do not fit the RAM (Q6_K: 101 GiB)")
    ap.add_argument("--pool-workers", type=int)
    ap.add_argument("--vram-reserve-mib", type=int, default=setup.VISION["gpu"]["reserve_mib"],
                    help="with --vision yes: VRAM the expert cache leaves free for the GPU picture reader")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--context", type=int, default=32768)
    ap.add_argument("--kv", choices=["int8", "q4_0", "k8v4", "fp16"], default="int8")
    ap.add_argument("--kv-streaming", choices=["on", "off"], default="off")
    ap.add_argument("--vision", choices=["yes", "no"], default="no")
    ap.add_argument("--api-key")
    ap.add_argument("--no-browser", dest="browser", action="store_false", default=None)
    ap.add_argument("--no-start", action="store_true", help="configure only, do not start the model")
    a = ap.parse_args(argv)
    if a.context > TRAINED_CONTEXT:                    # checked before any packing or download
        setup.fail(f"--context {a.context} is over Orca's trained {TRAINED_CONTEXT} tokens",
                   "a longer context needs rope scaling, which this script does not set")
    return a


def get_config(a: argparse.Namespace, engine: Path, lib_dirs: list) -> dict:
    """The run config docs/ORCA.md describes, for these options."""
    quant = QUANTS[a.quant]
    tag = f"orca-{a.quant.lower()}"
    pack = a.data_dir / "packs" / tag
    shard1 = str(a.gguf_dir / get_shard_name(a.quant, 1))
    ple = str(a.gguf_dir / get_shard_name(a.quant, quant["ple_shard"]))
    args = ["--pack", str(pack), "--native", shard1, "--ple-gguf", ple,
            "--expert-profile", str(setup.ROOT / "data" / "expert-profile.bin"), "--expert-cache", "auto",
            "--prefill", quant["prefill"], "--spec", "4", "--spec-min-p", "0.5",
            "--mtp", str(a.data_dir / "mtp" / "rt"), "--max-context", str(a.context), "--kv", a.kv]
    if a.kv_streaming == "on" and a.context >= 65536:   # setup's rule: only from 64K up
        args += ["--kv-resident", "32768"]
    if a.resident_budget_gib:
        args += ["--resident-budget-gib", str(a.resident_budget_gib)]
    if a.pool_workers:
        args += ["--pool-workers", str(a.pool_workers)]
    cfg = {"exe": str(engine / setup.EXE), "args": args, "cwd": str(setup.ROOT), "tokenizer": str(pack / "tokenizer"),
           "model_name": f"orcarouter-qwen3.8-flash-next-uncensored-{a.quant.lower()}",
           "log": str(setup.home() / f"strata-{tag}.log"), "lib_dirs": lib_dirs, "host": a.host, "port": a.port}
    if "env" in quant:
        cfg["env"] = quant["env"]
    if a.api_key:
        cfg["api_key"] = a.api_key
    if a.vision == "yes":
        args += ["--vision", "--vram-reserve-mib", str(a.vram_reserve_mib)]
        cfg["vision"] = {"exe": str(engine / setup.VEXE), "mmproj": str(a.gguf_dir / MMPROJ), "model": shard1,
                         "gpu": True, "max_tokens": setup.VISION["gpu"]["max_tokens"]}
    return cfg


def main(argv=None) -> int:
    a = parse_args(argv)
    engine = setup.ROOT / "engine"
    if not (engine / setup.EXE).is_file():
        setup.fail(f"no engine at {engine / setup.EXE}", "run ./setup.sh --setup once for any model to get it")
    shards = [a.gguf_dir / get_shard_name(a.quant, n) for n in range(1, QUANTS[a.quant]["shards"] + 1)]
    missing = [p for p in shards if not p.is_file()]
    if missing:
        setup.fail("missing: " + ", ".join(map(str, missing)),
                   f"download every {a.quant} shard of {REPO} first")
    if not (a.data_dir / "mtp" / "rt").is_dir():
        setup.fail(f"no draft head at {a.data_dir / 'mtp' / 'rt'}", "make it as docs/ORCA.md (Preparation) says")
    tag = f"orca-{a.quant.lower()}"
    pack = a.data_dir / "packs" / tag
    if not (pack / "native_experts.txt").is_file():    # iq_pack writes it last (#172)
        setup.say(f"packing {shards[0].name} into {pack} (docs/ORCA.md) ...")
        subprocess.run([sys.executable, str(Path(__file__).with_name("iq_pack.py")), "--gguf",
                        str(shards[0]), "--out", str(pack), "--compat-bf16"], check=True)
    if not (pack / "native_experts.txt").is_file():
        setup.fail(f"packing did not finish: {pack}")
    mmproj = a.gguf_dir / MMPROJ
    if a.vision == "yes" and not mmproj.is_file():
        setup.download(f"{setup.hf_endpoint()}/{REPO}/resolve/{REVISION}/{MMPROJ}", mmproj)
        with open(mmproj, "rb") as f:
            sha = hashlib.file_digest(f, "sha256").hexdigest()
        if sha != MMPROJ_SHA256:
            mmproj.unlink()
            mmproj.with_name(mmproj.name + ".done").unlink(missing_ok=True)
            setup.fail(f"{MMPROJ}: SHA-256 {sha} is not the pinned {MMPROJ_SHA256}",
                       "run it again to download it again")
    cfg = get_config(a, engine, setup.engine_lib_dirs(engine))
    if a.browser is not None:
        cfg["open_browser"] = a.browser
    cfg_path = setup.home() / f"strata-{tag}.json"
    setup.write_setup_config(cfg_path, cfg)
    script = setup.write_run_script(tag, cfg_path, a.port, cfg.get("open_browser") is not False)
    setup.ok(f"config: {cfg_path}")
    setup.ok(f"start it with: {script}")
    return 0 if a.no_start else setup.start(cfg_path, a.port)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        setup.say("\nstopped.")
        sys.exit(1)
