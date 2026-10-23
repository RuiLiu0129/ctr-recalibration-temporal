"""Tier-D staging (2026-10-22): processed dirs + configs for the seed-43/44
slice-length sweep (Experiment 1).

The temporal prepare is seed-independent (deterministic stride train
sample, encoder fit on train rows only, day/hour-defined calib+test
splits); only the manifest fingerprint carries the config seed. Tier-C
used exactly this staging for the origin-2 seed runs (copy the seed-42
processed dir, patch fingerprint seed + encoder_path, md5-verify). We do
the same from the seed-42 sweep dirs data/processed_sweep_o{o}_h{h} ->
data/processed_sweep_o{o}_h{h}_s{s}, and write configs
config_sweep_o{o}_h{h}_s{s}.yaml (seed + results_dir changed only).

Idempotent: existing staged dirs with a matching patched fingerprint are
left untouched.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.data_pipeline import _fingerprint  # noqa: E402
from src.run_experiments import load_config  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
HOURS = (1, 3, 12, 24)
ORIGINS = (1, 2)
SEEDS = (43, 44)


def md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def stage_one(o: int, h: int, s: int) -> None:
    src_dir = os.path.join(ROOT, f"data/processed_sweep_o{o}_h{h}")
    dst_dir = os.path.join(ROOT, f"data/processed_sweep_o{o}_h{h}_s{s}")
    cfg_src = load_config(os.path.join(ROOT, f"config_sweep_o{o}_h{h}.yaml"))
    assert int(cfg_src["seed"]) == 42

    # --- config -------------------------------------------------------------
    cfg = dict(cfg_src)
    cfg["seed"] = s
    cfg["results_dir"] = f"results/sweep_seeds/seed{s}/hours_{h}/origin{o}"
    cfg_path = os.path.join(ROOT, f"config_sweep_o{o}_h{h}_s{s}.yaml")
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    # fingerprint prepare_arm will compute for the new config (temporal arm)
    index = json.load(open(os.path.join(src_dir, "dataset_index.json")))
    fp_expected = _fingerprint("temporal", 4_000_000, cfg, int(index["total_rows"]))

    # --- processed dir --------------------------------------------------------
    man_path = os.path.join(dst_dir, "temporal", "manifest.json")
    if os.path.exists(man_path):
        man = json.load(open(man_path))
        if man.get("fingerprint") == fp_expected:
            print(f"o{o}_h{h}_s{s}: already staged, fingerprint OK")
            return
        print(f"o{o}_h{h}_s{s}: staged dir exists but fingerprint differs -- restaging")
        shutil.rmtree(dst_dir)
    shutil.copytree(src_dir, dst_dir)
    man = json.load(open(man_path))
    man["fingerprint"]["seed"] = s
    man["encoder_path"] = f"data/processed_sweep_o{o}_h{h}_s{s}/temporal/encoder.json"
    with open(man_path, "w") as f:
        json.dump(man, f, indent=2)
    assert man["fingerprint"] == fp_expected, (o, h, s, man["fingerprint"], fp_expected)
    # md5 spot check: train split + encoder identical to the seed-42 source
    for rel in ("temporal/train_X.npy", "temporal/encoder.json"):
        a, b = md5(os.path.join(src_dir, rel)), md5(os.path.join(dst_dir, rel))
        assert a == b, (rel, a, b)
    print(f"o{o}_h{h}_s{s}: staged + fingerprint + md5 OK")


def main() -> None:
    for s in SEEDS:
        for o in ORIGINS:
            for h in HOURS:
                stage_one(o, h, s)
    print("STAGING COMPLETE")


if __name__ == "__main__":
    main()
