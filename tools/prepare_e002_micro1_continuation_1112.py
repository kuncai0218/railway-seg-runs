"""Validate and clone a stopped E002 full checkpoint into a new micro1 run.

No CUDA, training, environment changes, source edits, or result deletion.
The caller verifies that the previous trainer has exited before --apply.
"""
from pathlib import Path
import os

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[name] = "2"

import argparse
import datetime
import hashlib
import json
import shutil
import sys

import torch


REQUIRED = {
    "model", "optimizer", "scaler", "optimizer_updates", "attempts",
    "completed_epochs", "best_validation_scs", "permutation", "cursor",
    "generator_state", "month_lists", "Python_RNG", "NumPy_RNG",
    "Torch_RNG", "CUDA_RNG", "source_hashes", "protocol_SHA256",
}


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(8 * 2**20):
            h.update(block)
    return h.hexdigest()


def finite_tensors(value):
    count = 0
    if isinstance(value, torch.Tensor):
        if value.is_floating_point() or value.is_complex():
            if not bool(torch.isfinite(value).all()):
                raise ValueError("nonfinite model/optimizer tensor")
        count += 1
    elif isinstance(value, dict):
        for item in value.values():
            count += finite_tensors(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            count += finite_tensors(item)
    return count


def validate(checkpoint, kit, parent):
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU-only preflight requires no initialized CUDA")
    config = json.loads((parent / "config.json").read_text(encoding="utf-8-sig"))
    expected = config["source_hashes"]
    before = sha(checkpoint)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    missing = sorted(REQUIRED - set(state))
    if missing:
        raise ValueError("not a complete continuation checkpoint; missing " + ",".join(missing))
    if expected != state["source_hashes"]:
        raise ValueError("checkpoint and parent config source hashes differ")
    for name, digest in expected.items():
        if Path(name).name != name or sha(kit / name) != digest:
            raise ValueError("frozen source mismatch: " + name)
    protocol_names = [name for name in expected if name.startswith("protocol") and name.endswith(".json")]
    if len(protocol_names) != 1 or sha(kit / protocol_names[0]) != state["protocol_SHA256"]:
        raise ValueError("protocol identity mismatch")
    protocol = json.loads((kit / protocol_names[0]).read_text(encoding="utf-8-sig"))
    if protocol != config["protocol"] or config["global_batch"] != 8:
        raise ValueError("parent scientific protocol/global batch mismatch")
    if not 0 < state["optimizer_updates"] <= state["attempts"]:
        raise ValueError("invalid actual optimizer update count")
    permutation = state["permutation"]
    cursor = state["cursor"]
    if permutation.ndim != 1 or cursor < 0 or cursor + 8 > len(permutation) or cursor % 8:
        raise ValueError("invalid continuation permutation/cursor")
    if len(state["month_lists"]) != len(protocol["train_regions"]):
        raise ValueError("monthly sampling state does not cover training regions")
    checked = finite_tensors(state["model"]) + finite_tensors(state["optimizer"])
    if not state["optimizer"].get("state") or not state["scaler"] or not state["CUDA_RNG"]:
        raise ValueError("empty optimizer/scaler/CUDA RNG state")
    if sha(checkpoint) != before:
        raise ValueError("checkpoint changed while being inspected; use a stopped run/archive")
    assert not torch.cuda.is_initialized()
    return state, config, before, checked


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kit-dir", type=Path, required=True)
    ap.add_argument("--parent-out", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--destination", type=Path, required=True)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--previous-trainer-stopped", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(2)
    kit, parent, checkpoint, destination = [p.resolve() for p in
        (a.kit_dir, a.parent_out, a.checkpoint, a.destination)]
    if checkpoint.parent != parent or destination == parent or destination.exists():
        raise ValueError("use a parent-run checkpoint and a new independent output directory")
    if a.apply and not a.previous_trainer_stopped:
        raise ValueError("verify actual previous trainer has exited before applying")
    state, config, digest, checked = validate(checkpoint, kit, parent)
    receipt = {
        "time": datetime.datetime.now().astimezone().isoformat(),
        "passed_CPU_preflight": True,
        "applied": bool(a.apply),
        "checkpoint": str(checkpoint),
        "checkpoint_SHA256": digest,
        "checkpoint_bytes": checkpoint.stat().st_size,
        "parent_output": str(parent),
        "destination": str(destination),
        "parent_actual_micro": config["micro_batch"],
        "continuation_actual_micro": 1,
        "continuation_accumulation": 8,
        "effective_batch": 8,
        "optimizer_updates_restored": state["optimizer_updates"],
        "attempts_restored": state["attempts"],
        "completed_epochs_restored": state["completed_epochs"],
        "complete_optimizer_scaler_sampler_months_CPU_CUDA_RNG_present": True,
        "finite_model_optimizer_tensors_checked": checked,
        "scientific_protocol_SHA256": state["protocol_SHA256"],
        "frozen_source_hashes": state["source_hashes"],
        "trainer_and_scientific_protocol_edited": False,
        "CUDA_initialized": torch.cuda.is_initialized(),
        "GPU_qualification_or_fault_cure_claimed": False,
        "trajectory_note": "Micro2 to micro1 keeps effective batch8 and valid-pixel loss weighting; dropout/random-kernel consumption and floating-point sums can change. This is a recorded engineering continuation, not bit-identical replay.",
        "script_SHA256": sha(Path(__file__)),
        "command_argv": [sys.executable, "-u", str(kit / "e002_train.py"),
                         "--micro", "1", "--out", str(destination)],
    }
    if a.apply:
        destination.mkdir(parents=True, exist_ok=False)
        shutil.copyfile(checkpoint, destination / "last.pth")
        if sha(destination / "last.pth") != digest or sha(checkpoint) != digest:
            raise ValueError("copy identity failed; source/results retained, do not launch")
        # Parent config is intentionally not copied: the frozen trainer writes
        # a fresh runconfig with actual micro1 and validates the same source SHA.
        with (destination / "continuation_parent_receipt.json").open("w", encoding="utf-8", newline="\n") as out:
            json.dump(receipt, out, ensure_ascii=False, indent=2)
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == "__main__":
    main()
