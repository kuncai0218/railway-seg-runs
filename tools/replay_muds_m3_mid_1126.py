"""Evaluate one completed MUDS M3 checkpoint while the original trainer continues."""

import argparse
import hashlib
import io
import json
import sys
import time
from datetime import datetime
from functools import lru_cache
from pathlib import Path


def read_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True, help="Frozen MUDS package directory containing train_muds.py")
    parser.add_argument("--run", type=Path, required=True, help="Active M3 run containing last.pth")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="New directory outside the training run")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--check", action="store_true", help="Score one validation block per split to check wiring")
    return parser.parse_args()


def read_snapshot(torch, args, api):
    run_contract = json.loads((args.run / "protocol.json").read_text(encoding="utf-8-sig"))
    assert run_contract["method"] == "m3" and bool(run_contract["smoke"]) == args.check
    assert run_contract["split_sha256"] == api.SPLIT_SHA
    for name, digest in run_contract["sources"].items():
        assert api.sha(args.runtime / name) == digest, name
    checkpoint = args.run / "last.pth"
    payload = checkpoint.read_bytes()
    state = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    assert state["contract"] == run_contract
    epoch = int(state["epoch"])
    assert epoch >= 1 and [row["epoch"] for row in state["history"]] == list(range(1, epoch + 1))
    provenance = {
        "selection_rule": "last fully saved training epoch at first read, fixed before test evaluation",
        "seed": run_contract["seed"], "epoch": epoch, "training_budget": run_contract["training_epochs"],
        "batch_size": 1, "split_sha256": run_contract["split_sha256"], "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(payload).hexdigest(),
        "frozen_at": datetime.now().astimezone().isoformat(),
        "validation": state["history"][-1]["validation"], "best_validation_miou": state["best"],
        "original_run_protocol": run_contract, "replay_sha256": api.sha(Path(__file__)),
        "replay_device": args.device, "replay_torch_version": str(torch.__version__),
    }
    return {"model": state["model"], "epoch": epoch}, provenance


def main():
    args = read_args()
    args.runtime, args.run, args.data, args.out = (p.resolve() for p in (args.runtime, args.run, args.data, args.out))
    if args.run == args.out or args.run in args.out.parents:
        raise ValueError("Choose a separate output directory outside the existing training run")
    sys.path.insert(0, str(args.runtime))
    import torch
    import train_muds as api

    state, provenance = read_snapshot(torch, args, api)
    args.seed = provenance["seed"]
    args.cache = None
    args.out.mkdir(parents=True, exist_ok=False)
    api.save_json(args.out / "selection.json", provenance)
    api.save_ckpt(args.out / "frozen_model.pth", state)
    provenance["frozen_model_sha256"] = api.sha(args.out / "frozen_model.pth")
    print(f"Frozen MUDS M3 seed={args.seed} epoch={state['epoch']} "
          f"source_sha256={provenance['checkpoint_sha256']}", flush=True)

    torch.set_num_threads(2)
    api.seed_everything(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.1, device)
    adapter, cfg = api.setup(args)
    model = api.build(adapter, cfg, "m3")
    assert sum(p.numel() for p in model.parameters()) == provenance["original_run_protocol"]["model_parameters"]
    model.load_state_dict(state["model"], strict=True)
    model.to(device).eval()
    part = "val" if args.check else "test"
    regions = json.loads((args.data / "split.json").read_text(encoding="utf-8-sig"))
    results = {"complete": False, "kind": "validation_check" if args.check else "interim_official_test",
               "checkpoint": provenance, "sets": {}}
    start = time.monotonic()
    for domain, blocks in (("in", 80), ("out", 640)):
        dataset = adapter.make_set(cfg, part, domain_shift=domain == "out", cache=False)
        assert len(dataset) == blocks and dataset.split == part
        assert dataset.sits_ids == sorted(regions[part if domain == "out" else "train"])
        if not args.check and domain == "in":
            assert [tuple(dataset.get_loc_per_split(i)) for i in (0, 1)] == [(7, 6), (6, 7)]
        source = lru_cache(maxsize=1)(dataset.load_data)
        dataset.load_data = lambda k, tile, months, path: source(k, tile, tuple(months), path)
        print(f"Scoring {part}_{domain}, blocks={1 if args.check else blocks}", flush=True)
        metrics = api.evaluate(model, dataset, args.device, adapter.SCDMetric, detail=not args.check,
                               limit=1 if args.check else None)
        results["sets"][domain] = {"split": part, "blocks": 1 if args.check else blocks,
                                   "regions": dataset.sits_ids, "metrics": metrics}
        results["complete"] = domain == "out"
        results["elapsed_seconds"] = time.monotonic() - start
        api.save_json(args.out / "interim_result.json", results)
        print(f"{part}_{domain}: mIoU={metrics['miou']:.4f} SCS={metrics['scs']:.4f} "
              f"strict_F1={metrics['strict_f1']:.4f} seconds={results['elapsed_seconds']:.1f}", flush=True)
        source.cache_clear()
    print(f"Result SHA256 {api.sha(args.out / 'interim_result.json')}", flush=True)


if __name__ == "__main__":
    main()
