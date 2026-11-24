"""Isolate the effects of the frozen terrain branch and its two gates."""
import argparse
from pathlib import Path

import numpy as np
import torch

from fusion_worker import Adapter, evaluate, indices, load_pack, read, write


MODELS = ("dem_suppress", "dem_shift_suppress")


@torch.inference_mode()
def replay(model, valid, groups):
    x, quality, anchor, _ = valid
    predictions = {k: [] for k in ("unmodified", "no_illumination_gate", "no_anchor_gate",
                                  "both_gates_removed", "terrain_relocated")}
    updates = {k: [] for k in predictions}
    terrain = groups[model.terrain_name]
    opposed = groups["wrong_terrain" if model.terrain_name == "terrain" else "terrain"]
    for xx, q, p in zip(x.split(8192), quality.split(8192), anchor.split(8192)):
        baseline = torch.logit(p.clamp(.001, .999))
        local = torch.exp(-((p - .5) / .24).square())
        without_illumination = q.clone()
        without_illumination[:, 6] = 1
        moved = xx.clone()
        moved[:, terrain] = xx[:, opposed]
        measured = model(xx, q, p) - baseline
        unit = model(xx, without_illumination, p) - baseline
        branches = {
            "unmodified": measured,
            "no_illumination_gate": unit,
            "no_anchor_gate": measured / local,
            "both_gates_removed": unit / local,
            "terrain_relocated": model(moved, q, p) - baseline,
        }
        for name, delta in branches.items():
            predictions[name].append(torch.sigmoid(baseline + delta).cpu().numpy())
            updates[name].append(delta.cpu().numpy())
    return ({k: np.concatenate(v) for k, v in predictions.items()},
            {k: np.concatenate(v) for k, v in updates.items()})


def diagnostics(model, valid, val, names, groups):
    probabilities, updates = replay(model, valid, groups)
    reference = val["anchor"].astype(np.float32)
    baseline = evaluate(reference, val, names)
    results = {}
    for key, score in probabilities.items():
        magnitude = np.abs(updates[key])
        decisions = (score >= .5) != (reference >= .5)
        results[key] = {
            "flipped_pixels": int(decisions.sum()),
            "updated_pixels_above_0p05_logit": int((magnitude>.05).sum()),
            "absolute_update_quantiles": np.quantile(magnitude,[.5,.9,.99]).tolist(),
            **evaluate(score, val, names),
        }
    return baseline, results


def main(args):
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.2)
    torch.set_float32_matmul_precision("high")
    schema = read(args.data / "schema.json")
    groups = indices(schema["bands"])
    payload = {}
    for fold in schema["development_folds"]:
        _, _, scale = load_pack(args.data, fold + "_train", groups)
        valid, val, _ = load_pack(args.data, fold + "_val", groups, scale)
        folds = {}
        for arm in MODELS:
            stem = f"{fold}_{arm}_s17_e14_m1.60"
            state = torch.load(args.run / (stem + ".pt"), map_location="cuda", weights_only=True)
            assert state["epoch"] == 14 and state["arm"] == arm
            model = Adapter(arm, groups, 1.6).cuda().eval()
            model.load_state_dict(state["model"])
            reference, results = diagnostics(model, valid, val, schema["strata"], groups)
            saved = read(args.run / (stem + ".metrics.json"))["strata"]["all"]
            verify = results["unmodified"]["strata"]["all"]
            assert all(saved[k] == verify[k] for k in ("tp", "fp", "fn"))
            folds[arm] = {"archive": reference, "variants": results}
            print(fold, arm, {n: r["flipped_pixels"] for n, r in results.items()}, flush=True)
        payload[fold] = folds
    args.output.parent.mkdir(exist_ok=True, parents=True)
    write(args.output, {"scope": "Frozen development checkpoints", "folds": payload})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
