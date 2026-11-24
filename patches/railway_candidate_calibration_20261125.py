"""Compare spatial proposals while keeping the archived decisions fixed."""
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from fusion_worker import metric, write
from spatial_worker import FOLDS, Samples, Spatial, to_cuda


FP_BUDGETS = (0, 4, 10, 25)
SHIFT = (37, -49)
MODES = ("optical_full", "optical_shared", "radar_aligned", "radar_shifted",
         "radar_aligned_pooled", "radar_shifted_pooled")
SCENE_MODES = ("optical_scene_tail_probability", "optical_scene_tail_normalization")
RADAR_SCENE_MODES = ("radar_aligned_scene_tail_normalization", "radar_shifted_scene_tail_normalization")


def load_model(path, fold, arm, seed):
    checkpoint = torch.load(path / f"{fold}_{arm}_s{seed}.pt", map_location="cuda", weights_only=True)
    assert checkpoint["arm"] == arm and checkpoint["last_epoch"] == checkpoint["epochs"]
    model = Spatial(arm).to(device="cuda", memory_format=torch.channels_last)
    model.load_state_dict(checkpoint["model"])
    return model.eval()


def shifted_inputs(data):
    changed = list(data)
    changed[4] = torch.roll(data[4], SHIFT, dims=(2,3))
    geometric = torch.roll(data[5], SHIFT, dims=(2,3))
    geometric[:, :, :SHIFT[0], :] = 0
    geometric[:, :, :, SHIFT[1]:] = 0
    changed[5] = geometric
    return changed


def pool_on_shared(probability, shared):
    supported = F.avg_pool2d(shared, 3, 1, 1)
    average = F.avg_pool2d(probability * shared, 3, 1, 1) / supported.clamp_min(.0001)
    return torch.where(supported >= 7/9, average, probability)


@torch.inference_mode()
def proposal_scores(optical, radar, inputs):
    data = to_cuda(tuple(v[None] for v in inputs))
    displaced = shifted_inputs(data)
    shared = data[5] * displaced[5]
    with torch.autocast("cuda",dtype=torch.float16):
        optical_seed = optical(data)[1]
        aligned = radar(data)[2]
        shifted = radar(displaced)[2]
    aligned_pool, shifted_pool = [pool_on_shared(p, shared) for p in (aligned, shifted)]
    flat = lambda value: value[0,0].cpu().numpy()
    variants = (optical_seed, optical_seed, aligned, shifted, aligned_pool, shifted_pool)
    return [flat(p) for p in variants], flat(shared)>.5


def pick(counts, budget, allowed):
    mask = np.flatnonzero((counts[1] <= budget) & allowed)
    return max(mask, key=lambda k:(int(counts[0,k]), -int(counts[1,k]), int(k)))


def metrics(counts, base):
    original_tp,original_fp,positives,negatives = map(int,base)
    tp,fp=map(int,counts)
    return {"extra_tp":tp,"extra_fp":fp,
            **metric(original_tp+tp, original_fp+fp, positives-original_tp-tp, negatives)}


def analyze(mode, by_event, base):
    scored = {tag:tuple(np.sort(np.concatenate(parts)) for parts in zip(*examples))
              for tag,examples in by_event.items()}
    positives = np.concatenate([values[0] for values in scored.values()])
    thresholds = np.r_[np.unique(positives), np.inf]
    curves = {tag:np.stack([len(rows)-np.searchsorted(rows,thresholds) for rows in values])
              for tag,values in scored.items()}
    candidates = sum(curves.values())
    all_cuts = np.ones(len(thresholds),dtype=bool)
    threshold_at = lambda i: float(thresholds[i]) if thresholds[i] != np.inf else None
    threshold_name = "rank_threshold" if mode in (*SCENE_MODES,*RADAR_SCENE_MODES) else "logit_threshold"
    result = {}
    for budget in FP_BUDGETS:
        chosen = pick(candidates,budget,all_cuts)
        held = np.zeros(2, np.int64)
        rules = {}
        for tag, example in curves.items():
            known = np.concatenate([values[0] for k,values in scored.items() if k!=tag])
            allowed = np.r_[np.isin(thresholds[:-1],known), True]
            cut = pick(candidates-example,budget,allowed)
            amount = example[:,cut]
            held += amount
            rules["/".join(tag)] = {threshold_name:threshold_at(cut),"extra_tp":int(amount[0]),"extra_fp":int(amount[1])}
        result[str(budget)] = {"pooled": {threshold_name:threshold_at(chosen), **metrics(candidates[:,chosen],base)},
                               "leave_event_out": {**metrics(held,base), "events":rules}}
    precision = candidates[0,:-1] / np.maximum(candidates[:,:-1].sum(axis=0),1)
    outside_ap = float((precision*(-np.diff(candidates[0]))).sum()/len(positives)) if len(positives) else None
    negative_count = sum(len(values[1]) for values in scored.values())
    return {"mode":mode,"support_outside":{"positives":len(positives),"background":negative_count},
            "false_positives_before_first_true":int(candidates[1,-2]) if len(positives) else None,
            "outside_ap":outside_ap,"budgets":result}


def normalize_scene_tail(score, outer):
    values = np.sort(score[outer & np.isfinite(score)].astype(np.float32))
    if not len(values):
        return score, score
    probability = -(len(values) - np.searchsorted(values, score, side="left") + 1) / (len(values) + 1)
    low, high = np.quantile(values, [.95, .995])
    scaled = (score.astype(np.float32) - high) / max(high - low, .01)
    return probability, scaled


def main(args):
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.28)
    torch.set_float32_matmul_precision("high")
    modes = (*MODES, *SCENE_MODES,*RADAR_SCENE_MODES) if args.scene_normalization else MODES
    samples = {name:defaultdict(list) for name in modes}
    base = np.zeros(4,dtype=np.int64)
    splits = {}
    for fold in FOLDS:
        pack = Samples(args.data / fold / "val")
        optical=load_model(args.run,fold,"optical_context",args.seed)
        radar=load_model(args.run,fold,"radar_rescue",args.seed)
        fold_counts=np.zeros(6,dtype=np.int64)
        for i, row in enumerate(pack.rows):
            *data,y=pack.entry(i)
            p,n = y[0]==1,y[0]==0
            candidate=data[7][0]>.5
            original=data[6][0]>=.5
            outer=~candidate & ~original
            scores, shared = proposal_scores(optical,radar,data)
            if args.scene_normalization:
                scores.extend(normalize_scene_tail(scores[0], outer))
                scores.extend(normalize_scene_tail(scores[j],outer & shared)[1] for j in (2,3))
            tag=(row["site"],row["event_group"])
            fold_counts += [int((p&original).sum()),int((n&original).sum()),int(p.sum()),int(n.sum()),
                            int((outer&p).sum()),int((outer&p&shared).sum())]
            for name, score in zip(modes,scores):
                mask=outer if name=="optical_full" or name in SCENE_MODES else outer&shared
                samples[name][tag].append((score[p&mask], score[n&mask]))
        base+=fold_counts[:4]
        splits[fold]=dict(zip(("original_tp","original_fp","positive","background","outside_positive",
                               "shared_outside_positive"),fold_counts.tolist()))
        pack.entry.cache_clear()
    assert base.tolist()==[1397,106,1929,2250527], base.tolist()
    outcome=[analyze(name,samples[name],base) for name in modes]
    report={"seed":args.seed,"scope":"fixed development folds only","shift_pixels":SHIFT,
            "baseline":metrics((0,0),base),"split_counts":splits,"comparisons":outcome}
    args.output.parent.mkdir(exist_ok=True,parents=True)
    write(args.output,report)
    for row in outcome:
        print(row["mode"],row["support_outside"],[(k,v["pooled"]["extra_tp"],v["pooled"]["extra_fp"],
              v["leave_event_out"]["extra_tp"],v["leave_event_out"]["extra_fp"])
              for k,v in row["budgets"].items()], flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--data",type=Path,default=Path("spatial_data"))
    p.add_argument("--run",type=Path,default=Path("spatial"))
    p.add_argument("--seed",type=int,default=17)
    p.add_argument("--scene-normalization",action="store_true")
    p.add_argument("--output",type=Path,required=True)
    main(p.parse_args())
