# -*- coding: utf-8 -*-
"""Evaluate PSE+BN run on official test set (selected best checkpoint + stop_epoch checkpoint).
Read-only on run directory; computes official SCDMetric + Strict Transition F1 + change error breakdown.
Usage:
  python -X utf8 extra_metrics_psebn.py --entry train_t34_psebn_local.py --out <run_dir> --seed 42 --stop_epoch 120 --save <out.json>
"""
import argparse
import importlib.util
import json
import sys
from functools import lru_cache
from pathlib import Path
import numpy as np

K = 6


def summarize(joint):
    a, b, c, d = np.indices((K, K, K, K))
    changed, detected = a != b, c != d
    ok_a, ok_b = a == c, b == d
    tc, pc = int(joint[changed].sum()), int(joint[detected].sum())
    tp = int(joint[changed & detected].sum())
    strict = int(joint[changed & detected & ok_a & ok_b].sum())
    fp, fn = pc - tp, tc - tp
    return {
        "真值变化数": tc,
        "预测变化数": pc,
        "检出的真变化": tp,
        "变化查准率": 100.0 * tp / max(pc, 1),
        "变化查全率": 100.0 * tp / max(tc, 1),
        "变化F1": 200.0 * tp / max(tc + pc, 1),
        "BC复核": 100.0 * tp / max(tp + fp + fn, 1),
        "严格转移F1": 200.0 * strict / max(tc + pc, 1),
        "检出的真变化中前后都对的比例": 100.0 * strict / max(tp, 1),
        "检出但前后类别有错（占真变化）": 100.0 * (tp - strict) / max(tc, 1),
        "漏检（占真变化）": 100.0 * fn / max(tc, 1),
        "误报（占真值不变）": 100.0 * fp / max(int(joint[~changed].sum()), 1),
    }


def gn_to_bn(model, torch):
    for n, ch in model.named_children():
        if isinstance(ch, torch.nn.GroupNorm):
            setattr(model, n, torch.nn.BatchNorm2d(ch.num_channels))
        else:
            gn_to_bn(ch, torch)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entry", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--save", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--stop_epoch", type=int, default=120)
    a = ap.parse_args()

    out, entry = Path(a.out), Path(a.entry).resolve()
    sys.argv = [str(entry)]
    sys.path.insert(0, str(entry.parent))
    spec = importlib.util.spec_from_file_location("t34_psebn_entry", entry)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)

    import torch
    from data.datamodule import ImageDataModule
    from metrics.scd_metrics import SCDMetric
    from pse_net import make_pse

    m.OUTPUT = out
    m.ARGS = argparse.Namespace(
        seed=a.seed, method="pse", out=str(out), smoke_steps=0, resume=False, stop_epoch=a.stop_epoch
    )
    cfg, split, contract = m.check_configuration()
    protocol = json.loads((out / "protocol.json").read_text("utf-8"))
    assert protocol == json.loads(json.dumps(contract)), "Protocol mismatch vs run directory!"

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = make_pse(cfg, "")
    gn_to_bn(model, torch)
    assert not any(isinstance(x, torch.nn.GroupNorm) for x in model.modules())
    model = model.to(device)

    sel = json.loads((out / "test_selected_checkpoints.json").read_text("utf-8"))
    official = json.loads((out / "official_test.json").read_text("utf-8"))["results"]
    stop_file = f"epoch{a.stop_epoch:03d}.pth"
    has_stop = (out / stop_file).exists()

    res = {}
    for domain in m.PER_DOMAIN:
        shift = domain == "out"
        dataset = m.released_adapter.make_set(cfg, "test", domain_shift=shift, cache=False)
        assert len(dataset) == official["test_" + domain]["blocks"]
        decode_tile = lru_cache(maxsize=1)(dataset.load_data)

        def load_tile(i, tile, months, path):
            return decode_tile(i, tile, tuple(months), path)

        dataset.load_data = load_tile
        owner = ImageDataModule(dataset, dataset, dataset, 2, 0)
        owner.setup("test")
        loader = owner.get_test_loader()

        eval_targets = [("selected", sel[domain]["weights"], sel[domain]["epoch"])]
        if has_stop:
            eval_targets.append((f"epoch{a.stop_epoch}", stop_file, a.stop_epoch))

        res[domain] = {}
        for tag, wname, ep in eval_targets:
            ckpt = torch.load(out / wname, map_location="cpu", weights_only=True)
            model.load_state_dict(ckpt["model"], strict=True)
            model.eval()
            metric = SCDMetric(cfg["dataset"]["num_classes"], cfg["dataset"]["class_names"], cfg["dataset"]["ignore_index"])
            joint = np.zeros((K, K, K, K), dtype=np.int64)
            with torch.inference_mode():
                for batch in loader:
                    x = batch["data"].to(device)
                    pos = batch["positions"].to(device)
                    y = batch["gt"].to(device)
                    logits = model(x, pos)
                    pred = logits.argmax(2)
                    metric.update(pred, y)
                    yn = y.cpu().numpy()
                    pn = pred.cpu().numpy()
                    a0, b0, c0, d0 = yn[:, :-1], yn[:, 1:], pn[:, :-1], pn[:, 1:]
                    good = (a0 >= 0) & (a0 < K) & (b0 >= 0) & (b0 < K)
                    enc = ((a0[good] * K + b0[good]) * K + c0[good]) * K + d0[good]
                    joint += np.bincount(enc, minlength=K**4).reshape(K, K, K, K)
            sc = {k: float(v) for k, v in metric.compute().items()}
            extra = summarize(joint)
            if tag == "selected":
                off_bc = float(official["test_" + domain]["scores"]["bc"])
                assert abs(extra["BC复核"] - off_bc) < 0.05, (extra["BC复核"], off_bc)
            res[domain][tag] = {"epoch": ep, "weights": wname, "official_scores": sc, "change_metrics": extra}
        decode_tile.cache_clear()

    out_obj = {
        "作业": out.name,
        "方法": "pse_bn",
        "种子": a.seed,
        "stop_epoch": a.stop_epoch,
        "lr_tmax": 250,
        "结果": res,
    }
    Path(a.save).parent.mkdir(parents=True, exist_ok=True)
    Path(a.save).write_text(json.dumps(out_obj, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[OK] Saved PSE+BN extra metrics ->", a.save)


if __name__ == "__main__":
    main()
