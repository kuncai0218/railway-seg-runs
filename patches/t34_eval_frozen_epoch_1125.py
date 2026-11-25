# -*- coding: utf-8 -*-
"""对指定作业、指定轮次权重，在正式测试域上同时算官方指标与完整补充指标。
用于“验证集 SCS 选轮”口径下的基线全指标重测。跨机重评只告警协议路径差异。
"""
import argparse
import importlib.util
import json
import os
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

K = 6
CLASS_KEYS = ("impervi", "agricult", "forest", "wetlands", "soil", "water")


def summarize(joint):
    a, b, c, d = np.indices((K, K, K, K))
    changed, detected = a != b, c != d
    ok_a, ok_b = a == c, b == d
    tc, pc = int(joint[changed].sum()), int(joint[detected].sum())
    tp = int(joint[changed & detected].sum())
    strict = int(joint[changed & detected & ok_a & ok_b].sum())
    fp, fn = pc - tp, tc - tp
    return {
        "true_changes": tc, "alarms": pc, "exact_hits": strict,
        "precision": 100 * tp / max(pc, 1), "recall": 100 * tp / max(tc, 1),
        "change_f1": 200 * tp / max(tc + pc, 1),
        "strict_f1": 200 * strict / max(tc + pc, 1),
        "pair": 100 * strict / max(tp, 1),
        "miss": 100 * fn / max(tc, 1),
        "stable_fpr": 100 * fp / max(int(joint[~changed].sum()), 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entry", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--method", choices=["orig", "pse"], default="orig")
    ap.add_argument("--epoch", type=int, required=True)
    ap.add_argument("--save", required=True)
    a = ap.parse_args()

    out = Path(a.out)
    entry = Path(a.entry).resolve()
    sys.argv = [str(entry)]
    sys.path.insert(0, str(entry.parent))
    spec = importlib.util.spec_from_file_location("t34_entry", entry)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    import torch

    m.OUTPUT = out
    if hasattr(m, "ARGS"):
        m.ARGS = argparse.Namespace(seed=a.seed, method=a.method, out=str(out), smoke_steps=0, resume=False)
    cfg, split, contract = m.check_configuration()
    protocol = out / "protocol.json"
    if json.loads(protocol.read_text("utf-8")) != json.loads(json.dumps(contract)):
        print("[WARN] 协议与作业目录不一致（跨机重评正常现象），继续评测", flush=True)

    device = "cuda:0"
    if a.method == "pse":
        from pse_net import make_pse
        opts = os.environ.get("T34_ABL_OPTS", "") if entry.name == "train_t34_abl.py" else ""
        model = make_pse(cfg, opts).to(device)
    else:
        model = m.load_model(cfg)
        if "normalization" in contract.get("patches_and_decisions", {}):
            def _to_gn(mm):
                for n, ch in mm.named_children():
                    if isinstance(ch, torch.nn.modules.batchnorm._BatchNorm):
                        c = ch.num_features
                        setattr(mm, n, torch.nn.GroupNorm(max(1, min(8, c // 8)), c))
                    else:
                        _to_gn(ch)
            _to_gn(model)
        model = model.to(device)

    weights = out / f"epoch{a.epoch:03d}.pth"
    ckpt = torch.load(weights, map_location="cpu", weights_only=True)
    assert ckpt["epoch"] == a.epoch and ckpt["protocol_sha256"] == m.sha256(protocol)
    model.load_state_dict(ckpt["model"], strict=True)
    make_set = m.msi_adapter.make_set if hasattr(m, "msi_adapter") else m.released_adapter.make_set

    evidence = {}
    for domain in m.PER_DOMAIN:
        shift = domain == "out"
        dataset = make_set(cfg, "test", domain_shift=shift, cache=False)
        assert dataset.split == "test"
        decode_tile = lru_cache(maxsize=1)(dataset.load_data)
        dataset.load_data = lambda i, t, mo, p: decode_tile(i, t, tuple(mo), p)
        owner = m.ImageDataModule(dataset, dataset, dataset, 2, 0)
        owner.setup("test")
        metric = m.SCDMetric(cfg["dataset"]["num_classes"], cfg["dataset"]["class_names"], cfg["dataset"]["ignore_index"])
        model.eval()
        joint = np.zeros(K ** 4, np.int64)
        with torch.no_grad():
            for batch in owner.get_test_loader():
                x, y, date = batch["data"].to(device), batch["gt"].to(device), batch["positions"].to(device)
                pred = model(x, date).argmax(2).view_as(y)
                metric.update(pred.cpu(), y.cpu())
                pr = pred.cpu().numpy().astype(np.int64)
                gt = y.cpu().numpy().astype(np.int64)
                v = (gt[:, 1:] != 6) & (gt[:, :-1] != 6)
                g0, g1, p0, p1 = gt[:, :-1][v], gt[:, 1:][v], pr[:, :-1][v], pr[:, 1:][v]
                joint += np.bincount(((g0 * K + g1) * K + p0) * K + p1, minlength=K ** 4)
        official = {k: float(v) for k, v in metric.compute().items()}
        s = summarize(joint.reshape(K, K, K, K))
        s.update({
            "miou": official["miou"], "bc": official["bc"], "sc": official["sc"], "scs": official["scs"],
            "class_iou": [official[k] for k in CLASS_KEYS],
        })
        evidence["test_" + domain] = s
        decode_tile.cache_clear()
        del dataset, owner, decode_tile
        print(f"[完整指标] ep{a.epoch} {domain} mIoU {s['miou']:.2f} SCS {s['scs']:.2f} 严格F1 {s['strict_f1']:.2f}", flush=True)

    Path(a.save).write_text(json.dumps({
        "purpose": "验证 SCS 选轮权重的完整正式测试指标",
        "run": str(out), "entry": str(entry), "epoch": a.epoch,
        "results": evidence, "finished": m.now(),
    }, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
