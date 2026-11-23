# -*- coding: utf-8 -*-
"""SNT-v2.1 中途评测（只评 val_in / val_out，**绝不碰 official test**）。

按作业目录里的 config_snt.json 原样重建模型（同 head/scpc/backbone_opts/backbone_norm），
加载指定轮次的权重，用 train_snt.py 自带的 evaluate_loader 算全部指标（口径与训练时逐字一致）：
mIoU、六类 IoU、SC、BC、SCS、严格转移F1、变化F1、查准、查全、稳定位置误报、告警数、真变化数、
精确命中数、ABA 精确召回、AAA/ABA 误报、以及植被↔土壤两个方向的精确召回。

用法：
  python midtest_v2.py --run runs_snt_v2/p2v21_M3_s42 --epoch 48 --split val_in
  python midtest_v2.py --run ... --epoch 12 --split val_out --save midtest_ep12_out.json
"""
import argparse
import json
import os
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import train_snt as T  # noqa: E402  复用同一套路径解析、模型构建与评估器


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="作业目录（含 config_snt.json 与 epochNNN.pth）")
    ap.add_argument("--epoch", type=int, required=True)
    ap.add_argument("--split", choices=["val_in", "val_out", "test_in", "test_out"], required=True,
                    help="test_* 是 official test（Phase 3 才该看；本次按用户 11-23 20:0x 明确要求开封）")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--save", default="")
    ap.add_argument("--aoi_bank", type=int, default=0, choices=[0, 1], help="0=与训练时逐轮验证同口径（默认）；1=启用 AOI bank（诊断用）")
    a = ap.parse_args()

    run = Path(a.run).resolve()
    cfg_rec = json.loads((run / "config_snt.json").read_text("utf-8"))
    A = cfg_rec["args"]
    print(f"[MID] {run.name} 第 {a.epoch} 轮 | {a.split} | head={A['head']} scpc={A['scpc']} "
          f"rad_aug={A['rad_aug']} bb_opts={A['backbone_opts']!r} norm={A['backbone_norm']}", flush=True)

    t34_dir, here_dir, author_dir, data_dir, cache_dir = T.resolve_paths(A["dataset"])
    os.environ["ST3D_REPO"] = str(author_dir)
    os.environ["ST3D_DATA"] = str(data_dir.parent)      # 与 train_snt 一致，否则 st3d_repro 会用 Windows 默认缓存路径
    os.environ["ST3D_CACHE"] = str(cache_dir)
    for p in (str(HERE), str(t34_dir), str(here_dir), str(author_dir)):
        if p not in sys.path:
            sys.path.insert(0, p)

    import st3d_repro as released_adapter  # noqa: E402
    from data.datamodule import ImageDataModule  # noqa: E402
    from pse_net import make_pse  # noqa: E402
    from utils.wrapper import load_yaml  # noqa: E402
    from snt_model import SNTNet  # noqa: E402

    yaml_name = "dynamicearthnet_msi.yaml" if A["dataset"] == "s2" else "dynamicearthnet.yaml"   # 与 train_snt 一致
    cfg = load_yaml(author_dir / "configs" / yaml_name)
    cfg["dataset"]["path"] = str(data_dir)
    cfg["dataset"]["split_file"] = str(data_dir / "split.json")
    cfg["computer"]["accelerator"] = a.device
    cfg["run"]["seed"] = A["seed"]
    num_classes = int(cfg["dataset"]["num_classes"])
    ignore_idx = int(cfg["dataset"].get("ignore_index", 6))

    bb_opts = A["backbone_opts"]
    backbone = make_pse(cfg, bb_opts)
    if A["backbone_norm"] == "bn":
        T.gn_to_bn(backbone)
    model = SNTNet(backbone=backbone, in_channels=int(cfg["model"]["num_channels"]), num_classes=num_classes,
                   head_mode=A["head"], use_scpc=bool(A["scpc"]), use_rad_aug=bool(A["rad_aug"]),
                   freeze_backbone=bool(A["freeze_backbone"]), causal=bool(A["causal"]),
                   ignore_index=ignore_idx, drop=float(A["drop"])).to(a.device)

    ck = torch.load(run / f"epoch{a.epoch:03d}.pth", map_location=a.device, weights_only=False)
    model.load_state_dict(ck["model"], strict=True)
    model.eval()
    n = sum(p.numel() for p in model.parameters())
    print(f"[MID] 权重已载入，参数量 {n:,}", flush=True)

    make_set_fn = released_adapter.make_set
    if a.split.startswith("test"):
        ds = make_set_fn(cfg, "test", domain_shift=(a.split == "test_out"), cache=False)
        decode_tile = T.lru_cache(maxsize=1)(ds.load_data)          # 与 train_snt 的 official test 同写法
        ds.load_data = lambda i, tile, months, path, _fn=decode_tile: _fn(i, tile, tuple(months), path)
    else:
        ds = make_set_fn(cfg, "val", domain_shift=(a.split == "val_out"))
    bs = int(cfg["dataset"]["global_batch_size"])
    dm = ImageDataModule(ds, ds, ds, bs, 0)
    dm.setup("test")
    loader = dm.get_test_loader()

    m = T.evaluate_loader(model, loader, ds, a.device, num_classes, use_aoi_bank=bool(a.aoi_bank), per_aoi=False)
    out = {"run": str(run), "epoch": a.epoch, "split": a.split, "aoi_bank": a.aoi_bank, "args": A, "metrics": m}
    print("\n===== 指标 =====")
    keys = ["miou", "sc", "bc", "scs", "strict_f1", "change_f1", "precision", "recall", "stable_fpr",
            "alarms", "true_changes", "exact_hits", "aba_exact_rec", "aaa_aba_fp",
            "agr_stable_fpr", "veg_to_soil_exact_rec", "soil_to_veg_exact_rec"]
    for k in keys:
        if k in m:
            v = m[k]
            print(f"  {k:22s} " + (f"{v:.4f}" if isinstance(v, float) else str(v)))
    if "class_iou" in m:
        names = ["impervi", "agricult", "forest", "wetlands", "soil", "water"]
        print("  class_iou             " + " ".join(f"{n}={v:.2f}" for n, v in zip(names, m["class_iou"])))

    fname = f"midtest_ep{a.epoch:03d}_{a.split}" + ("_bank" if a.aoi_bank else "") + ".json"
    save = Path(a.save) if a.save else (run / fname)
    # metrics 里可能有 nan/inf，train_snt.save_json 会因 allow_nan=False 报错，这里换成容错写法
    def _clean(o):
        if isinstance(o, float):
            return o if o == o and abs(o) != float("inf") else None
        if isinstance(o, dict):
            return {k: _clean(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_clean(v) for v in o]
        return o
    save.parent.mkdir(parents=True, exist_ok=True)
    save.write_text(json.dumps(_clean(out), ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n[MID] 已保存 {save}", flush=True)


if __name__ == "__main__":
    main()
