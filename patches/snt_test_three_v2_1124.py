# -*- coding: utf-8 -*-
"""对某个 SNT 作业，按用户 2026-11-23 20:1x 要求，跑三种权重 × 两个 official test 划分：
  ① 末轮（epochs.json 里最大轮次）
  ② 最佳验证 mIoU（val_out 最好；同时给出 val_in 最好作为对照）
  ③ 最佳验证 SCS（val_out 最好；另给 val_in 最好）
跳过已存在的 json（幂等），可反复调用。

用法：python test_three_v2.py --run runs_snt_v2/p2v21_M3_s42 [--min_epoch 10] [--dry]
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PY = sys.executable   # 本机=论文锁定环境；云端=t34_cu128：跟随调用者，跨平台
HERE = Path(__file__).resolve().parent
MID = HERE / "midtest_v2.py"


def best_epoch(rows, dom, key):
    b = max(rows, key=lambda r: (r.get(dom) or {}).get(key, -1e9))
    return int(b["epoch"]), float(b[dom][key])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--min_epoch", type=int, default=0, help="只在 >= 该轮里选“最佳”（躲开早期抖动，默认 0 = 不设限）")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--with_in", action="store_true", help="同时评 val_in 最好的两个权重（默认只按 val_out 选）")
    a = ap.parse_args()
    run = Path(a.run).resolve()
    rows = json.load(open(run / "epochs.json", encoding="utf-8"))
    if a.min_epoch:
        rows_sel = [r for r in rows if r["epoch"] >= a.min_epoch]
    else:
        rows_sel = rows
    last = int(rows[-1]["epoch"])
    picks = {
        "末轮": last,
        "最佳valmIoU": best_epoch(rows_sel, "val_out", "miou")[0],
        "最佳valSCS": best_epoch(rows_sel, "val_out", "scs")[0],
    }
    if a.with_in:
        picks["最佳valin_mIoU"] = best_epoch(rows_sel, "val_in", "miou")[0]
        picks["最佳valin_SCS"] = best_epoch(rows_sel, "val_in", "scs")[0]
    uniq = sorted(set(picks.values()))
    print(f"[TEST3] {run.name}: 已跑 {last} 轮；候选权重 {picks} -> {uniq}", flush=True)
    if a.dry:
        return 0

    for ep in uniq:
        for split in ("test_in", "test_out"):
            out = run / f"midtest_ep{ep:03d}_{split}.json"
            if out.exists():
                print(f"[TEST3] 跳过（已有）{out.name}", flush=True)
                continue
            t0 = time.time()
            r = subprocess.run([PY, "-u", "-X", "utf8", str(MID), "--run", str(run), "--epoch", str(ep), "--split", split],
                               capture_output=True, text=True, encoding="utf-8", errors="replace")
            line = [x.strip() for x in r.stdout.splitlines() if x.strip().startswith(("miou", "scs", "strict_f1"))]
            print(f"[TEST3] ep{ep} {split} rc={r.returncode} {(time.time()-t0)/60:.1f}min | " + " ; ".join(line), flush=True)
            if r.returncode != 0:
                print("[TEST3] 错误：" + (r.stderr or r.stdout)[-300:].replace("\n", " "), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
