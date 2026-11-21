# -*- coding: utf-8 -*-
"""T34 补充指标（11-20 用户要求）：对已完成作业，在正式测试集上用“验证集选中的权重”和“第 250 轮权重”各评一次，
除官方 SCDMetric 外另算：变化查准率 / 查全率 / F1、严格转移 F1（变化被检出且前后两期类别都对，按 11-08 约定的定义：
2×严格命中 /（真值变化数 + 预测变化数）），以及变化错误的分解（漏检、误报、前后类别错）。
口径与官方一致：相邻两期、两期标签都不是忽略值的像元才计入。只读作业目录，结果另存。不参与任何选择。
用法：python -X utf8 extra_metrics.py --entry <入口> --out <作业目录> --seed 42 --method orig --save <结果 json>"""
import argparse, importlib.util, json, sys
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
        "真值变化数": tc, "预测变化数": pc, "检出的真变化": tp,
        "变化查准率": 100 * tp / max(pc, 1), "变化查全率": 100 * tp / max(tc, 1),
        "变化F1": 200 * tp / max(tc + pc, 1), "BC复核": 100 * tp / max(tp + fp + fn, 1),
        "严格转移F1": 200 * strict / max(tc + pc, 1),
        "检出的真变化中前后都对的比例": 100 * strict / max(tp, 1),
        "检出但前后类别有错（占真变化）": 100 * (tp - strict) / max(tc, 1),
        "漏检（占真变化）": 100 * fn / max(tc, 1),
        "误报（占真值不变）": 100 * fp / max(int(joint[~changed].sum()), 1),
    }


def main():
    ap = argparse.ArgumentParser()
    for k in ("--entry", "--out", "--save"):
        ap.add_argument(k, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--method", choices=["orig", "pse"], default="orig")
    a = ap.parse_args()
    out, entry = Path(a.out), Path(a.entry).resolve()
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
    assert json.loads(protocol.read_text("utf-8")) == json.loads(json.dumps(contract)), "协议与作业目录不一致"
    device = "cuda:0"
    if a.method == "pse":
        from pse_net import make_pse
        model = make_pse(cfg, "").to(device)
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
    sel = json.loads((out / "test_selected_checkpoints.json").read_text("utf-8"))
    official = json.loads((out / "official_test.json").read_text("utf-8"))["results"]
    last = json.loads((out / "official_test_last.json").read_text("utf-8"))["results"] if (out / "official_test_last.json").exists() else None
    make_set = m.msi_adapter.make_set if hasattr(m, "msi_adapter") else m.released_adapter.make_set
    res = {}
    for domain in m.PER_DOMAIN:
        shift = domain == "out"
        dataset = make_set(cfg, "test", domain_shift=shift, cache=False)
        assert dataset.split == "test" and len(dataset) == official["test_" + domain]["blocks"]
        decode_tile = lru_cache(maxsize=1)(dataset.load_data)
        dataset.load_data = lambda i, t, mo, p: decode_tile(i, t, tuple(mo), p)
        owner = m.ImageDataModule(dataset, dataset, dataset, 2, 0)
        owner.setup("test")
        for tag, wfile, ref in (("正式", sel[domain]["weights"], official), ("最后一轮", f"epoch{m.EPOCHS:03d}.pth", last)):
            if tag == "最后一轮" and ref is None:
                continue
            ck = torch.load(out / wfile, map_location="cpu", weights_only=True)
            assert ck["protocol_sha256"] == m.sha256(protocol)
            model.load_state_dict(ck["model"], strict=True)
            model.eval()
            joint = np.zeros(K ** 4, np.int64)
            with torch.no_grad():
                for batch in owner.get_test_loader():
                    x, y, dt = batch["data"].to(device), batch["gt"].to(device), batch["positions"].to(device)
                    pr = model(x, dt).argmax(2).view_as(y).cpu().numpy().astype(np.int64)
                    gt = y.cpu().numpy().astype(np.int64)
                    v = (gt[:, 1:] != 6) & (gt[:, :-1] != 6)
                    g0, g1, p0, p1 = gt[:, :-1][v], gt[:, 1:][v], pr[:, :-1][v], pr[:, 1:][v]
                    joint += np.bincount(((g0 * K + g1) * K + p0) * K + p1, minlength=K ** 4)
            s = summarize(joint.reshape(K, K, K, K))
            s["BC官方"] = ref["test_" + domain]["scores"]["bc"]
            s["各类IoU官方"] = {c: ref["test_" + domain]["scores"][c] for c in cfg["dataset"]["class_names"]}
            s["mIoU官方"], s["SCS官方"] = ref["test_" + domain]["scores"]["miou"], ref["test_" + domain]["scores"]["scs"]
            s["轮次"] = ck["epoch"]
            assert abs(s["BC复核"] - s["BC官方"]) < 0.05, (s["BC复核"], s["BC官方"])  # 第 2 版（11-21）：GPU 卷积算法的细微非确定性会让极少数像元翻转，组长本机差 0.0007、甲差 1.8e-5；容差放到 0.05
            res[f"{tag}_{'同区' if domain == 'in' else '跨区'}"] = s
            print(tag, domain, {k: round(v, 2) for k, v in s.items() if isinstance(v, float)}, flush=True)
        decode_tile.cache_clear()
    Path(a.save).parent.mkdir(parents=True, exist_ok=True)
    Path(a.save).write_text(json.dumps({"作业": str(out), "入口": str(entry), "结果": res}, ensure_ascii=False, indent=1), "utf-8")


if __name__ == "__main__":
    main()
