# -*- coding: utf-8 -*-
"""T34 最后一轮权重的正式测试（11-19 用户要求，作参考单独列出，不参与任何选择）。
训练完成（official_test.json 已写出）后，对同一作业的 epoch250.pth 在同区、跨区正式测试集上各评一次，
所有代码路径（数据类、逐块切法、批大小、SCDMetric、run_eval）与入口 run_test 相同，只把权重换成第 250 轮。
开评前用入口自己的 check_configuration() 重新生成协议并要求与作业目录 protocol.json 完全相同。结果写 official_test_last.json，
不改动作业目录里任何已有文件。适用于：冻结入口 train_den_baseline.py、train_t34.py（本机与甲乙包）、train_t34_s2.py。
用法：python -X utf8 test_last.py --entry <入口 .py> --out <作业目录> [--seed 43 --method orig]"""
import argparse
import importlib.util
import json
import sys
from functools import lru_cache
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entry", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--method", choices=["orig", "pse"], default="orig")
    ap.add_argument("--dry", default="", help="只验流程：用冒烟作业目录的最后一个权重，结果写到这个文件，不读 official_test.json")
    a = ap.parse_args()
    out = Path(a.out)
    target = Path(a.dry) if a.dry else out / "official_test_last.json"
    assert a.dry or (out / "official_test.json").exists(), "训练与正式测试尚未完成"
    assert not target.exists(), "最后一轮已评过"
    entry = Path(a.entry).resolve()
    sys.argv = [str(entry)]
    sys.path.insert(0, str(entry.parent))  # 与直接运行入口时一样，入口所在目录在导入路径最前
    spec = importlib.util.spec_from_file_location("t34_entry", entry)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)  # 只执行模块顶层（环境变量、导入、确定性设置），不调用 main()
    import torch
    m.OUTPUT = out
    if hasattr(m, "ARGS"):
        smoke = json.loads((out / "protocol.json").read_text("utf-8")).get("smoke_steps", 0) if a.dry else 0
        m.ARGS = argparse.Namespace(seed=a.seed, method=a.method, out=str(out), smoke_steps=smoke, resume=False)
    cfg, split, contract = m.check_configuration()
    protocol = out / "protocol.json"
    assert json.loads(protocol.read_text("utf-8")) == json.loads(json.dumps(contract)), "协议与作业目录不一致"
    device = "cuda:0"
    if a.method == "pse":
        from pse_net import make_pse
        model = make_pse(cfg, "").to(device)
    else:
        model = m.load_model(cfg).to(device)
    metric = m.SCDMetric(cfg["dataset"]["num_classes"], cfg["dataset"]["class_names"], cfg["dataset"]["ignore_index"])
    last_epoch = max(int(p.stem[5:]) for p in out.glob("epoch[0-9][0-9][0-9].pth")) if a.dry else m.EPOCHS
    weights = out / f"epoch{last_epoch:03d}.pth"
    ckpt = torch.load(weights, map_location="cpu", weights_only=True)
    assert ckpt["epoch"] == last_epoch and ckpt["protocol_sha256"] == m.sha256(protocol)
    model.load_state_dict(ckpt["model"], strict=True)
    make_set = m.msi_adapter.make_set if hasattr(m, "msi_adapter") else m.released_adapter.make_set
    official = None if a.dry else json.loads((out / "official_test.json").read_text("utf-8"))["results"]
    evidence = {}
    for domain in m.PER_DOMAIN:
        shift = domain == "out"
        dataset = make_set(cfg, "test", domain_shift=shift, cache=False)
        assert dataset.split == "test"
        assert dataset.sits_ids == sorted(split["test" if shift else "train"])
        if official:
            assert len(dataset) == official["test_" + domain]["blocks"]
        if not shift:
            assert [tuple(dataset.get_loc_per_split(i)) for i in (0, 1)] == [(7, 6), (6, 7)]
        decode_tile = lru_cache(maxsize=1)(dataset.load_data)

        def load_tile(i, tile, months, path):
            return decode_tile(i, tile, tuple(months), path)
        dataset.load_data = load_tile
        owner = m.ImageDataModule(dataset, dataset, dataset, 2, 0)
        owner.setup("test")
        print(json.dumps({"phase": "test_last", "domain": domain, "blocks": len(dataset)}), flush=True)
        scores = m.released_adapter.run_eval(model, owner.get_test_loader(), metric, device)
        evidence["test_" + domain] = {"epoch": last_epoch, "weights": m.sha256(weights), "regions": dataset.sits_ids,
                                      "blocks": len(dataset), "scores": scores}
        decode_tile.cache_clear()
        del dataset, owner, decode_tile
    m.save_json(target, {"purpose": "最后一轮（第 250 轮）权重的正式测试，作参考单独报告，不参与选择；正式成绩仍为 official_test.json（按验证集选）",
                         "results": evidence, "entry": str(entry), "entry_sha256": m.sha256(entry),
                         "protocol_sha256": m.sha256(protocol), "finished": m.now()})
    print("完成", "（试跑，不显示分数）" if a.dry else {d: round(v["scores"]["miou"], 2) for d, v in evidence.items()}, flush=True)


if __name__ == "__main__":
    main()
