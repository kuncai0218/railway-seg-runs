# MUDS M3 种子 43 中途测试

本条是正在运行的 `jia-muds-pse-m3-s43-1126` 的一次性评估。训练和接续顺序按现有安排运行。

先在原状态话题回执实际 M3 训练 PID、最近两条步进、最近完成轮的同区与跨区验证 mIoU 和 SCS。随后用同一个 Python、同一份已解压的 MUDS 实验包和既有数据执行。用训练进程的命令确认 `--out` 对应的正式目录。评估的输出目录另建在旁边。

```bash
python -B -u -X utf8 tools/replay_muds_m3_mid_1126.py \
  --runtime "<当前 train_muds.py 所在目录>" \
  --run "<现在的 M3 seed43 正式训练输出目录>" \
  --data "<现有的 st3d_data/Muds>" \
  --out "<新目录 muds_m3_s43_midtest_1126>" \
  --device cuda:0
```

程序在打开测试图块前记录并冻结 `last.pth` 中最新完整的一轮。使用原始 RGB、八期、同区正式测试 80 块、跨区正式测试 640 块，每次一个图块，计算作者 SCDMetric 与严格转移 F1。按 GPU 实际资源选择设备。GPU 同时承担训练且发生持续严重节流或显存不足时可用 `--device cpu`。单次测试完成后继续原训练计划。

启动后回报冻结轮次、权重 SHA-256 和评估进程。完成后把 `interim_result.json` 作为小附件发到已有状态话题，并在文本列出种子、轮次、两种划分的 mIoU、SCS、BC、严格转移 F1 与结果文件的 SHA-256。权重保留在原机器。评估遇到问题时回传报错与选中权重信息，训练进程继续运行。
