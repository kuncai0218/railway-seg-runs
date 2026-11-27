# MUDS 错题拆解（11-27，只读验证集、不训练）

## 为什么做

MUDS 上新方法（M3）比骨干（A1）BC 高很多、SC 却低一些。组长要逐个新建事件拆开看：新楼是“准时认出、提前、晚了、还是根本没认出”，新方法在哪一层把它丢掉的（骨干分数 → 场景校准 → 时间链），以及几种不训练的改读法能不能让 SC、BC 一起涨。

原来想让你们把权重传回来（`jia-muds-weights-s43-1127` / `yi-muds-weights-s44-1127`），但 ntfy 单个附件上限只有 2 MB，权重每个约 7 MB 传不了，那两条**取消**。改为在你们本机用自己的权重跑拆解脚本，只把几百 KB 的结果传回来。

## 包

- `patches/muds_dissect_peer_1127.zip`，SHA256 `8813633f339e07d28686b6a85e38ffbadbef8c54dc488d357497789d2a7f844d`（22207 字节）。解压后是一个 `dissect/` 目录（8 个 .py，只依赖 numpy、torch 和配对包本身，不需要装任何新包）。
- **解压到你们跑 `train_muds.py` 的那个目录里**（配对包 `muds_m3_matched_1126` 解压出来的目录，里面有 `train_muds.py` 和 `upstream/`），解压后应有 `dissect/run_dissect.py`。

## 命令

在 `train_muds.py` 所在目录，用训练 MUDS 时同一个 Python：

```
python -X utf8 dissect/run_dissect.py --seed <种子> --data <Muds 数据目录> --out <一个新的空目录>
    --m3_in <新方法 best_in 权重> --m3_out <新方法 best_out 权重>
    --a1_in <骨干 best_in 权重> --a1_out <骨干 best_out 权重>
```

`--data` 就是训练时 `--data` 用的那个 Muds 目录（里面有 `split.json`）。四个权重就是 `official_test.json` 里写的那四个（指令条目里列了路径）。

它依次做：

1. 逐像元导出：新方法、骨干在同区验证（80 块）、跨区验证（640 块）上各推理一遍，另把跨区权重在同区验证上再推理一遍。用 GPU，但每个进程只占约 15% 显存，几分钟一份，正在跑的训练不用停。
2. 归类、逐层追踪、空间诊断、两种改读法的参数网格（只用 CPU，可能要几十分钟到一两个小时）。
3. 最后把所有结果 JSON 和日志打成 `<out>/dissect_results_s<种子>.zip`（几百 KB），打印字节数和 SHA256。

中断了重复同一条命令即可，已完成的步骤会跳过。导出的数组约 2.5 GB 留在 `--out` 目录，**组长确认收到结果之前不要删**。

## 回报

- 收到：本条编号、实际读到的 main 提交号、包 SHA256 核验结果。
- 完成：把 `dissect_results_s<种子>.zip` 作为 ntfy 附件发到原状态话题（几百 KB，不超过 2 MB 上限），正文写字节数、SHA256、总用时。
- 出错：停下，把 `<out>/run_dissect_s<种子>.log` 的最后 60 行贴回来，不要自己改脚本。
