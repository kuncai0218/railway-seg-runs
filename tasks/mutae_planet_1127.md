# DynamicEarthNet Planet：官方 Multi-UTAE 基线（统一协议，11-27）

## 为什么做

我们的骨干 A1 与 Multi-UTAE 属于同一家结构，审稿人一定会问“和作者原版 Multi-UTAE 比怎样”。统一协议下我们还没有 Multi-UTAE 的结果。这次用 SitsSCD 作者代码原样的 MultiUTAE（作者 `multiutae.yaml` 的超参），数据、批大小 2、250 轮、AdamW 1e-4 逐轮余弦、作者损失、按验证 mIoU 分域选轮、正式测试，全部与 11-19 的 T34 配对（原版 ST3DUNet、PSE）逐字相同，只换网络。

组长本机跑种子 42，甲跑 43，乙跑 44。

## 包

- 附加包：`patches/t34_mutae_addon_1127.zip`，SHA256 `befc0085705d81279ea25c38c70208c6c27af5878ae976247bf92cd7a4f10fe4`（20,372 字节）。
- **解压到本机已有的 `t34_peer_1119` 目录的上一级**（也就是 11-19 原包解压的位置），选“不覆盖已有文件”。它只新增文件，不改原包任何文件。解压后应当有：
  - `t34_peer_1119/train_t34_mutae.py`（与 `train_t34.py` 同目录）
  - `t34_peer_1119/test_last_mutae.py`
  - `t34_peer_1119/run_mutae.py`
  - `t34_peer_1119/sitsscd/models/networks/` 下 5 个文件（作者网络代码原样）
- 环境、数据、缓存全部沿用：同一个 Python 3.10 环境（torch 2.3.0+cu121，不升级任何包），`T34_DATA`、`T34_CACHE` 与之前 T34 / PSE / SNT 任务完全相同。缓存 11-19 已经生成，不用重做。

## 命令

在 `t34_peer_1119/` 目录下（`<S>` 为种子，`<runs>` 用之前 T34 作业的根目录）：

```
python -X utf8 run_mutae.py --seed <S> --runs <runs>
```

它依次做：

1. 冒烟 20 步（`<runs>/_smoke_mutae_s<S>`）。日志里 `model` 一行的 `trainable` 必须是 **16171526**，不一致就停下回报原文。
2. 正式训练 250 轮（`<runs>/den_planet_mutae_s<S>`）。训练结束时入口自动按同区、跨区各自的验证 mIoU 选轮，并跑正式测试，写出 `official_test.json`。
3. 用第 250 轮权重补测一次，写出 `official_test_last.json`（作参考，不参与任何选择）。

中断后重复同一命令即可：已完成的步骤跳过，有 `last.pth` 的会带 `--resume` 续跑。失败会停下并在 `<runs>/mutae_s<S>_state.json` 里写明，不会自动改配方。

预计时长：3090 上约一天到一天半（网络约 1617 万参数，比 PSE 大）。

## 回报

- 收到：本条编号、实际读到的 main 提交号、附加包 SHA256 核验结果、解压后上面 8 个新文件都在。
- 冒烟：`_smoke_mutae_s<S>/status.json` 最后一行和日志里 `model` 那一行。
- 正式训练期间每天至少一次：`status.json`（轮次、更新数、每秒步数）。
- 完成后：`protocol.json`、`epochs.json`、`test_selected_checkpoints.json`、`official_test.json`、`official_test_last.json`、三份日志的最后 50 行，以及选中权重和第 250 轮权重的 SHA256。权重留在本机，不上传。GitHub 不通时走 ntfy 附件，写明 zip 的 SHA256。
