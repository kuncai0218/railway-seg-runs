# DEN-Planet 修正版同头消融（11-27）

## 为什么做

现在的完整版（M3）在新地区把误报压下去了，但也漏掉了一部分真变化。要写成论文，必须用从头训练的对照讲清楚：收益来自时间链本身、来自看两端影像的有向转移，还是来自按类别的“正常外观变化”（常态）。同时修掉两个已知实现问题：
1. 端点交叉熵的类别权重乘了两次；
2. 端点证据有任意零点：某一期所有类别同加一个常数，会改变报警量。

这两处修正在 `snt_1127_clean/normalcy_crf.py` 里。组长已在本机做过 CPU 单元测试：关掉两项修正时，新旧代码输出逐位相同；修正后，对某一期端点分数整体平移，后验不变。

## 五个臂（`--head`）

所有臂都与已有的 M1 逐项一致：A1 骨干（`event=0,gate=0`）、SCPC 关、辐射增强关、120 轮（余弦按 250 轮）、学习率 1e-4、加权 CRF。只换头：

| `--head` | 含义 | 期望横幅里的 total |
|---|---|---|
| `h3` | 分类常态（主方法） | 1,747,313 |
| `none` | 无头，只有 A1 | 1,687,240 |
| `h0` | 固定转移：只有一张可学的 6×6 转移表，不看影像 | 1,719,091 |
| `h2` | 共享常态：所有类共用一个常态 | 1,738,493 |
| `hdir` | 有向观测：有向边和端点证据，不建常态 | 1,729,694 |

顺序：`h3` → `none` → `h0`。若队列里届时没有更新的条目，接着跑 `h2` → `hdir`。

## 包与环境

- 包：`patches/pkg_snt_clean_1127.zip`，SHA256 `d31b1bdd645f8bdb23734b2e8f6a33181d1da273a37cfb6d0da933aa17c60d5a`。解压到新目录，得到 `snt_1127_clean/`。
- 环境变量沿用之前跑 SNT v2.1（`*-t34-snt-m3-s44-1124` 等）时那一套：`SNT_T34_DIR`、`SNT_HERE_DIR` 指向本机已有的 `t34_peer_1119` 目录；`T34_DATA`、`T34_CACHE` 与 PSE+BN 任务完全相同。Python 环境不变，不要升级任何包。

## 命令（`<S>` 为种子，`<H>` 为头）

在 `snt_1127_clean/` 目录下执行。

冒烟（每个臂第一次启动前做一次）：

```
python -B -u -X utf8 train_snt.py --dataset planet --seed <S> --head <H> --scpc 0 --rad_aug 0 --eval_aoi_bank 0 --backbone_opts "event=0,gate=0" --epochs 120 --lr_tmax 250 --lr 1e-4 --head_lr 1e-4 --crf_weighted 1 --crf_chg_boost 2.5 --select_metric miou --smoke_steps 20 --out <runs>/_smoke_clean_<H>_s<S>
```

横幅必须以 `[SNT-clean-1127] head=<H>` 开头，total 与上表一致。不一致就停下，回报原文。

正式训练：同一命令去掉 `--smoke_steps 20`，`--out <runs>/clean_<H>_planet_s<S>`。中断后带原参数加 `--resume`。

正式测试（训练完立即做）：

```
python -u -X utf8 official_planet_clean.py --run <runs>/clean_<H>_planet_s<S>
```

它按训练时记录的 best：同区用 val_in 最佳 mIoU 轮、跨区用 val_out 最佳 mIoU 轮，另测第 120 轮，写出 4 个 `midtest_ep*_test_*.json` 和汇总 `official_clean.json`。

## 回传（每完成一个臂就交，不要攒着）

经原状态通道附件发回：`config_snt.json`、`epochs.json`、`status.json`、`official_clean.json`、4 个 `midtest_ep*_test_*.json`、日志最后 50 行，并写明 zip 的 SHA256。权重留在本机。GitHub 不通时走 ntfy 附件。

## 旧作业补测（只推理，在旧目录里做）

修正版代码会改变端点证据的算法，**不能**用它来测旧模型。旧的 v2.1 作业要用原来的 `snt_1123_v2` 目录和其中的 `midtest_v2.py`（即之前发过的 `snt_midtest_v2_1124.py`，SHA256 `f6d07c17…80ab6`）：

```
python -u -X utf8 midtest_v2.py --run <旧作业目录> --epoch <status.json 里 best.in.epoch> --split test_in
python -u -X utf8 midtest_v2.py --run <旧作业目录> --epoch <status.json 里 best.out.epoch> --split test_out
```

若对应结果文件已存在就跳过；若所需轮次的权重不在，回报缺哪个，不要补训。
