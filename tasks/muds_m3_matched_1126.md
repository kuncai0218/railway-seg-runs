# MUDS 原版、PSE 与 M3 对照

## 甲乙分工

| 节点 | 种子 | 首项 | 接续 |
|---|---:|---|---|
| 甲 | 43 | M3 | 同种子 PSE/A1，回收已有原版和旧 PSE |
| 乙 | 44 | PSE/A1 | 同种子 M3，回收已有原版 |

收到后先回报任务编号、实际读取的主分支提交号、包哈希、GPU 是否有活跃进程。继续已有正常任务，设备释放后执行表里的顺序。启动后回报实际 `step`、PID 和利用率样本。有同种子、同结构的可复用完整结果时先提供配置、检查点和日志，省去重复训练。

## 对照口径

- 使用现有 `st3d_data/Muds`。作者适配版划分为 40 个训练区、10 个验证区、10 个测试区，`split.json` SHA256 为 `008b2db51aabf66b23b9d9d644b394285a36687f3a0dcdbc6397c5f28abf7b7f`。RGB 三通道，按原加载器从可用月份均匀取八期，128×128，建筑／非建筑，忽略值 2。沿用已核验的 `st3d_cache/Muds`，未建缓存时可省略 `--cache` 并使用作者原始 JPEG。
- A1 使用 `event=0,gate=0` 的 PSE 骨干。M3 使用同一 A1 加冻结代码版 SNT v2.1 h3、SCPC、辐射增强和加权 CRF，改变数据输入为 MUDS 的三通道、两类。已有 `muds_pse_s42/43` 和 `muds_pseg0_*` 的事件分支配置不同，保留为历史对照。
- 两个新训练配对使用每轮 1,200 步、批大小 2、250 轮、AdamW `1e-4→1e-6` 逐轮余弦、相同数据抽样种子。语义 CE 类别权重 `[.25,.75]`。M3 保留既有头部损失和权重衰减。验证原生同区 80 块、跨区 640 块，推理逐块，按各自验证 mIoU 选轮。训练完成后固定已选权重进入正式测试，同区 80 块、跨区 640 块，保存作者 SCDMetric 和严格转移指标准确计数。
- 原版的历史三种子记录优先回收。种子 42 已有正式复算。种子 43/44 如有 250 轮验证日志和对应 `best_in.pth`、`best_out.pth`，直接在相同的测试入口回放，记录历史选轮的验证批量和配置差异。缺权重或日志时回报缺项，再在本配方补齐相应种子。正式比较写清原版是历史复算还是同入口重训。

## 命令

以下是甲的例子。乙把种子改成 44，先运行 PSE，再运行 M3。路径替换为已有数据、缓存和独立输出目录。原先的 Planet 四臂短试保持完成归档。

```bash
python -B -u -X utf8 train_muds.py --data "<已有 Muds>" --cache "<已有 Muds 缓存>" --method m3 --seed 43 --out "<冒烟目录>" --device cuda:0 --workers 0 --smoke
python -B -u -X utf8 train_muds.py --data "<已有 Muds>" --cache "<已有 Muds 缓存>" --method m3 --seed 43 --out "<M3 43 正式新目录>" --device cuda:0 --workers 2
python -B -u -X utf8 train_muds.py --data "<已有 Muds>" --cache "<已有 Muds 缓存>" --method pse --seed 43 --out "<PSE A1 43 正式新目录>" --device cuda:0 --workers 2
```

在 Windows 可沿用现有 Python 解释器。根据实际环境，可在同一节点两个正式臂开始前一起确定可工作的 DataLoader worker 数。训练若中断，带原参数和 `--resume` 读取 `last.pth`。已完成的权重、日志和数据保留。

历史原版的回放入口如下。需要重训原版时去掉 `--replay-original`，指向新正式目录。

```bash
python -B -u -X utf8 train_muds.py --data "<已有 Muds>" --method st3d --seed 43 --replay-original "<已有 muds_paper_s43>" --out "<回放新目录>" --device cuda:0
```

每项交付 `protocol.json`、训练曲线、`official_test.json`、运行日志及各权重 SHA256。历史 PSE 先找 `final.json` 和 `config.json` 原件，保留它的原测试口径。仅回传小结果文件和日志，权重保留原节点。沿用原状态话题回传。
