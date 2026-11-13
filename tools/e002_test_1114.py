"""E002 停训后补测试（11-14，组长决定：作者五折复现只作核对，三折都停，用现有权重各测一次）。

在训练用的同一个 Python 环境、同样的三个环境变量（E002_AUTHOR_REPO、E002_RAW_ROOT、E002_SSD_ROOT）下运行：
  python -X utf8 e002_test_1114.py --kit <放 e002_train.py / e002_data.py / protocol.json 的目录> --run <训练输出目录> --out <结果 JSON 路径>
做的事：
  1. 读 kit/protocol.json 里本折锁定的 11 个测试区（internal_test_regions_locked）和 11 个验证区；
  2. 分别载入 run/best_scs.pth（验证集 SCS 选出的权重，主结果）和 run/last.pth（停训时的权重，参照）；
  3. 按训练脚本 evaluate() 的同一口径（24 期、每区 64 块、fp16、argmax、ignore 6）在测试区和验证区上评估；
  4. 验证区的重算结果应与训练日志里 best 的验证分数一致（核对评估没走样）。
测试区如果不在 SSD 缓存里，自动改用作者原始 JPEG 读法，并先在一个验证样本上核对两种读法逐值相等。
只读权重、不改训练目录里的任何文件；结果 JSON 很小，回传它即可，权重不用传。
"""
import argparse, hashlib, json, sys, time, datetime as dt
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--kit', type=Path, required=True)
ap.add_argument('--run', type=Path, required=True)
ap.add_argument('--out', type=Path, required=True)
a = ap.parse_args()
sys.path.insert(0, str(a.kit.resolve()))

import torch
import e002_data
from e002_data import CachedAuthorDEN, collate_fn, RAW
from e002_metrics import SCDMetric
from data.data import DynamicEarthNet
from models.networks.multiutae import MultiUTAE


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        while b := f.read(8 << 20): h.update(b)
    return h.hexdigest()


proto_path = a.kit / 'protocol.json'
if not proto_path.is_file():
    proto_path = sorted(a.kit.glob('protocol_fold*.json'))[0]
proto = json.loads(proto_path.read_text(encoding='utf-8'))


class NativeEval(DynamicEarthNet):
    """作者原始 JPEG 读法；只把标签和图像目录限定到给定的区（与 CachedAuthorDEN 相同）。"""
    def __init__(self, regions):
        self.explicit_regions = sorted(regions); self._labels_loaded = None
        super().__init__(path=str(RAW), split='val', domain_shift=True, num_channels=4, num_classes=6,
                         img_size=128, true_size=1024, train_length=12, date_aug_range=2)
        self.image_folder = str(RAW / 'train')
    load_ground_truth = CachedAuthorDEN.load_ground_truth


def make(regions, tag, log):
    try:
        return CachedAuthorDEN(regions, 'val'), 'SSD 缓存'
    except AssertionError:
        log[tag + '_reader'] = 'native_JPEG'
        return NativeEval(regions), '作者原始 JPEG'


log = {}
val_ds, val_reader = make(proto['validation_regions'], 'val', log)
test_ds, test_reader = make(proto['internal_test_regions_locked'], 'test', log)
if test_reader != 'SSD 缓存' and val_reader == 'SSD 缓存':
    nat = NativeEval(proto['validation_regions'])
    for idx in [0, len(val_ds) // 2, len(val_ds) - 1]:
        x, y = val_ds[idx], nat[idx]
        for k in ['data', 'gt', 'positions']:
            assert torch.equal(x[k], y[k]), ('两种读法不一致', idx, k)
    log['native_vs_cache_val_samples_equal'] = 3

net = MultiUTAE(**proto['network']).cuda()


def to_cuda(samples):
    b = collate_fn(samples)
    return {k: v.cuda(non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in b.items()}


@torch.no_grad()
def run(ds):
    net.eval(); m = SCDMetric(6, ['impervi', 'agricult', 'forest', 'wetlands', 'soil', 'water'], ignore_index=6)
    for i in range(len(ds)):
        b = to_cuda([ds[i]])
        with torch.autocast('cuda', dtype=torch.float16): pred = net(b)['logits'].argmax(2)
        m.update(pred, b['gt'])
    return {k: float(v) for k, v in m.compute().items()}


res = {'time': dt.datetime.now().astimezone().isoformat(), 'author_fold': proto.get('author_fold'),
       'protocol_SHA256': sha(proto_path), 'script_SHA256': sha(Path(__file__)),
       'test_regions': sorted(proto['internal_test_regions_locked']), 'readers': {'val': val_reader, 'test': test_reader},
       'checks': log, 'results': {}}
for ck in ['best_scs.pth', 'last.pth']:
    st = torch.load(a.run / ck, map_location='cpu', weights_only=False)
    net.load_state_dict(st['model'])
    r = {'optimizer_updates': st['optimizer_updates'], 'logged_val': st.get('metrics'), 'checkpoint_SHA256': sha(a.run / ck)}
    for name, ds in [('test', test_ds), ('val', val_ds)]:
        t = time.time(); r[name] = run(ds); r[name + '_minutes'] = round((time.time() - t) / 60, 2)
        print(ck, name, {k: round(r[name][k], 2) for k in ['miou', 'bc', 'sc', 'scs']}, flush=True)
    res['results'][ck] = r
a.out.parent.mkdir(parents=True, exist_ok=True)
a.out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
print('结果已写入', a.out, 'SHA256', sha(a.out))
