"""顺序读一遍大文件，把数据预热进系统文件缓存（不修改文件）。用法：python warm_cache.py 文件1 [文件2 ...]"""
import os
import sys
import time

for p in sys.argv[1:]:
    t0, n = time.time(), 0
    with open(p, "rb") as f:
        while True:
            b = f.read(64 << 20)
            if not b:
                break
            n += len(b)
    print(f"{p}: {n / 2**30:.2f} GB，{time.time() - t0:.0f} 秒，{n / 2**20 / max(1e-6, time.time() - t0):.0f} MB/s", flush=True)
