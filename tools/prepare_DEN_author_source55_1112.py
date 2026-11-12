"""Verify/unpack public archives and decode only the active author fold's dates."""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import time
import zipfile
import zlib


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        while b := f.read(8 * 2**20):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    for key in ['manifest', 'manifest-sha', 'incoming', 'raw', 'cache', 'protocol', 'split-metadata']:
        ap.add_argument('--' + key, required=True)
    a = ap.parse_args()
    manifest_path = Path(a.manifest)
    assert sha(manifest_path) == a.manifest_sha
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    assert manifest['complete'] and manifest['archive_count'] == 55
    raw, cache = Path(a.raw).resolve(), Path(a.cache).resolve()
    raw.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    for record in manifest['archives']:
        archive = Path(a.incoming) / record['file']
        assert archive.stat().st_size == record['bytes'] and sha(archive) == record['SHA256']
        region = record['region']
        expected = {f'train/{region}/{region}_{day}_{band}.jpeg'
                    for day in range(730) for band in ['rgb', 'infra']}
        expected.add(f'labels/{region}.npy')
        with zipfile.ZipFile(archive) as z:
            assert len(z.namelist()) == len(expected) and set(z.namelist()) == expected
            assert z.testzip() is None
            for name in z.namelist():
                target = (raw / name).resolve()
                assert target.is_relative_to(raw) and not target.is_symlink()
                target.parent.mkdir(parents=True, exist_ok=True)
                info = z.getinfo(name)
                if target.exists():
                    assert target.stat().st_size == info.file_size
                    crc = 0
                    with target.open('rb') as f:
                        while b := f.read(2**20):
                            crc = zlib.crc32(b, crc)
                    assert crc == info.CRC, f'existing data mismatch retained: {target}'
                    continue
                temporary = target.with_name(target.name + f'.partial_{time.time_ns()}')
                with z.open(name) as source, temporary.open('xb') as output:
                    while b := source.read(2**20):
                        output.write(b)
                assert not target.exists()
                os.rename(temporary, target)
        print(json.dumps(dict(stage='verified_unpacked', region=region)), flush=True)
    split_bytes = Path(a.split_metadata).read_bytes()
    if (raw / 'split.json').exists():
        assert (raw / 'split.json').read_bytes() == split_bytes
    else:
        (raw / 'split.json').write_bytes(split_bytes)

    # Only these source-fold train/validation image pixels are decoded. Neither
    # the internal fold test nor original official validation/test is a loader.
    import numpy as np
    import torch
    import torchvision
    torch.set_num_threads(2)
    proto = json.loads(Path(a.protocol).read_text(encoding='utf-8'))
    regions = sorted(proto['train_regions'] + proto['validation_regions'])
    assert len(regions) == len(set(regions)) == 44
    assert not set(regions) & set(proto['internal_test_regions_locked'])
    original = json.loads(split_bytes)
    assert set(regions) <= set(original['train'])
    assert not set(regions) & set(original['val'] + original['test'])
    first = dt.date(2018, 1, 1)
    monthly = [(dt.date(year, month, 1) - first).days for year in [2018, 2019] for month in range(1, 13)]
    days = sorted({max(0, day + offset) for day in monthly for offset in range(-2, 3)})
    receipts = []
    shape = (8, 8, len(days), 128, 128, 4)
    for region in regions:
        target = cache / f'{region}.blk.npy'
        if not target.exists():
            temporary = cache / f'{region}.building_{time.time_ns()}.npy'
            mmap = np.lib.format.open_memmap(temporary, mode='w+', dtype=np.uint8, shape=shape)
            for j, day in enumerate(days):
                prefix = raw / 'train' / region / f'{region}_{day}'
                rgb = torchvision.io.read_image(str(prefix) + '_rgb.jpeg').numpy()
                infrared = torchvision.io.read_image(str(prefix) + '_infra.jpeg').numpy()
                assert rgb.shape == (3, 1024, 1024) and infrared.shape == (1, 1024, 1024)
                frame = np.concatenate([rgb, infrared], axis=0).transpose(1, 2, 0)
                mmap[:, :, j] = frame.reshape(8, 128, 8, 128, 4).transpose(0, 2, 1, 3, 4)
            mmap.flush()
            del mmap
            assert not target.exists()
            os.rename(temporary, target)
        array = np.load(target, mmap_mode='r')
        assert array.shape == shape and array.dtype == np.uint8
        del array
        receipts.append(dict(region=region, file=target.name, bytes=target.stat().st_size, SHA256=sha(target)))
        (cache / 'build_progress.json').write_text(json.dumps(dict(
            completed_regions=len(receipts), total_regions=44, last=receipts[-1])), encoding='utf-8')
        print(json.dumps(dict(stage='verified_cache', **receipts[-1])), flush=True)
    index = dict(days=days, tiles=regions, dtype='uint8', shape=list(shape),
                 manifest_SHA256=a.manifest_sha, source='raw JPEG decoded RGB+NIR; no normalization or resize',
                 decoded_internal_fold_test_pixels=False, archives_retained=True)
    (cache / 'index.json').write_text(json.dumps(index, indent=2), encoding='utf-8')
    (cache / 'receipt.json').write_text(json.dumps(dict(passed=True, index=index, archives=receipts), indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
