"""Receive the original D2LS experiment over the campus LAN and start it."""

import argparse
import hashlib
import hmac
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
from zipfile import ZipFile


DATA = ("inputs_f16.npy", "labels_u8.npy", "index.json", "norm.json", "splits_periods.json")
WEIGHTS = "convnext_base-6075fbad.pth"


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def campus_request(root, name, file, args, resume=False):
    command = ["curl", "--fail", "--silent", "--show-error", "--noproxy", "*",
               "--interface", args.interface, "--connect-timeout", "8", "--max-time", "1800",
               "--output", str(file), "--write-out", "peer=%{remote_ip} local=%{local_ip} bytes=%{size_download}"]
    if resume and file.exists():
        command.extend(["--continue-at", "-"])
    result = subprocess.run([*command, f"{root}/{name}"], check=True, capture_output=True, text=True)
    match = re.fullmatch(r"peer=(\S+) local=(\S+) bytes=(\d+)", result.stdout.strip())
    if not match or match[1] != args.address or match[2] != args.interface:
        raise RuntimeError("Campus transfer did not use the verified physical interface")
    print(name, result.stdout.strip(), flush=True)
    return int(match[3])


def fetch(root, name, dest, record, args, existing=None):
    valid = lambda p: p.is_file() and p.stat().st_size == record["bytes"] and sha(p) == record["sha256"]
    if valid(dest):
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    if existing is not None and valid(existing):
        shutil.copyfile(existing, dest)
        print("Verified and reused", existing, flush=True)
        return dest
    partial = dest.with_suffix(dest.suffix + ".partial")
    if not valid(partial):
        if partial.exists() and partial.stat().st_size >= record["bytes"]:
            partial.unlink()
        campus_request(root, name, partial, args, resume=True)
    if not valid(partial):
        raise ValueError(f"Incorrect download digest for {name}")
    partial.replace(dest)
    return dest


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--address", required=True)
    cli.add_argument("--port", type=int, required=True)
    cli.add_argument("--interface", required=True)
    cli.add_argument("--prior-pack", type=Path, required=True)
    cli.add_argument("--nonce", required=True)
    cli.add_argument("--probe", required=True)
    cli.add_argument("--manifest", required=True)
    cli.add_argument("--c23", type=Path)
    cli.add_argument("--convnext", type=Path)
    cli.add_argument("--seed", type=int, choices=(20261018, 20261019), required=True)
    cli.add_argument("--parts", nargs="+", choices=("full", "A", "B"), default=["full", "A", "B"])
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()

    with args.prior_pack.open("rb") as f:
        f.seek(262144)
        key = f.read(65536)
    if len(key) != 65536:
        raise ValueError("Expected the already verified r3_2023a_train.npz")
    token = hmac.new(hashlib.sha256(key).digest(), b"railway-spatial-v1:" + args.nonce.encode("ascii"), "sha256").hexdigest()
    root = f"http://{args.address}:{args.port}/{token}"
    work = args.output.resolve()
    work.mkdir(parents=True, exist_ok=True)
    probe = work / "route_probe.bin"
    if campus_request(root, probe.name, probe, args) != 4096 or sha(probe) != args.probe:
        raise ValueError("Campus 4 KiB probe differs")
    transfer = work / "transfer_manifest.json"
    campus_request(root, "manifest.json", transfer, args)
    if sha(transfer) != args.manifest:
        raise ValueError("Transfer manifest checksum differs")
    files = json.loads(transfer.read_text(encoding="utf-8"))["assets"]
    package_name = "frozen_d2ls_peer.zip"
    package = fetch(root, package_name, work / package_name, files[package_name], args)
    with ZipFile(package) as z:
        z.extractall(work)
    c23 = work / "c23"
    if args.c23 and all((args.c23 / name).is_file() and sha(args.c23 / name) == files[name]["sha256"] for name in DATA):
        c23 = args.c23.resolve()
        print("Verified existing C23", c23, flush=True)
    else:
        for name in DATA:
            fetch(root, name, c23 / name, files[name], args, None if args.c23 is None else args.c23 / name)
    fetch(root, WEIGHTS, work / "assets/weights" / WEIGHTS, files[WEIGHTS], args, args.convnext)
    cmd = [sys.executable, "-u", str(work / "retrain_frozen_d2ls.py"),
           "--manifest", str(work / "split_manifest.json"), "--record", str(work / "archived_run.json"),
           "--code", str(work / "code"), "--assets", str(work / "assets"), "--c23", str(c23),
           "--seed", str(args.seed), "--parts", *args.parts, "--output", str(work / "training")]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
