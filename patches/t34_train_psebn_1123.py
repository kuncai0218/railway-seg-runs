"""T34 统一入口：原版 ST3DUNet 与单源 PSE 在同一份论文配方下训练与正式测试（由 make_entry.py 从冻结入口生成）。"""
import argparse
import hashlib
import importlib.metadata as metadata
import json
import math
try:
    import msvcrt
except ImportError:  # Linux
    msvcrt = None
    import fcntl


def _lock(f, on):
    if msvcrt is not None:
        msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK if on else msvcrt.LK_UNLCK, 1)
    else:
        fcntl.flock(f.fileno(), (fcntl.LOCK_EX | fcntl.LOCK_NB) if on else fcntl.LOCK_UN)
import os
import platform
import random
import sys
import time
from datetime import datetime
from functools import lru_cache
from pathlib import Path

HERE = Path(__file__).resolve().parent  # 包目录（作者快照、适配器、本机缓存清单）
T34 = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(T34))
AUTHOR = HERE / "upstream"
SOURCE_DATA = Path(os.environ["T34_DATA"])  # SitsSCD 版 DynamicEarthNet 根目录（含 train/val/test/labels/split.json）
CACHE = Path(os.environ["T34_CACHE"])  # 本机 SSD 上的解码缓存目录
OUTPUT = None  # main() 里按 --out 设定
EPOCHS = 250
ARGS = None
PER_DOMAIN = ("in", "out")

os.environ["ST3D_REPO"] = str(AUTHOR)
os.environ["ST3D_DATA"] = str(SOURCE_DATA.parent)
os.environ["ST3D_CACHE"] = str(CACHE)
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ["TQDM_DISABLE"] = "1"

import numpy as np
import psutil
import torch
import torchvision
import st3d_repro as released_adapter
from data.datamodule import ImageDataModule
from main import seed_all
from metrics.scd_metrics import SCDMetric
from models.losses import Losses
from train import SitsScdWrapper
from utils.wrapper import load_model, load_yaml


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def replace_file(temporary, path):
    for attempt in range(5):
        try:
            temporary.replace(path)
            return
        except PermissionError as exc:
            if getattr(exc, "winerror", None) not in (5, 32) or attempt == 4:
                raise
            time.sleep(0.05 * (attempt + 1))


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    replace_file(temporary, path)


def save_weights(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    replace_file(temporary, path)


def report(phase, **details):
    entry = {"time": now(), "pid": os.getpid(), "created_unix": psutil.Process().create_time(),
             "phase": phase, **details}
    save_json(OUTPUT / "status.json", entry)
    print(json.dumps(entry, ensure_ascii=False, allow_nan=False), flush=True)


def check_configuration():
    source = json.loads((HERE / "source_snapshot.json").read_text("utf-8"))
    for path, expected in source["source_sha256"].items():
        if sha256(HERE / path) != expected:
            raise RuntimeError(f"Frozen source changed: {path}")
    # Prepare the run without constructing or opening any test split.
    split_path = SOURCE_DATA / "split.json"
    split = json.loads(split_path.read_text("utf-8"))
    expected_sizes = {"train":55,"val":10,"test":10}
    assert all(len(split[s])==expected_sizes[s] and len(set(split[s]))==expected_sizes[s] for s in expected_sizes)
    assert len(set().union(*(split[s] for s in expected_sizes))) == 75
    manifest = json.loads((HERE / "decode_cache.json").read_text("utf-8"))
    assert manifest["cache"] == str(CACHE) and manifest["split_sha256"] == sha256(split_path)
    assert manifest["used_months"] == [0,3,6,9,13,16,19,23]
    assert manifest["torchvision"] == torchvision.__version__ and len(manifest["tiles"])==65
    for part in ("train","val"):
        index_path = CACHE / "DynamicEarthNet" / part / "index.json"
        index = json.loads(index_path.read_text("utf-8"))
        assert index["tiles"] == sorted(split[part])
        for tile in split[part]:
            row = manifest["tiles"][tile]
            path = index_path.parent / (tile + ".blk.npy")
            stat = path.stat()
            assert row["partition"] == part and row["days"] == index["days"]
            assert stat.st_size == row["bytes"] and stat.st_mtime_ns == row["mtime_ns"], path
    runtime = {
        "python":platform.python_version(),
        "torch":str(torch.__version__),
        "torchvision":torchvision.__version__,
        "cuda":str(torch.version.cuda),
        "cudnn":torch.backends.cudnn.version(),
        "numpy":np.__version__,
        "opencv-python":metadata.version("opencv-python"),
        "rasterio":metadata.version("rasterio"),
        "pandas":metadata.version("pandas"),
        "torchmetrics":metadata.version("torchmetrics"),
        "device":torch.cuda.get_device_name(0),
    }
    assert runtime["python"].startswith("3.10.")
    assert runtime["torch"] == "2.3.0+cu121" and runtime["cuda"] == "12.1"
    assert runtime["torchvision"] == "0.18.0+cu121"
    assert runtime["numpy"] == "1.26.4" and runtime["opencv-python"] == "4.12.0.88" and runtime["rasterio"] == "1.4.3"
    cfg = load_yaml(AUTHOR / "configs" / "dynamicearthnet.yaml")
    cfg["dataset"]["path"] = str(SOURCE_DATA)
    cfg["dataset"]["split_file"] = str(split_path)
    cfg["computer"]["accelerator"] = "cuda:0"
    cfg["trainer"].update(max_epochs=EPOCHS, lr=0.0001, weight_decay=0.001, cosine=True, warmup=False)
    assert cfg["dataset"]["seq_len"] == 8 and cfg["dataset"]["global_batch_size"] == 2
    assert cfg["dataset"]["num_channels"] == 4 and cfg["dataset"]["image_size"] == 128
    assert cfg["run"]["seed"] == 42 and cfg["computer"]["num_workers"] == 4  # 作者 YAML 原值
    cfg["run"]["seed"] = ARGS.seed
    contract = {
        "purpose":"T34: DynamicEarthNet Planet, paper-settings training, method=" + ARGS.method + ", seed=" + str(ARGS.seed),
        "method":ARGS.method, "seed":ARGS.seed, "smoke_steps":ARGS.smoke_steps,
        "t34_entry_sha256":sha256(Path(__file__)),
        "pse":({"opts":"", "norm":"bn2d", "w_prior":0.3, "w_event":1.0, "ema":False, "pse_net_sha256":sha256(T34 / "pse_net.py")} if ARGS.method == "pse" else None),
        "upstream":source,
        "training_script_sha256":sha256(Path(__file__)),
        "configuration":cfg,
        "runtime":runtime,
        "split_file_sha256":sha256(split_path),
        "cache_manifest_sha256":sha256(HERE / "decode_cache.json"),
        "rng_and_precision":{
            "python_numpy_torch_seed":ARGS.seed, "autocast":False,
            "cudnn_deterministic":True, "cudnn_benchmark":False,
            "cudnn_allow_tf32":torch.backends.cudnn.allow_tf32,
            "matmul_allow_tf32":torch.backends.cuda.matmul.allow_tf32,
            "cuda_launch_blocking":1, "cpu_threads":torch.get_num_threads(),
        },
        "patches_and_decisions":{
            "missing_attention_mode":"ltae, as required by the released forward output shape",
            "normalization":"all GroupNorm layers in PSENet replaced by BatchNorm2d(C); LayerNorm in temporal attention unchanged",
            "scheduler":"step once at the end of each epoch, T_max=250, eta_min=1e-6 (supports early --stop_epoch truncation with identical LR curve)",
            "training_loop":"reuse official train_one_epoch and train loader, replace broken logging and checkpointing",
            "storage":"lossless SSD uint8 block cache generated from author dataset JPEGs with the pinned torchvision",
            "selection":"independently maximize val mIoU per spatial domain, first epoch wins ties",
            "final_test":"after final checkpoint, frozen best checkpoints against explicit split=test, batch=1, original SCDMetric, raw JPEG",
        },
    }
    return cfg, split, contract


def create_loaders(cfg, split):
    train = released_adapter.make_set(cfg, "train")
    validate = {d:released_adapter.make_set(cfg, "val", d=="out") for d in PER_DOMAIN}
    assert train.sits_ids == sorted(split["train"]) and len(train) == 3300
    assert validate["in"].sits_ids == train.sits_ids and len(validate["in"]) == 110
    assert validate["out"].sits_ids == sorted(split["val"]) and len(validate["out"]) == 640
    assert [tuple(validate["in"].get_loc_per_split(i)) for i in (0,1)] == [(7,7),(6,6)]
    loaders = {}
    for domain, ds in validate.items():
        args = cfg["dataset"]
        owner = ImageDataModule(train, ds, ds, args["global_batch_size"], cfg["computer"]["num_workers"])
        if domain == "in":
            owner.setup("fit")
            fit = owner.get_train_loader()
        owner.setup("test")
        loaders[domain] = owner.get_test_loader()
        assert loaders[domain].batch_size == 1 and loaders[domain].num_workers == 0
    assert fit.num_workers == 4 and not fit.persistent_workers and fit.drop_last and len(fit) == 1650
    return fit, loaders


def verify_decoded_tensors(cfg, fit, validation):
    evidence = []
    for split, domain, fast, indices in (
        ("train",False,fit.dataset,(0,1649,3299)),
        ("val",False,validation["in"].dataset,(0,109)),
        ("val",True,validation["out"].dataset,(37,229,639)),
    ):
        raw = released_adapter.make_set(cfg, split, domain_shift=domain, cache=False)
        for item in indices:
            seed = 777 + item
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            decoded = raw[item]
            after_raw = random.getstate()
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            cached = fast[item]
            assert all(torch.equal(decoded[k], cached[k]) for k in ("data", "gt", "positions"))
            assert after_raw == random.getstate() and decoded["name"] == cached["name"]
            evidence.append({"split":split,"domain_shift":domain,"item":item,"seed":seed,
                             "pixel_label_day_identity":True,"same_augmentation_rng_state":True})
        del raw
    save_json(OUTPUT / "raw_cache_parity.json", evidence)


def rng_state():
    return {"random":random.getstate(), "numpy":np.random.get_state(), "torch":torch.get_rng_state(),
            "cuda":torch.cuda.get_rng_state_all()}


def restore_rng(value):
    random.setstate(value["random"])
    np.random.set_state(value["numpy"])
    torch.set_rng_state(value["torch"])
    torch.cuda.set_rng_state_all(value["cuda"])


def train_epoch(trainer, model, optimizer, epoch):
    """原版组：作者 train_one_epoch 原样调用。PSE 组：同一循环写法 + 事件辅助损失。冒烟时两组都只跑 smoke_steps 步。"""
    if ARGS.method == "orig" and not ARGS.smoke_steps:
        return trainer.train_one_epoch(epoch)
    model.train()
    sums, n = {}, 0
    lf = lambda pred, gt: trainer.compute_loss(pred=pred, batch=gt, average=True)["loss"]  # noqa: E731
    for step, batch in enumerate(trainer.trainloader):
        if ARGS.smoke_steps and step >= ARGS.smoke_steps:
            break
        batch = {k: v.to(trainer.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        x, y, date = batch["data"], batch["gt"], batch["positions"]
        logits = model(x, date)
        losses = trainer.compute_loss(pred=logits, batch=y, average=True)
        loss = losses["loss"]
        parts = {}
        if ARGS.method == "pse":
            aux, parts = model.aux_loss(y, lf, 0.3, 1.0)
            loss = loss + aux
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        for k, v in {**losses, **parts}.items():
            sums[k] = sums.get(k, 0.0) + float(v.detach())
        n += 1
    return {k: v / max(n, 1) for k, v in sums.items()}


def train(cfg, contract, fit, validation, resume):
    seed_all(cfg["run"]["seed"])
    device = cfg["computer"]["accelerator"]
    if ARGS.method == "pse":
        from pse_net import make_pse
        assert cfg["model"]["num_channels"] == 4 and cfg["ST3DUNET"]["INPUT_CHANNELS"] == 4  # 单源：in_prior = 0
        model = make_pse(cfg, "")
        def _gn_to_bn(m):
            for n, ch in m.named_children():
                if isinstance(ch, torch.nn.GroupNorm):
                    setattr(m, n, torch.nn.BatchNorm2d(ch.num_channels))
                else:
                    _gn_to_bn(ch)
        _gn_to_bn(model)
        assert not any(isinstance(x, torch.nn.GroupNorm) for x in model.modules()), "GroupNorm still present!"
        assert any(isinstance(x, torch.nn.BatchNorm2d) for x in model.modules()), "No BatchNorm2d created!"
        model = model.to(device)
        assert model.iq == 0 and model.event
    else:
        model = load_model(cfg).to(device)
        assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 1_652_262
    report("model", method=ARGS.method, trainable=sum(p.numel() for p in model.parameters() if p.requires_grad))
    assert torch.backends.cudnn.deterministic and not torch.backends.cudnn.benchmark
    settings = cfg["trainer"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["lr"], weight_decay=settings["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=settings["lr"] * 0.01)
    trainer = SitsScdWrapper(
        model=model, loss_fn=Losses(**cfg["loss"]), val_metrics=SCDMetric(
            cfg["dataset"]["num_classes"], cfg["dataset"]["class_names"], cfg["dataset"]["ignore_index"]),
        trainloader=fit, testloader_in=validation["in"], testloader_out=validation["out"],
        optimizer=optimizer, scheduler=scheduler, device=device, seq_len=cfg["dataset"]["seq_len"],
    )
    last = OUTPUT / "last.pth"
    epoch_start = 1
    history = []
    best = {domain:None for domain in PER_DOMAIN}
    if resume:
        state = torch.load(last, map_location="cpu", weights_only=False)
        assert state["protocol_sha256"] == sha256(OUTPUT / "protocol.json")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        epoch_start, history, best = state["epoch"]+1, state["history"], state["best"]
        assert scheduler.last_epoch == state["epoch"]
        restore_rng(state["rng"])
        del state
    protocol_sha = sha256(OUTPUT / "protocol.json")
    steps_per_epoch = len(fit)
    last_epoch = 2 if ARGS.smoke_steps else int(ARGS.stop_epoch)
    assert 1 <= last_epoch <= EPOCHS, f'Invalid stop_epoch={last_epoch}'
    for epoch in range(epoch_start, last_epoch+1):
        start = time.monotonic()
        step = 0
        last_progress = start
        lr = optimizer.param_groups[0]["lr"]
        report("training", epoch=epoch, epochs=EPOCHS, step=0, steps_per_epoch=steps_per_epoch, learning_rate=lr)
        def after_step(_optimizer, _args, _kwargs):
            nonlocal step, last_progress
            step += 1
            sample = time.monotonic()
            if step == 1 or step == steps_per_epoch or sample-last_progress >= 20:
                if not optimizer.state:
                    raise RuntimeError("No AdamW parameter state after update.")
                last_progress = sample
                report("training", epoch=epoch, epochs=EPOCHS, step=step, steps_per_epoch=steps_per_epoch,
                       optimizer_updates=(epoch-1)*steps_per_epoch+step, learning_rate=lr,
                       steps_per_second=step/(sample-start),
                       gpu_peak_MiB=round(torch.cuda.max_memory_allocated()/2**20))
        hook = optimizer.register_step_post_hook(after_step)
        try:
            losses = train_epoch(trainer, model, optimizer, epoch)
        finally:
            hook.remove()
        assert (step == steps_per_epoch or ARGS.smoke_steps) and all(math.isfinite(loss) for loss in losses.values())
        scheduler.step()
        entry = {"epoch":epoch,"lr_used":lr,"lr_next":optimizer.param_groups[0]["lr"],
                 "loss":losses, "train_seconds":time.monotonic()-start}
        model_file = f"epoch{epoch:03d}.pth"
        keep_model = epoch % cfg["trainer"]["ckpt_interval"] == 0 or epoch == last_epoch or epoch == EPOCHS
        for domain in PER_DOMAIN:
            report("validation", epoch=epoch, domain=domain, updates=epoch*steps_per_epoch)
            scores = released_adapter.run_eval(model, validation[domain], trainer.metric_fn, trainer.device)
            entry["val_" + domain] = scores
            if best[domain] is None or scores["miou"] > best[domain]["score"]:
                keep_model = True
                best[domain] = {"epoch":epoch,"score":scores["miou"],"weights":model_file}
        history.append(entry)
        if keep_model:
            save_weights(OUTPUT / model_file, {"model": model.state_dict(), "epoch":epoch,
                                             "protocol_sha256":protocol_sha})
        save_weights(last, {"model":model.state_dict(),"optimizer":optimizer.state_dict(),
                           "scheduler":scheduler.state_dict(),"epoch":epoch,
                           "history":history,"best":best,"rng":rng_state(),"protocol_sha256":protocol_sha})
        save_json(OUTPUT / "epochs.json", history)
        report("epoch_checkpoint", epoch=epoch, epochs=EPOCHS, loss=losses["loss"], best=best)
    if not ARGS.smoke_steps:
        expected_lr = 1e-6 + 0.5 * (settings["lr"] - 1e-6) * (1.0 + math.cos(math.pi * last_epoch / EPOCHS))
        assert scheduler.last_epoch == last_epoch and abs(optimizer.param_groups[0]["lr"] - expected_lr) < 1e-12, (
            scheduler.last_epoch, last_epoch, optimizer.param_groups[0]["lr"], expected_lr
        )
    return model, trainer.metric_fn, best


def run_test(cfg, split, contract, model, metric, best):
    selected = {}
    for domain in PER_DOMAIN:
        weights = OUTPUT / best[domain]["weights"]
        selected[domain] = {**best[domain],"weight_sha256":sha256(weights)}
    save_json(OUTPUT / "test_selected_checkpoints.json", selected)
    evidence = {}
    for domain in PER_DOMAIN:
        shift = domain == "out"
        epoch = best[domain]["epoch"]
        ckpt = torch.load(OUTPUT / best[domain]["weights"], map_location="cpu", weights_only=True)
        assert ckpt["epoch"] == epoch and ckpt["protocol_sha256"] == sha256(OUTPUT / "protocol.json")
        model.load_state_dict(ckpt["model"], strict=True)
        dataset = released_adapter.make_set(cfg, "test", domain_shift=shift, cache=False)
        assert dataset.split == "test"
        assert dataset.sits_ids == sorted(split["test" if shift else "train"])
        assert len(dataset) == (640 if shift else 110)
        if not shift:
            assert [tuple(dataset.get_loc_per_split(i)) for i in (0,1)] == [(7,6),(6,7)]
        # Evaluation always uses the tile's same eight dates; reuse those identical JPEG decodes.
        decode_tile = lru_cache(maxsize=1)(dataset.load_data)
        def load_tile(i, tile, months, path):
            return decode_tile(i, tile, tuple(months), path)
        dataset.load_data = load_tile
        owner = ImageDataModule(dataset, dataset, dataset, 2, 0)
        owner.setup("test")
        report("official_test", domain=domain, blocks=len(dataset), epoch=epoch)
        scores = released_adapter.run_eval(model, owner.get_test_loader(), metric, "cuda:0")
        evidence["test_" + domain] = {"epoch":epoch, "selected_using_val_miou":best[domain]["score"],
                                     "weights":selected[domain]["weight_sha256"],"regions":dataset.sits_ids,
                                     "blocks":len(dataset), "scores":scores}
        decode_tile.cache_clear()
        del dataset, owner, decode_tile
    save_json(OUTPUT / "official_test.json", {"results":evidence,"runtime":contract["runtime"],
                                             "protocol_sha256":sha256(OUTPUT / "protocol.json"),"finished":now()})
    report("complete", completed_epochs=int(ARGS.stop_epoch), lr_tmax=EPOCHS, metrics=evidence)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, choices=[42, 43, 44], required=True)
    parser.add_argument("--method", choices=["orig", "pse"], required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--smoke_steps", type=int, default=0, help="冒烟：每轮只跑这么多步、2 轮、不做正式测试")
    parser.add_argument("--stop_epoch", type=int, default=120, help="截停轮次（默认 120，保持 T_max=250 余弦学习率曲线不变）")
    args = parser.parse_args()
    global OUTPUT, ARGS
    ARGS = args
    OUTPUT = Path(args.out)
    if not torch.cuda.is_available():
        raise RuntimeError("The pinned PyTorch runtime needs a CUDA device.")
    torch.set_num_threads(2)
    OUTPUT.mkdir(exist_ok=True)
    with (OUTPUT / "run.lock").open("a+b") as lock:
        if lock.tell() == 0:
            lock.write(b"1")
            lock.flush()
        lock.seek(0)
        _lock(lock, True)
        try:
            protocol = OUTPUT / "protocol.json"
            if (OUTPUT / "last.pth").exists() != args.resume:
                raise RuntimeError("For an existing checkpoint explicitly use --resume; a new run requires clean output.")
            if (OUTPUT / "official_test.json").exists():
                raise RuntimeError("Official test already completed.")
            report("preflight")
            cfg, split, contract = check_configuration()
            if args.resume:
                if json.loads(protocol.read_text("utf-8")) != contract:
                    raise RuntimeError("Protocol differs from the existing checkpoint.")
            else:
                save_json(protocol, contract)
            fit, validation = create_loaders(cfg, split)
            if not args.resume:
                verify_decoded_tensors(cfg, fit, validation)
            model, metric, best = train(cfg, contract, fit, validation, args.resume)
            del fit, validation
            if ARGS.smoke_steps:
                report("smoke_complete", best=best)
            else:
                run_test(cfg, split, contract, model, metric, best)
        except Exception as exc:
            report("failed", error=repr(exc))
            raise
        finally:
            lock.seek(0)
            _lock(lock, False)


if __name__ == "__main__":
    main()

