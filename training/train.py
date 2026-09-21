"""Fine-tune EfficientNetV2-L on ImageNet-1k + the Perturb adversarial dataset."""
from __future__ import annotations

import argparse
import io
import json
import logging
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from PIL import Image
from torch.utils.data import DataLoader, Dataset, IterableDataset

from common import (
    DEFAULT_ADV_DATASET,
    DEFAULT_IMAGENET_DATASET,
    LabelMapper,
    eval_transform,
    load_model,
    save_model,
    train_transform,
)

logger = logging.getLogger("train")


# --------------------------------------------------------------------------- args


@dataclass
class Args:
    imagenet: str
    adv_dataset: str
    init: str
    output_dir: str
    epochs: int
    steps_per_epoch: int
    batch_size: int
    adv_rows: int
    adv_per_row: int
    adv_weight: float
    consistency_weight: float
    label_smoothing: float
    lr: float
    weight_decay: float
    warmup_steps: int
    grad_accum: int
    train_res: int
    amp: bool
    train_bn: bool
    num_workers: int
    eval_imagenet_samples: int
    eval_adv_rows: int
    eval_every: int
    save_every: int
    keep: str
    seed: int
    label_map: str | None
    hf_token: str | None


def parse_args() -> Args:
    load_dotenv()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--imagenet", default=os.getenv("IMAGENET_DATASET", DEFAULT_IMAGENET_DATASET),
                   help="HF dataset id (streamed) or local ImageFolder root with train/ and val/")
    p.add_argument("--adv-dataset", default=os.getenv("ADV_DATASET", DEFAULT_ADV_DATASET))
    p.add_argument("--init", default="pretrained", help="'pretrained' (torchvision), a model dir, or an HF model repo")
    p.add_argument("--output-dir", default=f"runs/{time.strftime('%Y%m%d-%H%M%S')}")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--steps-per-epoch", type=int, default=2000, help="optimiser steps per epoch (streams have no length)")
    p.add_argument("--batch-size", type=int, default=8, help="clean ImageNet images per micro-batch")
    p.add_argument("--adv-rows", type=int, default=4, help="adversarial-dataset rows per micro-batch (0 disables)")
    p.add_argument("--adv-per-row", type=int, default=2, help="adversarial versions sampled per row")
    p.add_argument("--adv-weight", type=float, default=1.0)
    p.add_argument("--consistency-weight", type=float, default=0.5, help="KL(clean || adversarial) weight")
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--train-res", type=int, default=480, help="crop size for clean ImageNet (adversarial rows always use 480)")
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.add_argument("--train-bn", action="store_true",
                   help="update BatchNorm running statistics (default: frozen; only sensible with micro-batches >= 32)")
    p.add_argument("--num-workers", type=int, default=4, help="ImageNet loader workers; ~1-2 GB host RAM each")
    p.add_argument("--eval-imagenet-samples", type=int, default=2000)
    p.add_argument("--eval-adv-rows", type=int, default=300)
    p.add_argument("--eval-every", type=int, default=0, help="steps between evaluations (0 = end of each epoch only)")
    p.add_argument("--save-every", type=int, default=200,
                   help="steps between weight-only checkpoints to <output-dir>/last (0 = only at evaluations)")
    p.add_argument("--keep", choices=("both", "best"), default="both",
                   help="checkpoints kept on disk, each overwritten in place: 'both' = last/ and best/ (2 x 476 MB), "
                        "'best' = only best/ (476 MB; no crash-recovery checkpoint)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--label-map", default=None, help="JSON {label_name: imagenet1k_index} overrides")
    p.add_argument("--hf-token", default=os.getenv("HF_TOKEN") or None)
    a = p.parse_args()
    return Args(**vars(a))


# --------------------------------------------------------------------------- data


def _pil(value: Any) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, dict) and value.get("bytes") is not None:
        return Image.open(io.BytesIO(value["bytes"])).convert("RGB")
    if isinstance(value, dict) and value.get("path"):
        return Image.open(value["path"]).convert("RGB")
    raise TypeError(f"unsupported image value: {type(value)}")


def _worker_id() -> int:
    # HF streaming datasets shard themselves across DataLoader workers (and stop
    # surplus workers when there are fewer shards), so we only need the id for seeding.
    info = torch.utils.data.get_worker_info()
    return info.id if info is not None else 0

# Rows per Arrow batch when streaming parquet; the default (one whole row group per
# worker) OOM-kills workers on hosts with modest RAM.
STREAM_BATCH_ROWS = 64


def _stream(repo: str, split: str, token: str | None):
    from datasets import load_dataset

    try:
        return load_dataset(repo, split=split, streaming=True, token=token, batch_size=STREAM_BATCH_ROWS)
    except (TypeError, ValueError):
        # not a parquet-backed dataset; the builder doesn't take batch_size
        return load_dataset(repo, split=split, streaming=True, token=token)


class ImageNetHF(IterableDataset):

    def __init__(self, repo: str, split: str, transform, *, token: str | None, shuffle: bool, seed: int, limit: int | None = None):
        self.repo, self.split, self.transform, self.token = repo, split, transform, token
        self.shuffle, self.seed, self.limit = shuffle, seed, limit

    def __iter__(self) -> Iterator[tuple[torch.Tensor, int]]:
        ds = _stream(self.repo, self.split, self.token)
        info = torch.utils.data.get_worker_info()
        workers = info.num_workers if info is not None else 1
        if workers > 1 and getattr(ds, "n_shards", workers) < workers and hasattr(ds, "reshard"):
            # A split stored as fewer parquet files than workers would leave the extra
            # workers idle; resharding splits it by row group instead.
            try:
                ds = ds.reshard()
            except Exception as exc:  # pragma: no cover - best effort
                logger.warning(f"reshard failed ({exc}); continuing with {ds.n_shards} shard(s)")
        if self.shuffle:
            ds = ds.shuffle(seed=self.seed + _worker_id(), buffer_size=256)
        count = 0
        for ex in ds:
            if ex.get("label", -1) is None or int(ex["label"]) < 0:
                continue
            yield self.transform(_pil(ex["image"])), int(ex["label"])
            count += 1
            if self.limit and count >= self.limit:
                return


def _list_lengths(ds, column: str) -> list[int]:
    import pyarrow.compute as pc

    try:
        return pc.list_value_length(ds.data.table.column(column)).to_pylist()
    except Exception:
        return [len(ex[column]) for ex in ds.select_columns([column])]


class AdversarialRows(Dataset):

    def __init__(self, repo: str, split: str, *, per_row: int, mapper: LabelMapper, token: str | None,
                 seed: int, limit: int | None = None):
        from datasets import Image as HFImage, Sequence, load_dataset

        ds = load_dataset(repo, split=split, token=token)
        # keep PNG bytes; decode only what we sample, in the loader workers
        ds = ds.cast_column("image", HFImage(decode=False))
        ds = ds.cast_column("adversarial", Sequence(HFImage(decode=False)))
        self.ds = ds
        self.per_row, self.mapper, self.seed = per_row, mapper, seed
        self.transform = eval_transform()

        labels = [mapper(name) for name in ds["label_name"]]
        counts = _list_lengths(ds, "adversarial")
        self.rows = [i for i, (lab, n) in enumerate(zip(labels, counts)) if lab is not None and n > 0]
        if limit:
            self.rows = self.rows[:limit]
        self.labels = labels
        self.n_adv = sum(counts[i] for i in self.rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.ds[self.rows[index]]
        advs = list(row["adversarial"])
        if self.per_row and len(advs) > self.per_row:
            advs = random.sample(advs, self.per_row)
        clean = self.transform(_pil(row["image"]))
        return clean, [self.transform(_pil(a)) for a in advs], self.labels[self.rows[index]]


def collate_adv(batch):
    clean = torch.stack([b[0] for b in batch])
    labels = torch.tensor([b[2] for b in batch], dtype=torch.long)
    adv, owner = [], []
    for i, (_, advs, _) in enumerate(batch):
        adv.extend(advs)
        owner.extend([i] * len(advs))
    adv_t = torch.stack(adv) if adv else torch.empty(0, *clean.shape[1:])
    return clean, adv_t, torch.tensor(owner, dtype=torch.long), labels


def imagenet_loaders(args: Args) -> tuple[DataLoader, DataLoader]:
    root = Path(args.imagenet)
    if root.is_dir():
        from torchvision.datasets import ImageFolder

        train = ImageFolder(str(root / "train"), transform=train_transform(args.train_res))
        val_full = ImageFolder(str(root / "val"), transform=eval_transform())
        idx = list(range(len(val_full)))
        random.Random(args.seed).shuffle(idx)
        val = torch.utils.data.Subset(val_full, idx[: args.eval_imagenet_samples])
        train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                  pin_memory=True, drop_last=True, persistent_workers=args.num_workers > 0)
        val_loader = DataLoader(val, batch_size=args.batch_size * 2, num_workers=eval_workers(args), pin_memory=True)
        return train_loader, val_loader

    train = ImageNetHF(args.imagenet, "train", train_transform(args.train_res), token=args.hf_token, shuffle=True, seed=args.seed)
    # streaming shards are split across workers, so the per-worker limit is a share of the total
    val = ImageNetHF(args.imagenet, "validation", eval_transform(), token=args.hf_token, shuffle=True, seed=args.seed + 1,
                     limit=math.ceil(args.eval_imagenet_samples / max(1, eval_workers(args))))
    train_loader = DataLoader(train, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val, batch_size=args.batch_size * 2, num_workers=eval_workers(args), pin_memory=True)
    return train_loader, val_loader


def eval_workers(args: Args) -> int:
    # Eval loaders run while the training workers are still alive; keep the extra
    # processes (and their RAM) small.
    return min(2, args.num_workers)


def adversarial_loaders(args: Args, mapper: LabelMapper) -> tuple[DataLoader | None, DataLoader]:
    workers = min(args.num_workers, 2)
    train = None
    if args.adv_rows > 0:
        train_ds = AdversarialRows(args.adv_dataset, "train", per_row=args.adv_per_row, mapper=mapper, token=args.hf_token,
                                   seed=args.seed)
        logger.info(f"adversarial train split: {len(train_ds)} rows, {train_ds.n_adv} adversarial images (cached locally)")
        train = DataLoader(train_ds, batch_size=args.adv_rows, shuffle=True, drop_last=True, num_workers=workers,
                           collate_fn=collate_adv, pin_memory=True, persistent_workers=workers > 0)
    val_ds = AdversarialRows(args.adv_dataset, "validation", per_row=0, mapper=mapper, token=args.hf_token,
                             seed=args.seed, limit=args.eval_adv_rows)
    logger.info(f"adversarial validation split: {len(val_ds)} rows, {val_ds.n_adv} adversarial images")
    val = DataLoader(val_ds, batch_size=2, num_workers=min(1, args.num_workers), collate_fn=collate_adv, pin_memory=True)
    return train, val


def forever(loader: DataLoader):
    while True:
        yielded = False
        try:
            for batch in loader:
                yielded = True
                yield batch
        except RuntimeError as exc:
            if "DataLoader worker" in str(exc) and "Killed" in str(exc):
                logger.error(
                    "A data-loading worker was SIGKILLed: the host ran out of RAM (not GPU memory). "
                    "Each streaming worker needs roughly 1-2 GB; lower --num-workers (check `free -h`)."
                )
            raise
        if not yielded:
            raise RuntimeError("data loader produced no batches")


# --------------------------------------------------------------------------- eval


def set_train_mode(model: torch.nn.Module, train_bn: bool) -> None:
    model.train()
    if not train_bn:
        for module in model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.eval()


@torch.no_grad()
def evaluate(model, imagenet_val: DataLoader, adv_val: DataLoader, device: torch.device, amp: bool, train_bn: bool) -> dict[str, float]:
    model.eval()
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16

    correct = total = 0
    for x, y in imagenet_val:
        x, y = x.to(device, non_blocking=True), y.to(device)
        with torch.autocast(device.type, dtype=dtype, enabled=amp and device.type == "cuda"):
            pred = model(x).argmax(1)
        correct += int((pred == y).sum())
        total += int(y.numel())
    metrics = {"imagenet_top1": correct / max(1, total), "imagenet_samples": total}

    clean_ok = adv_ok = flipped = rows = n_adv = 0
    for clean, adv, owner, y in adv_val:
        clean, y = clean.to(device, non_blocking=True), y.to(device)
        with torch.autocast(device.type, dtype=dtype, enabled=amp and device.type == "cuda"):
            pc = model(clean).argmax(1)
            pa = model(adv.to(device, non_blocking=True)).argmax(1) if adv.numel() else pc[:0]
        owner = owner.to(device)
        clean_ok += int((pc == y).sum())
        adv_ok += int((pa == y[owner]).sum())
        flipped += int((pa != pc[owner]).sum())
        rows += int(y.numel())
        n_adv += int(pa.numel())
    metrics.update(
        adv_clean_top1=clean_ok / max(1, rows),
        adv_top1=adv_ok / max(1, n_adv),
        adv_flip_rate=flipped / max(1, n_adv),
        adv_rows=rows,
        adv_images=n_adv,
    )
    set_train_mode(model, train_bn)
    return metrics


# --------------------------------------------------------------------------- train


def _f(t: torch.Tensor) -> float:
    return float(t.detach())


def _host_memory_limit_bytes() -> int | None:
    for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            raw = Path(path).read_text().strip()
        except OSError:
            continue
        if raw.isdigit() and int(raw) < (1 << 60):
            return int(raw)
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return None


def _log_host_memory(num_workers: int) -> None:
    limit = _host_memory_limit_bytes()
    if limit is None:
        return
    # main process (~4 GB with CUDA) + ImageNet workers + adversarial/eval workers, ~1.5 GB each
    processes = 4 + num_workers + 3
    per_process_gb = limit / 1e9 / processes
    msg = f"host RAM limit {limit / 1e9:.1f} GB for ~{processes} process-equivalents ({per_process_gb:.1f} GB each)"
    if per_process_gb < 1.5:
        logger.warning(f"{msg}: likely to OOM-kill loader workers, lower --num-workers (currently {num_workers})")
    else:
        logger.info(msg)


def lr_at(step: int, total: int, base: float, warmup: int) -> float:
    if step < warmup:
        return base * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return base * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    for noisy in ("httpx", "httpcore", "huggingface_hub", "filelock", "fsspec"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    import datasets

    datasets.disable_progress_bars()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    logger.info(f"device={device} init={args.init} imagenet={args.imagenet} adv={args.adv_dataset}")
    _log_host_memory(args.num_workers)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "training_args.json").write_text(json.dumps(asdict(args) | {"hf_token": None}, indent=2), encoding="utf-8")

    model = load_model(None if args.init == "pretrained" else args.init, device=device, token=args.hf_token)
    model = model.to(memory_format=torch.channels_last)
    set_train_mode(model, args.train_bn)
    if not args.train_bn:
        logger.info("BatchNorm statistics frozen (small micro-batches); pass --train-bn to update them")

    mapper = LabelMapper.from_file(args.label_map)
    imnet_train, imnet_val = imagenet_loaders(args)
    adv_train, adv_val = adversarial_loaders(args, mapper)

    decay, no_decay = [], []
    for name, param in model.named_parameters():
        (no_decay if param.ndim <= 1 or name.endswith(".bias") else decay).append(param)
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay}, {"params": no_decay, "weight_decay": 0.0}], lr=args.lr
    )
    use_amp = args.amp and device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)

    total_steps = args.epochs * args.steps_per_epoch
    imnet_iter = forever(imnet_train)
    adv_iter = forever(adv_train) if adv_train is not None else None
    metrics_log = (out / "metrics.jsonl").open("a", encoding="utf-8")
    best_score = -1.0
    step = 0

    def log_metrics(kind: str, payload: dict[str, Any]) -> None:
        record = {"kind": kind, "step": step, "time": time.time(), **payload}
        metrics_log.write(json.dumps(record) + "\n")
        metrics_log.flush()

    def run_eval() -> None:
        nonlocal best_score
        m = evaluate(model, imnet_val, adv_val, device, use_amp, args.train_bn)
        logger.info(
            f"[eval step {step}] imagenet_top1={m['imagenet_top1']:.4f} adv_clean_top1={m['adv_clean_top1']:.4f} "
            f"adv_top1={m['adv_top1']:.4f} adv_flip_rate={m['adv_flip_rate']:.4f} "
            f"(imagenet n={m['imagenet_samples']}, adv rows={m['adv_rows']} images={m['adv_images']})"
        )
        log_metrics("eval", m)
        score = 0.5 * m["adv_top1"] + 0.5 * m["imagenet_top1"]
        if args.keep == "both":
            save_model(model, out / "last", extra={"step": step, "metrics": m})
        if score > best_score:
            best_score = score
            save_model(model, out / "best", extra={"step": step, "metrics": m})
            logger.info(f"new best (score={score:.4f}) saved to {out / 'best'}")

    logger.info(f"training for {total_steps} steps (micro-batch: {args.batch_size} imagenet + {args.adv_rows} adv rows x{args.adv_per_row}, accum {args.grad_accum})")
    t0 = time.time()
    running: dict[str, float] = {}

    def train_step(epoch: int) -> None:
        nonlocal step
        lr = lr_at(step, total_steps, args.lr, args.warmup_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)

        for _ in range(args.grad_accum):
            x, y = next(imnet_iter)
            if step == 0 and not running:
                logger.info(f"first ImageNet batch arrived after {time.time() - t0:.0f}s (streaming warm-up)")
            x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
                loss_imnet = F.cross_entropy(model(x), y, label_smoothing=args.label_smoothing)
                loss = loss_imnet
                parts = {"loss_imagenet": _f(loss_imnet)}

                if adv_iter is not None:
                    clean, adv, owner, ya = next(adv_iter)
                    clean = clean.to(device, non_blocking=True).to(memory_format=torch.channels_last)
                    adv = adv.to(device, non_blocking=True).to(memory_format=torch.channels_last)
                    owner, ya = owner.to(device), ya.to(device)
                    logits = model(torch.cat([clean, adv]))
                    lc, la = logits[: clean.shape[0]], logits[clean.shape[0]:]
                    loss_clean = F.cross_entropy(lc, ya, label_smoothing=args.label_smoothing)
                    loss_adv = F.cross_entropy(la, ya[owner], label_smoothing=args.label_smoothing) if la.shape[0] else lc.sum() * 0
                    loss = loss + args.adv_weight * (loss_clean + loss_adv)
                    parts.update(loss_adv_clean=_f(loss_clean), loss_adv=_f(loss_adv))
                    if args.consistency_weight > 0 and la.shape[0]:
                        kl = F.kl_div(F.log_softmax(la.float(), 1), F.softmax(lc[owner].detach().float(), 1), reduction="batchmean")
                        loss = loss + args.consistency_weight * kl
                        parts["loss_consistency"] = _f(kl)
            scaler.scale(loss / args.grad_accum).backward()
            for k, v in parts.items():
                running[k] = running.get(k, 0.0) + v / args.grad_accum

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        step += 1

        if step % 20 == 0:
            avg = {k: v / 20 for k, v in running.items()}
            running.clear()
            elapsed = time.time() - t0
            logger.info(
                f"epoch {epoch + 1}/{args.epochs} step {step}/{total_steps} lr={lr:.2e} "
                + " ".join(f"{k}={v:.4f}" for k, v in avg.items())
                + f" ({elapsed / step:.2f}s/step)"
            )
            log_metrics("train", avg | {"lr": lr})
        if args.eval_every and step % args.eval_every == 0:
            run_eval()
        elif args.save_every and args.keep == "both" and step % args.save_every == 0:
            save_model(model, out / "last", extra={"step": step})
            logger.info(f"checkpoint saved to {out / 'last'} (step {step})")

    try:
        for epoch in range(args.epochs):
            for _ in range(args.steps_per_epoch):
                train_step(epoch)
            if not args.eval_every or step % args.eval_every != 0:
                run_eval()
    except KeyboardInterrupt:
        # Ctrl-C / SIGINT (pm2 stop, tmux kill): keep the work done so far.
        save_model(model, out / "last", extra={"step": step, "interrupted": True})
        logger.warning(f"interrupted at step {step}; weights saved to {out / 'last'} "
                       f"(evaluate with: python evaluate.py --models pretrained {out / 'last'})")
        return 130

    if mapper.unmapped:
        logger.warning(f"{len(mapper.unmapped)} adversarial label names could not be mapped to ImageNet-1k and were skipped: "
                       f"{sorted(mapper.unmapped)[:5]} (use --label-map to override)")
    logger.info(f"done. submit-ready model: {out / 'best'}" + (f"  (latest: {out / 'last'})" if args.keep == "both" else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
