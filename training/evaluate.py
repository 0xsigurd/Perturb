"""Compare one or more models against the original EfficientNetV2-L on the same data."""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from torch.utils.data import DataLoader

import train
from common import DEFAULT_ADV_DATASET, DEFAULT_IMAGENET_DATASET, MEAN, STD, LabelMapper, load_model

logger = logging.getLogger("evaluate")

PGD_STEPS = 10


def parse_args() -> argparse.Namespace:
    load_dotenv()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=["pretrained"],
                   help="'pretrained', model dirs, or HF repos (optionally repo@revision). 'pretrained' is the baseline")
    p.add_argument("--imagenet", default=os.getenv("IMAGENET_DATASET", DEFAULT_IMAGENET_DATASET))
    p.add_argument("--imagenet-samples", type=int, default=2000)
    p.add_argument("--adv-dataset", default=os.getenv("ADV_DATASET", DEFAULT_ADV_DATASET))
    p.add_argument("--adv-split", default="test", help="test (default, never used for model selection) or validation")
    p.add_argument("--adv-rows", type=int, default=0, help="limit adversarial rows (0 = whole split)")
    p.add_argument("--pgd-samples", type=int, default=64, help="clean images attacked with PGD per model (0 disables)")
    p.add_argument("--pgd-eps", type=float, nargs="+", default=[1.0, 2.0, 4.0], help="Linf budgets in 1/255 units")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--label-map", default=None)
    p.add_argument("--hf-token", default=os.getenv("HF_TOKEN") or None)
    p.add_argument("--out", default=None, help="write all metrics as JSON here")
    return p.parse_args()


# --------------------------------------------------------------------------- helpers


class Acc:

    def __init__(self) -> None:
        self.sums: dict[str, float] = {}
        self.counts: dict[str, int] = {}

    def add(self, key: str, value: float, n: int = 1) -> None:
        self.sums[key] = self.sums.get(key, 0.0) + value
        self.counts[key] = self.counts.get(key, 0) + n

    def mean(self, key: str) -> float:
        return self.sums[key] / max(1, self.counts[key])


def margins(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    true = logits.gather(1, y[:, None]).squeeze(1)
    other = logits.clone()
    other.scatter_(1, y[:, None], float("-inf"))
    return true - other.max(1).values


def pgd_linf(model, x: torch.Tensor, y: torch.Tensor, eps_pixel: float, steps: int, autocast_ctx) -> torch.Tensor:
    std = torch.tensor(STD, device=x.device).view(1, 3, 1, 1)
    mean = torch.tensor(MEAN, device=x.device).view(1, 3, 1, 1)
    eps = eps_pixel / std
    alpha = eps / 4
    lo, hi = (0 - mean) / std, (1 - mean) / std
    x_adv = (x + (torch.rand_like(x) * 2 - 1) * eps).clamp(lo, hi).detach()
    for _ in range(steps):
        x_adv.requires_grad_(True)
        with autocast_ctx():
            loss = F.cross_entropy(model(x_adv).float(), y)
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv.detach() + alpha * grad.sign()
        x_adv = torch.max(torch.min(x_adv, x + eps), x - eps).clamp(lo, hi).detach()
    return x_adv


# --------------------------------------------------------------------------- main


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    for noisy in ("httpx", "httpcore", "huggingface_hub", "filelock", "fsspec"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    import datasets

    datasets.disable_progress_bars()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = args.amp and device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16

    def autocast_ctx():
        return torch.autocast(device.type, dtype=amp_dtype, enabled=use_amp)

    # models -----------------------------------------------------------------
    models: dict[str, torch.nn.Module] = {}
    for spec in args.models:
        source, _, revision = spec.partition("@")
        name = "pretrained" if source == "pretrained" else spec
        m = load_model(None if source == "pretrained" else source, device=device, revision=revision or None, token=args.hf_token)
        m.eval().requires_grad_(False)
        models[name] = m.to(memory_format=torch.channels_last)
        logger.info(f"loaded {name}")
    acc = {name: Acc() for name in models}

    # ImageNet ---------------------------------------------------------------
    root = Path(args.imagenet)
    if root.is_dir():
        from torchvision.datasets import ImageFolder

        full = ImageFolder(str(root / "val"), transform=train.eval_transform())
        idx = torch.randperm(len(full), generator=torch.Generator().manual_seed(args.seed))[: args.imagenet_samples].tolist()
        imnet = DataLoader(torch.utils.data.Subset(full, idx), batch_size=args.batch_size, num_workers=args.num_workers)
    else:
        ds = train.ImageNetHF(args.imagenet, "validation", train.eval_transform(), token=args.hf_token, shuffle=True,
                              seed=args.seed, limit=math.ceil(args.imagenet_samples / max(1, args.num_workers)))
        imnet = DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers)

    t0 = time.time()
    seen = 0
    if not root.is_dir():
        logger.info(f"streaming {args.imagenet_samples} ImageNet validation images (warm-up of 1-3 min before the first batch)")
    with torch.no_grad():
        for x, y in imnet:
            if seen == 0:
                logger.info(f"first ImageNet batch after {time.time() - t0:.0f}s")
            x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            y = y.to(device)
            for name, m in models.items():
                with autocast_ctx():
                    logits = m(x).float()
                top5 = logits.topk(5, 1).indices
                acc[name].add("imagenet_top1", float((top5[:, 0] == y).sum()), y.numel())
                acc[name].add("imagenet_top5", float((top5 == y[:, None]).any(1).sum()), y.numel())
                acc[name].add("imagenet_nll", float(F.cross_entropy(logits, y, reduction="sum")), y.numel())
            seen += y.numel()
            if seen % (args.batch_size * 25) == 0:
                logger.info(f"imagenet {seen}/{args.imagenet_samples}")
    logger.info(f"imagenet: {seen} samples in {time.time() - t0:.0f}s")

    # adversarial dataset ----------------------------------------------------
    mapper = LabelMapper.from_file(args.label_map)
    adv_ds = train.AdversarialRows(args.adv_dataset, args.adv_split, per_row=0, mapper=mapper, token=args.hf_token,
                                   seed=args.seed, limit=args.adv_rows or None)
    logger.info(f"adversarial {args.adv_split} split: {len(adv_ds)} rows, {adv_ds.n_adv} adversarial images")
    adv_loader = DataLoader(adv_ds, batch_size=2, num_workers=min(1, args.num_workers), collate_fn=train.collate_adv)
    pgd_x, pgd_y = [], []
    with torch.no_grad():
        for clean, adv, owner, y in adv_loader:
            clean = clean.to(device).to(memory_format=torch.channels_last)
            adv = adv.to(device).to(memory_format=torch.channels_last)
            owner, y = owner.to(device), y.to(device)
            if args.pgd_samples and sum(t.shape[0] for t in pgd_x) < args.pgd_samples:
                pgd_x.append(clean.cpu())
                pgd_y.append(y.cpu())
            for name, m in models.items():
                with autocast_ctx():
                    lc = m(clean).float()
                    la = torch.cat([m(adv[i:i + args.batch_size]).float() for i in range(0, adv.shape[0], args.batch_size)]) if adv.numel() else lc[:0]
                pc, pa = lc.argmax(1), la.argmax(1)
                a = acc[name]
                a.add("adv_clean_top1", float((pc == y).sum()), y.numel())
                a.add("adv_top1", float((pa == y[owner]).sum()), pa.numel())
                flips = pa != pc[owner]
                a.add("adv_flip_rate", float(flips.sum()), pa.numel())
                a.add("adv_margin_mean", float(margins(la, y[owner]).sum()), pa.numel())
                for r in range(y.numel()):
                    a.add("adv_rows_all_robust", float(not bool(flips[owner == r].any())), 1)

    # adaptive PGD -----------------------------------------------------------
    if args.pgd_samples and pgd_x:
        X = torch.cat(pgd_x)[: args.pgd_samples]
        Y = torch.cat(pgd_y)[: args.pgd_samples]
        logger.info(f"PGD-Linf ({PGD_STEPS} steps) on {X.shape[0]} clean images, eps={args.pgd_eps} /255")
        pgd_bs = max(1, args.batch_size // 2)
        for name, m in models.items():
            t1 = time.time()
            for eps in args.pgd_eps:
                for i in range(0, X.shape[0], pgd_bs):
                    x = X[i:i + pgd_bs].to(device).to(memory_format=torch.channels_last)
                    y = Y[i:i + pgd_bs].to(device)
                    with torch.no_grad(), autocast_ctx():
                        pc = m(x).argmax(1)
                    x_adv = pgd_linf(m, x, y, eps / 255.0, PGD_STEPS, autocast_ctx)
                    with torch.no_grad(), autocast_ctx():
                        pa = m(x_adv).argmax(1)
                    acc[name].add(f"pgd_top1@{eps:g}", float((pa == y).sum()), y.numel())
                    acc[name].add(f"pgd_flip_rate@{eps:g}", float((pa != pc).sum()), y.numel())
            logger.info(f"PGD done for {name} in {time.time() - t1:.0f}s")

    # report -----------------------------------------------------------------
    keys = list(dict.fromkeys(k for a in acc.values() for k in a.sums))
    names = list(models)
    results = {name: {k: acc[name].mean(k) for k in keys if k in acc[name].sums} for name in names}
    width = max(len(n) for n in names)
    print()
    print(f"{'metric':<24}" + "".join(f"{n[-width:]:>{width + 2}}" for n in names) + ("   delta vs pretrained" if "pretrained" in names and len(names) > 1 else ""))
    for k in keys:
        row = f"{k:<24}" + "".join(f"{results[n].get(k, float('nan')):>{width + 2}.4f}" for n in names)
        if "pretrained" in names and len(names) > 1:
            deltas = [results[n][k] - results["pretrained"][k] for n in names if n != "pretrained" and k in results[n]]
            row += "   " + " ".join(f"{d:+.4f}" for d in deltas)
        print(row)
    print()
    print("higher is better: imagenet_top1/top5, adv_clean_top1, adv_top1, adv_margin_mean, adv_rows_all_robust, pgd_top1@*")
    print("lower is better : imagenet_nll, adv_flip_rate, pgd_flip_rate@*")

    if args.out:
        payload = {"models": args.models, "imagenet_samples": seen, "adv_split": args.adv_split,
                   "adv_rows": len(adv_ds), "adv_images": adv_ds.n_adv, "pgd_samples": args.pgd_samples,
                   "pgd_steps": PGD_STEPS, "results": results}
        Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
