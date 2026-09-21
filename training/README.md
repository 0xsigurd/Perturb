# Adversarial Training (miner model track)

Default scripts for the model track of the Perturb subnet. Every day at 00:00 UTC validators evaluate
the models miners committed on-chain (see the main README, *Adversarial training track*); the best one
receives 20% of miner emissions for that day. These scripts produce, check and submit such a model:

| script | what it does |
|---|---|
| `train.py` | fine-tunes EfficientNetV2-L on ImageNet-1k **plus** the growing [Perturb adversarial dataset](https://huggingface.co/datasets/perturb-ai/efficientnet-v2-l-adv-dataset), producing a submit-ready model directory |
| `evaluate.py` | compares your model with the original EfficientNetV2-L side by side: clean ImageNet accuracy, accuracy/flip rate on the held-out adversarial test split, and white-box PGD robustness |
| `submit.py` | uploads that model to Hugging Face and commits `{repo_id, revision, hash(model+hotkey)}` to the Bittensor chain with your hotkey — private repo first, public one block after a successful chain commit. All settings come from `.env` |

The model, weights format and preprocessing are exactly what validators run
(`torchvision.efficientnet_v2_l`; float tensor → bicubic resize 480 → center crop 480 → normalize
with mean = std = 0.5, i.e. `EfficientNet_V2_L_Weights.IMAGENET1K_V1.transforms()`), so what you
evaluate locally is bit-for-bit what gets scored.

## Setup

```bash
cd training                     # from the Perturb repo root; the repo's .venv already has every dependency
cp .env.example .env            # HF_TOKEN, HF_REPO_ID, wallet, netuid
```

The scripts import `perturbnet.model_commit` from the repo root (they add it to `sys.path`
themselves), so run them from `training/` or as `python training/train.py ...` from the root.

ImageNet-1k on the Hub (`ILSVRC/imagenet-1k`) is gated: open its page once, accept the terms,
and put a token with read access in `HF_TOKEN`. Alternatively point `--imagenet` at a local copy
in ImageFolder layout (`<root>/train/<wnid>/*.JPEG`, `<root>/val/<wnid>/*.JPEG`).

## train.py

```bash
python train.py --output-dir runs/v1 --epochs 2 --steps-per-epoch 3000
python train.py --imagenet /data/imagenet --output-dir runs/v1          # local ImageNet
python train.py --init runs/v1/best --output-dir runs/v2                 # continue from a run or an HF model repo
```

Every optimiser step combines a micro-batch of clean ImageNet images with a micro-batch of
adversarial-dataset rows (each row = the clean image + `--adv-per-row` of its verified adversarial
versions, all through the validator's deterministic preprocessing so the pixel-exact perturbations
are preserved):

```text
loss = CE(imagenet)
     + adv_weight        * ( CE(clean from adv rows) + CE(adversarial versions) )
     + consistency_weight * KL( p(clean) || p(adversarial) )
```

The KL term (TRADES-style) pulls the prediction on each adversarial image toward the model's own
prediction on its clean counterpart, which is the property the network scores: an adversarial
image should *not* change the top-1 class.

Key flags (defaults in parentheses):

| flag | purpose |
|---|---|
| `--batch-size 8`, `--adv-rows 4`, `--adv-per-row 2`, `--grad-accum 4` | micro-batch composition; effective batch = 4 × (8 + 4 + 8) images. Lower `--batch-size`/`--adv-rows` if you run out of memory; EfficientNetV2-L at 480 px needs ~1 GB per image in bf16 training |
| `--adv-weight 1.0`, `--consistency-weight 0.5` | how hard to push on the adversarial data |
| `--lr 2e-5`, `--warmup-steps 200`, `--weight-decay 1e-5`, `--label-smoothing 0.1` | AdamW + cosine; small LR because we start from the pretrained weights and must keep clean accuracy |
| `--train-res 480` | crop for clean ImageNet (adversarial rows are always 480 to match the validator) |
| `--train-bn` | update BatchNorm running statistics. Off by default: with micro-batches of 4–8 images the batch statistics are meaningless (the pretrained model drops to ~0% top-1 in `train()` mode), so BN runs on its pretrained statistics while its affine weights still train. Only enable with micro-batches ≥ 32 |
| `--eval-imagenet-samples 2000`, `--eval-adv-rows 300`, `--eval-every 0` | in-training evaluation size / cadence (0 = end of each epoch) |
| `--save-every 200`, `--keep both\|best` | checkpointing, see below |
| `--label-map file.json` | override the ImageNet-100 → ImageNet-1k label mapping if a class name fails to map (the script logs any) |

In-training evaluation (on the adversarial **validation** split) reports `imagenet_top1`,
`adv_clean_top1`, `adv_top1` and `adv_flip_rate`; metrics are appended to `<output-dir>/metrics.jsonl`.

### Checkpoints and disk usage

Disk usage is bounded: there are at most two checkpoint directories per run, and each save
overwrites the previous one in place (atomically, so a crash mid-save can't corrupt it).

| directory | written when | size |
|---|---|---|
| `<output-dir>/last` | every `--save-every` steps (default 200), at every evaluation, and on Ctrl-C | 476 MB |
| `<output-dir>/best` | at an evaluation whose score (mean of `adv_top1` and `imagenet_top1`) beats the previous best | 476 MB |

`--keep best` drops `last/` entirely (476 MB per run total, but no crash-recovery checkpoint). Each
directory is submit-ready:

```text
model.safetensors     torchvision efficientnet_v2_l state dict
config.json           architecture, preprocessing, metrics at save time
```

Runs with different `--output-dir` values accumulate, so delete old `runs/*` once you've submitted.

### Stopping a run early

Ctrl-C (or `pm2 stop`) saves the current weights to `<output-dir>/last` and exits. Note that `best/`
only exists once at least one evaluation has run (`--eval-every N`, or the end of an epoch), so after
an early stop `last/` is usually the checkpoint to evaluate and submit.

## evaluate.py

Compares any number of models against the original EfficientNetV2-L on identical images:

```bash
python evaluate.py --models pretrained runs/v1/last                    # baseline vs your checkpoint
python evaluate.py --models pretrained runs/v1/best runs/v2/best        # several runs at once
python evaluate.py --models pretrained yourname/repo@<revision>          # a published submission
python evaluate.py --models pretrained runs/v1/best --imagenet-samples 5000 --pgd-samples 128 --out eval.json
```

Output is one table, metrics as rows, models as columns, with the delta against `pretrained`:

| metric | meaning | want |
|---|---|---|
| `imagenet_top1`, `imagenet_top5`, `imagenet_nll` | clean ImageNet-1k validation (`--imagenet-samples`, default 2000) — must not regress | ↑ ↑ ↓ |
| `adv_clean_top1` | accuracy on the adversarial rows' clean source images | ↑ |
| `adv_top1` | accuracy on the adversarial versions (label = the clean image's label) | ↑ |
| `adv_flip_rate` | fraction of adversarial versions whose prediction differs from the clean prediction — what the network scores | ↓ |
| `adv_margin_mean` | mean true-class logit minus best other logit on adversarial versions (> 0 = correct) | ↑ |
| `adv_rows_all_robust` | fraction of source images for which *every* adversarial version is classified like the clean one | ↑ |
| `pgd_top1@1`, `@2`, `@4` | accuracy under a 10-step white-box PGD-L∞ attack on the model under test with budget 1/2/4 of 255 (`--pgd-samples`, default 64) | ↑ |
| `pgd_flip_rate@…` | fraction of those images PGD manages to flip | ↓ |

By default the adversarial metrics use the dataset's **test** split, which training never touches
(training uses `train`; in-training evaluation uses `validation`), so they're an honest estimate.
The PGD rows matter because the dataset's attacks were all crafted against the *original* model: a
fine-tuned model can be immune to those and still fold to fresh attacks against itself, which is
exactly what miners on the network will craft. PGD perturbs the 480×480 tensor the validator sees
rather than the native-resolution image, so treat it as a proxy for the task, not the task itself.

Cost on a 3090: ~3 min for 2000 ImageNet samples, seconds for the adversarial split, ~1 min per
model per PGD budget at 64 samples.

Labels: the adversarial dataset carries ImageNet-100 class names; `train.py` maps them to
ImageNet-1k indices by WordNet synonym (all 100 classes resolve) so the two sources share one
label space.

## submit.py

No flags. Fill in `.env` (see `.env.example`) and run:

```bash
python submit.py
```

Required `.env` keys: `HF_TOKEN`, `HF_REPO_ID`. Defaults for the rest:

| key | default |
|---|---|
| `MODEL_DIR` | `runs/v1/last` |
| `WALLET_NAME` | `miner` |
| `WALLET_HOTKEY` | `default` |
| `NETUID` | `26` |
| `SUBTENSOR_NETWORK` | `finney` |

Steps, in order:

1. Model upload (no chain access; the wallet's hotkey address is read from disk to write the model card)
   - require `HF_TOKEN` and `HF_REPO_ID`
   - create the Hugging Face model repo if it does not exist, then set it **private**
   - upload `MODEL_DIR` (`model.safetensors`, `config.json`, generated `README.md`)
   - take the upload's commit sha as `revision`
2. On-chain commit
   - load `WALLET_NAME` / `WALLET_HOTKEY`
   - refuse if the hotkey is not registered on `NETUID`
   - `set_commitment` with `{r: repo_id, rv: revision, h: sha256(model \|\| hotkey)[:16]}`
   - on success, wait one block, then make the repo **public**
   - on failure the repo stays private and you re-run

The on-chain hash is `sha256(model.safetensors bytes concatenated with the hotkey)`, not the file
hash alone, so a copied model cannot reuse another miner's commitment. Because the weights stay
private until that commitment is on chain (plus one block), nobody can copy the model and attribute
it first. If the chain commit fails the repo stays private and you re-run.

### What validators check

Validators evaluate the commitment state as it was **a day earlier** (the API's `previous`
snapshot, `GET /training/commitments`), so a new submission enters the evaluation the day after it
is committed and must stay unchanged until then. A model is skipped when:

| reason | cause |
|---|---|
| `chain_commitment_changed` | you re-committed since the snapshot; the new model is evaluated the next day |
| `chain_commitment_missing` | the hotkey has no on-chain commitment any more |
| `hash_mismatch` | `sha256(model.safetensors || hotkey)[:16]` at that revision differs from `h` on chain |
| `load_failed:*` | `model.safetensors` missing, too large, not a plain `efficientnet_v2_l` state dict, or the repo is private |

The repo must be public by the time validators evaluate it (submit.py makes it public one block
after the chain commit). Keep `HF_REPO_ID` ≤ 44 characters so the payload fits the 128-byte budget.

## Running unattended (pm2 / tmux)

Training is a finite job, not a service, so the goal is only to survive your SSH session dropping.
Any of these work:

```bash
# pm2 — one-shot: --no-autorestart is essential, otherwise pm2 restarts train.py from
# scratch the moment it finishes (or crashes) and overwrites the run directory
pm2 start train.py --name adv-train --interpreter .venv/bin/python --no-autorestart -- \
    --output-dir runs/v1 --batch-size 4 --adv-rows 2 --grad-accum 8
pm2 logs adv-train            # follow progress
pm2 describe adv-train        # status; "stopped" with exit code 0 means it finished
pm2 delete adv-train          # clean up when done

# tmux
tmux new -s train
python train.py --output-dir runs/v1 ... 2>&1 | tee runs/v1/train.log     # Ctrl-b d to detach

# nohup
nohup python train.py --output-dir runs/v1 ... > runs/v1/train.log 2>&1 &
```

If a run dies part-way, restart it with `--init runs/v1/last --output-dir runs/v1b`: it continues
from the last saved weights (written every `--save-every` steps, default 200, and at every
evaluation), with a fresh optimizer state and LR schedule. `pm2 stop` sends SIGINT, which is handled
like Ctrl-C: the current weights are saved to `last/` before exit.

## Hardware notes

- Defaults target a single 24–48 GB GPU with bf16 autocast. On 24 GB use `--batch-size 4 --adv-rows 2 --grad-accum 8`.
- Streaming ImageNet from the Hub is network-bound; more `--num-workers` keeps the GPU fed, but each
  worker holds roughly 1–2 GB of **host RAM** (parquet buffers + decoded images). If a worker dies with
  `DataLoader worker ... is killed by signal: Killed`, that is the kernel OOM killer: check `free -h`
  and lower `--num-workers` (default 4; 2 on a 16 GB box). A local copy is faster if you have one.
- The adversarial dataset is downloaded once per run into the Hugging Face cache (`~/.cache/huggingface`)
  and read with random access — it is cycled many times per run, so streaming it would re-download it
  on every pass. It grows every ~2 minutes; a new run picks up whatever exists at that moment, so re-run
  `train.py --init <previous best>` periodically.
- Startup is dominated by ImageNet streaming warm-up: each worker has to fetch its first parquet row
  group (~100 MB) before the first batch, so expect 1–3 minutes of `GPU-Util 0%` on a slow link. The
  log prints `first ImageNet batch arrived after Ns` when training actually starts.
