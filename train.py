from __future__ import annotations

import argparse
import importlib.util
import random
import sys
from contextlib import nullcontext
from pathlib import Path

# Allow running as a script: python hubert/train.py
if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from hubert.data import HubertWindowDataset, make_collate_fn
from hubert.inference import DEFAULT_LAYER_IDX, DEFAULT_MIN_COVERAGE, DEFAULT_MIN_RMS_RATIO
from hubert.model import FrozenHubertSvModel


def split_wavs(
    wav_paths: list[Path],
    train_fraction: float,
    seed: int,
) -> tuple[set[Path], set[Path]]:
    rng = np.random.default_rng(seed)
    indices = np.arange(len(wav_paths))
    rng.shuffle(indices)
    train_size = max(1, int(len(wav_paths) * train_fraction))
    train_idx = indices[:train_size]
    val_idx = indices[train_size:] if train_size < len(wav_paths) else indices[:train_size]
    train_wavs = {wav_paths[i] for i in train_idx}
    val_wavs = {wav_paths[i] for i in val_idx}
    return train_wavs, val_wavs


def _make_optimizer(params, lr: float, weight_decay: float, device: torch.device):
    adam_kwargs = dict(lr=lr, weight_decay=weight_decay)
    if device.type == "cuda":
        adam_kwargs["fused"] = True
    try:
        return optim.Adam(params, **adam_kwargs)
    except TypeError:
        adam_kwargs.pop("fused", None)
        return optim.Adam(params, **adam_kwargs)


def train_model(args: argparse.Namespace) -> Path:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
    print(f"Using device: {device}")

    model = FrozenHubertSvModel(
        hubert_name=args.hubert_name,
        layer_idx=args.layer_idx,
        attn_hidden_dim=args.attn_hidden_dim,
        out_dim=args.out_dim,
        sample_rate=args.sample_rate,
    ).to(device)

    dataset = HubertWindowDataset(
        wav_dir=Path(args.wav_dir),
        emb_list=Path(args.emb_list),
        window_sec=args.window_sec,
        hop_sec=args.hop_sec,
        min_coverage=args.min_coverage,
        min_rms_ratio=args.min_rms_ratio,
        target_sr=args.sample_rate,
        cache_dir=Path(args.cache_dir) if args.cache_dir else None,
    )
    if args.precache:
        dataset.precompute_cache(show_progress=True)
    train_wavs, val_wavs = split_wavs(dataset._wav_paths, args.train_fraction, args.seed)
    train_indices = [
        i for i, entry in enumerate(dataset.entries) if entry.wav_path in train_wavs
    ]
    val_indices = [
        i for i, entry in enumerate(dataset.entries) if entry.wav_path in val_wavs
    ]
    train_ds = Subset(dataset, train_indices)
    val_ds = Subset(dataset, val_indices)

    collate_fn = make_collate_fn(model.feature_extractor, target_sr=args.sample_rate)
    loader_common: dict[str, object] = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_fn,
    )
    if args.num_workers > 0:
        loader_common["persistent_workers"] = args.persistent_workers
        loader_common["prefetch_factor"] = args.prefetch_factor

    train_cache: list[tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]] | None = None
    train_stream_ds = train_ds
    if device.type == "cuda" and args.gpu_cache_batches > 0 and len(train_indices) > 0:
        cache_size = min(len(train_indices), args.gpu_cache_batches * args.batch_size)
        if cache_size > 0:
            rng = np.random.default_rng(args.seed)
            cache_indices = rng.choice(train_indices, size=cache_size, replace=False).tolist()
            cache_set = set(cache_indices)
            stream_indices = [i for i in train_indices if i not in cache_set]
            train_stream_ds = Subset(dataset, stream_indices) if stream_indices else None

            cache_loader_common = dict(loader_common)
            if args.num_workers > 0:
                cache_loader_common["persistent_workers"] = False
            cache_loader = DataLoader(Subset(dataset, cache_indices), shuffle=True, **cache_loader_common)

            print(f"Attempting to cache {len(cache_indices)} samples on GPU...")
            train_cache = []
            try:
                cache_bar = tqdm(
                    cache_loader,
                    total=len(cache_loader),
                    desc="Caching train batches to GPU",
                    leave=False,
                )
                for batch in cache_bar:
                    if batch is None:
                        continue
                    inputs, targets = batch
                    input_values = inputs["input_values"].to(device, non_blocking=True)
                    attention_mask = inputs.get("attention_mask")
                    if attention_mask is not None:
                        attention_mask = attention_mask.to(device, non_blocking=True)
                    targets = targets.to(device, non_blocking=True)
                    train_cache.append((input_values, attention_mask, targets))
                print(f"Cached {len(train_cache)} train batches on GPU.")
            except RuntimeError as exc:
                print(
                    "Warning: failed to cache training batches on GPU "
                    f"({exc}); falling back to DataLoader."
                )
                train_cache = None
                train_stream_ds = train_ds
                torch.cuda.empty_cache()

    train_stream_loader = (
        DataLoader(train_stream_ds, shuffle=True, **loader_common)
        if train_stream_ds is not None
        else None
    )
    val_loader = DataLoader(val_ds, shuffle=False, **loader_common)

    stream_len = len(train_stream_loader) if train_stream_loader is not None else 0
    cache_len = len(train_cache) if train_cache is not None else 0
    if stream_len == 0 and cache_len == 0:
        raise RuntimeError("No training batches produced. Try lowering min_rms_ratio.")

    if device.type == "cuda" and hasattr(torch, "compile") and args.torch_compile:
        try:
            if importlib.util.find_spec("triton") is None:
                raise RuntimeError("triton is not installed")
            model = torch.compile(model)
        except Exception as exc:  # pragma: no cover - compile is optional
            print(f"Warning: torch.compile failed, using eager mode ({exc})")

    mag_alpha = args.mag_alpha if args.mag_alpha is not None else args.norm_alpha
    optimizer = _make_optimizer(model.trainable_parameters, args.lr, args.weight_decay, device)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda" if use_amp else "cpu", enabled=use_amp)
    if use_amp:
        def amp_ctx():
            return torch.amp.autocast(device_type="cuda")
    else:
        def amp_ctx():
            return nullcontext()

    eps = 1e-6
    progress = tqdm(range(args.num_epochs), desc="Training", total=args.num_epochs)
    for epoch in progress:
        model.train()
        model.hubert.eval()
        running_loss = 0.0
        num_batches = 0

        if train_cache is not None:
            random.shuffle(train_cache)
            train_cache_batches = tqdm(
                train_cache,
                total=len(train_cache),
                desc=f"Epoch {epoch + 1}/{args.num_epochs} [train-cache]",
                leave=False,
            )
            for batch in train_cache_batches:
                if batch is None:
                    continue
                input_values, attention_mask, targets = batch

                optimizer.zero_grad(set_to_none=True)
                with amp_ctx():
                    dir_pred, mag_pred, _pooled = model(input_values, attention_mask)
                    dir_pred = F.normalize(dir_pred.float(), dim=1, eps=eps)
                    targets_fp32 = targets.float()
                    tgt_norm = torch.linalg.norm(targets_fp32, dim=1, keepdim=True)
                    tgt_dir = targets_fp32 / (tgt_norm + eps)
                    cos = 1.0 - F.cosine_similarity(dir_pred, tgt_dir, dim=1).mean()
                    mag_loss = F.smooth_l1_loss(
                        mag_pred.float().squeeze(1),
                        tgt_norm.squeeze(1),
                        beta=args.huber_beta,
                    )
                    loss = cos + mag_alpha * mag_loss

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                running_loss += float(loss.item())
                num_batches += 1
                train_cache_batches.set_postfix(loss=f"{loss.item():.4f}")

        if train_stream_loader is not None and len(train_stream_loader) > 0:
            train_batches = tqdm(
                train_stream_loader,
                total=len(train_stream_loader),
                desc=f"Epoch {epoch + 1}/{args.num_epochs} [train]",
                leave=False,
            )
            for batch in train_batches:
                if batch is None:
                    continue
                inputs, targets = batch
                input_values = inputs["input_values"].to(device, non_blocking=True)
                attention_mask = inputs.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                with amp_ctx():
                    dir_pred, mag_pred, _pooled = model(input_values, attention_mask)
                    dir_pred = F.normalize(dir_pred.float(), dim=1, eps=eps)
                    targets_fp32 = targets.float()
                    tgt_norm = torch.linalg.norm(targets_fp32, dim=1, keepdim=True)
                    tgt_dir = targets_fp32 / (tgt_norm + eps)
                    cos = 1.0 - F.cosine_similarity(dir_pred, tgt_dir, dim=1).mean()
                    mag_loss = F.smooth_l1_loss(
                        mag_pred.float().squeeze(1),
                        tgt_norm.squeeze(1),
                        beta=args.huber_beta,
                    )
                    loss = cos + mag_alpha * mag_loss

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                running_loss += float(loss.item())
                num_batches += 1
                train_batches.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = running_loss / max(1, num_batches)

        model.eval()
        val_running = 0.0
        val_batches = 0
        with torch.inference_mode():
            val_batches_bar = tqdm(
                val_loader,
                total=len(val_loader),
                desc=f"Epoch {epoch + 1}/{args.num_epochs} [val]",
                leave=False,
            )
            for batch in val_batches_bar:
                if batch is None:
                    continue
                inputs, targets = batch
                input_values = inputs["input_values"].to(device, non_blocking=True)
                attention_mask = inputs.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                with amp_ctx():
                    dir_pred, mag_pred, _pooled = model(input_values, attention_mask)
                    dir_pred = F.normalize(dir_pred.float(), dim=1, eps=eps)
                    targets_fp32 = targets.float()
                    tgt_norm = torch.linalg.norm(targets_fp32, dim=1, keepdim=True)
                    tgt_dir = targets_fp32 / (tgt_norm + eps)
                    cos = 1.0 - F.cosine_similarity(dir_pred, tgt_dir, dim=1).mean()
                    mag_loss = F.smooth_l1_loss(
                        mag_pred.float().squeeze(1),
                        tgt_norm.squeeze(1),
                        beta=args.huber_beta,
                    )
                    loss = cos + mag_alpha * mag_loss

                val_running += float(loss.item())
                val_batches += 1
                val_batches_bar.set_postfix(loss=f"{loss.item():.4f}")

        val_loss = val_running / max(1, val_batches)
        scheduler.step()
        progress.set_postfix(train=f"{train_loss:.4f}", val=f"{val_loss:.4f}")

        if args.verbose:
            progress.write(
                f"Epoch {epoch + 1}/{args.num_epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}"
            )

    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    state = model.state_dict()
    if any(k.startswith("_orig_mod.") for k in state.keys()):
        state = {k.replace("_orig_mod.", "", 1): v for k, v in state.items()}

    torch.save(
        {
            "model_state_dict": state,
            "config": {
                "hubert_name": args.hubert_name,
                "layer_idx": args.layer_idx,
                "sample_rate": args.sample_rate,
                "attn_hidden_dim": args.attn_hidden_dim,
                "out_dim": args.out_dim,
            },
        },
        save_path,
    )
    print(f"Training complete. Saved model to {save_path}")
    return save_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train frozen HuBERT + attentive pooling + MLP.")
    parser.add_argument("--wav_dir", type=str, default="wav", help="Directory of wav files")
    parser.add_argument("--emb_list", type=str, default="emb_list.json", help="JSON list with SV embeddings")
    parser.add_argument("--save_path", type=str, default="checkpoints/hubert_sv_model.pth", help="Output checkpoint path")

    parser.add_argument("--hubert_name", type=str, default="facebook/hubert-base-ls960")
    parser.add_argument("--layer_idx", type=int, default=DEFAULT_LAYER_IDX, help="Hidden state index to use (layer 8 by default)")
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="cache_16k",
        help="Optional cache directory for 16k mono wavs to avoid repeated resampling",
    )
    parser.add_argument(
        "--precache",
        action="store_true",
        default=True,
        help="Precompute 16k cache before training (default: on)",
    )
    parser.add_argument(
        "--no_precache",
        action="store_false",
        dest="precache",
        help="Disable precomputing cache before training",
    )

    parser.add_argument("--window_sec", type=float, default=8.0)
    parser.add_argument("--hop_sec", type=float, default=4.0)
    parser.add_argument("--min_coverage", type=float, default=DEFAULT_MIN_COVERAGE)
    parser.add_argument("--min_rms_ratio", type=float, default=DEFAULT_MIN_RMS_RATIO)

    parser.add_argument("--train_fraction", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=114514)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--persistent_workers",
        action="store_true",
        default=True,
        help="Keep DataLoader workers alive between epochs (only when num_workers>0)",
    )
    parser.add_argument(
        "--no_persistent_workers",
        action="store_false",
        dest="persistent_workers",
        help="Disable persistent workers",
    )
    parser.add_argument(
        "--prefetch_factor",
        type=int,
        default=2,
        help="Batches prefetched per worker (only when num_workers>0)",
    )
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument(
        "--gpu_cache_batches",
        type=int,
        default=0,
        help="Cache this many training batches on GPU to reduce IO (0 = disable)",
    )
    parser.add_argument(
        "--norm_alpha",
        type=float,
        default=0.25,
        help="Deprecated. Use --mag_alpha instead.",
    )
    parser.add_argument(
        "--mag_alpha",
        type=float,
        default=None,
        help="Weight for magnitude (Huber) loss. Overrides --norm_alpha if set.",
    )
    parser.add_argument(
        "--huber_beta",
        type=float,
        default=1.0,
        help="Huber beta for magnitude loss.",
    )

    parser.add_argument("--attn_hidden_dim", type=int, default=256)
    parser.add_argument("--out_dim", type=int, default=32)

    parser.add_argument(
        "--torch_compile",
        action="store_true",
        default=True,
        help="Enable torch.compile when available (default: on)",
    )
    parser.add_argument(
        "--no_torch_compile",
        action="store_false",
        dest="torch_compile",
        help="Disable torch.compile",
    )
    parser.add_argument("--verbose", action="store_true", help="Print epoch losses")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_model(args)


if __name__ == "__main__":
    main()
