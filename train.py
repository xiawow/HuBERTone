from __future__ import annotations

import argparse
import importlib.util
import random
import sys
from contextlib import nullcontext
from pathlib import Path

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
from hubert.sampler import PKBatchSampler


def supcon_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Supervised Contrastive Loss (InfoNCE with positive pairs from same class).

    Args:
        features: [B, D] L2-normalized embeddings
        labels: [B] class labels (音色版本 ID)
        temperature: temperature scaling factor
    """
    device = features.device
    batch_size = features.shape[0]

    if batch_size <= 1:
        return torch.tensor(0.0, device=device)

    sim_matrix = torch.matmul(features, features.T) / temperature

    labels = labels.unsqueeze(0)
    mask_pos = (labels.T == labels).float()
    mask_pos.fill_diagonal_(0)

    logits_max, _ = sim_matrix.max(dim=1, keepdim=True)
    logits = sim_matrix - logits_max.detach()

    exp_logits = torch.exp(logits)
    exp_logits = exp_logits * (1 - torch.eye(batch_size, device=device))

    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

    pos_count = mask_pos.sum(dim=1)
    valid_mask = pos_count > 0
    if not valid_mask.any():
        return torch.tensor(0.0, device=device)

    mean_log_prob = (mask_pos * log_prob).sum(dim=1) / (pos_count + 1e-12)
    loss = -mean_log_prob[valid_mask].mean()

    return loss


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

    layer_indices = tuple(args.layer_indices) if args.layer_indices else (4, 8, 12)
    model = FrozenHubertSvModel(
        hubert_name=args.hubert_name,
        layer_idx=args.layer_idx,
        layer_indices=layer_indices,
        use_multi_layer=args.use_multi_layer,
        attn_hidden_dim=args.attn_hidden_dim,
        out_dim=args.out_dim,
        dropout=args.dropout,
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

    train_version_labels = [dataset.entries[i].wav_path.stem.split("_")[-1] for i in train_indices]
    use_pk_sampling = args.use_pk_sampling and len(set(train_version_labels)) >= args.pk_P

    if use_pk_sampling:
        try:
            train_sampler = PKBatchSampler(
                version_labels=train_version_labels,
                P=args.pk_P,
                K=args.pk_K,
                seed=args.seed,
            )
            train_loader = DataLoader(
                train_ds,
                batch_sampler=train_sampler,
                num_workers=args.num_workers,
                pin_memory=device.type == "cuda",
                collate_fn=collate_fn,
            )
            print(f"Using PK sampling: P={args.pk_P}, K={args.pk_K}")
        except ValueError as e:
            print(f"Warning: PK sampling failed ({e}), falling back to standard sampling")
            use_pk_sampling = False
            train_loader = DataLoader(
                train_ds,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                pin_memory=device.type == "cuda",
                collate_fn=collate_fn,
            )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            collate_fn=collate_fn,
        )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_fn,
    )

    if len(train_loader) == 0:
        raise RuntimeError("No training batches produced. Try lowering min_rms_ratio.")

    if device.type == "cuda" and hasattr(torch, "compile") and args.torch_compile:
        try:
            if importlib.util.find_spec("triton") is None:
                raise RuntimeError("triton is not installed")
            model = torch.compile(model)
        except Exception as exc:
            print(f"Warning: torch.compile failed, using eager mode ({exc})")

    mag_alpha = args.mag_alpha if args.mag_alpha is not None else args.norm_alpha
    con_alpha = args.con_alpha
    optimizer = _make_optimizer(model.trainable_parameters, args.lr, args.weight_decay, device)

    if args.warmup_epochs > 0:
        warmup_scheduler = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            total_iters=args.warmup_epochs,
        )
        cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.num_epochs - args.warmup_epochs,
        )
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[args.warmup_epochs],
        )
        print(f"Using warmup for {args.warmup_epochs} epochs")
    else:
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

        train_batches = tqdm(
            train_loader,
            total=len(train_loader),
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

                if use_pk_sampling and con_alpha > 0:
                    version_labels_batch = []
                    for i in range(input_values.shape[0]):
                        idx = train_indices[i % len(train_indices)]
                        version = dataset.entries[idx].wav_path.stem.split("_")[-1]
                        version_labels_batch.append(version)
                    label_tensor = torch.tensor(
                        [hash(v) % 10000 for v in version_labels_batch],
                        device=device,
                    )
                    con_loss = supcon_loss(dir_pred, label_tensor, temperature=args.temperature)
                else:
                    con_loss = torch.tensor(0.0, device=device)

                loss = cos + mag_alpha * mag_loss + con_alpha * con_loss

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
                "layer_indices": list(layer_indices),
                "use_multi_layer": args.use_multi_layer,
                "sample_rate": args.sample_rate,
                "attn_hidden_dim": args.attn_hidden_dim,
                "out_dim": args.out_dim,
                "dropout": args.dropout,
            },
        },
        save_path,
    )
    print(f"Training complete. Saved model to {save_path}")
    return save_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train frozen HuBERT + multi-layer fusion + contrastive learning.")
    parser.add_argument("--wav_dir", type=str, default="wav", help="Directory of wav files")
    parser.add_argument("--emb_list", type=str, default="emb_list.json", help="JSON list with SV embeddings")
    parser.add_argument("--save_path", type=str, default="checkpoints/hubert_sv_model.pth", help="Output checkpoint path")

    parser.add_argument("--hubert_name", type=str, default="facebook/hubert-base-ls960")
    parser.add_argument("--layer_idx", type=int, default=DEFAULT_LAYER_IDX, help="Single layer index (when use_multi_layer=False)")
    parser.add_argument("--layer_indices", type=int, nargs="+", default=[4, 8, 12], help="Layer indices for multi-layer fusion")
    parser.add_argument("--use_multi_layer", action="store_true", default=True, help="Enable multi-layer fusion")
    parser.add_argument("--no_multi_layer", action="store_false", dest="use_multi_layer", help="Disable multi-layer fusion")
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
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate for MLP head")

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

    parser.add_argument("--warmup_epochs", type=int, default=5, help="Number of warmup epochs")
    parser.add_argument("--con_alpha", type=float, default=0.5, help="Weight for contrastive loss")
    parser.add_argument("--temperature", type=float, default=0.07, help="Temperature for contrastive loss")
    parser.add_argument("--use_pk_sampling", action="store_true", default=True, help="Use PK sampling for batches")
    parser.add_argument("--no_pk_sampling", action="store_false", dest="use_pk_sampling", help="Disable PK sampling")
    parser.add_argument("--pk_P", type=int, default=8, help="Number of versions per batch (PK sampling)")
    parser.add_argument("--pk_K", type=int, default=4, help="Samples per version per batch (PK sampling)")

    parser.add_argument("--verbose", action="store_true", help="Print epoch losses")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_model(args)


if __name__ == "__main__":
    main()
