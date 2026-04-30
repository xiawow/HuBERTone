from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as a script: python hubert/infer.py
if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torchaudio

from hubert.model import FrozenHubertSvModel
from hubert.utils import (
    TARGET_SR,
    float32_to_hex,
    rms_energy,
    sliding_window_positions,
    weighted_average,
)


def load_checkpoint(model_path: Path, device: torch.device, layer_idx: int, hubert_name: str) -> FrozenHubertSvModel:
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        config = checkpoint.get("config", {})
        hubert_name = config.get("hubert_name", hubert_name)
        layer_idx = int(config.get("layer_idx", layer_idx))
        model_state = checkpoint["model_state_dict"]
    else:
        model_state = checkpoint

    model = FrozenHubertSvModel(hubert_name=hubert_name, layer_idx=layer_idx)
    if any(k.startswith("_orig_mod.") for k in model_state.keys()):
        model_state = {k.replace("_orig_mod.", "", 1): v for k, v in model_state.items()}
    model.load_state_dict(model_state, strict=False)
    model.to(device)
    model.eval()
    return model


def load_wav(path: Path, target_sr: int = TARGET_SR) -> torch.Tensor:
    wav, sr = torchaudio.load(str(path))
    wav = wav.mean(dim=0)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav.to(torch.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inference with frozen HuBERT + attentive pooling + MLP.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument("--input_wav", type=str, required=True, help="Path to input .wav")

    parser.add_argument("--hubert_name", type=str, default="facebook/hubert-base-ls960")
    parser.add_argument("--layer_idx", type=int, default=8)
    parser.add_argument("--sample_rate", type=int, default=16000)

    parser.add_argument("--scale", type=float, default=1.0, help="Scale factor for the output embedding norm")
    parser.add_argument("--window_sec", type=float, default=1.6, help="Sliding window size in seconds")
    parser.add_argument("--hop_sec", type=float, default=0.8, help="Sliding window hop in seconds")
    parser.add_argument("--min_coverage", type=float, default=0.75, help="Minimum coverage for the last window")
    parser.add_argument("--min_rms_ratio", type=float, default=0.2, help="Minimum RMS ratio vs full utterance")
    parser.add_argument("--gauss_sigma_ratio", type=float, default=0.4, help="Gaussian sigma as fraction of length")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_checkpoint(Path(args.model_path), device, args.layer_idx, args.hubert_name)

    wav = load_wav(Path(args.input_wav), target_sr=args.sample_rate)
    positions = sliding_window_positions(
        num_samples=wav.numel(),
        window_sec=args.window_sec,
        hop_sec=args.hop_sec,
        min_coverage=args.min_coverage,
        sample_rate=args.sample_rate,
    )
    if not positions:
        raise RuntimeError("No windows created from input audio.")

    full_rms = rms_energy(wav)
    min_rms = max(full_rms * args.min_rms_ratio, 1e-6)
    win_samples = int(args.window_sec * args.sample_rate)

    segments: list[np.ndarray] = []
    centers: list[float] = []
    for start, end, center in positions:
        seg = wav[start:end]
        if seg.numel() < win_samples:
            seg = torch.nn.functional.pad(seg, (0, win_samples - seg.numel()))
        if rms_energy(seg) < min_rms:
            continue
        segments.append(seg.detach().cpu().numpy())
        centers.append(center)

    if not segments:
        raise RuntimeError("All windows filtered as silence. Adjust min_rms_ratio.")

    inputs = model.feature_extractor(
        segments,
        sampling_rate=args.sample_rate,
        padding=True,
        return_tensors="pt",
    )
    input_values = inputs["input_values"].to(device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    with torch.inference_mode():
        dir_pred, mag_pred, _pooled = model(input_values, attention_mask)
    dir_pred = dir_pred.detach().cpu().numpy()
    mag_pred = mag_pred.detach().cpu().numpy().squeeze(1)
    norms = np.linalg.norm(dir_pred, axis=1, keepdims=True)
    sv_dirs = dir_pred / (norms + 1e-6)

    mu = wav.numel() / 2.0
    sigma = max(wav.numel() * args.gauss_sigma_ratio, 1.0)
    centers_arr = np.asarray(centers, dtype=np.float32)
    gauss_weights = np.exp(-0.5 * ((centers_arr - mu) / sigma) ** 2)
    dir_agg = weighted_average(sv_dirs, gauss_weights)
    dir_norm = np.linalg.norm(dir_agg)
    if dir_norm > 0:
        dir_agg = dir_agg / dir_norm
    r = float(np.average(mag_pred, weights=gauss_weights))
    aggregated = dir_agg * r
    if args.scale != 1.0:
        aggregated = aggregated * args.scale

    print(aggregated.reshape(1, -1))
    sv_embedding_hex = float32_to_hex(aggregated)
    print(f"sv_embedding (Hex): {sv_embedding_hex}")


if __name__ == "__main__":
    main()
