from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torchaudio

from .model import FrozenHubertSvModel
from .utils import (
    TARGET_SR,
    float32_to_hex,
    rms_energy,
    sliding_window_positions,
    weighted_average,
)

DEFAULT_LAYER_IDX = 8
DEFAULT_WINDOW_SEC = 1.6
DEFAULT_HOP_SEC = 0.8
DEFAULT_MIN_COVERAGE = 0.75
DEFAULT_MIN_RMS_RATIO = 0.2
DEFAULT_GAUSS_SIGMA_RATIO = 0.4
DEFAULT_SCALE = 1.0

_MODEL_CACHE: dict[tuple[str, str, int, str], FrozenHubertSvModel] = {}


def load_checkpoint(
    model_path: Path,
    device: torch.device,
    layer_idx: int = DEFAULT_LAYER_IDX,
    hubert_name: str = "facebook/hubert-base-ls960",
) -> FrozenHubertSvModel:
    cache_key = (str(model_path.resolve()), hubert_name, int(layer_idx), device.type)
    cached = _MODEL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        config = checkpoint.get("config", {})
        if not hubert_name:
            hubert_name = str(config.get("hubert_name", hubert_name))
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

    _MODEL_CACHE[cache_key] = model
    return model


def load_wav(path: Path, target_sr: int = TARGET_SR) -> torch.Tensor:
    wav, sr = torchaudio.load(str(path))
    wav = wav.mean(dim=0)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav.to(torch.float32)


def extract_embedding(
    model: FrozenHubertSvModel,
    wav: torch.Tensor,
    device: torch.device,
    window_sec: float = DEFAULT_WINDOW_SEC,
    hop_sec: float = DEFAULT_HOP_SEC,
    min_coverage: float = DEFAULT_MIN_COVERAGE,
    min_rms_ratio: float = DEFAULT_MIN_RMS_RATIO,
    gauss_sigma_ratio: float = DEFAULT_GAUSS_SIGMA_RATIO,
    scale: float = DEFAULT_SCALE,
    direct_average: bool = False,
) -> tuple[np.ndarray, str]:
    positions = sliding_window_positions(
        num_samples=wav.numel(),
        window_sec=window_sec,
        hop_sec=hop_sec,
        min_coverage=min_coverage,
        sample_rate=TARGET_SR,
    )
    if not positions:
        raise RuntimeError("No windows created from input audio.")

    full_rms = rms_energy(wav)
    min_rms = max(full_rms * min_rms_ratio, 1e-6)
    win_samples = int(window_sec * TARGET_SR)

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
        sampling_rate=TARGET_SR,
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
    sigma = max(wav.numel() * gauss_sigma_ratio, 1.0)
    centers_arr = np.asarray(centers, dtype=np.float32)
    gauss_weights = np.exp(-0.5 * ((centers_arr - mu) / sigma) ** 2)

    if direct_average:
        sv_pred = sv_dirs * mag_pred[:, None]
        aggregated = weighted_average(sv_pred, gauss_weights)
    else:
        dir_agg = weighted_average(sv_dirs, gauss_weights)
        dir_norm = np.linalg.norm(dir_agg)
        if dir_norm > 0:
            dir_agg = dir_agg / dir_norm
        r = float(np.average(mag_pred, weights=gauss_weights))
        aggregated = dir_agg * r

    if scale != 1.0:
        aggregated = aggregated * float(scale)

    hex_text = float32_to_hex(aggregated)
    return aggregated, hex_text
