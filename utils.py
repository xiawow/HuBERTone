import os
import struct
from pathlib import Path
from typing import Iterable

import numpy as np
import torch


TARGET_SR = 16000


def set_default_hf_home(repo_root: Path | None = None) -> Path:
    """Pin HuggingFace cache to the repo-local hf_cache folder."""
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[1]
    repo_root = Path(repo_root)
    hf_home = repo_root if repo_root.name == "hf_cache" else repo_root / "hf_cache"
    os.environ["HF_HOME"] = str(hf_home)
    return hf_home


def hex_to_float32(hex_str: str) -> np.ndarray:
    hex_str = hex_str.replace("0x", "").replace(" ", "")
    if len(hex_str) != 256:
        raise ValueError("SV embedding hex must be 256 characters.")
    return np.frombuffer(bytes.fromhex(hex_str), dtype="<f4")


def float32_to_hex(float_array: Iterable[float]) -> str:
    hex_str = ""
    for val in float_array:
        packed = struct.pack("<f", float(val))
        hex_str += "".join([f"{byte:02x}" for byte in packed])
    return hex_str


def load_sv_map(json_path: Path) -> dict[str, np.ndarray]:
    import json

    with open(json_path, "r", encoding="utf-8") as f:
        items = json.load(f)
    sv_map: dict[str, np.ndarray] = {}
    for item in items:
        version = str(item.get("version", "")).strip()
        data = item.get("data")
        if version and data:
            try:
                sv_map[version] = hex_to_float32(data)
            except ValueError as exc:
                print(f"Warning: {exc} (version={version})")
    return sv_map


def extract_version_from_name(wav_path: Path) -> str:
    parts = wav_path.stem.split("_")
    if not parts:
        return ""
    return parts[-1]


def lengths_to_mask(lengths: torch.Tensor, max_len: int | None = None) -> torch.Tensor:
    if lengths.ndim != 1:
        raise ValueError(f"lengths must be 1D, got shape={tuple(lengths.shape)}")
    if max_len is None:
        max_len = int(lengths.max().item())
    arange = torch.arange(max_len, device=lengths.device)
    return arange.unsqueeze(0) < lengths.unsqueeze(1)


def sliding_window_positions(
    num_samples: int,
    window_sec: float,
    hop_sec: float,
    min_coverage: float,
    sample_rate: int = TARGET_SR,
) -> list[tuple[int, int, float]]:
    """Return (start, end, center) sample positions."""
    win_samples = int(window_sec * sample_rate)
    hop_samples = int(hop_sec * sample_rate)
    if win_samples <= 0 or hop_samples <= 0:
        raise ValueError("window_sec and hop_sec must be > 0")

    if num_samples <= win_samples:
        center = num_samples / 2.0
        return [(0, win_samples, center)]

    positions: list[tuple[int, int, float]] = []
    max_start = num_samples - win_samples
    for start in range(0, max_start + 1, hop_samples):
        end = start + win_samples
        center = start + win_samples / 2.0
        positions.append((start, end, center))

    last_start = (max_start // hop_samples) * hop_samples
    if last_start != max_start:
        coverage = (num_samples - max_start) / win_samples
        if coverage >= min_coverage:
            start = max_start
            end = start + win_samples
            center = start + win_samples / 2.0
            positions.append((start, end, center))
    return positions


def rms_energy(x: torch.Tensor) -> float:
    if x.numel() == 0:
        return 0.0
    return float(torch.sqrt(torch.mean(x * x)).item())


def weighted_average(vectors: np.ndarray, weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float32)
    if weights.size == 0:
        raise ValueError("No weights provided")
    if np.all(weights == 0):
        weights = np.ones_like(weights)
    weights = weights / np.sum(weights)
    return np.sum(vectors * weights[:, None], axis=0)
