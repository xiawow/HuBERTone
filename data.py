from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset

from .utils import (
    TARGET_SR,
    extract_version_from_name,
    load_sv_map,
    rms_energy,
    sliding_window_positions,
)


@dataclass(frozen=True)
class WindowEntry:
    wav_path: Path
    start: int
    end: int
    center: float
    target: torch.Tensor


def _resampled_length(num_frames: int, src_sr: int, target_sr: int) -> int:
    if src_sr == target_sr:
        return int(num_frames)
    return int(round(num_frames * float(target_sr) / float(src_sr)))


def _read_wav_metadata(wav_path: Path) -> tuple[int, int] | None:
    """Fast metadata read for PCM wav via the stdlib wave module."""
    try:
        with wave.open(str(wav_path), "rb") as wf:
            return wf.getframerate(), wf.getnframes()
    except Exception:
        return None


def _read_src_metadata(wav_path: Path) -> tuple[int, int]:
    """Get (sample_rate, num_frames) with progressively slower fallbacks."""
    meta = _read_wav_metadata(wav_path)
    if meta is not None:
        return meta

    info_fn = getattr(torchaudio, "info", None)
    if callable(info_fn):
        info = info_fn(str(wav_path))
        return int(info.sample_rate), int(info.num_frames)

    # Older torchaudio builds may not expose torchaudio.info; fall back to load.
    wav, sr = torchaudio.load(str(wav_path))
    num_frames = wav.shape[-1]
    return int(sr), int(num_frames)


class HubertWindowDataset(Dataset):
    """Dataset that expands each wav into sliding-window segments.

    Label lookup and filename parsing follow preprocessing1.py:
    - version is the final underscore-separated token in the wav stem.
    - emb_list.json maps version -> 32-dim SV embedding in hex.
    """

    def __init__(
        self,
        wav_dir: Path,
        emb_list: Path,
        window_sec: float = 8.0,
        hop_sec: float = 4.0,
        min_coverage: float = 0.75,
        min_rms_ratio: float = 0.2,
        target_sr: int = TARGET_SR,
        cache_dir: Path | None = None,
    ) -> None:
        super().__init__()
        self.wav_dir = wav_dir
        self.target_sr = target_sr
        self.window_sec = window_sec
        self.hop_sec = hop_sec
        self.min_coverage = min_coverage
        self.min_rms_ratio = min_rms_ratio
        self.window_samples = int(window_sec * target_sr)
        self.cache_dir = cache_dir

        sv_map = load_sv_map(emb_list)
        wav_paths = sorted(wav_dir.rglob("*.wav"))

        entries: list[WindowEntry] = []
        skipped = 0
        self._src_sr: dict[Path, int] = {}
        self._num_frames: dict[Path, int] = {}
        self._full_rms: dict[Path, float] = {}
        self._cached_path: dict[Path, Path] = {}
        self._wav_paths: list[Path] = []
        wav_path_set: set[Path] = set()
        for wav_path in wav_paths:
            version = extract_version_from_name(wav_path)
            sv_emb = sv_map.get(version)
            if sv_emb is None:
                skipped += 1
                print(f"Warning: no SV embedding for {wav_path.name} (version={version})")
                continue

            try:
                src_sr, num_frames = _read_src_metadata(wav_path)
                self._src_sr[wav_path] = src_sr
                self._num_frames[wav_path] = num_frames
                est_len = _resampled_length(num_frames, src_sr, target_sr)
            except Exception as exc:  # pragma: no cover - defensive against backend issues
                skipped += 1
                print(f"Warning: failed to read info for {wav_path} ({exc})")
                continue

            if wav_path not in wav_path_set:
                wav_path_set.add(wav_path)
                self._wav_paths.append(wav_path)

            positions = sliding_window_positions(
                est_len,
                window_sec=window_sec,
                hop_sec=hop_sec,
                min_coverage=min_coverage,
                sample_rate=target_sr,
            )
            target_tensor = torch.tensor(sv_emb, dtype=torch.float32)
            for start, end, center in positions:
                entries.append(
                    WindowEntry(
                        wav_path=wav_path,
                        start=start,
                        end=end,
                        center=center,
                        target=target_tensor,
                    )
                )

        if not entries:
            raise RuntimeError("No valid window entries found. Check wav_dir and emb_list.json.")

        self.entries = entries
        if self.cache_dir is not None:
            print(f"Cache enabled: {self.cache_dir}")
        if skipped:
            print(f"Dataset build: {len(entries)} windows, skipped {skipped} wavs.")
        else:
            print(f"Dataset build: {len(entries)} windows.")

    def __len__(self) -> int:
        return len(self.entries)

    def precompute_cache(self, show_progress: bool = True) -> None:
        if self.cache_dir is None:
            print("Cache disabled; skipping precompute.")
            return

        if not self._wav_paths:
            print("No wav paths available for caching.")
            return

        if show_progress:
            from tqdm import tqdm

            iterator = tqdm(self._wav_paths, desc="Caching 16k wavs")
        else:
            iterator = self._wav_paths

        for wav_path in iterator:
            self._ensure_cached(wav_path)
            self._get_full_rms(wav_path)

    def _cache_path(self, wav_path: Path) -> Path:
        if self.cache_dir is None:
            return wav_path
        try:
            rel = wav_path.relative_to(self.wav_dir)
        except ValueError:
            rel = wav_path.name
        cache_path = self.cache_dir / rel
        if cache_path.suffix.lower() != ".wav":
            cache_path = cache_path.with_suffix(".wav")
        return cache_path

    def _ensure_cached(self, wav_path: Path) -> Path:
        if self.cache_dir is None:
            return wav_path
        cached = self._cached_path.get(wav_path)
        if cached is not None and cached.exists():
            return cached

        cache_path = self._cache_path(wav_path)
        if cache_path.exists():
            self._cached_path[wav_path] = cache_path
            return cache_path

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        wav, sr = torchaudio.load(str(wav_path))
        wav = wav.mean(dim=0)
        if sr != self.target_sr:
            wav = torchaudio.functional.resample(wav, sr, self.target_sr)
        wav = wav.to(torch.float32).unsqueeze(0).cpu()
        torchaudio.save(str(cache_path), wav, sample_rate=self.target_sr)
        self._cached_path[wav_path] = cache_path
        return cache_path

    def _get_src_sr(self, wav_path: Path) -> int:
        if self.cache_dir is not None:
            return self.target_sr
        sr = self._src_sr.get(wav_path)
        if sr is not None:
            return sr
        sr, num_frames = _read_src_metadata(wav_path)
        self._src_sr[wav_path] = sr
        self._num_frames[wav_path] = num_frames
        return sr

    def _get_full_rms(self, wav_path: Path) -> float:
        cached = self._full_rms.get(wav_path)
        if cached is not None:
            return cached
        source_path = self._ensure_cached(wav_path) if self.cache_dir is not None else wav_path
        wav, sr = torchaudio.load(str(source_path))
        wav = wav.mean(dim=0)
        if sr != self.target_sr:
            wav = torchaudio.functional.resample(wav, sr, self.target_sr)
        full_rms = rms_energy(wav)
        self._full_rms[wav_path] = full_rms
        return full_rms

    def _load_segment(self, wav_path: Path, start: int, end: int) -> torch.Tensor:
        """Load a segment using frame offsets when supported."""
        source_path = self._ensure_cached(wav_path) if self.cache_dir is not None else wav_path
        src_sr = self._get_src_sr(wav_path)
        src_start = int(start * src_sr / self.target_sr)
        src_end = int(end * src_sr / self.target_sr)
        num_frames = max(1, src_end - src_start)

        try:
            wav, sr = torchaudio.load(str(source_path), frame_offset=src_start, num_frames=num_frames)
        except TypeError:
            wav, sr = torchaudio.load(str(source_path))
            wav = wav[..., src_start : src_start + num_frames]

        wav = wav.mean(dim=0)
        if sr != self.target_sr:
            wav = torchaudio.functional.resample(wav, sr, self.target_sr)
        wav = wav.to(torch.float32)

        # Ensure a consistent window length after resampling.
        if wav.numel() > self.window_samples:
            wav = wav[: self.window_samples]
        elif wav.numel() < self.window_samples:
            pad_len = self.window_samples - wav.numel()
            wav = torch.nn.functional.pad(wav, (0, pad_len))
        return wav

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor] | None:
        entry = self.entries[idx]

        full_rms = self._get_full_rms(entry.wav_path)
        min_rms = max(full_rms * self.min_rms_ratio, 1e-6)

        seg = self._load_segment(entry.wav_path, entry.start, entry.end)

        if rms_energy(seg) < min_rms:
            return None

        return seg, entry.target


def make_collate_fn(
    feature_extractor=None,
    target_sr: int = TARGET_SR,
    hubert_name: str = "facebook/hubert-base-ls960",
    do_normalize: bool = True,
) -> Callable:
    """Collate that pads and normalizes using the HF feature extractor.

    This version is simple and works well with num_workers=0 on Windows.
    """
    _ = hubert_name, do_normalize  # kept for backward-compatible call sites
    if feature_extractor is None:
        raise ValueError("feature_extractor is required for the single-thread collate_fn.")

    def collate(batch: list[tuple[torch.Tensor, torch.Tensor] | None]):
        batch = [item for item in batch if item is not None]
        if not batch:
            return None

        wavs, targets = zip(*batch)
        wav_list = [w.detach().cpu().numpy() for w in wavs]
        inputs = feature_extractor(
            wav_list,
            sampling_rate=target_sr,
            padding=True,
            return_tensors="pt",
        )
        target_tensor = torch.stack(targets, dim=0)
        return inputs, target_tensor

    return collate
