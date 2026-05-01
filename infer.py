from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from hubert.inference import (
    DEFAULT_GAUSS_SIGMA_RATIO,
    DEFAULT_HOP_SEC,
    DEFAULT_LAYER_IDX,
    DEFAULT_MIN_COVERAGE,
    DEFAULT_MIN_RMS_RATIO,
    DEFAULT_SCALE,
    DEFAULT_WINDOW_SEC,
    extract_embedding,
    load_checkpoint,
    load_wav,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inference with frozen HuBERT + attentive pooling + MLP.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument("--input_wav", type=str, required=True, help="Path to input .wav")

    parser.add_argument("--hubert_name", type=str, default="facebook/hubert-base-ls960")
    parser.add_argument("--layer_idx", type=int, default=DEFAULT_LAYER_IDX)
    parser.add_argument("--sample_rate", type=int, default=16000)

    parser.add_argument("--scale", type=float, default=DEFAULT_SCALE, help="Scale factor for the output embedding norm")
    parser.add_argument("--window_sec", type=float, default=DEFAULT_WINDOW_SEC, help="Sliding window size in seconds")
    parser.add_argument("--hop_sec", type=float, default=DEFAULT_HOP_SEC, help="Sliding window hop in seconds")
    parser.add_argument("--min_coverage", type=float, default=DEFAULT_MIN_COVERAGE, help="Minimum coverage for the last window")
    parser.add_argument("--min_rms_ratio", type=float, default=DEFAULT_MIN_RMS_RATIO, help="Minimum RMS ratio vs full utterance")
    parser.add_argument("--gauss_sigma_ratio", type=float, default=DEFAULT_GAUSS_SIGMA_RATIO, help="Gaussian sigma as fraction of length")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_checkpoint(Path(args.model_path), device, args.layer_idx, args.hubert_name)
    wav = load_wav(Path(args.input_wav), target_sr=args.sample_rate)

    aggregated, hex_text = extract_embedding(
        model=model,
        wav=wav,
        device=device,
        window_sec=args.window_sec,
        hop_sec=args.hop_sec,
        min_coverage=args.min_coverage,
        min_rms_ratio=args.min_rms_ratio,
        gauss_sigma_ratio=args.gauss_sigma_ratio,
        scale=args.scale,
    )

    print(aggregated.reshape(1, -1))
    print(f"sv_embedding (Hex): {hex_text}")


if __name__ == "__main__":
    main()
