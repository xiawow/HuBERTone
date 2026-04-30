from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import gradio as gr
import numpy as np
import torch
import torchaudio

# Allow running as a script from the repo root or from hubert/: python ap.py
if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent))
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from hubert.model import FrozenHubertSvModel
from hubert.utils import (
    TARGET_SR,
    float32_to_hex,
    rms_energy,
    sliding_window_positions,
    weighted_average,
)


_MODEL_CACHE: dict[tuple[str, str, int, str], FrozenHubertSvModel] = {}


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint(
    model_path: Path,
    device: torch.device,
    layer_idx: int,
    hubert_name: str,
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


def _to_mono_resampled(audio_path: str, target_sr: int = TARGET_SR) -> torch.Tensor:
    wav, sr = torchaudio.load(str(audio_path))
    wav = wav.mean(dim=0)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav.to(torch.float32)


def run_infer(
    audio: str | None,
    model_path: str,
    hubert_name: str,
    layer_idx: int,
    window_sec: float,
    hop_sec: float,
    min_coverage: float,
    min_rms_ratio: float,
    gauss_sigma_ratio: float,
    scale: float,
    direct_average: bool,
) -> tuple[str, str]:
    print(
        "[infer] audio=%s, model_path=%s, hubert_name=%s, layer_idx=%s, window_sec=%.3f, hop_sec=%.3f, "
        "min_coverage=%.3f, min_rms_ratio=%.3f, gauss_sigma_ratio=%.3f, scale=%.3f, direct_average=%s"
        % (
            audio,
            model_path,
            hubert_name,
            layer_idx,
            window_sec,
            hop_sec,
            min_coverage,
            min_rms_ratio,
            gauss_sigma_ratio,
            scale,
            direct_average,
        )
    )
    if audio is None:
        return "请先上传音频。", ""

    model_path_p = Path(model_path)
    if not model_path_p.exists():
        return f"模型不存在: {model_path_p}", ""

    device = _device()
    model = load_checkpoint(model_path_p, device, layer_idx, hubert_name)

    wav = _to_mono_resampled(audio, target_sr=TARGET_SR)
    positions = sliding_window_positions(
        num_samples=wav.numel(),
        window_sec=window_sec,
        hop_sec=hop_sec,
        min_coverage=min_coverage,
        sample_rate=TARGET_SR,
    )
    if not positions:
        return "未能从音频创建滑窗。", ""

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
        return "所有滑窗都被静音过滤了（调低 min_rms_ratio 试试）。", ""

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

    vec_text = np.array2string(aggregated.reshape(1, -1), precision=6, suppress_small=False)
    hex_text = float32_to_hex(aggregated)
    return vec_text, hex_text


def build_ui(default_model_path: str, default_hubert_name: str, default_layer_idx: int) -> gr.Blocks:
    with gr.Blocks(title="HuBERT SV 推理") as demo:
        gr.Markdown("## svembbing 推理")

        with gr.Row():
            audio_in = gr.Audio(label="输入音频", type="filepath")
            with gr.Column():
                model_path = gr.Textbox(
                    label="模型路径 (.pth)",
                    value=default_model_path,
                    interactive=False,
                )
                hubert_name = gr.Textbox(
                    label="HuBERT 路径",
                    value=default_hubert_name,
                    interactive=False,
                )
                layer_idx = gr.Number(
                    label="层索引 (hidden_states)",
                    value=default_layer_idx,
                    precision=0,
                    interactive=False,
                )

        with gr.Row():
            window_sec = gr.Slider(0.4, 12.0, value=4.0, step=0.2, label="window_sec")
            hop_sec = gr.Slider(0.2, 12.0, value=0.5, step=0.2, label="hop_sec")
            min_coverage = gr.Slider(0.1, 1.0, value=0.75, step=0.05, label="min_coverage (没问题就别动它)")
            min_rms_ratio = gr.Slider(0.0, 1.0, value=0.2, step=0.05, label="min_rms_ratio (没问题就别动它)")

        with gr.Row():
            gauss_sigma_ratio = gr.Slider(0.1, 1.5, value=0.4, step=0.05, label="gauss_sigma_ratio")
            scale = gr.Slider(0.5, 2.0, value=1.0, step=0.05, label="scale")

        direct_average = gr.Checkbox(
            label="直接加权平均（不做方向归一化）",
            value=False,
        )

        run_btn = gr.Button("开始推理", variant="primary")

        with gr.Row():
            vec_out = gr.Textbox(label="SV 向量 (1x32)", lines=4)
            hex_out = gr.Textbox(label="SV Hex", lines=4)

        run_btn.click(
            fn=run_infer,
            inputs=[
                audio_in,
                model_path,
                hubert_name,
                layer_idx,
                window_sec,
                hop_sec,
                min_coverage,
                min_rms_ratio,
                gauss_sigma_ratio,
                scale,
                direct_average,
            ],
            outputs=[vec_out, hex_out],
        )

    return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gradio web app for HuBERT SV inference.")
    parser.add_argument("--model_path", type=str, default="hubert_sv_model.pth")
    parser.add_argument("--hubert_name", type=str, default="facebook/hubert-base-ls960")
    parser.add_argument("--layer_idx", type=int, default=9)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Enable Gradio share link")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    demo = build_ui(args.model_path, args.hubert_name, args.layer_idx)
    demo.queue().launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
