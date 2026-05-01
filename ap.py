from __future__ import annotations

import argparse
import sys
from pathlib import Path

import gradio as gr
import numpy as np
import torch

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent))
    sys.path.append(str(Path(__file__).resolve().parent.parent))

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
from hubert.utils import TARGET_SR


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
        % (audio, model_path, hubert_name, layer_idx, window_sec, hop_sec,
           min_coverage, min_rms_ratio, gauss_sigma_ratio, scale, direct_average)
    )
    if audio is None:
        return "请先上传音频。", ""

    model_path_p = Path(model_path)
    if not model_path_p.exists():
        return f"模型不存在: {model_path_p}", ""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_checkpoint(model_path_p, device, layer_idx, hubert_name)
    wav = load_wav(Path(audio), target_sr=TARGET_SR)

    try:
        aggregated, hex_text = extract_embedding(
            model=model,
            wav=wav,
            device=device,
            window_sec=window_sec,
            hop_sec=hop_sec,
            min_coverage=min_coverage,
            min_rms_ratio=min_rms_ratio,
            gauss_sigma_ratio=gauss_sigma_ratio,
            scale=scale,
            direct_average=direct_average,
        )
    except RuntimeError as e:
        return str(e), ""

    vec_text = np.array2string(aggregated.reshape(1, -1), precision=6, suppress_small=False)
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
                audio_in, model_path, hubert_name, layer_idx,
                window_sec, hop_sec, min_coverage, min_rms_ratio,
                gauss_sigma_ratio, scale, direct_average,
            ],
            outputs=[vec_out, hex_out],
        )

    return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gradio web app for HuBERT SV inference.")
    parser.add_argument("--model_path", type=str, default="checkpoints/hubert_sv_model.pth")
    parser.add_argument("--hubert_name", type=str, default="facebook/hubert-base-ls960")
    parser.add_argument("--layer_idx", type=int, default=DEFAULT_LAYER_IDX)
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
