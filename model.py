from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import HubertModel, Wav2Vec2FeatureExtractor

from .utils import TARGET_SR, lengths_to_mask, set_default_hf_home


class AttentiveStatsPooling(nn.Module):
    """Attentive Statistics Pooling (ASP): weighted mean + weighted std."""

    def __init__(self, input_dim: int, attn_hidden_dim: int = 256, eps: float = 1e-6) -> None:
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(input_dim, attn_hidden_dim),
            nn.Tanh(),
            nn.Linear(attn_hidden_dim, 1),
        )
        self.eps = eps

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected [B, T, D], got shape={tuple(x.shape)}")

        scores = self.attn(x).squeeze(-1)

        if mask is not None:
            if mask.shape != scores.shape:
                raise ValueError(
                    f"Mask shape must match [B, T]. mask={tuple(mask.shape)} scores={tuple(scores.shape)}"
                )
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)

        weights = F.softmax(scores, dim=1)
        weights = weights.unsqueeze(-1)

        mean = torch.sum(x * weights, dim=1)
        var = torch.sum(weights * (x - mean.unsqueeze(1)) ** 2, dim=1)
        std = torch.sqrt(var + self.eps)
        return torch.cat([mean, std], dim=1)


class DirectionMagnitudeHead(nn.Module):
    """Residual MLP head that predicts direction and magnitude separately."""

    def __init__(self, input_dim: int = 768, hidden_dim: int = 384, dir_dim: int = 32) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.dir_head = nn.Linear(hidden_dim, dir_dim)
        self.mag_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h1 = F.silu(self.ln1(self.fc1(x)))
        h2 = F.silu(self.ln2(self.fc2(h1)))
        x = h1 + h2
        dir_pred = self.dir_head(x)
        mag_pred = F.softplus(self.mag_head(x))
        return dir_pred, mag_pred


@dataclass(frozen=True)
class HubertConfig:
    hubert_name: str = "facebook/hubert-base-ls960"
    layer_idx: int = 8
    sample_rate: int = TARGET_SR


class FrozenHubertSvModel(nn.Module):
    """wav -> frozen HuBERT (layer 8) -> ASP -> (direction, magnitude)."""

    def __init__(
        self,
        hubert_name: str = "facebook/hubert-base-ls960",
        layer_idx: int = 8,
        attn_hidden_dim: int = 256,
        out_dim: int = 32,
        hf_home: Path | None = None,
        sample_rate: int = TARGET_SR,
    ) -> None:
        super().__init__()
        if hf_home is not None:
            set_default_hf_home(hf_home)
        else:
            set_default_hf_home()

        self.config = HubertConfig(hubert_name=hubert_name, layer_idx=layer_idx, sample_rate=sample_rate)
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(hubert_name)
        self.hubert = HubertModel.from_pretrained(hubert_name)

        for param in self.hubert.parameters():
            param.requires_grad = False
        self.hubert.eval()

        hidden_size = self.hubert.config.hidden_size
        self.pool = AttentiveStatsPooling(input_dim=hidden_size, attn_hidden_dim=attn_hidden_dim)
        self.mlp = DirectionMagnitudeHead(input_dim=hidden_size * 2, dir_dim=out_dim)

    @property
    def trainable_parameters(self):
        return list(self.pool.parameters()) + list(self.mlp.parameters())

    def _frame_mask_from_attention(self, attention_mask: torch.Tensor | None) -> torch.Tensor | None:
        if attention_mask is None:
            return None
        lengths = attention_mask.sum(dim=1)
        frame_lengths = self.hubert._get_feat_extract_output_lengths(lengths)
        max_frames = int(frame_lengths.max().item())
        return lengths_to_mask(frame_lengths, max_frames)

    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # HuBERT stays frozen; we do not track its gradients.
        with torch.no_grad():
            outputs = self.hubert(
                input_values=input_values,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )

        hidden_states = outputs.hidden_states
        layer_idx = self.config.layer_idx
        if hidden_states is None or layer_idx >= len(hidden_states):
            raise ValueError(f"Requested layer_idx={layer_idx}, but only {len(hidden_states) if hidden_states else 0} hidden states available.")

        frames = hidden_states[layer_idx]
        frame_mask = self._frame_mask_from_attention(attention_mask)
        if frame_mask is not None:
            frame_mask = frame_mask.to(frames.device)

        pooled = self.pool(frames, frame_mask)
        dir_pred, mag_pred = self.mlp(pooled)
        return dir_pred, mag_pred, pooled
