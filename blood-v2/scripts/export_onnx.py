#!/usr/bin/env python3
"""Export Blood Mahjong policy model to ONNX format.

Usage:
    python scripts/export_onnx.py --checkpoint checkpoints/model.pth --output model.onnx
    python scripts/export_onnx.py --checkpoint checkpoints/model.pth --output model.onnx --quantize

The exported model takes a flat observation vector and action mask, and
outputs action logits.  This enables deployment on ONNX Runtime, TensorRT,
or WASM-based inference engines.
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn

log = logging.getLogger(__name__)

NUM_TILE_TYPES = 27
OBS_CHANNELS = 464
OBS_SIZE = OBS_CHANNELS * NUM_TILE_TYPES
ACTION_DIM = 34


class OnnxWrapper(nn.Module):
    """Thin wrapper that packages inference model for ONNX export.

    Input:
        obs: (B, OBS_SIZE)   float32
        mask: (B, ACTION_DIM) float32  (1.0 = legal, 0.0 = illegal)

    Output:
        logits: (B, ACTION_DIM) float32  (masked; illegal = -1e9)
    """

    def __init__(self, policy_model):
        super().__init__()
        self.policy = policy_model

    def forward(self, obs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        logits = self.policy(obs)
        logits = logits.masked_fill(mask < 0.5, -1e9)
        return logits


def export(checkpoint_path: str, output_path: str, quantize: bool = False):
    from blood.model.inference import PolicyModel

    model = PolicyModel.from_sf2_checkpoint(checkpoint_path, device="cpu")
    wrapper = OnnxWrapper(model)
    wrapper.eval()

    obs_size = model._obs_channels * NUM_TILE_TYPES
    batch = 1
    dummy_obs = torch.randn(batch, obs_size)
    dummy_mask = torch.ones(batch, ACTION_DIM)

    log.info("Exporting to %s ...", output_path)
    torch.onnx.export(
        wrapper,
        (dummy_obs, dummy_mask),
        output_path,
        input_names=["obs", "action_mask"],
        output_names=["logits"],
        dynamic_axes={
            "obs": {0: "batch"},
            "action_mask": {0: "batch"},
            "logits": {0: "batch"},
        },
        opset_version=17,
    )
    log.info("ONNX model saved: %s", output_path)

    if quantize:
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType
            q_path = output_path.replace(".onnx", "_int8.onnx")
            quantize_dynamic(output_path, q_path, weight_type=QuantType.QInt8)
            log.info("Quantized model saved: %s", q_path)
        except ImportError:
            log.warning("onnxruntime-tools not installed; skipping quantization")

    _verify(output_path, wrapper)


def _verify(onnx_path: str, wrapper: nn.Module):
    try:
        import onnxruntime as ort
    except ImportError:
        log.info("onnxruntime not installed; skipping verification")
        return

    session = ort.InferenceSession(onnx_path)
    obs_size = wrapper.policy._obs_channels * NUM_TILE_TYPES

    for batch_size in [1, 4, 16, 64]:
        obs = torch.randn(batch_size, obs_size)
        mask = torch.ones(batch_size, ACTION_DIM)
        ort_out = session.run(
            None,
            {"obs": obs.numpy(), "action_mask": mask.numpy()},
        )[0]

        with torch.no_grad():
            pt_out = wrapper(obs, mask).numpy()

        diff = abs(ort_out - pt_out).max()
        log.info("Batch=%d: max absolute difference (PyTorch vs ONNX): %.6f", batch_size, diff)
        assert diff < 1e-4, f"ONNX verification failed (batch={batch_size}): max diff = {diff}"

    log.info("ONNX verification passed (batch=1, 4, 16, 64)!")


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Export Blood Mahjong model to ONNX")
    parser.add_argument("--checkpoint", required=True, help="SF2 checkpoint path")
    parser.add_argument("--output", default="blood_policy.onnx", help="Output ONNX path")
    parser.add_argument("--quantize", action="store_true", help="Also export INT8 quantized version")
    args = parser.parse_args()

    export(args.checkpoint, args.output, args.quantize)


if __name__ == "__main__":
    main()
