"""
ESCNet segmentation wrapper (ADDITIVE).

Loads the fine-tuned ESCNet checkpoint via configurable paths.
Does not modify ESCNet official source or the fine-tuned weights.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "escnet.json"


def _load_config() -> dict:
    cfg = {}
    if DEFAULT_CONFIG.is_file():
        cfg = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    root = os.environ.get("TUKU_ESCNET_ROOT") or cfg.get("escnet_root") or ""
    ckpt = os.environ.get("TUKU_ESCNET_CHECKPOINT") or cfg.get("checkpoint") or ""
    return {"escnet_root": root, "checkpoint": ckpt}


def resolve_escnet_paths() -> Tuple[Path, Path]:
    cfg = _load_config()
    root = Path(cfg["escnet_root"])
    ckpt = Path(cfg["checkpoint"])
    if not root.is_dir():
        # Try WSL mount from Windows
        alt = Path(r"\\wsl$\Ubuntu\home\batka\ESCNet")
        if alt.is_dir():
            root = alt
            ckpt = alt / "checkpoints" / "escnet_finetune" / "escnet_cod10k_best.pth"
    if not root.is_dir():
        raise FileNotFoundError(
            "ESCNet root not found. Set TUKU_ESCNET_ROOT or config/escnet.json."
        )
    if not ckpt.is_file():
        raise FileNotFoundError(
            f"ESCNet checkpoint not found: {ckpt}. "
            "Set TUKU_ESCNET_CHECKPOINT or config/escnet.json."
        )
    return root, ckpt


def select_device(preferred: Optional[Union[str, torch.device]] = None, allow_cpu: bool = True):
    if preferred is not None:
        return torch.device(preferred)
    if torch.cuda.is_available():
        try:
            t = torch.zeros(1, device="cuda")
            _ = t + 1
            return torch.device("cuda")
        except Exception:
            if not allow_cpu:
                raise
            return torch.device("cpu")
    if allow_cpu:
        return torch.device("cpu")
    raise RuntimeError("CUDA required and allow_cpu=False")


class ESCNetSegmenter:
    """predict(image_bgr) -> uint8 binary mask {0,255} at original resolution."""

    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        escnet_root: Optional[Union[str, Path]] = None,
        device: Optional[Union[str, torch.device]] = None,
        allow_cpu: bool = True,
        image_size: int = 416,
    ):
        root, default_ckpt = resolve_escnet_paths()
        self.escnet_root = Path(escnet_root) if escnet_root else root
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else default_ckpt
        self.image_size = image_size
        self.device = select_device(device, allow_cpu=allow_cpu)

        root_s = str(self.escnet_root)
        if root_s not in sys.path:
            sys.path.insert(0, root_s)

        from models.ESCNet import ESCNet  # type: ignore  # noqa: E402
        from utils import check_state_dict  # type: ignore  # noqa: E402

        self.config = SimpleNamespace(
            backbone="pvt_v2_b4",
            img_size=image_size,
            lateral_channels=[512, 320, 128, 64],
            compile=False,
        )
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        self.model = ESCNet(self.config, pretrained=False).to(self.device)
        raw = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        if isinstance(raw, dict) and "model_state_dict" in raw:
            state = raw["model_state_dict"]
        elif isinstance(raw, dict) and "state_dict" in raw:
            state = raw["state_dict"]
        else:
            state = raw
        state = check_state_dict(state)
        missing, unexpected = self.model.load_state_dict(state, strict=True)
        if missing or unexpected:
            raise RuntimeError(
                f"ESCNet state_dict mismatch missing={missing} unexpected={unexpected}"
            )
        self.model.eval()

    @torch.inference_mode()
    def predict(
        self,
        image: np.ndarray,
        threshold: float = 0.5,
        assume_bgr: bool = True,
    ) -> np.ndarray:
        if image is None:
            raise ValueError("image is None")
        if assume_bgr:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb = image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        pil = Image.fromarray(rgb)
        orig_w, orig_h = pil.size
        inp = self.transform(pil).unsqueeze(0).to(self.device)
        _edge, preds = self.model(inp)
        probs = preds[-1].sigmoid()
        binary = (probs >= threshold).float()
        binary_up = F.interpolate(
            binary, size=(orig_h, orig_w), mode="bilinear", align_corners=True
        )
        mask = (binary_up[0, 0].float().cpu().numpy() >= threshold).astype(np.uint8) * 255
        return mask

    def __call__(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Return finest mask logits for shared pipeline get_mask() compatibility."""
        _edge, preds = self.model(image_tensor)
        return preds[-1]
