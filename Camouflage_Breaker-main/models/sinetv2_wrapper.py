"""
SINet-V2 segmentation wrapper (ADDITIVE).

Loads the official SINet-V2 Network from models/sinetv2/source without
modifying ResUNet or ResNet50. Inference uses the finest output (res2).
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types
from typing import Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_SOURCE = os.path.join(PROJECT_ROOT, "models", "sinetv2", "source")
DEFAULT_LIB = os.path.join(DEFAULT_SOURCE, "lib")
DEFAULT_RES2NET = os.path.join(
    PROJECT_ROOT,
    "models",
    "sinetv2",
    "res2net50_v1b_26w_4s-3cf99910.pth",
)
DEFAULT_OFFICIAL = os.path.join(
    PROJECT_ROOT,
    "models",
    "sinetv2",
    "snapshot",
    "SINet_V2",
    "Net_epoch_best.pth",
)
DEFAULT_FINETUNED = os.path.join(
    PROJECT_ROOT,
    "saved_models",
    "sinetv2",
    "sinetv2_cod10k_best.pth",
)


def select_torch_device(
    preferred: Optional[Union[str, torch.device]] = None,
    allow_cpu: bool = False,
) -> torch.device:
    """
    Prefer CUDA when kernels actually run on this GPU.
    If allow_cpu=True (Streamlit path), fall back to CPU when CUDA is unavailable.
    """
    if preferred is not None:
        device = torch.device(preferred)
        if device.type == "cpu":
            return device
        if device.type == "cuda" and torch.cuda.is_available():
            try:
                t = torch.zeros(1, device=device)
                _ = t + 1
                return device
            except Exception:
                if allow_cpu:
                    return torch.device("cpu")
                raise RuntimeError(
                    f"Requested CUDA device is not usable: {device}"
                )
        if allow_cpu:
            return torch.device("cpu")
        return torch.device("cpu")

    if not torch.cuda.is_available():
        if allow_cpu:
            return torch.device("cpu")
        raise RuntimeError("CUDA is required but torch.cuda.is_available() is False.")

    try:
        probe = torch.zeros(1, device="cuda")
        _ = probe + 1
        return torch.device("cuda")
    except Exception as exc:
        if allow_cpu:
            return torch.device("cpu")
        raise RuntimeError(
            f"CUDA probe failed on this GPU; refusing CPU fallback. Error: {exc}"
        ) from exc


def _find_res2net_checkpoint(search_root: str) -> Optional[str]:
    target = "res2net50_v1b_26w_4s-3cf99910.pth"
    preferred = os.path.join(search_root, "models", "sinetv2", target)
    if os.path.isfile(preferred):
        return preferred
    for root, _, files in os.walk(os.path.join(search_root, "models", "sinetv2")):
        if target in files:
            return os.path.join(root, target)
    return None


def _load_module(module_name: str, file_path: str):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module:\n{file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _ensure_sinet_imports(source_dir: str, lib_dir: str, res2net_path: Optional[str]):
    if not os.path.isdir(lib_dir):
        raise FileNotFoundError(f"SINet-V2 lib folder not found:\n{lib_dir}")

    if "lib" not in sys.modules:
        lib_package = types.ModuleType("lib")
        lib_package.__path__ = [lib_dir]
        sys.modules["lib"] = lib_package
    else:
        lib_package = sys.modules["lib"]
        if not hasattr(lib_package, "__path__"):
            lib_package.__path__ = [lib_dir]

    if source_dir not in sys.path:
        sys.path.insert(0, source_dir)
    if lib_dir not in sys.path:
        sys.path.insert(0, lib_dir)

    res2net_file = os.path.join(lib_dir, "Res2Net_v1b.py")
    if not os.path.isfile(res2net_file):
        raise FileNotFoundError(f"Res2Net source not found:\n{res2net_file}")

    res2net_module = _load_module("lib.Res2Net_v1b", res2net_file)
    lib_package.Res2Net_v1b = res2net_module

    if res2net_path and os.path.isfile(res2net_path):
        original = res2net_module.res2net50_v1b_26w_4s

        def local_res2net50_v1b_26w_4s(pretrained=True, **kwargs):
            model = original(pretrained=False, **kwargs)
            if pretrained:
                checkpoint = torch.load(res2net_path, map_location="cpu")
                if isinstance(checkpoint, dict):
                    if "state_dict" in checkpoint:
                        checkpoint = checkpoint["state_dict"]
                    elif "model_state_dict" in checkpoint:
                        checkpoint = checkpoint["model_state_dict"]
                cleaned = {}
                for key, value in checkpoint.items():
                    if key.startswith("module."):
                        key = key[7:]
                    cleaned[key] = value
                model.load_state_dict(cleaned, strict=False)
            return model

        res2net_module.res2net50_v1b_26w_4s = local_res2net50_v1b_26w_4s

    network_file = os.path.join(lib_dir, "Network_Res2Net_GRA_NCD.py")
    if not os.path.isfile(network_file):
        raise FileNotFoundError(f"SINet-V2 network source not found:\n{network_file}")

    network_module = _load_module("lib.Network_Res2Net_GRA_NCD", network_file)
    return network_module.Network


def resolve_checkpoint_path(checkpoint_path: Optional[str] = None) -> str:
    """Prefer fine-tuned weights; fall back to official COD checkpoint."""
    candidates = []
    if checkpoint_path:
        candidates.append(checkpoint_path)
    candidates.extend([DEFAULT_FINETUNED, DEFAULT_OFFICIAL])
    for path in candidates:
        if path and os.path.isfile(path):
            return path
    raise FileNotFoundError(
        "No SINet-V2 checkpoint found. Expected one of:\n"
        + "\n".join(candidates)
    )


def extract_state_dict(checkpoint) -> dict:
    if not isinstance(checkpoint, dict):
        return checkpoint

    if "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
    else:
        # Raw state dict (keys look like layer names)
        sample_key = next(iter(checkpoint.keys()))
        if isinstance(checkpoint[sample_key], torch.Tensor):
            state = checkpoint
        else:
            raise KeyError(
                "Checkpoint dict has no model_state_dict/state_dict "
                f"and does not look like a raw state dict. Keys: {list(checkpoint.keys())[:10]}"
            )

    cleaned = {}
    for key, value in state.items():
        if key.startswith("module."):
            key = key[7:]
        cleaned[key] = value
    return cleaned


class SINetV2Segmenter:
    """Official SINet-V2 wrapper for camouflaged-object segmentation."""

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: Optional[Union[str, torch.device]] = None,
        image_size: int = 352,
        channel: int = 32,
        source_dir: Optional[str] = None,
        res2net_path: Optional[str] = None,
        allow_cpu: bool = False,
    ):
        self.image_size = image_size
        self.device = select_torch_device(device, allow_cpu=allow_cpu)

        source_dir = source_dir or DEFAULT_SOURCE
        lib_dir = os.path.join(source_dir, "lib")
        res2net_path = res2net_path or _find_res2net_checkpoint(PROJECT_ROOT) or DEFAULT_RES2NET
        checkpoint_path = resolve_checkpoint_path(checkpoint_path)

        Network = _ensure_sinet_imports(source_dir, lib_dir, res2net_path)

        # Load architecture without re-downloading ImageNet weights;
        # official / fine-tuned COD checkpoint already contains backbone weights.
        self.model = Network(channel=channel, imagenet_pretrained=False)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = extract_state_dict(checkpoint)
        incompatible = self.model.load_state_dict(state_dict, strict=True)
        if incompatible is not None and (
            getattr(incompatible, "missing_keys", None)
            or getattr(incompatible, "unexpected_keys", None)
        ):
            missing = list(getattr(incompatible, "missing_keys", []) or [])
            unexpected = list(getattr(incompatible, "unexpected_keys", []) or [])
            if missing or unexpected:
                raise RuntimeError(
                    "SINet-V2 checkpoint / architecture mismatch.\n"
                    f"Missing keys: {missing[:20]}\n"
                    f"Unexpected keys: {unexpected[:20]}"
                )

        self.model.to(self.device)
        self.model.eval()
        self.checkpoint_path = checkpoint_path
        self.res2net_path = res2net_path

        self.mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)

    def preprocess(
        self,
        image: np.ndarray,
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """Return normalized tensor and original (H, W)."""
        if image is None:
            raise ValueError("image is None")

        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # Accept RGB or BGR; convert BGR→RGB when channels look OpenCV-like.
        # Callers may pass RGB (PIL) or BGR (cv2). Prefer RGB for the network.
        original_hw = (image.shape[0], image.shape[1])
        image_rgb = image
        # Heuristic: if caller used cv2.imread they pass BGR; pipeline often
        # converts. We document that predict() expects BGR (OpenCV) by default
        # via `assume_bgr`, and forward() accepts tensors already prepared.

        image_resized = cv2.resize(
            image_rgb,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_LINEAR,
        )
        tensor = torch.from_numpy(image_resized.astype(np.float32) / 255.0)
        tensor = tensor.permute(2, 0, 1)
        tensor = (tensor - self.mean) / self.std
        return tensor.unsqueeze(0).to(self.device), original_hw

    def forward_logits(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Run SINet-V2 and return the finest logits map (res2) only.
        image_tensor: (B, 3, H, W)
        """
        with torch.no_grad():
            outputs = self.model(image_tensor)
            if isinstance(outputs, (tuple, list)):
                if len(outputs) < 4:
                    raise RuntimeError(
                        f"Expected SINet-V2 multi-output (res5..res2), got {len(outputs)} tensors."
                    )
                res2 = outputs[3]  # res5, res4, res3, res2
            else:
                res2 = outputs
            return res2

    def predict_mask(
        self,
        image: np.ndarray,
        threshold: float = 0.5,
        assume_bgr: bool = True,
        return_probability: bool = False,
    ) -> np.ndarray:
        """
        Predict a full-resolution mask for an image.

        Returns:
            uint8 mask {0,1} shaped (H, W), or float probability if return_probability.
        """
        if assume_bgr:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image

        original_hw = (image_rgb.shape[0], image_rgb.shape[1])
        image_resized = cv2.resize(
            image_rgb,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_LINEAR,
        )
        tensor = torch.from_numpy(image_resized.astype(np.float32) / 255.0)
        tensor = tensor.permute(2, 0, 1)
        tensor = (tensor - self.mean) / self.std
        tensor = tensor.unsqueeze(0).to(self.device)

        logits = self.forward_logits(tensor)
        if logits.shape[-2:] != (self.image_size, self.image_size):
            logits = F.interpolate(
                logits,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )

        prob = torch.sigmoid(logits)
        prob_up = F.interpolate(
            prob,
            size=original_hw,
            mode="bilinear",
            align_corners=False,
        )
        prob_np = prob_up.squeeze().detach().cpu().numpy()

        if return_probability:
            return prob_np.astype(np.float32)

        return (prob_np > threshold).astype(np.uint8)

    def predict(self, image: np.ndarray, threshold: float = 0.5, assume_bgr: bool = True) -> np.ndarray:
        """Alias matching the project interface: returns binary mask (H, W)."""
        return self.predict_mask(
            image,
            threshold=threshold,
            assume_bgr=assume_bgr,
            return_probability=False,
        )

    def __call__(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Torch-module-like call used by the shared pipeline get_mask().
        Returns logits for res2 only so existing sigmoid/threshold code works.
        """
        return self.forward_logits(image_tensor)
