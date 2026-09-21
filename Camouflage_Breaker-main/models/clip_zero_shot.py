"""
Zero-shot COD class recognition via OpenCLIP (additive).

Used to refine ResNet50 labels on hard / out-of-distribution camouflage
photos without modifying production checkpoints.
"""
from __future__ import annotations

import re
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image


def _humanize(name: str) -> str:
    name = name.replace("Reccoon", "Raccoon")
    return re.sub(r"(?<!^)(?=[A-Z])", " ", name)


class ClipZeroShotClassifier:
    """OpenCLIP ViT-L/14 zero-shot over a fixed COD class list."""

    def __init__(
        self,
        class_names: list[str],
        device: Optional[torch.device] = None,
        model_name: str = "ViT-L-14",
        pretrained: str = "openai",
    ):
        import open_clip

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.class_names = list(class_names)
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model = self.model.to(device).eval()

        templates = [
            "a photo of a {}",
            "a camouflaged {}",
            "a {} blending into the background",
            "wildlife photo of a {}",
        ]
        with torch.no_grad():
            feats = []
            for cls in self.class_names:
                label = _humanize(cls).lower()
                prompts = [t.format(label) for t in templates]
                tokens = self.tokenizer(prompts).to(device)
                emb = self.model.encode_text(tokens)
                emb = emb / emb.norm(dim=-1, keepdim=True)
                feats.append(emb.mean(0))
            text = torch.stack(feats, dim=0)
            self.text_features = text / text.norm(dim=-1, keepdim=True)

    def _encode_bgr(self, image_bgr: np.ndarray) -> torch.Tensor:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        tensor = self.preprocess(pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.model.encode_image(tensor)
            feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat

    def predict(
        self,
        crop_bgr: np.ndarray,
        full_bgr: Optional[np.ndarray] = None,
        top_k: int = 5,
    ) -> dict:
        if crop_bgr is None or getattr(crop_bgr, "size", 0) == 0:
            return {
                "class_name": "No object detected",
                "confidence": 0.0,
                "predicted_index": None,
                "top_k": [],
            }

        with torch.no_grad():
            feat = self._encode_bgr(crop_bgr)
            if full_bgr is not None and getattr(full_bgr, "size", 0) > 0:
                feat = feat + self._encode_bgr(full_bgr)
                feat = feat / feat.norm(dim=-1, keepdim=True)
            logits = (100.0 * feat @ self.text_features.T)[0]
            probs = torch.softmax(logits, dim=0)
            conf, idx = torch.max(probs, dim=0)
            k = min(top_k, probs.numel())
            vals, idxs = torch.topk(probs, k)

        top = [
            {
                "class_name": self.class_names[int(i)],
                "confidence": float(v) * 100.0,
                "index": int(i),
            }
            for v, i in zip(vals, idxs)
        ]
        pred_i = int(idx.item())
        return {
            "class_name": self.class_names[pred_i],
            "confidence": float(conf.item()) * 100.0,
            "predicted_index": pred_i,
            "top_k": top,
        }
