# ============================================================
# Camouflage Breaker - Complete Inference Pipeline
# ============================================================

import os
import sys
import json

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms


# ------------------------------------------------------------
# Project root
# ------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.insert(0, PROJECT_ROOT)

from models.resunet import ResUNet


class ResNet50Classifier(nn.Module):

    def __init__(self, num_classes=69):

        super().__init__()

        self.backbone = models.resnet50(
            weights=None
        )

        in_features = self.backbone.fc.in_features

        self.backbone.fc = nn.Sequential(

            nn.Dropout(0.5),

            nn.Linear(
                in_features,
                512
            ),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(
                512,
                num_classes
            )
        )

    def forward(self, x):

        return self.backbone(x)


class CamouflageBreakerPipeline:

    def __init__(
        self,
        seg_model_path=None,
        classifier_model_path=None,
        class_mapping_path=None,
        segmenter="resunet",
        # Backward-compatible alias used by older app.py calls
        cls_model_path=None,
    ):

        # ----------------------------------------------------
        # Default paths
        # ----------------------------------------------------

        segmenter = (segmenter or "resunet").strip().lower()
        if segmenter not in {"resunet", "sinetv2", "escnet"}:
            raise ValueError(
                f"Unsupported segmenter '{segmenter}'. "
                "Use 'resunet', 'sinetv2', or 'escnet'."
            )
        self.segmenter = segmenter

        if classifier_model_path is None and cls_model_path is not None:
            classifier_model_path = cls_model_path

        if seg_model_path is None:
            if segmenter == "sinetv2":
                candidate = os.path.join(
                    PROJECT_ROOT,
                    "saved_models",
                    "sinetv2",
                    "sinetv2_cod10k_best.pth"
                )
                official = os.path.join(
                    PROJECT_ROOT,
                    "models",
                    "sinetv2",
                    "snapshot",
                    "SINet_V2",
                    "Net_epoch_best.pth"
                )
                seg_model_path = (
                    candidate
                    if os.path.exists(candidate)
                    else official
                )
            elif segmenter == "escnet":
                # Resolved inside ESCNetSegmenter via config/escnet.json / env
                seg_model_path = None
            else:
                seg_model_path = os.path.join(
                    PROJECT_ROOT,
                    "saved_models",
                    "resunet_best.pth"
                )

        if classifier_model_path is None:

            classifier_model_path = os.path.join(
                PROJECT_ROOT,
                "saved_models",
                "classifier_best.pth"
            )

        if class_mapping_path is None:

            class_mapping_path = os.path.join(
                PROJECT_ROOT,
                "saved_models",
                "class_mapping.json"
            )

        self.seg_model_path = seg_model_path
        self.classifier_model_path = classifier_model_path
        self.class_mapping_path = class_mapping_path

        # ----------------------------------------------------
        # Device
        # ----------------------------------------------------

        # CUDA if usable; otherwise CPU (Streamlit must launch without CUDA).
        if torch.cuda.is_available():
            try:
                torch.zeros(1, device="cuda")
                _ = torch.zeros(1, device="cuda") + 1
                self.device = torch.device("cuda")
            except Exception as exc:
                print(
                    f"WARNING: CUDA probe failed ({exc}); falling back to CPU."
                )
                self.device = torch.device("cpu")
        else:
            self.device = torch.device("cpu")

        print("=" * 70)
        print("CAMOUFLAGE BREAKER INFERENCE PIPELINE")
        print("=" * 70)

        print(
            f"Using device: {self.device}"
        )
        print(
            f"Segmenter: {self.segmenter}"
        )

        # ----------------------------------------------------
        # Load segmentation model (ResUNet / SINet-V2 / ESCNet)
        # ----------------------------------------------------

        if self.segmenter == "sinetv2":

            print(
                "\nLoading SINet-V2 segmentation model..."
            )

            from models.sinetv2_wrapper import SINetV2Segmenter

            self.seg_model = SINetV2Segmenter(
                checkpoint_path=seg_model_path,
                device=self.device,
                allow_cpu=(self.device.type == "cpu"),
            )

            print(
                "[OK] SINet-V2 loaded"
            )
            print(
                f"  Checkpoint: {self.seg_model.checkpoint_path}"
            )

        elif self.segmenter == "escnet":

            print(
                "\nLoading ESCNet segmentation model..."
            )

            from models.escnet_wrapper import ESCNetSegmenter

            self.seg_model = ESCNetSegmenter(
                checkpoint_path=seg_model_path,
                device=self.device,
                allow_cpu=(self.device.type == "cpu"),
            )
            self.seg_model_path = str(self.seg_model.checkpoint_path)

            print(
                "[OK] ESCNet loaded"
            )
            print(
                f"  Checkpoint: {self.seg_model.checkpoint_path}"
            )

        else:

            print(
                "\nLoading ResUNet segmentation model..."
            )

            if not os.path.exists(
                seg_model_path
            ):

                raise FileNotFoundError(
                    f"ResUNet model not found:\n"
                    f"{seg_model_path}"
                )

            self.seg_model = ResUNet(
                encoder_name="resnet50",
                encoder_weights=None
            )

            seg_checkpoint = torch.load(
                seg_model_path,
                map_location=self.device
            )

            if (
                isinstance(seg_checkpoint, dict)
                and
                "model_state_dict" in seg_checkpoint
            ):

                self.seg_model.load_state_dict(
                    seg_checkpoint[
                        "model_state_dict"
                    ]
                )

            else:

                self.seg_model.load_state_dict(
                    seg_checkpoint
                )

            self.seg_model.to(
                self.device
            )

            self.seg_model.eval()

            print(
                "[OK] ResUNet loaded"
            )

        # ----------------------------------------------------
        # Load Classifier Checkpoint (optional if missing)
        # ----------------------------------------------------

        print(
            "\nLoading trained ResNet50 classifier..."
        )

        self.classifier = None
        self.classifier_available = False
        self.num_classes = 69
        self.idx_to_class = {}
        self.class_to_idx = {}

        if not os.path.exists(
            classifier_model_path
        ):

            print(
                "WARNING: Classifier model not found:\n"
                f"{classifier_model_path}\n"
                "Segmentation will still run; classification disabled."
            )

        else:

            checkpoint = torch.load(
                classifier_model_path,
                map_location=self.device
            )

            # ----------------------------------------------------
            # Read number of classes
            # ----------------------------------------------------

            if (
                isinstance(checkpoint, dict)
                and
                "num_classes" in checkpoint
            ):

                self.num_classes = int(
                    checkpoint["num_classes"]
                )

            else:

                self.num_classes = 69

            # ----------------------------------------------------
            # Create EXACT same classifier architecture
            # used during Colab training
            # ----------------------------------------------------

            self.classifier = ResNet50Classifier(
                num_classes=self.num_classes
            )

            # ----------------------------------------------------
            # Load state dictionary
            # ----------------------------------------------------

            if (
                isinstance(checkpoint, dict)
                and
                "model_state_dict" in checkpoint
            ):

                state_dict = checkpoint[
                    "model_state_dict"
                ]

            else:

                state_dict = checkpoint

            self.classifier.load_state_dict(
                state_dict
            )

            self.classifier.to(
                self.device
            )

            self.classifier.eval()
            self.classifier_available = True

            print(
                f"[OK] ResNet50 classifier loaded "
                f"({self.num_classes} classes)"
            )

        # ----------------------------------------------------
        # Load class mapping
        # ----------------------------------------------------

        print(
            "\nLoading class mapping..."
        )

        if not os.path.exists(
            class_mapping_path
        ):

            print(
                "WARNING: Class mapping not found:\n"
                f"{class_mapping_path}"
            )
            self.idx_to_class = {
                str(i): f"class_{i}" for i in range(self.num_classes)
            }
            self.class_to_idx = {
                v: int(k) for k, v in self.idx_to_class.items()
            }

        else:

            with open(
                class_mapping_path,
                "r",
                encoding="utf-8"
            ) as f:

                mapping = json.load(f)

            if (
                isinstance(mapping, dict)
                and
                "idx_to_class" in mapping
            ):

                self.idx_to_class = mapping[
                    "idx_to_class"
                ]

                self.class_to_idx = mapping.get(
                    "class_to_idx"
                ) or {
                    v: int(k)
                    for k, v in self.idx_to_class.items()
                }

            else:
                # Flat mapping produced by extract_classes.py:
                # {"1": "BatFish", ...}
                self.idx_to_class = {
                    str(k): v for k, v in mapping.items()
                }
                self.class_to_idx = {
                    v: int(k) for k, v in self.idx_to_class.items()
                }

            print(
                f"[OK] Loaded "
                f"{len(self.idx_to_class)} class names"
            )

        # ----------------------------------------------------
        # Classification preprocessing
        # ----------------------------------------------------

        self.cls_transform = transforms.Compose([

            transforms.ToPILImage(),

            transforms.Resize(
                (224, 224)
            ),

            transforms.ToTensor(),

            transforms.Normalize(
                mean=[
                    0.485,
                    0.456,
                    0.406
                ],

                std=[
                    0.229,
                    0.224,
                    0.225
                ]
            )
        ])

        print(
            "\n[OK] Pipeline ready"
        )

        print("=" * 70)

    # ========================================================
    # PREPROCESS IMAGE FOR RESUNET
    # ========================================================

    def preprocess_image(
        self,
        image
    ):

        image_rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        image_resized = cv2.resize(
            image_rgb,
            (352, 352)
        )

        image_float = (
            image_resized.astype(
                np.float32
            ) / 255.0
        )

        image_tensor = torch.from_numpy(
            image_float
        )

        image_tensor = image_tensor.permute(
            2,
            0,
            1
        )

        mean = torch.tensor(
            [0.485, 0.456, 0.406],
            dtype=torch.float32
        ).view(
            3,
            1,
            1
        )

        std = torch.tensor(
            [0.229, 0.224, 0.225],
            dtype=torch.float32
        ).view(
            3,
            1,
            1
        )

        image_tensor = (
            image_tensor - mean
        ) / std

        image_tensor = image_tensor.unsqueeze(
            0
        )

        return image_tensor.to(
            self.device
        )

    # ========================================================
    # SEGMENTATION
    # ========================================================

    def get_probability_map(self, image_tensor):
        """Soft camouflage probability map in [0, 1] at network resolution."""
        with torch.no_grad():
            output = self.seg_model(image_tensor)
            probability = torch.sigmoid(output)
            return probability.squeeze().cpu().numpy()

    def adaptive_threshold(self, soft_mask, threshold=0.5):
        """
        Keep the locked 0.5 default when it fires.
        On hard COD images where 0.5 is empty but the network still has signal,
        fall back to a lower adaptive cut so faint animals are not discarded.
        """
        soft = np.asarray(soft_mask, dtype=np.float32)
        binary = (soft >= float(threshold)).astype(np.uint8)
        if binary.any():
            return binary, float(threshold)

        peak = float(soft.max()) if soft.size else 0.0
        if peak < 0.12:
            return binary, float(threshold)

        # Prefer a stable fraction of the peak, floored for stability.
        adaptive = max(0.18, min(0.45, peak * 0.45))
        binary = (soft >= adaptive).astype(np.uint8)
        if not binary.any():
            # Last resort: keep the strongest 2% of pixels.
            flat = soft.reshape(-1)
            k = max(1, int(0.02 * flat.size))
            cutoff = float(np.partition(flat, -k)[-k])
            binary = (soft >= cutoff).astype(np.uint8)
            adaptive = cutoff
        return binary, float(adaptive)

    def refine_binary_mask(self, mask):
        """Remove speckles; keep the dominant connected camouflage region."""
        binary = (np.asarray(mask) > 0).astype(np.uint8)
        if not binary.any():
            return binary

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)
        if not cleaned.any():
            cleaned = binary

        num, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
        if num <= 1:
            return cleaned

        areas = stats[1:, cv2.CC_STAT_AREA]
        keep = 1 + int(np.argmax(areas))
        # Keep secondary blobs only if they are large relative to the main one.
        main_area = float(areas.max())
        refined = np.zeros_like(cleaned)
        for idx in range(1, num):
            if float(stats[idx, cv2.CC_STAT_AREA]) >= max(64.0, 0.08 * main_area):
                refined[labels == idx] = 1
        if not refined.any():
            refined[labels == keep] = 1
        return refined

    def get_mask(
        self,
        image_tensor,
        threshold=0.5
    ):

        soft = self.get_probability_map(image_tensor)
        binary_mask, _ = self.adaptive_threshold(soft, threshold=threshold)
        return binary_mask

    # ========================================================
    # RESIZE MASK
    # ========================================================

    def resize_mask(
        self,
        mask,
        image
    ):

        return cv2.resize(
            mask,
            (
                image.shape[1],
                image.shape[0]
            ),
            interpolation=cv2.INTER_NEAREST
        )

    # ========================================================
    # BOUNDARY
    # ========================================================

    def draw_boundary(
        self,
        image,
        mask,
        thickness=3
    ):

        result = image.copy()

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:

            if cv2.contourArea(
                contour
            ) > 20:

                cv2.drawContours(
                    result,
                    [contour],
                    -1,
                    (0, 0, 255),
                    thickness
                )

        return result

    # ========================================================
    # COLORED OVERLAY
    # ========================================================

    def create_overlay(
        self,
        image,
        mask,
        alpha=0.35
    ):

        colored_mask = np.zeros_like(
            image
        )

        colored_mask[
            mask > 0
        ] = (
            0,
            0,
            255
        )

        result = cv2.addWeighted(
            image,
            1 - alpha,
            colored_mask,
            alpha,
            0
        )

        return result

    # ========================================================
    # CROP OBJECT
    # ========================================================

    def crop_object(
        self,
        image,
        mask,
        padding=20
    ):

        coordinates = np.where(
            mask > 0
        )

        if len(
            coordinates[0]
        ) == 0:

            return None

        y_min = coordinates[0].min()
        y_max = coordinates[0].max()

        x_min = coordinates[1].min()
        x_max = coordinates[1].max()

        height, width = image.shape[:2]

        y_min = max(
            0,
            y_min - padding
        )

        y_max = min(
            height,
            y_max + padding + 1
        )

        x_min = max(
            0,
            x_min - padding
        )

        x_max = min(
            width,
            x_max + padding + 1
        )

        crop = image[
            y_min:y_max,
            x_min:x_max
        ]

        if crop.size == 0:

            return None

        return crop

    # ========================================================
    # CLASSIFICATION
    # ========================================================

    def _get_clip_classifier(self):
        """Lazy shared OpenCLIP recognizer (one copy across pipelines)."""
        if getattr(self, "_clip_classifier", None) is not None:
            return self._clip_classifier
        if not hasattr(CamouflageBreakerPipeline, "_CLIP_SHARED"):
            CamouflageBreakerPipeline._CLIP_SHARED = None
        if CamouflageBreakerPipeline._CLIP_SHARED is None:
            from models.clip_zero_shot import ClipZeroShotClassifier

            class_names = [
                self.idx_to_class[str(i)]
                for i in range(len(self.idx_to_class))
            ]
            print("\nLoading OpenCLIP animal recognizer (ViT-L/14)…")
            CamouflageBreakerPipeline._CLIP_SHARED = ClipZeroShotClassifier(
                class_names=class_names,
                device=self.device,
            )
            print("[OK] OpenCLIP recognizer ready")
        self._clip_classifier = CamouflageBreakerPipeline._CLIP_SHARED
        return self._clip_classifier

    def classify_object(
        self,
        crop,
        full_image=None,
        use_clip_refinement=True,
    ):

        empty = {
            "class_name": "No object detected",
            "confidence": 0.0,
            "predicted_index": None,
            "top_k": [],
            "resnet_class_name": "No object detected",
            "resnet_confidence": 0.0,
            "clip_class_name": None,
            "clip_confidence": 0.0,
            "source": None,
        }

        if crop is None:
            return empty

        if not getattr(self, "classifier_available", False) or self.classifier is None:
            empty["class_name"] = "Classifier checkpoint missing"
            empty["resnet_class_name"] = "Classifier checkpoint missing"
            return empty

        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        image_tensor = self.cls_transform(crop_rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.classifier(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            resnet_conf, predicted_index = torch.max(probabilities, dim=0)
            top_vals, top_idxs = torch.topk(
                probabilities, k=min(5, probabilities.numel())
            )

        predicted_index = int(predicted_index.item())
        resnet_confidence = float(resnet_conf.item()) * 100.0
        resnet_class = self.idx_to_class.get(
            str(predicted_index),
            f"Class_{predicted_index + 1}",
        )
        resnet_top = [
            {
                "class_name": self.idx_to_class.get(str(int(i)), f"Class_{int(i)+1}"),
                "confidence": float(v) * 100.0,
                "index": int(i),
            }
            for v, i in zip(top_vals, top_idxs)
        ]

        result = {
            "class_name": resnet_class,
            "confidence": resnet_confidence,
            "predicted_index": predicted_index,
            "top_k": resnet_top,
            "resnet_class_name": resnet_class,
            "resnet_confidence": resnet_confidence,
            "clip_class_name": None,
            "clip_confidence": 0.0,
            "source": "resnet50",
        }

        if not use_clip_refinement:
            return result

        try:
            clip = self._get_clip_classifier()
            clip_out = clip.predict(crop, full_bgr=full_image, top_k=5)
        except Exception as exc:
            print(f"WARNING: OpenCLIP refinement skipped ({exc})")
            return result

        result["clip_class_name"] = clip_out["class_name"]
        result["clip_confidence"] = float(clip_out["confidence"])

        # Also score the full frame; oversized COD masks can dilute the crop.
        if full_image is not None:
            try:
                clip_full = clip.predict(full_image, full_bgr=None, top_k=5)
                if float(clip_full["confidence"]) > float(clip_out["confidence"]):
                    clip_out = clip_full
                    result["clip_class_name"] = clip_out["class_name"]
                    result["clip_confidence"] = float(clip_out["confidence"])
            except Exception:
                pass

        clip_name = clip_out["class_name"]
        clip_conf = float(clip_out["confidence"])

        insect_like = {
            "Ant", "Bee", "Beetle", "Bug", "Butterfly", "Caterpillar",
            "Centipede", "Cicada", "Dragonfly", "Grasshopper", "Katydid",
            "Mantis", "Moth", "Owlfly", "Spider", "StickInsect", "Worm",
        }
        vertebrate_like = {
            "Bat", "Bird", "Bittern", "Cat", "Chameleon", "Cheetah", "Crocodile",
            "Deer", "Dog", "Duck", "Frog", "Frogmouth", "Gecko", "Giraffe",
            "Grouse", "Heron", "Human", "Kangaroo", "Leopard", "Lion", "Lizard",
            "Mockingbird", "Monkey", "Owl", "Rabbit", "Reccoon", "Sciuridae",
            "Sheep", "Snake", "Tiger", "Toad", "Turtle", "Wolf",
        }

        prefer_clip = False
        if clip_name != resnet_class:
            if clip_conf >= 70.0:
                prefer_clip = True
            elif clip_conf >= resnet_confidence:
                prefer_clip = True
            elif (
                resnet_class in insect_like
                and clip_name in vertebrate_like
                and clip_conf >= 28.0
            ):
                # Hard camouflage photos often fool ResNet into insect labels.
                prefer_clip = True
        elif clip_conf >= resnet_confidence:
            prefer_clip = True

        if prefer_clip:
            result["class_name"] = clip_name
            result["confidence"] = clip_conf
            result["predicted_index"] = clip_out["predicted_index"]
            result["top_k"] = clip_out["top_k"]
            result["source"] = "openclip"
        return result

    # ========================================================
    # COMPLETE PREDICTION
    # ========================================================

    def predict(
        self,
        image,
        threshold=0.5
    ):

        # ----------------------------------------------------
        # Load image
        # ----------------------------------------------------

        if isinstance(
            image,
            str
        ):

            image_path = image

            image = cv2.imread(
                image_path
            )

            if image is None:

                raise ValueError(
                    f"Could not load image:\n"
                    f"{image_path}"
                )

        # ----------------------------------------------------
        # Original
        # ----------------------------------------------------

        original = image.copy()

        # ----------------------------------------------------
        # Segmentation (ResUNet / SINet-V2 / ESCNet)
        # ----------------------------------------------------

        if self.segmenter == "escnet":
            # ESCNet wrapper returns full-resolution uint8 {0,255}
            mask = self.seg_model.predict(
                image,
                threshold=threshold,
                assume_bgr=True,
            )
            if mask.max() <= 1:
                mask = (mask * 255).astype(np.uint8)
            else:
                mask = (mask > 0).astype(np.uint8) * 255
            mask = (self.refine_binary_mask(mask) * 255).astype(np.uint8)
        else:
            image_tensor = self.preprocess_image(
                image
            )

            soft_small = self.get_probability_map(image_tensor)
            soft_full = cv2.resize(
                soft_small.astype(np.float32),
                (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
            binary, used_thr = self.adaptive_threshold(
                soft_full, threshold=threshold
            )
            binary = self.refine_binary_mask(binary)
            mask = (binary * 255).astype(np.uint8)
            self._last_threshold_used = used_thr

        # ----------------------------------------------------
        # Check detection
        # ----------------------------------------------------

        object_pixels = np.sum(
            mask > 0
        )

        if object_pixels == 0:

            return {

                "original":
                    original,

                "mask":
                    mask,

                "boundary":
                    original.copy(),

                "overlay":
                    original.copy(),

                "crop":
                    None,

                "class_name":
                    "No object detected",

                "confidence":
                    0.0,

                "predicted_index":
                    None,

                "object_detected":
                    False,

                "top_k": [],
                "resnet_class_name": "No object detected",
                "resnet_confidence": 0.0,
                "clip_class_name": None,
                "clip_confidence": 0.0,
                "classifier_source": None,
            }

        # ----------------------------------------------------
        # Boundary
        # ----------------------------------------------------

        boundary = self.draw_boundary(
            original,
            mask
        )

        # ----------------------------------------------------
        # Overlay
        # ----------------------------------------------------

        overlay = self.create_overlay(
            original,
            mask
        )

        # ----------------------------------------------------
        # Crop
        # ----------------------------------------------------

        crop = self.crop_object(
            original,
            mask
        )

        # ----------------------------------------------------
        # ResNet50 + OpenCLIP refinement
        # ----------------------------------------------------

        cls = self.classify_object(
            crop,
            full_image=original,
            use_clip_refinement=True,
        )

        return {

            "original":
                original,

            "mask":
                mask,

            "boundary":
                boundary,

            "overlay":
                overlay,

            "crop":
                crop,

            "class_name":
                cls["class_name"],

            "confidence":
                cls["confidence"],

            "predicted_index":
                cls["predicted_index"],

            "object_detected":
                True,

            "top_k": cls.get("top_k", []),
            "resnet_class_name": cls.get("resnet_class_name"),
            "resnet_confidence": cls.get("resnet_confidence", 0.0),
            "clip_class_name": cls.get("clip_class_name"),
            "clip_confidence": cls.get("clip_confidence", 0.0),
            "classifier_source": cls.get("source"),
        }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 70)
    print("TESTING CAMOUFLAGE BREAKER")
    print("=" * 70)

    pipeline = CamouflageBreakerPipeline()

    test_dir = os.path.join(
        PROJECT_ROOT,
        "dataset",
        "Test",
        "Image"
    )

    if not os.path.exists(
        test_dir
    ):

        print(
            f"\nTest folder not found:\n"
            f"{test_dir}"
        )

        sys.exit()

    test_images = [

        os.path.join(
            test_dir,
            filename
        )

        for filename in sorted(
            os.listdir(test_dir)
        )

        if filename.lower().endswith(
            (
                ".jpg",
                ".jpeg",
                ".png"
            )
        )
    ]

    if len(test_images) == 0:

        print(
            "\nNo test images found."
        )

        sys.exit()

    print(
        f"\nFound {len(test_images)} test images."
    )

    print(
        "Testing first 5 images...\n"
    )

    output_dir = os.path.join(
        PROJECT_ROOT,
        "outputs"
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    for image_path in test_images[:5]:

        print("-" * 70)

        filename = os.path.basename(
            image_path
        )

        print(
            f"Image: {filename}"
        )

        try:

            result = pipeline.predict(
                image_path
            )

            print(
                f"Object detected: "
                f"{result['object_detected']}"
            )

            print(
                f"Prediction: "
                f"{result['class_name']}"
            )

            print(
                f"Confidence: "
                f"{result['confidence']:.2f}%"
            )

            base_name = os.path.splitext(
                filename
            )[0]

            cv2.imwrite(
                os.path.join(
                    output_dir,
                    f"{base_name}_boundary.jpg"
                ),
                result["boundary"]
            )

            cv2.imwrite(
                os.path.join(
                    output_dir,
                    f"{base_name}_overlay.jpg"
                ),
                result["overlay"]
            )

            if result["crop"] is not None:

                cv2.imwrite(
                    os.path.join(
                        output_dir,
                        f"{base_name}_crop.jpg"
                    ),
                    result["crop"]
                )

            print(
                "[OK] Results saved"
            )

        except Exception as e:

            print(
                f"❌ Error: {e}"
            )

    print("\n" + "=" * 70)
    print("PIPELINE TEST COMPLETE")
    print("=" * 70)

    print(
        f"Results folder:\n"
        f"{output_dir}"
    )