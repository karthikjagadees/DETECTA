import os
import sys
import json
import cv2
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
from tqdm import tqdm

# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from models.resunet import ResUNet


# ============================================================
# CONFIGURATION
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATASET_ROOT = os.path.join(PROJECT_ROOT, "dataset")

RESUNET_PATH = os.path.join(
    PROJECT_ROOT,
    "saved_models",
    "resunet_best.pth"
)

CLASSIFIER_PATH = os.path.join(
    PROJECT_ROOT,
    "saved_models",
    "classifier_best.pth"
)

CLASS_MAPPING_PATH = os.path.join(
    PROJECT_ROOT,
    "saved_models",
    "class_mapping.json"
)

IMAGE_SIZE_SEG = 352
IMAGE_SIZE_CLS = 224

BATCH_SIZE = 8


# ============================================================
# RESUNET EVALUATION DATASET
# ============================================================

class SegmentationDataset(Dataset):

    def __init__(self, root_dir, split="Test", image_size=352):

        self.image_dir = os.path.join(
            root_dir, split, "Image"
        )

        self.mask_dir = os.path.join(
            root_dir, split, "GT_Object"
        )

        self.image_size = image_size

        self.image_files = sorted([
            f for f in os.listdir(self.image_dir)
            if f.lower().endswith(
                (".jpg", ".jpeg", ".png", ".bmp")
            )
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):

        filename = self.image_files[idx]

        image_path = os.path.join(
            self.image_dir,
            filename
        )

        base_name = os.path.splitext(filename)[0]

        mask_path = os.path.join(
            self.mask_dir,
            base_name + ".png"
        )

        # -------------------------
        # Load image
        # -------------------------

        image = cv2.imread(image_path)

        if image is None:
            raise ValueError(
                f"Could not read image: {image_path}"
            )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        # -------------------------
        # Load mask
        # -------------------------

        mask = cv2.imread(
            mask_path,
            cv2.IMREAD_GRAYSCALE
        )

        if mask is None:
            raise ValueError(
                f"Could not read mask: {mask_path}"
            )

        # -------------------------
        # Resize
        # -------------------------

        image = cv2.resize(
            image,
            (self.image_size, self.image_size)
        )

        mask = cv2.resize(
            mask,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_NEAREST
        )

        # -------------------------
        # Normalize image
        # -------------------------

        image = image.astype(
            np.float32
        ) / 255.0

        mean = np.array(
            [0.485, 0.456, 0.406],
            dtype=np.float32
        )

        std = np.array(
            [0.229, 0.224, 0.225],
            dtype=np.float32
        )

        image = (image - mean) / std

        image = torch.from_numpy(
            image
        ).permute(2, 0, 1)

        # -------------------------
        # Binary mask
        # -------------------------

        mask = (mask > 0).astype(
            np.float32
        )

        mask = torch.from_numpy(
            mask
        ).unsqueeze(0)

        return image, mask


# ============================================================
# LOAD RESUNET
# ============================================================

def load_resunet():

    print("\nLoading ResUNet...")

    model = ResUNet(
        encoder_name="resnet50",
        encoder_weights=None
    )

    checkpoint = torch.load(
        RESUNET_PATH,
        map_location=DEVICE
    )

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):
        state_dict = checkpoint[
            "model_state_dict"
        ]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)

    model = model.to(DEVICE)
    model.eval()

    print("✓ ResUNet loaded")

    return model


# ============================================================
# SEGMENTATION METRICS
# ============================================================

def calculate_iou(pred, target):

    pred = pred.astype(bool)
    target = target.astype(bool)

    intersection = np.logical_and(
        pred,
        target
    ).sum()

    union = np.logical_or(
        pred,
        target
    ).sum()

    if union == 0:

        if target.sum() == 0:
            return 1.0

        return 0.0

    return intersection / union


def calculate_dice(pred, target):

    pred = pred.astype(bool)
    target = target.astype(bool)

    intersection = np.logical_and(
        pred,
        target
    ).sum()

    total = pred.sum() + target.sum()

    if total == 0:

        if target.sum() == 0:
            return 1.0

        return 0.0

    return (2.0 * intersection) / total


# ============================================================
# EVALUATE RESUNET
# ============================================================

def evaluate_segmentation():

    print("\n" + "=" * 60)
    print("RESUNET SEGMENTATION EVALUATION")
    print("=" * 60)

    dataset = SegmentationDataset(
        DATASET_ROOT,
        split="Test",
        image_size=IMAGE_SIZE_SEG
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    print(
        f"Test images: {len(dataset)}"
    )

    model = load_resunet()

    total_iou = 0.0
    total_dice = 0.0

    count = 0

    with torch.no_grad():

        for images, masks in tqdm(
            loader,
            desc="Evaluating segmentation"
        ):

            images = images.to(DEVICE)

            outputs = model(images)

            probabilities = torch.sigmoid(
                outputs
            )

            predictions = (
                probabilities > 0.5
            ).cpu().numpy()

            targets = (
                masks > 0.5
            ).cpu().numpy()

            batch_size = images.size(0)

            for i in range(batch_size):

                pred = predictions[i, 0]
                target = targets[i, 0]

                iou = calculate_iou(
                    pred,
                    target
                )

                dice = calculate_dice(
                    pred,
                    target
                )

                total_iou += iou
                total_dice += dice

                count += 1

    mean_iou = total_iou / count
    mean_dice = total_dice / count

    print("\nSegmentation Results")
    print("-" * 40)
    print(
        f"IoU Score  : {mean_iou:.4f}"
    )
    print(
        f"Dice Score : {mean_dice:.4f}"
    )

    return mean_iou, mean_dice


# ============================================================
# CLASSIFICATION DATASET
# ============================================================

class ClassificationDataset(Dataset):

    def __init__(
        self,
        root_dir,
        split="Test"
    ):

        self.image_dir = os.path.join(
            root_dir,
            split,
            "Image"
        )

        self.txt_path = os.path.join(
            root_dir,
            split,
            f"CAM-NonCAM_Instance_{split}.txt"
        )

        self.samples = []

        self.label_names = set()

        self._parse_labels()

        self.transform = transforms.Compose([
            transforms.Resize(
                (IMAGE_SIZE_CLS, IMAGE_SIZE_CLS)
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

    def _parse_labels(self):

        if not os.path.exists(self.txt_path):

            raise FileNotFoundError(
                f"Label file not found:\n"
                f"{self.txt_path}"
            )

        with open(
            self.txt_path,
            "r",
            encoding="utf-8"
        ) as f:

            lines = f.readlines()

        label_cache = {}

        i = 0

        while i < len(lines):

            line = lines[i].strip()

            if line.startswith("[INFO]"):

                parts = line.split()

                if len(parts) >= 3:

                    filename = parts[1]

                    cam_flag = int(parts[2])

                    base_name = os.path.splitext(
                        filename
                    )[0]

                    if i + 1 < len(lines):

                        class_line = lines[
                            i + 1
                        ].strip()

                        class_parts = (
                            class_line.split()
                        )

                        if len(class_parts) >= 1:

                            class_path = (
                                class_parts[0]
                            )

                            path_parts = (
                                class_path.split("/")
                            )

                            if len(path_parts) >= 3:

                                category = (
                                    path_parts[2]
                                )

                            elif len(path_parts) == 2:

                                category = (
                                    path_parts[1]
                                )

                            else:

                                category = "unknown"

                            label_cache[
                                base_name
                            ] = {
                                "category": category,
                                "cam_flag": cam_flag
                            }

                            self.label_names.add(
                                category
                            )

                            i += 2

                            continue

            i += 1

        # -------------------------
        # Build samples
        # -------------------------

        for filename in os.listdir(
            self.image_dir
        ):

            if not filename.lower().endswith(
                (".jpg", ".jpeg", ".png", ".bmp")
            ):
                continue

            base_name = os.path.splitext(
                filename
            )[0]

            if base_name not in label_cache:
                continue

            info = label_cache[
                base_name
            ]

            # Only CAM objects
            if info["cam_flag"] != 1:
                continue

            self.samples.append(
                (
                    filename,
                    info["category"]
                )
            )

        self.samples.sort()

        self.classes = sorted(
            self.label_names
        )

        self.class_to_idx = {
            name: idx
            for idx, name in enumerate(
                self.classes
            )
        }

        print(
            f"[INFO] Classification samples: "
            f"{len(self.samples)}"
        )

        print(
            f"[INFO] Classes found: "
            f"{len(self.classes)}"
        )

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, idx):

        filename, category = (
            self.samples[idx]
        )

        path = os.path.join(
            self.image_dir,
            filename
        )

        image = Image.open(
            path
        ).convert("RGB")

        image = self.transform(
            image
        )

        label = self.class_to_idx[
            category
        ]

        return image, label


# ============================================================
# LOAD CLASSIFIER
# ============================================================

def load_classifier():

    print(
        "\nLoading trained ResNet50 classifier..."
    )

    checkpoint = torch.load(
        CLASSIFIER_PATH,
        map_location=DEVICE
    )

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):

        state_dict = checkpoint[
            "model_state_dict"
        ]

        num_classes = checkpoint.get(
            "num_classes",
            69
        )

    else:

        state_dict = checkpoint
        num_classes = 69

    class ResNet50Classifier(
        torch.nn.Module
    ):

        def __init__(
            self,
            num_classes
        ):

            super().__init__()

            self.backbone = (
                models.resnet50(
                    weights=None
                )
            )

            in_features = (
                self.backbone.fc.in_features
            )

            self.backbone.fc = (
                torch.nn.Sequential(
                    torch.nn.Dropout(0.5),
                    torch.nn.Linear(
                        in_features,
                        512
                    ),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(0.3),
                    torch.nn.Linear(
                        512,
                        num_classes
                    )
                )
            )

        def forward(self, x):

            return self.backbone(x)

    model = ResNet50Classifier(
        num_classes
    )

    model.load_state_dict(
        state_dict
    )

    model = model.to(DEVICE)
    model.eval()

    print(
        f"✓ ResNet50 loaded "
        f"({num_classes} classes)"
    )

    return model


# ============================================================
# EVALUATE CLASSIFIER
# ============================================================

def evaluate_classification():

    print("\n" + "=" * 60)
    print("RESNET50 CLASSIFICATION EVALUATION")
    print("=" * 60)

    dataset = ClassificationDataset(
        DATASET_ROOT,
        split="Test"
    )

    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,
        num_workers=0
    )

    model = load_classifier()

    y_true = []
    y_pred = []

    with torch.no_grad():

        for images, labels in tqdm(
            loader,
            desc="Evaluating classification"
        ):

            images = images.to(DEVICE)

            outputs = model(
                images
            )

            predictions = torch.argmax(
                outputs,
                dim=1
            ).cpu().numpy()

            labels = labels.numpy()

            y_true.extend(
                labels.tolist()
            )

            y_pred.extend(
                predictions.tolist()
            )

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    print("\nClassification Results")
    print("-" * 40)

    print(
        f"Accuracy  : {accuracy:.4f}"
    )

    print(
        f"Precision : {precision:.4f}"
    )

    print(
        f"Recall    : {recall:.4f}"
    )

    print(
        f"F1 Score  : {f1:.4f}"
    )

    # -------------------------
    # Save confusion matrix
    # -------------------------

    output_dir = os.path.join(
        PROJECT_ROOT,
        "outputs"
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    np.save(
        os.path.join(
            output_dir,
            "confusion_matrix.npy"
        ),
        cm
    )

    # -------------------------
    # Save metrics
    # -------------------------

    metrics = {
        "accuracy": float(
            accuracy
        ),
        "precision": float(
            precision
        ),
        "recall": float(
            recall
        ),
        "f1_score": float(
            f1
        ),
        "number_of_test_samples": len(
            y_true
        ),
        "number_of_classes": len(
            dataset.classes
        )
    }

    with open(
        os.path.join(
            output_dir,
            "classification_metrics.json"
        ),
        "w"
    ) as f:

        json.dump(
            metrics,
            f,
            indent=4
        )

    return (
        accuracy,
        precision,
        recall,
        f1
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("CAMOUFLAGE BREAKER - MODEL EVALUATION")
    print("=" * 70)

    print(
        f"Using device: {DEVICE}"
    )

    # -------------------------
    # Segmentation
    # -------------------------

    seg_results = evaluate_segmentation()

    # -------------------------
    # Classification
    # -------------------------

    cls_results = evaluate_classification()

    # -------------------------
    # Final summary
    # -------------------------

    print("\n" + "=" * 70)
    print("FINAL MODEL EVALUATION")
    print("=" * 70)

    print(
        f"\nResUNet IoU       : "
        f"{seg_results[0]:.4f}"
    )

    print(
        f"ResUNet Dice      : "
        f"{seg_results[1]:.4f}"
    )

    print(
        f"\nResNet50 Accuracy : "
        f"{cls_results[0]:.4f}"
    )

    print(
        f"ResNet50 Precision: "
        f"{cls_results[1]:.4f}"
    )

    print(
        f"ResNet50 Recall   : "
        f"{cls_results[2]:.4f}"
    )

    print(
        f"ResNet50 F1 Score : "
        f"{cls_results[3]:.4f}"
    )

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)

    print(
        "\nSaved files:"
    )

    print(
        "outputs/confusion_matrix.npy"
    )

    print(
        "outputs/classification_metrics.json"
    )