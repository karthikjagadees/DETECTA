# ============================================================
# CAMOUFLAGE BREAKER
# COMPLETE PIPELINE EVALUATION
# ============================================================

import os
import sys
import json
import re
import csv

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from tqdm import tqdm


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# IMPORT RESUNET
# ============================================================

from models.resunet import ResUNet


# ============================================================
# PATHS
# ============================================================

TEST_IMAGE_DIR = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "Test",
    "Image"
)

TEST_MASK_DIR = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "Test",
    "GT_Object"
)

TEST_TXT = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "Test",
    "CAM-NonCAM_Instance_Test.txt"
)

TRAIN_TXT = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "Train",
    "CAM-NonCAM_Instance_Train.txt"
)

SEGMENTATION_MODEL = os.path.join(
    PROJECT_ROOT,
    "saved_models",
    "resunet_best.pth"
)

CLASSIFIER_MODEL = os.path.join(
    PROJECT_ROOT,
    "saved_models",
    "classifier_best.pth"
)

CLASS_MAPPING = os.path.join(
    PROJECT_ROOT,
    "saved_models",
    "class_mapping.json"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "pipeline_evaluation"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# RESNET50 CLASSIFIER
# ============================================================

class ResNet50Classifier(nn.Module):

    def __init__(self, num_classes=69):

        super().__init__()

        self.backbone = models.resnet50(
            weights=None
        )

        in_features = (
            self.backbone.fc.in_features
        )

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


# ============================================================
# NORMALIZE CLASS NAME
# ============================================================

def normalize_class_name(name):

    return re.sub(
        r"[^a-z0-9]",
        "",
        str(name).lower()
    )


# ============================================================
# READ COD10K TXT
# ============================================================

def read_annotation_file(txt_path):

    records = []

    current_record = None

    with open(
        txt_path,
        "r",
        encoding="utf-8"
    ) as file:

        for raw_line in file:

            line = raw_line.strip()

            if not line:
                continue

            # ------------------------------------------------
            # INFO LINE
            # ------------------------------------------------

            if line.startswith("[INFO]"):

                parts = line.split()

                if len(parts) < 3:
                    continue

                filename = parts[1]

                try:
                    cam_flag = int(parts[2])
                except ValueError:
                    continue

                current_record = {

                    "filename": filename,

                    "stem": os.path.splitext(
                        filename
                    )[0],

                    "cam_flag": cam_flag,

                    "classes": []
                }

                records.append(
                    current_record
                )

                continue

            # ------------------------------------------------
            # ANNOTATION LINE
            # ------------------------------------------------

            if current_record is not None:

                parts = line.split()

                if len(parts) >= 1:

                    annotation_path = parts[0]

                    if "/" in annotation_path:

                        class_name = (
                            annotation_path
                            .split("/")[-1]
                        )

                        if class_name not in current_record["classes"]:

                            current_record["classes"].append(
                                class_name
                            )

    return records


# ============================================================
# GET IMAGE FILES
# ============================================================

def build_file_lookup(folder):

    lookup = {}

    valid_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp"
    }

    for filename in os.listdir(folder):

        path = os.path.join(
            folder,
            filename
        )

        if not os.path.isfile(path):
            continue

        extension = os.path.splitext(
            filename
        )[1].lower()

        if extension not in valid_extensions:
            continue

        stem = os.path.splitext(
            filename
        )[0]

        lookup[stem.lower()] = path

    return lookup


# ============================================================
# LOAD CLASS MAPPING
# ============================================================

def load_class_mapping():

    print(
        "\nLoading class mapping..."
    )

    # --------------------------------------------------------
    # First preference:
    # checkpoint class mapping
    # --------------------------------------------------------

    checkpoint = torch.load(
        CLASSIFIER_MODEL,
        map_location="cpu"
    )

    checkpoint_mapping = None

    if isinstance(
        checkpoint,
        dict
    ):

        if (
            "class_to_idx"
            in checkpoint
        ):

            checkpoint_mapping = (
                checkpoint[
                    "class_to_idx"
                ]
            )

    if checkpoint_mapping is not None:

        class_to_idx = {}

        for name, index in checkpoint_mapping.items():

            class_to_idx[
                str(name)
            ] = int(index)

        print(
            "✓ Class mapping loaded from classifier checkpoint"
        )

        return class_to_idx

    # --------------------------------------------------------
    # Second preference:
    # class_mapping.json
    # --------------------------------------------------------

    if os.path.exists(
        CLASS_MAPPING
    ):

        with open(
            CLASS_MAPPING,
            "r",
            encoding="utf-8"
        ) as file:

            mapping = json.load(file)

        class_to_idx = mapping[
            "class_to_idx"
        ]

        class_to_idx = {
            str(k): int(v)
            for k, v in class_to_idx.items()
        }

        print(
            "✓ Class mapping loaded from class_mapping.json"
        )

        return class_to_idx

    # --------------------------------------------------------
    # Final fallback:
    # reconstruct from TRAIN TXT
    # --------------------------------------------------------

    print(
        "Class mapping not found in checkpoint."
    )

    print(
        "Reconstructing from training TXT..."
    )

    train_records = read_annotation_file(
        TRAIN_TXT
    )

    class_names = set()

    for record in train_records:

        if record["cam_flag"] > 0:

            for class_name in record["classes"]:

                class_names.add(
                    class_name
                )

    class_names = sorted(
        class_names
    )

    class_to_idx = {

        name: index

        for index, name
        in enumerate(class_names)
    }

    print(
        f"✓ Reconstructed {len(class_to_idx)} classes"
    )

    return class_to_idx


# ============================================================
# BUILD CLASS ALIASES
# ============================================================

def build_class_aliases(
    class_to_idx
):

    aliases = {}

    for class_name, index in class_to_idx.items():

        normalized = normalize_class_name(
            class_name
        )

        aliases[
            normalized
        ] = index

    return aliases


# ============================================================
# FIND CLASS INDEX
# ============================================================

def get_class_index(
    class_name,
    class_to_idx,
    aliases
):

    # Exact match
    if class_name in class_to_idx:

        return class_to_idx[
            class_name
        ]

    # Normalized match
    normalized = normalize_class_name(
        class_name
    )

    if normalized in aliases:

        return aliases[
            normalized
        ]

    return None


# ============================================================
# LOAD SEGMENTATION MODEL
# ============================================================

def load_segmentation_model():

    print(
        "\nLoading ResUNet..."
    )

    model = ResUNet(
        encoder_name="resnet50",
        encoder_weights=None
    )

    checkpoint = torch.load(
        SEGMENTATION_MODEL,
        map_location=DEVICE
    )

    if (
        isinstance(
            checkpoint,
            dict
        )
        and
        "model_state_dict"
        in checkpoint
    ):

        state_dict = checkpoint[
            "model_state_dict"
        ]

    else:

        state_dict = checkpoint

    model.load_state_dict(
        state_dict
    )

    model.to(
        DEVICE
    )

    model.eval()

    print(
        "✓ ResUNet loaded"
    )

    return model


# ============================================================
# LOAD CLASSIFIER
# ============================================================

def load_classifier():

    print(
        "\nLoading ResNet50 classifier..."
    )

    checkpoint = torch.load(
        CLASSIFIER_MODEL,
        map_location=DEVICE
    )

    if (
        isinstance(
            checkpoint,
            dict
        )
        and
        "num_classes"
        in checkpoint
    ):

        num_classes = int(
            checkpoint[
                "num_classes"
            ]
        )

    else:

        num_classes = 69

    model = ResNet50Classifier(
        num_classes=num_classes
    )

    if (
        isinstance(
            checkpoint,
            dict
        )
        and
        "model_state_dict"
        in checkpoint
    ):

        state_dict = checkpoint[
            "model_state_dict"
        ]

    else:

        state_dict = checkpoint

    # --------------------------------------------------------
    # Handle possible DataParallel prefix
    # --------------------------------------------------------

    cleaned_state_dict = {}

    for key, value in state_dict.items():

        if key.startswith(
            "module."
        ):

            key = key[
                len("module.") :
            ]

        cleaned_state_dict[
            key
        ] = value

    model.load_state_dict(
        cleaned_state_dict
    )

    model.to(
        DEVICE
    )

    model.eval()

    print(
        f"✓ ResNet50 loaded ({num_classes} classes)"
    )

    return model


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_for_segmentation(
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
        )
        / 255.0
    )

    tensor = torch.from_numpy(
        image_float
    )

    tensor = tensor.permute(
        2,
        0,
        1
    )

    mean = torch.tensor(
        [
            0.485,
            0.456,
            0.406
        ],
        dtype=torch.float32
    ).view(
        3,
        1,
        1
    )

    std = torch.tensor(
        [
            0.229,
            0.224,
            0.225
        ],
        dtype=torch.float32
    ).view(
        3,
        1,
        1
    )

    tensor = (
        tensor - mean
    ) / std

    tensor = tensor.unsqueeze(
        0
    )

    return tensor.to(
        DEVICE
    )


# ============================================================
# PREDICT MASK
# ============================================================

def predict_mask(
    model,
    image,
    threshold=0.5
):

    tensor = preprocess_for_segmentation(
        image
    )

    with torch.no_grad():

        output = model(
            tensor
        )

        probability = torch.sigmoid(
            output
        )

    mask = (
        probability
        .squeeze()
        .cpu()
        .numpy()
    )

    mask = (
        mask >= threshold
    ).astype(
        np.uint8
    )

    mask = cv2.resize(
        mask,
        (
            image.shape[1],
            image.shape[0]
        ),
        interpolation=cv2.INTER_NEAREST
    )

    return mask


# ============================================================
# PREPARE GT MASK
# ============================================================

def load_ground_truth_mask(
    mask_path,
    image
):

    mask = cv2.imread(
        mask_path,
        cv2.IMREAD_GRAYSCALE
    )

    if mask is None:
        return None

    mask = cv2.resize(
        mask,
        (
            image.shape[1],
            image.shape[0]
        ),
        interpolation=cv2.INTER_NEAREST
    )

    mask = (
        mask > 127
    ).astype(
        np.uint8
    )

    return mask


# ============================================================
# IOU
# ============================================================

def calculate_iou(
    predicted,
    ground_truth
):

    predicted = predicted > 0
    ground_truth = ground_truth > 0

    intersection = np.logical_and(
        predicted,
        ground_truth
    ).sum()

    union = np.logical_or(
        predicted,
        ground_truth
    ).sum()

    if union == 0:

        return 1.0

    return (
        intersection
        / union
    )


# ============================================================
# DICE
# ============================================================

def calculate_dice(
    predicted,
    ground_truth
):

    predicted = predicted > 0
    ground_truth = ground_truth > 0

    intersection = np.logical_and(
        predicted,
        ground_truth
    ).sum()

    total = (
        predicted.sum()
        +
        ground_truth.sum()
    )

    if total == 0:

        return 1.0

    return (
        2.0
        *
        intersection
        /
        total
    )


# ============================================================
# CROP OBJECT
# ============================================================

def crop_object(
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


# ============================================================
# CLASSIFICATION TRANSFORM
# ============================================================

classifier_transform = transforms.Compose([

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


# ============================================================
# CLASSIFY CROP
# ============================================================

def classify_crop(
    model,
    crop,
    idx_to_class
):

    if crop is None:

        return (
            None,
            0.0,
            None
        )

    crop_rgb = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2RGB
    )

    tensor = classifier_transform(
        crop_rgb
    )

    tensor = tensor.unsqueeze(
        0
    ).to(
        DEVICE
    )

    with torch.no_grad():

        output = model(
            tensor
        )

        probabilities = torch.softmax(
            output,
            dim=1
        )

        confidence, index = torch.max(
            probabilities,
            dim=1
        )

    index = int(
        index.item()
    )

    confidence = (
        float(
            confidence.item()
        )
        *
        100.0
    )

    class_name = idx_to_class.get(
        index,
        f"Class_{index}"
    )

    return (
        class_name,
        confidence,
        index
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\n"
        + "=" * 70
    )

    print(
        "CAMOUFLAGE BREAKER"
    )

    print(
        "COMPLETE PIPELINE EVALUATION"
    )

    print(
        "=" * 70
    )

    print(
        f"\nDevice: {DEVICE}"
    )

    if torch.cuda.is_available():

        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    # --------------------------------------------------------
    # CHECK FILES
    # --------------------------------------------------------

    print(
        "\nChecking required files..."
    )

    required_files = {

        "Test Images":
            TEST_IMAGE_DIR,

        "Test Masks":
            TEST_MASK_DIR,

        "Test TXT":
            TEST_TXT,

        "Train TXT":
            TRAIN_TXT,

        "ResUNet Model":
            SEGMENTATION_MODEL,

        "ResNet50 Model":
            CLASSIFIER_MODEL
    }

    for name, path in required_files.items():

        if not os.path.exists(path):

            raise FileNotFoundError(
                f"\n[ERROR] {name} not found:\n{path}"
            )

        print(
            f"[OK] {name}"
        )

    # --------------------------------------------------------
    # CLASS MAPPING
    # --------------------------------------------------------

    class_to_idx = load_class_mapping()

    print(
        f"Number of classes: {len(class_to_idx)}"
    )

    # --------------------------------------------------------
    # INDEX TO CLASS
    # --------------------------------------------------------

    idx_to_class = {

        int(index): name

        for name, index
        in class_to_idx.items()
    }

    print(
        "\nExample classes:"
    )

    for index in sorted(
        idx_to_class
    )[:10]:

        print(
            f"  {index}: {idx_to_class[index]}"
        )

    # --------------------------------------------------------
    # CLASS ALIASES
    # --------------------------------------------------------

    aliases = build_class_aliases(
        class_to_idx
    )

    # --------------------------------------------------------
    # READ TEST TXT
    # --------------------------------------------------------

    print(
        "\nReading COD10K annotation TXT..."
    )

    records = read_annotation_file(
        TEST_TXT
    )

    print(
        f"Total TXT records: {len(records)}"
    )

    # --------------------------------------------------------
    # CAM RECORDS
    # --------------------------------------------------------

    cam_records = [

        record

        for record in records

        if record["cam_flag"] > 0
    ]

    print(
        f"CAM records: {len(cam_records)}"
    )

    # --------------------------------------------------------
    # BUILD LOOKUPS
    # --------------------------------------------------------

    print(
        "\nBuilding image and mask lookup tables..."
    )

    image_lookup = build_file_lookup(
        TEST_IMAGE_DIR
    )

    mask_lookup = build_file_lookup(
        TEST_MASK_DIR
    )

    print(
        f"Actual images: {len(image_lookup)}"
    )

    print(
        f"Actual masks : {len(mask_lookup)}"
    )

    # --------------------------------------------------------
    # MATCH RECORDS
    # --------------------------------------------------------

    print(
        "\nMatching CAM records..."
    )

    evaluation_records = []

    missing_images = []
    missing_masks = []
    mapping_errors = []

    for record in cam_records:

        stem = record[
            "stem"
        ].lower()

        image_path = image_lookup.get(
            stem
        )

        mask_path = mask_lookup.get(
            stem
        )

        if image_path is None:

            missing_images.append(
                record["filename"]
            )

            continue

        if mask_path is None:

            missing_masks.append(
                record["filename"]
            )

            continue

        # ----------------------------------------------------
        # Get true class
        # ----------------------------------------------------

        if len(
            record["classes"]
        ) == 0:

            mapping_errors.append(
                (
                    record["filename"],
                    "NO_CLASS"
                )
            )

            continue

        true_class = record[
            "classes"
        ][0]

        true_index = get_class_index(
            true_class,
            class_to_idx,
            aliases
        )

        if true_index is None:

            mapping_errors.append(
                (
                    record["filename"],
                    true_class
                )
            )

            continue

        evaluation_records.append({

            "filename":
                record["filename"],

            "image_path":
                image_path,

            "mask_path":
                mask_path,

            "true_class":
                true_class,

            "true_index":
                true_index
        })

    print(
        "\n"
        + "-" * 70
    )

    print(
        f"CAM records          : {len(cam_records)}"
    )

    print(
        f"Matched records      : {len(evaluation_records)}"
    )

    print(
        f"Missing images       : {len(missing_images)}"
    )

    print(
        f"Missing masks        : {len(missing_masks)}"
    )

    print(
        f"Class mapping errors : {len(mapping_errors)}"
    )

    if len(mapping_errors) > 0:

        print(
            "\nFirst 10 class mapping problems:"
        )

        for problem in mapping_errors[:10]:

            print(
                f"  {problem}"
            )

    if len(evaluation_records) == 0:

        raise RuntimeError(
            "No valid evaluation records found."
        )

    print(
        f"\n✓ {len(evaluation_records)} valid records ready"
    )

    # --------------------------------------------------------
    # LOAD MODELS
    # --------------------------------------------------------

    segmentation_model = (
        load_segmentation_model()
    )

    classifier_model = (
        load_classifier()
    )

    # --------------------------------------------------------
    # RESULT STORAGE
    # --------------------------------------------------------

    segmentation_ious = []
    segmentation_dices = []

    true_labels = []
    predicted_labels = []

    prediction_rows = []

    # --------------------------------------------------------
    # EVALUATION
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "RUNNING COMPLETE PIPELINE"
    )

    print(
        "=" * 70
    )

    for record in tqdm(
        evaluation_records,
        desc="Evaluating Pipeline"
    ):

        image = cv2.imread(
            record["image_path"]
        )

        if image is None:
            continue

        # ----------------------------------------------------
        # Ground truth mask
        # ----------------------------------------------------

        ground_truth = load_ground_truth_mask(
            record["mask_path"],
            image
        )

        if ground_truth is None:
            continue

        # ----------------------------------------------------
        # RESUNET
        # ----------------------------------------------------

        predicted_mask = predict_mask(
            segmentation_model,
            image,
            threshold=0.5
        )

        # ----------------------------------------------------
        # SEGMENTATION METRICS
        # ----------------------------------------------------

        iou = calculate_iou(
            predicted_mask,
            ground_truth
        )

        dice = calculate_dice(
            predicted_mask,
            ground_truth
        )

        segmentation_ious.append(
            iou
        )

        segmentation_dices.append(
            dice
        )

        # ----------------------------------------------------
        # PREDICTED OBJECT CROP
        # ----------------------------------------------------

        crop = crop_object(
            image,
            predicted_mask
        )

        # ----------------------------------------------------
        # RESNET50
        # ----------------------------------------------------

        (
            predicted_class,
            confidence,
            predicted_index
        ) = classify_crop(
            classifier_model,
            crop,
            idx_to_class
        )

        if predicted_class is None:

            continue

        # ----------------------------------------------------
        # CLASSIFICATION RESULTS
        # ----------------------------------------------------

        true_labels.append(
            record["true_index"]
        )

        predicted_labels.append(
            predicted_index
        )

        prediction_rows.append({

            "filename":
                record["filename"],

            "true_class":
                record["true_class"],

            "predicted_class":
                predicted_class,

            "confidence":
                confidence,

            "segmentation_iou":
                iou,

            "segmentation_dice":
                dice,

            "object_detected":
                crop is not None
        })

    # ========================================================
    # FINAL SEGMENTATION RESULTS
    # ========================================================

    mean_iou = float(
        np.mean(
            segmentation_ious
        )
    )

    mean_dice = float(
        np.mean(
            segmentation_dices
        )
    )

    # ========================================================
    # FINAL CLASSIFICATION RESULTS
    # ========================================================

    accuracy = accuracy_score(
        true_labels,
        predicted_labels
    )

    precision = precision_score(
        true_labels,
        predicted_labels,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        true_labels,
        predicted_labels,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        true_labels,
        predicted_labels,
        average="weighted",
        zero_division=0
    )

    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    used_labels = sorted(
        set(true_labels)
        |
        set(predicted_labels)
    )

    cm = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=used_labels
    )

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "COMPLETE PIPELINE RESULTS"
    )

    print(
        "=" * 70
    )

    print(
        "\nSEGMENTATION"
    )

    print(
        "-" * 70
    )

    print(
        f"Evaluation samples : {len(segmentation_ious)}"
    )

    print(
        f"Mean IoU           : {mean_iou:.4f} "
        f"({mean_iou * 100:.2f}%)"
    )

    print(
        f"Mean Dice          : {mean_dice:.4f} "
        f"({mean_dice * 100:.2f}%)"
    )

    print(
        "\nCLASSIFICATION"
    )

    print(
        "-" * 70
    )

    print(
        f"Evaluation samples : {len(true_labels)}"
    )

    print(
        f"Accuracy           : {accuracy:.4f} "
        f"({accuracy * 100:.2f}%)"
    )

    print(
        f"Precision          : {precision:.4f} "
        f"({precision * 100:.2f}%)"
    )

    print(
        f"Recall             : {recall:.4f} "
        f"({recall * 100:.2f}%)"
    )

    print(
        f"F1 Score           : {f1:.4f} "
        f"({f1 * 100:.2f}%)"
    )

    print(
        "\nCONFUSION MATRIX"
    )

    print(
        "-" * 70
    )

    print(
        f"Shape: {cm.shape}"
    )

    # ========================================================
    # CLASSIFICATION REPORT
    # ========================================================

    report = classification_report(
        true_labels,
        predicted_labels,
        labels=used_labels,
        target_names=[
            idx_to_class[index]
            for index in used_labels
        ],
        zero_division=0
    )

    print(
        "\nPER-CLASS RESULTS"
    )

    print(
        "=" * 70
    )

    print(
        report
    )

    # ========================================================
    # SAVE CSV
    # ========================================================

    csv_path = os.path.join(
        OUTPUT_DIR,
        "pipeline_predictions.csv"
    )

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "filename",
                "true_class",
                "predicted_class",
                "confidence",
                "segmentation_iou",
                "segmentation_dice",
                "object_detected"
            ]
        )

        writer.writeheader()

        writer.writerows(
            prediction_rows
        )

    # ========================================================
    # SAVE SUMMARY
    # ========================================================

    summary = {

        "pipeline": "ResUNet + ResNet50",

        "device": str(
            DEVICE
        ),

        "segmentation": {

            "samples":
                len(segmentation_ious),

            "mean_iou":
                mean_iou,

            "mean_dice":
                mean_dice
        },

        "classification": {

            "samples":
                len(true_labels),

            "accuracy":
                float(accuracy),

            "precision":
                float(precision),

            "recall":
                float(recall),

            "f1_score":
                float(f1)
        },

        "records": {

            "total_cam_records":
                len(cam_records),

            "valid_records":
                len(evaluation_records),

            "missing_images":
                len(missing_images),

            "missing_masks":
                len(missing_masks),

            "class_mapping_errors":
                len(mapping_errors)
        }
    }

    summary_path = os.path.join(
        OUTPUT_DIR,
        "pipeline_summary.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            summary,
            file,
            indent=4
        )

    # ========================================================
    # SAVE CONFUSION MATRIX
    # ========================================================

    cm_path = os.path.join(
        OUTPUT_DIR,
        "confusion_matrix.csv"
    )

    with open(
        cm_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(
            file
        )

        header = [
            "True / Predicted"
        ]

        header.extend(
            [
                idx_to_class[index]
                for index in used_labels
            ]
        )

        writer.writerow(
            header
        )

        for row_index, row in enumerate(cm):

            row_data = [

                idx_to_class[
                    used_labels[row_index]
                ]

            ]

            row_data.extend(
                row.tolist()
            )

            writer.writerow(
                row_data
            )

    # ========================================================
    # FINAL
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "✓ COMPLETE PIPELINE EVALUATION FINISHED"
    )

    print(
        "=" * 70
    )

    print(
        "\nResults saved to:"
    )

    print(
        OUTPUT_DIR
    )

    print(
        f"\n✓ {csv_path}"
    )

    print(
        f"✓ {summary_path}"
    )

    print(
        f"✓ {cm_path}"
    )

    print(
        "\n"
        + "=" * 70
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()