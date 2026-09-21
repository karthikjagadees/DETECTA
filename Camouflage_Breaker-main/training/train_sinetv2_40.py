import os
import sys
import glob
import shutil
import random
import subprocess

import cv2
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# ============================================================
# PROJECT PATH
# ============================================================

if os.path.exists("/content/Camouflage_Breaker"):
    PROJECT_ROOT = "/content/Camouflage_Breaker"
else:
    PROJECT_ROOT = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )


TRAIN_IMAGE_DIR = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "Train",
    "Image"
)

TRAIN_MASK_DIR = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "Train",
    "GT_Object"
)

SOURCE_DIR = os.path.join(
    PROJECT_ROOT,
    "models",
    "sinetv2",
    "source"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "saved_models"
)

OUTPUT_MODEL = os.path.join(
    OUTPUT_DIR,
    "sinetv2_cod10k_40epoch_best.pth"
)


# ============================================================
# TRAINING SETTINGS
# ============================================================

IMAGE_SIZE = 352
BATCH_SIZE = 8
EPOCHS = 40

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

VAL_RATIO = 0.10
SEED = 42
NUM_WORKERS = 2


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.benchmark = True


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("SINET-V2 COD10K - 40 EPOCH TRAINING")
print("=" * 70)

print("Project :", PROJECT_ROOT)
print("Device  :", DEVICE)

if DEVICE.type == "cuda":
    print(
        "GPU     :",
        torch.cuda.get_device_name(0)
    )

print()


# ============================================================
# CHECK DATASET
# ============================================================

if not os.path.isdir(TRAIN_IMAGE_DIR):
    raise FileNotFoundError(
        f"Training image folder not found:\n"
        f"{TRAIN_IMAGE_DIR}"
    )

if not os.path.isdir(TRAIN_MASK_DIR):
    raise FileNotFoundError(
        f"Training mask folder not found:\n"
        f"{TRAIN_MASK_DIR}"
    )


# ============================================================
# FIND SINET-V2 SOURCE
# ============================================================

if not os.path.isdir(
    os.path.join(
        SOURCE_DIR,
        "lib"
    )
):

    print("SINet-V2 source not found.")
    print("Cloning official SINet-V2 repository...")

    os.makedirs(
        os.path.dirname(SOURCE_DIR),
        exist_ok=True
    )

    TEMP_REPO = "/content/SINet-V2"

    if os.path.exists(TEMP_REPO):
        shutil.rmtree(TEMP_REPO)

    subprocess.run(
        [
            "git",
            "clone",
            "https://github.com/GewelsJI/SINet-V2.git",
            TEMP_REPO
        ],
        check=True
    )

    shutil.copytree(
        TEMP_REPO,
        SOURCE_DIR,
        dirs_exist_ok=True
    )


if not os.path.isdir(
    os.path.join(
        SOURCE_DIR,
        "lib"
    )
):
    raise FileNotFoundError(
        "SINet-V2 source could not be located."
    )


print(
    "SINet source:",
    SOURCE_DIR
)


# ============================================================
# IMPORT SINET-V2 SOURCE
# ============================================================

if SOURCE_DIR not in sys.path:
    sys.path.insert(
        0,
        SOURCE_DIR
    )

LIB_DIR = os.path.join(
    SOURCE_DIR,
    "lib"
)

if LIB_DIR not in sys.path:
    sys.path.insert(
        0,
        LIB_DIR
    )


# ============================================================
# FIND PRETRAINED WEIGHTS
# ============================================================

def find_file(filename):

    search_locations = [
        PROJECT_ROOT,
        "/content",
        "/content/drive/MyDrive"
    ]

    matches = []

    for root in search_locations:

        if not os.path.exists(root):
            continue

        pattern = os.path.join(
            root,
            "**",
            filename
        )

        matches.extend(
            glob.glob(
                pattern,
                recursive=True
            )
        )

    matches = list(
        dict.fromkeys(matches)
    )

    project_matches = [
        path
        for path in matches
        if path.startswith(PROJECT_ROOT)
    ]

    if project_matches:
        return project_matches[0]

    if matches:
        return matches[0]

    return None


SINET_PRETRAINED = find_file(
    "Net_epoch_best.pth"
)

RES2NET_PRETRAINED = find_file(
    "res2net50_v1b_26w_4s-3cf99910.pth"
)


if SINET_PRETRAINED is None:
    raise FileNotFoundError(
        "Net_epoch_best.pth was not found."
    )

if RES2NET_PRETRAINED is None:
    raise FileNotFoundError(
        "res2net50_v1b_26w_4s-3cf99910.pth "
        "was not found."
    )


print(
    "SINet checkpoint :",
    SINET_PRETRAINED
)

print(
    "Res2Net checkpoint:",
    RES2NET_PRETRAINED
)

print()


# ============================================================
# PATCH RES2NET CHECKPOINT PATH
# ============================================================

import lib.Res2Net_v1b as res2net_module


original_res2net50 = (
    res2net_module.res2net50_v1b_26w_4s
)


def patched_res2net50_v1b_26w_4s(
    *args,
    **kwargs
):

    model = original_res2net50(
        *args,
        **kwargs
    )

    checkpoint = torch.load(
        RES2NET_PRETRAINED,
        map_location="cpu"
    )

    if isinstance(
        checkpoint,
        dict
    ):

        if "state_dict" in checkpoint:
            checkpoint = checkpoint[
                "state_dict"
            ]

        elif "model_state_dict" in checkpoint:
            checkpoint = checkpoint[
                "model_state_dict"
            ]

    cleaned = {}

    for key, value in checkpoint.items():

        new_key = key

        if new_key.startswith(
            "module."
        ):
            new_key = new_key[
                len("module.") :
            ]

        cleaned[new_key] = value

    model.load_state_dict(
        cleaned,
        strict=False
    )

    return model


res2net_module.res2net50_v1b_26w_4s = (
    patched_res2net50_v1b_26w_4s
)


# ============================================================
# IMPORT SINET NETWORK
# ============================================================

from lib.Network_Res2Net_GRA_NCD import Network


# ============================================================
# DATASET
# ============================================================

class COD10KDataset(Dataset):

    def __init__(
        self,
        image_dir,
        mask_dir
    ):

        self.image_dir = image_dir
        self.mask_dir = mask_dir

        extensions = [
            "*.jpg",
            "*.jpeg",
            "*.png",
            "*.JPG",
            "*.JPEG",
            "*.PNG"
        ]

        image_paths = []

        for extension in extensions:

            image_paths.extend(
                glob.glob(
                    os.path.join(
                        image_dir,
                        extension
                    )
                )
            )

        image_paths = sorted(
            list(
                set(image_paths)
            )
        )

        if len(image_paths) == 0:
            raise RuntimeError(
                "No training images were found."
            )

        self.samples = []

        for image_path in image_paths:

            filename = os.path.basename(
                image_path
            )

            base_name = os.path.splitext(
                filename
            )[0]

            possible_masks = [
                os.path.join(
                    mask_dir,
                    base_name + ".png"
                ),
                os.path.join(
                    mask_dir,
                    base_name + ".jpg"
                ),
                os.path.join(
                    mask_dir,
                    base_name + ".jpeg"
                )
            ]

            mask_path = None

            for candidate in possible_masks:

                if os.path.exists(candidate):

                    mask_path = candidate
                    break

            if mask_path is not None:

                self.samples.append(
                    (
                        image_path,
                        mask_path
                    )
                )

        if len(self.samples) == 0:
            raise RuntimeError(
                "No valid image-mask pairs were found."
            )

        print(
            "Valid image-mask pairs:",
            len(self.samples)
        )

    def __len__(self):

        return len(
            self.samples
        )

    def load_sample(
        self,
        index,
        augment=False
    ):

        image_path, mask_path = (
            self.samples[index]
        )

        image = cv2.imread(
            image_path,
            cv2.IMREAD_COLOR
        )

        mask = cv2.imread(
            mask_path,
            cv2.IMREAD_GRAYSCALE
        )

        if image is None:
            raise RuntimeError(
                f"Could not read image:\n"
                f"{image_path}"
            )

        if mask is None:
            raise RuntimeError(
                f"Could not read mask:\n"
                f"{mask_path}"
            )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        mask = (
            mask > 127
        ).astype(
            np.float32
        )

        image = cv2.resize(
            image,
            (
                IMAGE_SIZE,
                IMAGE_SIZE
            ),
            interpolation=cv2.INTER_LINEAR
        )

        mask = cv2.resize(
            mask,
            (
                IMAGE_SIZE,
                IMAGE_SIZE
            ),
            interpolation=cv2.INTER_NEAREST
        )

        if augment:

            if random.random() < 0.5:

                image = np.fliplr(
                    image
                ).copy()

                mask = np.fliplr(
                    mask
                ).copy()

            if random.random() < 0.5:

                image = np.flipud(
                    image
                ).copy()

                mask = np.flipud(
                    mask
                ).copy()

        image = (
            image.astype(
                np.float32
            ) / 255.0
        )

        image = torch.from_numpy(
            image
        ).permute(
            2,
            0,
            1
        )

        mask = torch.from_numpy(
            mask
        ).unsqueeze(
            0
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

        image = (
            image - mean
        ) / std

        return image, mask

    def __getitem__(
        self,
        index
    ):

        return self.load_sample(
            index,
            augment=False
        )


# ============================================================
# TRAIN / VALIDATION DATASETS
# ============================================================

class SubsetDataset(Dataset):

    def __init__(
        self,
        base_dataset,
        indices,
        augment
    ):

        self.base_dataset = base_dataset
        self.indices = list(indices)
        self.augment = augment

    def __len__(self):

        return len(
            self.indices
        )

    def __getitem__(
        self,
        index
    ):

        real_index = self.indices[
            index
        ]

        return self.base_dataset.load_sample(
            real_index,
            augment=self.augment
        )


# ============================================================
# CREATE DATASET
# ============================================================

full_dataset = COD10KDataset(
    TRAIN_IMAGE_DIR,
    TRAIN_MASK_DIR
)

total_size = len(
    full_dataset
)

val_size = int(
    total_size * VAL_RATIO
)

train_size = (
    total_size - val_size
)


# ============================================================
# FIXED TRAIN / VALIDATION SPLIT
# ============================================================

indices = list(
    range(total_size)
)

generator = torch.Generator().manual_seed(
    SEED
)

indices_tensor = torch.randperm(
    total_size,
    generator=generator
)

indices = indices_tensor.tolist()

train_indices = indices[
    :train_size
]

val_indices = indices[
    train_size:
]


train_dataset = SubsetDataset(
    full_dataset,
    train_indices,
    augment=True
)

val_dataset = SubsetDataset(
    full_dataset,
    val_indices,
    augment=False
)


# ============================================================
# DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=(
        DEVICE.type == "cuda"
    ),
    drop_last=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=(
        DEVICE.type == "cuda"
    )
)


print()
print("Dataset")
print("-" * 30)
print("Total :", total_size)
print("Train :", len(train_dataset))
print("Val   :", len(val_dataset))
print()


# ============================================================
# CREATE MODEL
# ============================================================

model = Network(
    channel=32
)

model = model.to(
    DEVICE
)


# ============================================================
# LOAD OFFICIAL SINET-V2 CHECKPOINT
# ============================================================

checkpoint = torch.load(
    SINET_PRETRAINED,
    map_location="cpu"
)


if isinstance(
    checkpoint,
    dict
):

    if "state_dict" in checkpoint:

        checkpoint = checkpoint[
            "state_dict"
        ]

    elif "model_state_dict" in checkpoint:

        checkpoint = checkpoint[
            "model_state_dict"
        ]


cleaned_checkpoint = {}

for key, value in checkpoint.items():

    new_key = key

    if new_key.startswith(
        "module."
    ):

        new_key = new_key[
            len("module.") :
        ]

    cleaned_checkpoint[
        new_key
    ] = value


load_result = model.load_state_dict(
    cleaned_checkpoint,
    strict=False
)


print(
    "Pretrained SINet-V2 loaded."
)

print(
    "Missing keys   :",
    len(load_result.missing_keys)
)

print(
    "Unexpected keys:",
    len(load_result.unexpected_keys)
)

print()


# ============================================================
# STRUCTURE LOSS
# ============================================================

def structure_loss(
    prediction,
    target
):

    if prediction.shape[-2:] != target.shape[-2:]:

        target = F.interpolate(
            target,
            size=prediction.shape[-2:],
            mode="nearest"
        )

    weight = (
        1
        + 5
        * torch.abs(
            F.avg_pool2d(
                target,
                kernel_size=31,
                stride=1,
                padding=15
            )
            - target
        )
    )

    prediction_probability = torch.sigmoid(
        prediction
    )

    weighted_bce = F.binary_cross_entropy_with_logits(
        prediction,
        target,
        reduction="none"
    )

    weighted_bce = (
        weight * weighted_bce
    ).sum(
        dim=(2, 3)
    ) / weight.sum(
        dim=(2, 3)
    )

    intersection = (
        prediction_probability
        * target
        * weight
    ).sum(
        dim=(2, 3)
    )

    weighted_union = (
        (
            prediction_probability
            + target
        )
        * weight
    ).sum(
        dim=(2, 3)
    )

    weighted_iou = 1 - (
        intersection + 1
    ) / (
        weighted_union
        - intersection
        + 1
    )

    return (
        weighted_bce
        + weighted_iou
    ).mean()


# ============================================================
# HANDLE SINET OUTPUTS
# ============================================================

def get_outputs(
    model_output
):

    if isinstance(
        model_output,
        (list, tuple)
    ):

        return list(
            model_output
        )

    return [
        model_output
    ]


# ============================================================
# IOU
# ============================================================

def calculate_iou(
    prediction,
    target
):

    prediction = (
        prediction > 0.5
    ).float()

    target = (
        target > 0.5
    ).float()

    intersection = (
        prediction * target
    ).sum(
        dim=(1, 2, 3)
    )

    union = (
        prediction
        + target
        - prediction * target
    ).sum(
        dim=(1, 2, 3)
    )

    iou = (
        intersection + 1e-7
    ) / (
        union + 1e-7
    )

    return iou


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS,
    eta_min=1e-6
)


# ============================================================
# TRAINING
# ============================================================

best_val_iou = 0.0
best_epoch = 0

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


for epoch in range(
    1,
    EPOCHS + 1
):

    model.train()

    running_loss = 0.0

    progress = tqdm(
        train_loader,
        desc=(
            f"Epoch "
            f"{epoch}/{EPOCHS}"
        )
    )

    for images, masks in progress:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        masks = masks.to(
            DEVICE,
            non_blocking=True
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        model_output = model(
            images
        )

        outputs = get_outputs(
            model_output
        )

        total_loss = 0.0

        for output in outputs:

            total_loss += structure_loss(
                output,
                masks
            )

        total_loss /= len(
            outputs
        )

        total_loss.backward()

        optimizer.step()

        running_loss += (
            total_loss.item()
        )

        progress.set_postfix(
            loss=(
                f"{total_loss.item():.4f}"
            )
        )

    scheduler.step()

    average_train_loss = (
        running_loss
        / len(train_loader)
    )


    # ========================================================
    # VALIDATION
    # ========================================================

    model.eval()

    total_iou = 0.0
    total_images = 0

    with torch.no_grad():

        for images, masks in val_loader:

            images = images.to(
                DEVICE,
                non_blocking=True
            )

            masks = masks.to(
                DEVICE,
                non_blocking=True
            )

            model_output = model(
                images
            )

            outputs = get_outputs(
                model_output
            )

            prediction = torch.sigmoid(
                outputs[-1]
            )

            if prediction.shape[-2:] != masks.shape[-2:]:

                prediction = F.interpolate(
                    prediction,
                    size=masks.shape[-2:],
                    mode="bilinear",
                    align_corners=False
                )

            batch_iou = calculate_iou(
                prediction,
                masks
            )

            batch_size = (
                images.size(0)
            )

            total_iou += (
                batch_iou.sum().item()
            )

            total_images += batch_size

    val_iou = (
        total_iou
        / total_images
    )

    current_lr = (
        optimizer.param_groups[0]["lr"]
    )


    print()
    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"Train Loss: "
        f"{average_train_loss:.4f} | "
        f"Val IoU: "
        f"{val_iou:.4f} | "
        f"LR: "
        f"{current_lr:.7f}"
    )


    # ========================================================
    # SAVE BEST MODEL
    # ========================================================

    if val_iou > best_val_iou:

        best_val_iou = val_iou
        best_epoch = epoch

        torch.save(
            model.state_dict(),
            OUTPUT_MODEL
        )

        print(
            f"BEST MODEL SAVED "
            f"(IoU={best_val_iou:.4f})"
        )

    print()


# ============================================================
# FINAL RESULT
# ============================================================

print("=" * 70)
print("40-EPOCH TRAINING COMPLETE")
print("=" * 70)

print(
    f"Best Validation IoU : "
    f"{best_val_iou:.4f}"
)

print(
    f"Best Epoch          : "
    f"{best_epoch}"
)

print(
    "Model               :"
)

print(
    OUTPUT_MODEL
)

print(
    "Exists              :",
    os.path.exists(
        OUTPUT_MODEL
    )
)

if os.path.exists(
    OUTPUT_MODEL
):

    size_mb = (
        os.path.getsize(
            OUTPUT_MODEL
        )
        / (1024 * 1024)
    )

    print(
        f"Size                : "
        f"{size_mb:.2f} MB"
    )

print("=" * 70)