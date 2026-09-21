import os
import sys
import time
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm
import json
import numpy as np

# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# CONFIGURATION
# ============================================================

class Config:

    CROPS_ROOT = os.path.join(PROJECT_ROOT, "dataset", "crops")

    SPLIT = "Train"

    IMAGE_SIZE = 224

    BATCH_SIZE = 16

    NUM_WORKERS = 0

    EPOCHS = int(os.environ.get("CLS_EPOCHS", "30"))

    LEARNING_RATE = float(os.environ.get("CLS_LR", "1e-4"))

    WEIGHT_DECAY = 1e-5

    # CONTINUE_TRAIN=1 → resume from classifier_best.pth
    CONTINUE_TRAIN = os.environ.get("CONTINUE_TRAIN", "0").strip() in (
        "1", "true", "True", "yes"
    )

    VAL_RATIO = 0.10

    SEED = 42

    NUM_CLASSES = 69

    CHECKPOINT_DIR = os.path.join(
        PROJECT_ROOT,
        "saved_models"
    )

    CHECKPOINT_NAME = "classifier_best.pth"

    CLASS_MAPPING_NAME = "class_mapping.json"

    PATIENCE = 5


# ============================================================
# SEED
# ============================================================

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# RESNET50 CLASSIFIER
# ============================================================

class ResNet50Classifier(nn.Module):

    def __init__(self, num_classes=69):

        super().__init__()

        self.backbone = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V2
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


# ============================================================
# DATASET
# ============================================================

class CropDataset(Dataset):

    def __init__(
        self,
        crops_dir,
        split="Train",
        transform=None
    ):

        self.crops_dir = os.path.join(
            crops_dir,
            split
        )

        self.transform = transform

        self.samples = []

        self.class_to_idx = {}

        self.idx_to_class = {}

        self._load_labels()


    # --------------------------------------------------------
    # LOAD COD10K LABELS
    # --------------------------------------------------------

    def _load_labels(self):

        txt_path = os.path.join(

            PROJECT_ROOT,

            "dataset",

            self.split_name(),

            f"CAM-NonCAM_Instance_{self.split_name()}.txt"
        )

        if not os.path.exists(txt_path):

            raise FileNotFoundError(
                f"Label file not found:\n{txt_path}"
            )


        # ----------------------------------------------------
        # Read TXT
        # ----------------------------------------------------

        with open(
            txt_path,
            "r",
            encoding="utf-8"
        ) as f:

            lines = f.readlines()


        label_map = {}

        current_filename = None

        current_category = None


        # ----------------------------------------------------
        # Parse COD10K TXT
        # ----------------------------------------------------

        i = 0

        while i < len(lines):

            line = lines[i].strip()


            if line.startswith("[INFO]"):

                parts = line.split()

                if len(parts) >= 3:

                    filename = parts[1]

                    cam_flag = parts[2]

                    base = os.path.splitext(
                        filename
                    )[0]


                    # Next line contains class path

                    if i + 1 < len(lines):

                        class_line = lines[i + 1].strip()

                        if (
                            class_line
                            and not class_line.startswith("[INFO]")
                            and "/" in class_line.split()[0]
                        ):

                            class_path = class_line.split()[0]

                            category = class_path.split("/")[-1]

                            if category and category != "[INFO]":

                                label_map[base] = category


                            i += 2

                            continue


            i += 1


        if len(label_map) == 0:

            raise RuntimeError(
                "No labels were loaded from COD10K TXT file."
            )


        # ----------------------------------------------------
        # Create crop samples first, then class mapping from
        # classes that actually appear in training crops.
        # ----------------------------------------------------

        if not os.path.exists(self.crops_dir):

            raise FileNotFoundError(
                f"Crop directory not found:\n{self.crops_dir}"
            )


        raw_samples = []

        for filename in sorted(
            os.listdir(self.crops_dir)
        ):

            if not filename.endswith("_crop.jpg"):

                continue


            base = filename.replace(
                "_crop.jpg",
                ""
            )


            if base not in label_map:

                continue


            class_name = label_map[base]

            image_path = os.path.join(
                self.crops_dir,
                filename
            )


            raw_samples.append(
                (
                    image_path,
                    class_name
                )
            )


        if len(raw_samples) == 0:

            raise RuntimeError(
                "No labeled crop samples found."
            )


        all_classes = sorted(
            {
                class_name
                for _, class_name
                in raw_samples
            }
        )


        self.class_to_idx = {
            class_name: index
            for index, class_name
            in enumerate(all_classes)
        }


        self.idx_to_class = {
            str(index): class_name
            for class_name, index
            in self.class_to_idx.items()
        }


        self.samples = [
            (
                image_path,
                self.class_to_idx[class_name]
            )
            for image_path, class_name
            in raw_samples
        ]


        print(
            f"[INFO] {self.split_name()} "
            f"classes: {len(self.class_to_idx)}"
        )

        print(
            f"[INFO] {self.split_name()} "
            f"crop samples: {len(self.samples)}"
        )


    def split_name(self):

        return os.path.basename(
            self.crops_dir
        )


    def __len__(self):

        return len(self.samples)


    def __getitem__(self, index):

        image_path, label = self.samples[index]


        image = Image.open(
            image_path
        ).convert("RGB")


        if self.transform:

            image = self.transform(image)


        return image, label


# ============================================================
# TRANSFORMS
# ============================================================

def get_train_transform():

    return transforms.Compose([

        transforms.Resize(
            (Config.IMAGE_SIZE, Config.IMAGE_SIZE)
        ),

        transforms.RandomHorizontalFlip(
            p=0.5
        ),

        transforms.RandomRotation(
            15
        ),

        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2
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


def get_val_transform():

    return transforms.Compose([

        transforms.Resize(
            (Config.IMAGE_SIZE, Config.IMAGE_SIZE)
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
# TRAINING
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device
):

    model.train()

    total_loss = 0

    correct = 0

    total = 0


    progress = tqdm(
        loader,
        desc="Training"
    )


    for images, labels in progress:

        images = images.to(device)

        labels = labels.to(device)


        optimizer.zero_grad()


        outputs = model(images)


        loss = criterion(
            outputs,
            labels
        )


        loss.backward()

        optimizer.step()


        total_loss += loss.item()


        predictions = outputs.argmax(
            dim=1
        )


        correct += (
            predictions == labels
        ).sum().item()


        total += labels.size(0)


        accuracy = (
            100.0 * correct / total
        )


        progress.set_postfix(
            loss=f"{loss.item():.4f}",
            acc=f"{accuracy:.2f}%"
        )


    return (
        total_loss / len(loader),
        100.0 * correct / total
    )


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def validate(
    model,
    loader,
    criterion,
    device
):

    model.eval()

    total_loss = 0

    correct = 0

    total = 0

    all_preds = []

    all_labels = []


    progress = tqdm(
        loader,
        desc="Validation"
    )


    for images, labels in progress:

        images = images.to(device)

        labels = labels.to(device)


        outputs = model(images)


        loss = criterion(
            outputs,
            labels
        )


        total_loss += loss.item()


        predictions = outputs.argmax(
            dim=1
        )


        correct += (
            predictions == labels
        ).sum().item()


        total += labels.size(0)

        all_preds.extend(
            predictions.cpu().tolist()
        )

        all_labels.extend(
            labels.cpu().tolist()
        )


    from sklearn.metrics import (
        f1_score,
        precision_score,
        recall_score,
    )

    precision = float(
        precision_score(
            all_labels,
            all_preds,
            average="macro",
            zero_division=0,
        )
    )

    recall = float(
        recall_score(
            all_labels,
            all_preds,
            average="macro",
            zero_division=0,
        )
    )

    f1 = float(
        f1_score(
            all_labels,
            all_preds,
            average="macro",
            zero_division=0,
        )
    )

    return (
        total_loss / len(loader),
        100.0 * correct / total,
        precision,
        recall,
        f1,
    )


# ============================================================
# SAVE CHECKPOINT
# ============================================================

def save_checkpoint(
    model,
    optimizer,
    epoch,
    best_accuracy,
    class_to_idx,
    idx_to_class
):

    os.makedirs(
        Config.CHECKPOINT_DIR,
        exist_ok=True
    )


    checkpoint_path = os.path.join(

        Config.CHECKPOINT_DIR,

        Config.CHECKPOINT_NAME
    )


    checkpoint = {

        "epoch": epoch,

        "best_accuracy": best_accuracy,

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "class_to_idx":
            class_to_idx,

        "idx_to_class":
            idx_to_class,

        "num_classes":
            Config.NUM_CLASSES
    }


    torch.save(
        checkpoint,
        checkpoint_path + ".tmp"
    )
    os.replace(
        checkpoint_path + ".tmp",
        checkpoint_path
    )


    # Also save class mapping separately (same order as training dataset)

    mapping_path = os.path.join(

        Config.CHECKPOINT_DIR,

        Config.CLASS_MAPPING_NAME
    )


    with open(
        mapping_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            {
                "class_to_idx": class_to_idx,
                "idx_to_class": idx_to_class,
                "num_classes": Config.NUM_CLASSES,
            },
            f,
            indent=2
        )


    print(
        f"[OK] Model saved: {checkpoint_path}"
    )

    print(
        f"[OK] Class mapping saved: {mapping_path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)

    print(
        "CAMOUFLAGE BREAKER - RESNET50 CLASSIFIER"
    )

    print("=" * 70)


    config = Config()


    set_seed(
        config.SEED
    )


    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(

        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    if device.type != "cuda":
        raise RuntimeError(
            "CUDA is required for classifier training but is not available."
        )

    try:
        probe = torch.zeros(1, device=device)
        _ = probe + 1
    except Exception as exc:
        raise RuntimeError(
            f"CUDA probe failed; refusing CPU fallback. Error: {exc}"
        ) from exc


    print(
        f"[INFO] Device: {device}"
    )
    print(
        f"[INFO] GPU: {torch.cuda.get_device_name(0)}"
    )


    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    print(
        "\n[INFO] Loading crop dataset..."
    )


    full_dataset = CropDataset(

        config.CROPS_ROOT,

        config.SPLIT,

        transform=None
    )


    total_samples = len(
        full_dataset
    )


    if total_samples == 0:

        raise RuntimeError(
            "No crop images found."
        )


    print(
        f"[INFO] Total samples: "
        f"{total_samples}"
    )


    print(
        f"[INFO] Number of classes: "
        f"{len(full_dataset.class_to_idx)}"
    )


    # --------------------------------------------------------
    # Check classes
    # --------------------------------------------------------

    if len(full_dataset.class_to_idx) != config.NUM_CLASSES:

        raise RuntimeError(

            f"Expected {config.NUM_CLASSES} classes, "

            f"but found "
            f"{len(full_dataset.class_to_idx)}."
        )


    # --------------------------------------------------------
    # Train / Validation Split
    # --------------------------------------------------------

    val_size = int(
        total_samples *
        config.VAL_RATIO
    )


    train_size = (
        total_samples -
        val_size
    )


    train_subset, val_subset = random_split(

        full_dataset,

        [train_size, val_size],

        generator=torch.Generator().manual_seed(
            config.SEED
        )
    )


    # --------------------------------------------------------
    # IMPORTANT:
    # Create separate dataset objects
    # so train and validation transforms
    # do not overwrite each other.
    # --------------------------------------------------------

    train_dataset = CropDataset(

        config.CROPS_ROOT,

        config.SPLIT,

        transform=get_train_transform()
    )


    val_dataset = CropDataset(

        config.CROPS_ROOT,

        config.SPLIT,

        transform=get_val_transform()
    )


    train_dataset.samples = [
        train_dataset.samples[i]
        for i in train_subset.indices
    ]


    val_dataset.samples = [
        val_dataset.samples[i]
        for i in val_subset.indices
    ]


    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    train_loader = DataLoader(

        train_dataset,

        batch_size=config.BATCH_SIZE,

        shuffle=True,

        num_workers=config.NUM_WORKERS,

        pin_memory=torch.cuda.is_available()
    )


    val_loader = DataLoader(

        val_dataset,

        batch_size=config.BATCH_SIZE,

        shuffle=False,

        num_workers=config.NUM_WORKERS,

        pin_memory=torch.cuda.is_available()
    )


    print(
        f"[INFO] Training samples: "
        f"{len(train_dataset)}"
    )

    print(
        f"[INFO] Validation samples: "
        f"{len(val_dataset)}"
    )


    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print(
        "\n[INFO] Building ResNet50..."
    )


    model = ResNet50Classifier(

        num_classes=config.NUM_CLASSES
    )


    model = model.to(device)


    total_parameters = sum(

        parameter.numel()

        for parameter
        in model.parameters()
    )


    print(
        f"[INFO] Parameters: "
        f"{total_parameters:,}"
    )


    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss()


    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.Adam(

        model.parameters(),

        lr=config.LEARNING_RATE,

        weight_decay=config.WEIGHT_DECAY
    )


    # --------------------------------------------------------
    # Scheduler
    # --------------------------------------------------------

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(

        optimizer,

        mode="max",

        factor=0.5,

        patience=3
    )


    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    best_accuracy = 0.0

    best_metrics = {}

    patience_counter = 0

    ckpt_path = os.path.join(config.CHECKPOINT_DIR, config.CHECKPOINT_NAME)
    if config.CONTINUE_TRAIN and os.path.isfile(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        best_accuracy = float(ckpt.get("best_accuracy", 0.0))
        print(
            f"[OK] CONTINUE from {ckpt_path} "
            f"(prior best_acc={best_accuracy:.4f}, "
            f"epochs={config.EPOCHS}, lr={config.LEARNING_RATE})"
        )

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_precision": [],
        "val_recall": [],
        "val_f1": [],
        "lr": [],
        "epoch_time_sec": [],
    }

    total_train_start = time.time()


    print(
        "\n[INFO] Starting training..."
    )


    print(
        "-" * 70
    )


    for epoch in range(
        config.EPOCHS
    ):

        start_time = time.time()


        train_loss, train_accuracy = train_one_epoch(

            model,

            train_loader,

            criterion,

            optimizer,

            device
        )


        (
            val_loss,
            val_accuracy,
            val_precision,
            val_recall,
            val_f1,
        ) = validate(

            model,

            val_loader,

            criterion,

            device
        )


        scheduler.step(
            val_accuracy
        )

        current_lr = optimizer.param_groups[0]["lr"]


        elapsed = (
            time.time()
            - start_time
        )

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)
        history["val_precision"].append(val_precision)
        history["val_recall"].append(val_recall)
        history["val_f1"].append(val_f1)
        history["lr"].append(current_lr)
        history["epoch_time_sec"].append(elapsed)


        print()

        print(
            f"Epoch "
            f"{epoch + 1:02d}/"
            f"{config.EPOCHS}"
        )

        print(
            f"Time: "
            f"{elapsed:.1f}s"
        )

        print(
            f"Train Loss: "
            f"{train_loss:.4f}"
        )

        print(
            f"Train Accuracy: "
            f"{train_accuracy:.2f}%"
        )

        print(
            f"Val Loss: "
            f"{val_loss:.4f}"
        )

        print(
            f"Val Accuracy: "
            f"{val_accuracy:.2f}%"
        )

        print(
            f"Val P/R/F1: "
            f"{val_precision:.4f}/"
            f"{val_recall:.4f}/"
            f"{val_f1:.4f}"
        )

        print(
            f"Learning Rate: "
            f"{current_lr:.6f}"
        )


        # ----------------------------------------------------
        # Save best model
        # ----------------------------------------------------

        if val_accuracy > best_accuracy:

            best_accuracy = val_accuracy

            patience_counter = 0

            best_metrics = {
                "epoch": epoch + 1,
                "learning_rate": current_lr,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "accuracy": val_accuracy,
                "precision": val_precision,
                "recall": val_recall,
                "f1": val_f1,
            }


            save_checkpoint(

                model,

                optimizer,

                epoch,

                best_accuracy,

                full_dataset.class_to_idx,

                full_dataset.idx_to_class
            )


            print(
                "[*] New best model!"
            )

        else:

            patience_counter += 1

            print(
                f"[INFO] No improvement "
                f"({patience_counter}/"
                f"{config.PATIENCE})"
            )


        print(
            "-" * 70
        )


        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        if patience_counter >= config.PATIENCE:

            print(
                "\n[STOP] Early stopping."
            )

            break


    # --------------------------------------------------------
    # COMPLETE
    # --------------------------------------------------------

    total_time = time.time() - total_train_start

    print()

    print("=" * 70)

    print(
        "CLASSIFIER TRAINING COMPLETE"
    )

    print(
        f"Best Validation Accuracy: "
        f"{best_accuracy:.2f}%"
    )

    print("=" * 70)

    os.makedirs(
        os.path.join(PROJECT_ROOT, "outputs"),
        exist_ok=True
    )

    train_meta = {
        "model": "ResNet50Classifier",
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0),
        "seed": config.SEED,
        "epochs_planned": config.EPOCHS,
        "epochs_ran": len(history["train_loss"]),
        "batch_size": config.BATCH_SIZE,
        "learning_rate": config.LEARNING_RATE,
        "num_classes": config.NUM_CLASSES,
        "best_val_metrics": best_metrics,
        "training_time_sec": total_time,
        "checkpoint": os.path.join(
            config.CHECKPOINT_DIR,
            config.CHECKPOINT_NAME
        ),
        "class_mapping": os.path.join(
            config.CHECKPOINT_DIR,
            config.CLASS_MAPPING_NAME
        ),
        "history": history,
    }

    with open(
        os.path.join(
            PROJECT_ROOT,
            "outputs",
            "classifier_training_metrics.json"
        ),
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(train_meta, f, indent=2)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()