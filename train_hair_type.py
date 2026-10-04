from pathlib import Path
import copy
import json

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader


# ============================================================
# AI HAIR INTELLIGENCE PROJECT
# HAIR TYPE CLASSIFICATION
# ============================================================

BASE_DIR = Path(
    r"C:\Users\kalya\OneDrive\Desktop\hair_analysis"
)

DATA_DIR = (
    BASE_DIR
    / "processed_datasets"
    / "hair_type"
)

MODEL_DIR = (
    BASE_DIR
    / "models"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ------------------------------------------------------------
# Settings
# ------------------------------------------------------------

IMAGE_SIZE = 224

BATCH_SIZE = 16

NUM_EPOCHS = 15

LEARNING_RATE = 0.0001

NUM_WORKERS = 0

RANDOM_SEED = 42

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

torch.manual_seed(RANDOM_SEED)


# ============================================================
# DATA TRANSFORMS
# ============================================================

train_transforms = transforms.Compose([
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    transforms.RandomRotation(
        10
    ),

    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15,
        saturation=0.10
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


eval_transforms = transforms.Compose([
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
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
# LOAD DATASETS
# ============================================================

train_dir = DATA_DIR / "train"
val_dir = DATA_DIR / "val"
test_dir = DATA_DIR / "test"


print("=" * 70)
print("AI HAIR INTELLIGENCE PROJECT")
print("HAIR TYPE CLASSIFICATION")
print("=" * 70)

print(
    f"\nDevice: {DEVICE}"
)

print(
    f"Dataset: {DATA_DIR}"
)


train_dataset = datasets.ImageFolder(
    train_dir,
    transform=train_transforms
)

val_dataset = datasets.ImageFolder(
    val_dir,
    transform=eval_transforms
)

test_dataset = datasets.ImageFolder(
    test_dir,
    transform=eval_transforms
)


# ============================================================
# CLASS INFORMATION
# ============================================================

class_names = train_dataset.classes

num_classes = len(
    class_names
)

print(
    f"\nClasses: {class_names}"
)

print(
    f"Number of classes: {num_classes}"
)

print(
    f"Training images: "
    f"{len(train_dataset)}"
)

print(
    f"Validation images: "
    f"{len(val_dataset)}"
)

print(
    f"Test images: "
    f"{len(test_dataset)}"
)


# ============================================================
# DATA LOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)


# ============================================================
# MODEL
# ============================================================

print(
    "\nLoading ResNet18..."
)

weights = models.ResNet18_Weights.DEFAULT

model = models.resnet18(
    weights=weights
)

# Replace final classification layer
model.fc = nn.Linear(
    model.fc.in_features,
    num_classes
)

model = model.to(
    DEVICE
)


# ============================================================
# LOSS / OPTIMIZER
# ============================================================

criterion = nn.CrossEntropyLoss()

optimizer = optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=0.0001
)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=2
)


# ============================================================
# TRAINING
# ============================================================

best_val_accuracy = 0.0

best_model_state = copy.deepcopy(
    model.state_dict()
)


for epoch in range(
    NUM_EPOCHS
):

    print(
        f"\nEpoch "
        f"{epoch + 1}/{NUM_EPOCHS}"
    )

    print(
        "-" * 50
    )

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(
            DEVICE
        )

        labels = labels.to(
            DEVICE
        )

        optimizer.zero_grad()

        outputs = model(
            images
        )

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += (
            loss.item()
            * images.size(0)
        )

        predictions = (
            outputs.argmax(
                dim=1
            )
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    train_loss = (
        running_loss / total
    )

    train_accuracy = (
        correct / total
    )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    val_loss_total = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(
                DEVICE
            )

            labels = labels.to(
                DEVICE
            )

            outputs = model(
                images
            )

            loss = criterion(
                outputs,
                labels
            )

            val_loss_total += (
                loss.item()
                * images.size(0)
            )

            predictions = (
                outputs.argmax(
                    dim=1
                )
            )

            val_correct += (
                predictions == labels
            ).sum().item()

            val_total += labels.size(0)

    val_loss = (
        val_loss_total
        / val_total
    )

    val_accuracy = (
        val_correct
        / val_total
    )

    scheduler.step(
        val_accuracy
    )

    print(
        f"Train Loss: "
        f"{train_loss:.4f}"
    )

    print(
        f"Train Accuracy: "
        f"{train_accuracy * 100:.2f}%"
    )

    print(
        f"Val Loss: "
        f"{val_loss:.4f}"
    )

    print(
        f"Val Accuracy: "
        f"{val_accuracy * 100:.2f}%"
    )

    # --------------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------------

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = (
            val_accuracy
        )

        best_model_state = (
            copy.deepcopy(
                model.state_dict()
            )
        )

        print(
            "✓ New best model"
        )


# ============================================================
# LOAD BEST MODEL
# ============================================================

model.load_state_dict(
    best_model_state
)


# ============================================================
# TEST
# ============================================================

print("\n" + "=" * 70)
print("FINAL TEST")
print("=" * 70)

model.eval()

test_correct = 0
test_total = 0

class_correct = [
    0
] * num_classes

class_total = [
    0
] * num_classes

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(
            DEVICE
        )

        labels = labels.to(
            DEVICE
        )

        outputs = model(
            images
        )

        predictions = (
            outputs.argmax(
                dim=1
            )
        )

        test_correct += (
            predictions == labels
        ).sum().item()

        test_total += labels.size(0)

        for label, prediction in zip(
            labels,
            predictions
        ):

            label_index = (
                label.item()
            )

            class_total[
                label_index
            ] += 1

            if label == prediction:

                class_correct[
                    label_index
                ] += 1


test_accuracy = (
    test_correct
    / test_total
)

print(
    f"\nTest Accuracy: "
    f"{test_accuracy * 100:.2f}%"
)

print(
    "\nPer-class accuracy:"
)

for index, class_name in enumerate(
    class_names
):

    if class_total[index] > 0:

        accuracy = (
            class_correct[index]
            / class_total[index]
        )

    else:

        accuracy = 0.0

    print(
        f"  {class_name:<20} "
        f"{accuracy * 100:.2f}% "
        f"({class_correct[index]}/"
        f"{class_total[index]})"
    )


# ============================================================
# SAVE MODEL
# ============================================================

model_path = (
    MODEL_DIR
    / "hair_type_resnet18.pth"
)

torch.save(
    {
        "model_state_dict":
            model.state_dict(),

        "class_names":
            class_names,

        "image_size":
            IMAGE_SIZE,

        "test_accuracy":
            test_accuracy,
    },
    model_path
)


# ============================================================
# SAVE CLASS INFORMATION
# ============================================================

class_info_path = (
    MODEL_DIR
    / "hair_type_classes.json"
)

with open(
    class_info_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        {
            "classes": class_names,
            "image_size": IMAGE_SIZE
        },
        file,
        indent=4
    )


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("HAIR TYPE TRAINING COMPLETED")
print("=" * 70)

print(
    f"\nBest validation accuracy: "
    f"{best_val_accuracy * 100:.2f}%"
)

print(
    f"Test accuracy: "
    f"{test_accuracy * 100:.2f}%"
)

print(
    f"\nModel saved to:"
)

print(
    model_path
)

print(
    "\nClass information saved to:"
)

print(
    class_info_path
)