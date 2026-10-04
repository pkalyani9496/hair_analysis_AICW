import os
import json
import copy
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = r"C:\Users\kalya\OneDrive\Desktop\hair_analysis"

DATA_DIR = os.path.join(
    BASE_DIR,
    "processed_datasets",
    "hair_disease"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
TEST_DIR = os.path.join(DATA_DIR, "test")

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "hair_disease_resnet18.pth"
)

CLASS_PATH = os.path.join(
    MODEL_DIR,
    "hair_disease_classes.json"
)

IMAGE_SIZE = 224

# CPU-friendly settings
BATCH_SIZE = 8
NUM_EPOCHS = 10

LEARNING_RATE = 0.001

NUM_WORKERS = 0


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("AI HAIR INTELLIGENCE PROJECT")
print("HAIR DISEASE CLASSIFICATION")
print("=" * 70)

print()
print(f"PyTorch version: {torch.__version__}")
print(f"Device: {device}")

if torch.cuda.is_available():
    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )
else:
    print("GPU not available. Using CPU.")

print()


# ============================================================
# CHECK DATASET
# ============================================================

for directory in [
    TRAIN_DIR,
    VAL_DIR,
    TEST_DIR
]:

    if not os.path.exists(directory):

        raise FileNotFoundError(
            f"Directory not found:\n{directory}"
        )


# ============================================================
# TRANSFORMS
# ============================================================

train_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    transforms.RandomRotation(
        degrees=10
    ),

    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15,
        saturation=0.15
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


eval_transform = transforms.Compose([

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

print("Loading datasets...")

train_dataset = datasets.ImageFolder(
    TRAIN_DIR,
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    VAL_DIR,
    transform=eval_transform
)

test_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=eval_transform
)


# ============================================================
# CLASSES
# ============================================================

class_names = train_dataset.classes

num_classes = len(class_names)

print()
print("Classes:")

for i, class_name in enumerate(
    class_names
):

    print(
        f"  {i}: {class_name}"
    )

print()

print(
    f"Number of classes: {num_classes}"
)

print(
    f"Training images:   {len(train_dataset)}"
)

print(
    f"Validation images: {len(val_dataset)}"
)

print(
    f"Test images:       {len(test_dataset)}"
)

print()


# ============================================================
# VERIFY CLASSES
# ============================================================

if (
    train_dataset.classes
    != val_dataset.classes
):

    raise RuntimeError(
        "Train and validation classes do not match."
    )


if (
    train_dataset.classes
    != test_dataset.classes
):

    raise RuntimeError(
        "Train and test classes do not match."
    )


# ============================================================
# SAVE CLASS INFORMATION
# ============================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

with open(
    CLASS_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        {
            "classes": class_names,
            "num_classes": num_classes
        },
        f,
        indent=4
    )

print(
    f"Class information saved to:\n{CLASS_PATH}"
)

print()


# ============================================================
# DATA LOADERS
# ============================================================

train_loader = DataLoader(

    train_dataset,

    batch_size=BATCH_SIZE,

    shuffle=True,

    num_workers=NUM_WORKERS,

    pin_memory=False
)


val_loader = DataLoader(

    val_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=NUM_WORKERS,

    pin_memory=False
)


test_loader = DataLoader(

    test_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=NUM_WORKERS,

    pin_memory=False
)


# ============================================================
# LOAD RESNET18
# ============================================================

print("Loading pretrained ResNet18...")

weights = models.ResNet18_Weights.DEFAULT

model = models.resnet18(
    weights=weights
)


# ============================================================
# FREEZE BACKBONE
# ============================================================

print("Freezing ResNet18 backbone...")

for parameter in model.parameters():

    parameter.requires_grad = False


# ============================================================
# REPLACE CLASSIFIER
# ============================================================

in_features = model.fc.in_features

model.fc = nn.Linear(
    in_features,
    num_classes
)

model = model.to(device)

print(
    "Classifier configured."
)

print()


# ============================================================
# LOSS
# ============================================================

criterion = nn.CrossEntropyLoss()


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(

    model.fc.parameters(),

    lr=LEARNING_RATE,

    weight_decay=0.01
)


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch():

    model.train()

    running_loss = 0.0

    correct = 0

    total = 0

    total_batches = len(train_loader)

    print()

    print(
        f"Training {total_batches} batches..."
    )

    for batch_index, (
        images,
        labels
    ) in enumerate(train_loader):

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

        running_loss += (
            loss.item()
            * images.size(0)
        )

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

        # Print every 25 batches
        if (
            (batch_index + 1) % 25 == 0
            or
            (batch_index + 1) == total_batches
        ):

            accuracy = (
                correct / total
            ) * 100

            print(
                f"  Batch "
                f"{batch_index + 1:4d}/"
                f"{total_batches} "
                f"| Loss: {loss.item():.4f} "
                f"| Accuracy: {accuracy:.2f}%",
                flush=True
            )

    epoch_loss = (
        running_loss / total
    )

    epoch_accuracy = (
        correct / total
    ) * 100

    return (
        epoch_loss,
        epoch_accuracy
    )


# ============================================================
# EVALUATION
# ============================================================

def evaluate(loader):

    model.eval()

    running_loss = 0.0

    correct = 0

    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)

            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            running_loss += (
                loss.item()
                * images.size(0)
            )

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    loss_value = (
        running_loss / total
    )

    accuracy = (
        correct / total
    ) * 100

    return (
        loss_value,
        accuracy
    )


# ============================================================
# TRAINING
# ============================================================

print("=" * 70)
print("STARTING TRAINING")
print("=" * 70)

print()

best_val_accuracy = 0.0

best_model_weights = copy.deepcopy(
    model.state_dict()
)

training_start = time.time()

for epoch in range(NUM_EPOCHS):

    epoch_start = time.time()

    print()
    print("=" * 70)

    print(
        f"EPOCH {epoch + 1}/{NUM_EPOCHS}"
    )

    print("=" * 70)

    train_loss, train_accuracy = (
        train_one_epoch()
    )

    print()
    print("Running validation...")

    val_loss, val_accuracy = evaluate(
        val_loader
    )

    epoch_time = (
        time.time() - epoch_start
    )

    print()

    print(
        f"Train Loss: "
        f"{train_loss:.4f}"
    )

    print(
        f"Train Accuracy: "
        f"{train_accuracy:.2f}%"
    )

    print(
        f"Validation Loss: "
        f"{val_loss:.4f}"
    )

    print(
        f"Validation Accuracy: "
        f"{val_accuracy:.2f}%"
    )

    print(
        f"Epoch Time: "
        f"{epoch_time / 60:.2f} minutes"
    )

    # --------------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------------

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        best_model_weights = copy.deepcopy(
            model.state_dict()
        )

        torch.save(
            {
                "model_state_dict":
                    best_model_weights,

                "class_names":
                    class_names,

                "num_classes":
                    num_classes,

                "image_size":
                    IMAGE_SIZE
            },
            MODEL_PATH
        )

        print()
        print(
            f"✓ New best model saved!"
        )

        print(
            f"  Validation Accuracy: "
            f"{val_accuracy:.2f}%"
        )


# ============================================================
# LOAD BEST MODEL
# ============================================================

print()
print("=" * 70)
print("LOADING BEST MODEL")
print("=" * 70)

model.load_state_dict(
    best_model_weights
)


# ============================================================
# TEST
# ============================================================

print()
print("=" * 70)
print("FINAL TEST")
print("=" * 70)

test_loss, test_accuracy = evaluate(
    test_loader
)

print()

print(
    f"Test Loss: "
    f"{test_loss:.4f}"
)

print(
    f"Test Accuracy: "
    f"{test_accuracy:.2f}%"
)


# ============================================================
# PER-CLASS ACCURACY
# ============================================================

print()
print("=" * 70)
print("PER-CLASS TEST ACCURACY")
print("=" * 70)

model.eval()

class_correct = [
    0 for _ in range(num_classes)
]

class_total = [
    0 for _ in range(num_classes)
]

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        labels = labels.to(device)

        outputs = model(images)

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        for label, prediction in zip(
            labels,
            predictions
        ):

            label_index = label.item()

            class_total[
                label_index
            ] += 1

            if label == prediction:

                class_correct[
                    label_index
                ] += 1


for i, class_name in enumerate(
    class_names
):

    if class_total[i] > 0:

        class_accuracy = (
            class_correct[i]
            / class_total[i]
        ) * 100

    else:

        class_accuracy = 0.0

    print(
        f"{class_name:<25} "
        f"{class_accuracy:6.2f}% "
        f"("
        f"{class_correct[i]}/"
        f"{class_total[i]}"
        f")"
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

training_time = (
    time.time() - training_start
)

print()
print("=" * 70)
print("HAIR DISEASE TRAINING COMPLETED")
print("=" * 70)

print()

print(
    f"Best validation accuracy: "
    f"{best_val_accuracy:.2f}%"
)

print(
    f"Test accuracy: "
    f"{test_accuracy:.2f}%"
)

print(
    f"Training time: "
    f"{training_time / 60:.2f} minutes"
)

print()

print("Model saved to:")

print(MODEL_PATH)

print()

print("Class information saved to:")

print(CLASS_PATH)

print()

print("=" * 70)
print("DONE")
print("=" * 70)