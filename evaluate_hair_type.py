import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score
)

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = r"C:\Users\kalya\OneDrive\Desktop\hair_analysis"

DATA_DIR = os.path.join(
    BASE_DIR,
    "processed_datasets",
    "hair_type"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "hair_type_resnet18.pth"
)

CLASSES_PATH = os.path.join(
    BASE_DIR,
    "models",
    "hair_type_classes.json"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "models",
    "hair_type_evaluation"
)

IMAGE_SIZE = 224
BATCH_SIZE = 32

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("HAIR TYPE MODEL EVALUATION")
print("=" * 60)

print(f"PyTorch version: {torch.__version__}")
print(f"Device: {device}")

if torch.cuda.is_available():
    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )
else:
    print("GPU: Not available - using CPU")

# ============================================================
# LOAD CLASS NAMES
# ============================================================

with open(
    CLASSES_PATH,
    "r",
    encoding="utf-8"
) as f:
    class_data = json.load(f)

if isinstance(class_data, list):
    class_names = class_data

elif isinstance(class_data, dict):

    if "classes" in class_data:
        class_names = class_data["classes"]

    elif "class_names" in class_data:
        class_names = class_data["class_names"]

    else:
        class_names = list(class_data.values())

else:
    raise ValueError(
        "Unknown class JSON format."
    )

print("\nClasses:")

for i, name in enumerate(class_names):
    print(f"{i}: {name}")

# ============================================================
# TEST TRANSFORMS
# ============================================================

test_transform = transforms.Compose([
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ============================================================
# LOAD TEST DATA
# ============================================================

test_dir = os.path.join(
    DATA_DIR,
    "test"
)

test_dataset = datasets.ImageFolder(
    test_dir,
    transform=test_transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print(
    f"\nTest images: {len(test_dataset)}"
)

print("\nDataset class ordering:")

print(test_dataset.class_to_idx)

# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading ResNet18 model...")

model = models.resnet18(
    weights=None
)

model.fc = torch.nn.Linear(
    model.fc.in_features,
    len(class_names)
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

if (
    isinstance(checkpoint, dict)
    and "model_state_dict" in checkpoint
):
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
else:
    model.load_state_dict(checkpoint)

model = model.to(device)

model.eval()

print("Model loaded successfully.")

# ============================================================
# RUN PREDICTIONS
# ============================================================

all_labels = []
all_predictions = []

print("\nRunning predictions...")

with torch.no_grad():

    for batch_idx, (images, labels) in enumerate(
        test_loader
    ):

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        all_labels.extend(
            labels.cpu().numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        if (batch_idx + 1) % 5 == 0:

            print(
                f"Processed batch "
                f"{batch_idx + 1}/"
                f"{len(test_loader)}"
            )

# ============================================================
# CONVERT TO NUMPY
# ============================================================

all_labels = np.array(
    all_labels
)

all_predictions = np.array(
    all_predictions
)

# ============================================================
# ACCURACY
# ============================================================

accuracy = accuracy_score(
    all_labels,
    all_predictions
)

print("\n" + "=" * 60)
print("OVERALL RESULTS")
print("=" * 60)

print(
    f"Test Accuracy: "
    f"{accuracy * 100:.2f}%"
)

# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    all_labels,
    all_predictions,
    target_names=class_names,
    digits=4,
    zero_division=0
)

print("\nClassification Report:")
print(report)

report_path = os.path.join(
    OUTPUT_DIR,
    "classification_report.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "HAIR TYPE MODEL EVALUATION\n"
    )

    f.write("=" * 60 + "\n\n")

    f.write(
        f"Test Accuracy: "
        f"{accuracy * 100:.2f}%\n\n"
    )

    f.write(report)

print(
    f"Classification report saved to:\n"
    f"{report_path}"
)

# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    all_labels,
    all_predictions
)

print("\nConfusion Matrix:")
print(cm)

cm_path = os.path.join(
    OUTPUT_DIR,
    "confusion_matrix.npy"
)

np.save(
    cm_path,
    cm
)

# ============================================================
# CONFUSION MATRIX IMAGE
# ============================================================

fig, ax = plt.subplots(
    figsize=(9, 8)
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

disp.plot(
    ax=ax,
    xticks_rotation=45,
    cmap="Blues",
    values_format="d"
)

plt.title(
    "Hair Type Model - Confusion Matrix"
)

plt.tight_layout()

cm_image_path = os.path.join(
    OUTPUT_DIR,
    "confusion_matrix.png"
)

plt.savefig(
    cm_image_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close()

print(
    f"Confusion matrix image saved to:\n"
    f"{cm_image_path}"
)

# ============================================================
# PER-CLASS ACCURACY
# ============================================================

print("\n" + "=" * 60)
print("PER-CLASS ACCURACY")
print("=" * 60)

for i, class_name in enumerate(
    class_names
):

    total = np.sum(
        all_labels == i
    )

    correct = np.sum(
        (all_labels == i)
        &
        (all_predictions == i)
    )

    if total > 0:
        class_accuracy = (
            correct / total
        )
    else:
        class_accuracy = 0

    print(
        f"{class_name}: "
        f"{class_accuracy * 100:.2f}% "
        f"({correct}/{total})"
    )

# ============================================================
# COMMON CONFUSIONS
# ============================================================

print("\n" + "=" * 60)
print("MOST COMMON CLASS CONFUSIONS")
print("=" * 60)

confusions = []

for true_class in range(
    len(class_names)
):

    for predicted_class in range(
        len(class_names)
    ):

        if true_class == predicted_class:
            continue

        count = cm[
            true_class,
            predicted_class
        ]

        if count > 0:

            confusions.append(
                (
                    count,
                    class_names[true_class],
                    class_names[predicted_class]
                )
            )

confusions.sort(
    reverse=True
)

for (
    count,
    true_name,
    predicted_name
) in confusions:

    print(
        f"{true_name} -> "
        f"{predicted_name}: "
        f"{count} images"
    )

print("\n" + "=" * 60)
print("EVALUATION COMPLETE")
print("=" * 60)

print(
    f"\nAll evaluation files saved in:\n"
    f"{OUTPUT_DIR}"
)