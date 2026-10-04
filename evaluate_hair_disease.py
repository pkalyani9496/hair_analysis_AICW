import os
import json
import torch
import torch.nn.functional as F
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
    "hair_disease"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "hair_disease_resnet18.pth"
)

CLASSES_PATH = os.path.join(
    BASE_DIR,
    "models",
    "hair_disease_classes.json"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "models",
    "disease_evaluation"
)

IMAGE_SIZE = 224
BATCH_SIZE = 32
VALIDATION_TARGET_PRECISION = 0.95
MINIMUM_CLASS_ACCEPTED = 20
NO_ACCEPTANCE_THRESHOLD = 1.01

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# DEVICE
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("HAIR DISEASE MODEL EVALUATION")
print("=" * 60)

print(f"PyTorch version: {torch.__version__}")
print(f"Device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("GPU: Not available - using CPU")

# ============================================================
# LOAD CLASS NAMES
# ============================================================

with open(CLASSES_PATH, "r", encoding="utf-8") as f:
    class_data = json.load(f)

# Handle either a list or dictionary format
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
    raise ValueError("Unknown class JSON format.")

print("\nClasses:")
for i, name in enumerate(class_names):
    print(f"{i}: {name}")

# ============================================================
# TRANSFORMS
# ============================================================

test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ============================================================
# LOAD VALIDATION AND TEST DATASETS
# ============================================================

validation_dir = os.path.join(DATA_DIR, "val")
test_dir = os.path.join(DATA_DIR, "test")

validation_dataset = datasets.ImageFolder(
    validation_dir,
    transform=test_transform
)

test_dataset = datasets.ImageFolder(
    test_dir,
    transform=test_transform
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print(f"\nValidation images: {len(validation_dataset)}")
print(f"Test images: {len(test_dataset)}")

expected_class_order = [
    name
    for name, index in sorted(
        validation_dataset.class_to_idx.items(),
        key=lambda item: item[1]
    )
]

if validation_dataset.class_to_idx != test_dataset.class_to_idx:
    raise ValueError("Validation and test class ordering does not match.")

if expected_class_order != class_names:
    raise ValueError(
        "Dataset class ordering does not match the model class file."
    )

print("\nDataset class ordering:")
print(validation_dataset.class_to_idx)

# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading ResNet18 model...")

model = models.resnet18(weights=None)

model.fc = torch.nn.Linear(
    model.fc.in_features,
    len(class_names)
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

# Support both state_dict formats
if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)

model = model.to(device)
model.eval()

print("Model loaded successfully.")

# ============================================================
# VALIDATION CALIBRATION
# ============================================================

def collect_logits(loader):

    logits = []
    labels = []

    with torch.inference_mode():

        for images, batch_labels in loader:

            outputs = model(images.to(device))
            logits.append(outputs.cpu())
            labels.append(batch_labels.cpu())

    return torch.cat(logits), torch.cat(labels)


def fit_temperature(logits, labels):

    log_temperature = torch.nn.Parameter(
        torch.zeros((), device=logits.device)
    )

    optimizer = torch.optim.LBFGS(
        [log_temperature],
        lr=0.1,
        max_iter=50,
    )

    def closure():

        optimizer.zero_grad()

        temperature = torch.exp(
            log_temperature
        ).clamp(0.05, 20.0)

        loss = F.cross_entropy(
            logits / temperature,
            labels,
        )

        loss.backward()
        return loss

    optimizer.step(closure)

    return float(
        torch.exp(log_temperature.detach())
        .clamp(0.05, 20.0)
        .item()
    )


def select_class_thresholds(probabilities, labels):

    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    thresholds = {}
    validation_stats = {}

    for class_index, class_name in enumerate(class_names):

        class_predictions = predictions == class_index
        class_confidence = confidence[class_predictions]
        class_correct = (
            labels[class_predictions] == class_index
        )

        threshold = NO_ACCEPTANCE_THRESHOLD
        accepted_count = 0
        accepted_precision = None

        for candidate in np.unique(class_confidence):

            accepted = class_confidence >= candidate
            count = int(np.sum(accepted))

            if count < MINIMUM_CLASS_ACCEPTED:
                continue

            precision = float(
                np.mean(class_correct[accepted])
            )

            if precision >= VALIDATION_TARGET_PRECISION:
                threshold = float(candidate)
                accepted_count = count
                accepted_precision = precision
                break

        thresholds[class_name] = threshold
        validation_stats[class_name] = {
            "accepted": accepted_count,
            "precision": accepted_precision,
        }

    return thresholds, validation_stats


print("\nRunning validation predictions for calibration...")
validation_logits, validation_labels = collect_logits(
    validation_loader
)

temperature = fit_temperature(
    validation_logits,
    validation_labels,
)

validation_probabilities = torch.softmax(
    validation_logits / temperature,
    dim=1,
).numpy()

thresholds, validation_stats = select_class_thresholds(
    validation_probabilities,
    validation_labels.numpy(),
)

calibration = {
    "method": "temperature_scaling",
    "temperature": temperature,
    "target_precision": VALIDATION_TARGET_PRECISION,
    "minimum_class_accepted": MINIMUM_CLASS_ACCEPTED,
    "no_acceptance_threshold": NO_ACCEPTANCE_THRESHOLD,
    "class_confidence_thresholds": thresholds,
    "validation": validation_stats,
}

calibration_path = os.path.join(
    OUTPUT_DIR,
    "calibration.json"
)

with open(calibration_path, "w", encoding="utf-8") as f:
    json.dump(calibration, f, indent=2)

print(f"Fitted temperature: {temperature:.4f}")
print(
    f"Calibration settings saved to:\n{calibration_path}"
)

for class_name, stats in validation_stats.items():
    print(
        f"{class_name}: threshold="
        f"{thresholds[class_name]:.4f}, "
        f"validation accepted={stats['accepted']}, "
        f"precision={stats['precision']}"
    )

# ============================================================
# TEST PREDICTIONS
# ============================================================

print("\nRunning predictions...")

test_logits, test_labels = collect_logits(test_loader)

# ============================================================
# NUMPY ARRAYS
# ============================================================

all_labels = test_labels.numpy()
all_predictions = test_logits.argmax(dim=1).numpy()

test_probabilities = torch.softmax(
    test_logits / temperature,
    dim=1,
).numpy()

test_confidence = test_probabilities.max(axis=1)
test_thresholds = np.array(
    [thresholds[class_names[index]] for index in all_predictions]
)
test_accepted = test_confidence >= test_thresholds
test_coverage = float(np.mean(test_accepted))
selective_accuracy = (
    float(np.mean(all_predictions[test_accepted] == all_labels[test_accepted]))
    if np.any(test_accepted)
    else None
)

selective_class_stats = {}

for class_index, class_name in enumerate(class_names):

    class_predictions = all_predictions == class_index
    accepted = class_predictions & test_accepted
    accepted_count = int(np.sum(accepted))
    predicted_count = int(np.sum(class_predictions))

    selective_class_stats[class_name] = {
        "accepted": accepted_count,
        "coverage": (
            accepted_count / predicted_count
            if predicted_count
            else 0.0
        ),
        "precision": (
            float(np.mean(all_labels[accepted] == class_index))
            if accepted_count
            else None
        ),
    }

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
    f"Test Accuracy: {accuracy * 100:.2f}%"
)
print(f"Test coverage at validation cutoffs: {test_coverage * 100:.2f}%")
print(
    "Selective test accuracy: "
    + (
        f"{selective_accuracy * 100:.2f}%"
        if selective_accuracy is not None
        else "no predictions accepted"
    )
)

print("\nSelective results by predicted class:")

for class_name, stats in selective_class_stats.items():
    precision_text = (
        f"{stats['precision'] * 100:.2f}%"
        if stats["precision"] is not None
        else "no predictions accepted"
    )
    print(
        f"{class_name}: coverage={stats['coverage'] * 100:.2f}%, "
        f"precision={precision_text}, "
        f"accepted={stats['accepted']}"
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

report += (
    "\nSelective prediction results\n"
    f"Test coverage: {test_coverage:.4f}\n"
    "Selective test accuracy: "
    + (
        f"{selective_accuracy:.4f}\n"
        if selective_accuracy is not None
        else "no predictions accepted\n"
    )
    + "\nSelective results by predicted class\n"
    + "\n".join(
        f"{class_name}: coverage={stats['coverage']:.4f}, "
        f"precision={stats['precision']}, accepted={stats['accepted']}"
        for class_name, stats in selective_class_stats.items()
    )
    + "\n"
)

print("\nClassification Report:")
print(report)

# Save report
report_path = os.path.join(
    OUTPUT_DIR,
    "classification_report.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as f:

    f.write("HAIR DISEASE MODEL EVALUATION\n")
    f.write("=" * 60 + "\n\n")

    f.write(
        f"Test Accuracy: {accuracy * 100:.2f}%\n\n"
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

# Save raw confusion matrix
cm_path = os.path.join(
    OUTPUT_DIR,
    "confusion_matrix.npy"
)

np.save(cm_path, cm)

# ============================================================
# PLOT CONFUSION MATRIX
# ============================================================

fig, ax = plt.subplots(
    figsize=(14, 12)
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
    "Hair Disease Model - Confusion Matrix"
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

for i, class_name in enumerate(class_names):

    total = np.sum(all_labels == i)
    correct = np.sum(
        (all_labels == i) &
        (all_predictions == i)
    )

    if total > 0:
        class_accuracy = correct / total
    else:
        class_accuracy = 0

    print(
        f"{class_name}: "
        f"{class_accuracy * 100:.2f}% "
        f"({correct}/{total})"
    )

# ============================================================
# MOST COMMON CONFUSIONS
# ============================================================

print("\n" + "=" * 60)
print("MOST COMMON CLASS CONFUSIONS")
print("=" * 60)

confusions = []

for true_class in range(len(class_names)):

    for predicted_class in range(len(class_names)):

        if true_class == predicted_class:
            continue

        count = cm[true_class, predicted_class]

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

for count, true_name, predicted_name in confusions[:15]:

    print(
        f"{true_name} -> {predicted_name}: "
        f"{count} images"
    )

print("\n" + "=" * 60)
print("EVALUATION COMPLETE")
print("=" * 60)

print(
    f"\nAll evaluation files saved in:\n"
    f"{OUTPUT_DIR}"
)