import os
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader  # type: ignore[reportMissingImports]
from torchvision import transforms


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = r"C:\Users\kalya\OneDrive\Desktop\hair_analysis"

DATA_DIR = os.path.join(
    BASE_DIR,
    "processed_datasets",
    "hair_segmentation"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "hair_segmentation_unet.pth"
)

IMAGE_SIZE = 224
BATCH_SIZE = 4
NUM_WORKERS = 0


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("HAIR SEGMENTATION MODEL EVALUATION")
print("=" * 60)

print(f"PyTorch version: {torch.__version__}")
print(f"Device: {device}")


# ============================================================
# DATASET
# ============================================================

class HairSegmentationDataset(Dataset):

    def __init__(self, root_dir, image_size=224):

        self.image_dir = os.path.join(
            root_dir,
            "images"
        )

        self.mask_dir = os.path.join(
            root_dir,
            "masks"
        )

        self.image_size = image_size

        self.images = sorted([
            f for f in os.listdir(self.image_dir)
            if f.lower().endswith(
                (".jpg", ".jpeg", ".png")
            )
        ])

        self.pairs = []

        for image_name in self.images:

            base_name = os.path.splitext(
                image_name
            )[0]

            mask_name = base_name + ".png"

            mask_path = os.path.join(
                self.mask_dir,
                mask_name
            )

            if os.path.exists(mask_path):

                self.pairs.append(
                    (
                        os.path.join(
                            self.image_dir,
                            image_name
                        ),
                        mask_path
                    )
                )

        print(
            f"Found {len(self.pairs)} "
            f"image/mask pairs."
        )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):

        image_path, mask_path = self.pairs[index]

        image = Image.open(
            image_path
        ).convert("RGB")

        mask = Image.open(
            mask_path
        ).convert("L")

        image = image.resize(
            (self.image_size, self.image_size),
            Image.Resampling.BILINEAR
        )

        mask = mask.resize(
            (self.image_size, self.image_size),
            Image.Resampling.NEAREST
        )

        image = transforms.ToTensor()(image)

        image = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )(image)

        mask = transforms.ToTensor()(mask)

        mask = (mask > 0.5).float()

        return image, mask


# ============================================================
# U-NET
# ============================================================

class DoubleConv(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels
    ):

        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(
                out_channels
            ),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(
                out_channels
            ),

            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):

    def __init__(self):

        super().__init__()

        self.enc1 = DoubleConv(3, 32)

        self.enc2 = DoubleConv(
            32,
            64
        )

        self.enc3 = DoubleConv(
            64,
            128
        )

        self.enc4 = DoubleConv(
            128,
            256
        )

        self.pool = nn.MaxPool2d(
            2,
            2
        )

        self.bottleneck = DoubleConv(
            256,
            512
        )

        self.up4 = nn.ConvTranspose2d(
            512,
            256,
            2,
            2
        )

        self.dec4 = DoubleConv(
            512,
            256
        )

        self.up3 = nn.ConvTranspose2d(
            256,
            128,
            2,
            2
        )

        self.dec3 = DoubleConv(
            256,
            128
        )

        self.up2 = nn.ConvTranspose2d(
            128,
            64,
            2,
            2
        )

        self.dec2 = DoubleConv(
            128,
            64
        )

        self.up1 = nn.ConvTranspose2d(
            64,
            32,
            2,
            2
        )

        self.dec1 = DoubleConv(
            64,
            32
        )

        self.output = nn.Conv2d(
            32,
            1,
            1
        )

    def forward(self, x):

        e1 = self.enc1(x)

        e2 = self.enc2(
            self.pool(e1)
        )

        e3 = self.enc3(
            self.pool(e2)
        )

        e4 = self.enc4(
            self.pool(e3)
        )

        b = self.bottleneck(
            self.pool(e4)
        )

        d4 = self.up4(b)

        d4 = torch.cat(
            [d4, e4],
            dim=1
        )

        d4 = self.dec4(d4)

        d3 = self.up3(d4)

        d3 = torch.cat(
            [d3, e3],
            dim=1
        )

        d3 = self.dec3(d3)

        d2 = self.up2(d3)

        d2 = torch.cat(
            [d2, e2],
            dim=1
        )

        d2 = self.dec2(d2)

        d1 = self.up1(d2)

        d1 = torch.cat(
            [d1, e1],
            dim=1
        )

        d1 = self.dec1(d1)

        return self.output(d1)


# ============================================================
# METRICS
# ============================================================

def dice_score(predictions, targets):

    predictions = (
        torch.sigmoid(predictions) > 0.5
    ).float()

    predictions = predictions.view(
        predictions.size(0),
        -1
    )

    targets = targets.view(
        targets.size(0),
        -1
    )

    intersection = (
        predictions * targets
    ).sum(dim=1)

    dice = (
        2.0 * intersection + 1.0
    ) / (
        predictions.sum(dim=1)
        + targets.sum(dim=1)
        + 1.0
    )

    return dice.mean().item()


def iou_score(predictions, targets):

    predictions = (
        torch.sigmoid(predictions) > 0.5
    ).float()

    predictions = predictions.view(
        predictions.size(0),
        -1
    )

    targets = targets.view(
        targets.size(0),
        -1
    )

    intersection = (
        predictions * targets
    ).sum(dim=1)

    union = (
        predictions
        + targets
        - predictions * targets
    ).sum(dim=1)

    iou = (
        intersection + 1.0
    ) / (
        union + 1.0
    )

    return iou.mean().item()


# ============================================================
# LOAD TEST DATA
# ============================================================

print("\nLoading test dataset...")

test_dataset = HairSegmentationDataset(
    os.path.join(
        DATA_DIR,
        "test"
    ),
    IMAGE_SIZE
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)

print(
    f"Test pairs: {len(test_dataset)}"
)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading trained U-Net...")

model = UNet().to(device)

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

    if "best_val_dice" in checkpoint:

        print(
            f"Saved best validation Dice: "
            f"{checkpoint['best_val_dice']:.4f}"
        )

else:

    model.load_state_dict(
        checkpoint
    )

model.eval()

print("Model loaded successfully.")


# ============================================================
# TEST
# ============================================================

print("\nRunning test evaluation...")

total_dice = 0.0
total_iou = 0.0

with torch.no_grad():

    for batch_idx, (
        images,
        masks
    ) in enumerate(test_loader):

        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)

        batch_dice = dice_score(
            outputs,
            masks
        )

        batch_iou = iou_score(
            outputs,
            masks
        )

        total_dice += (
            batch_dice
            * images.size(0)
        )

        total_iou += (
            batch_iou
            * images.size(0)
        )

        if (
            batch_idx + 1
        ) % 10 == 0:

            print(
                f"Processed batch "
                f"{batch_idx + 1}/"
                f"{len(test_loader)}"
            )


# ============================================================
# FINAL RESULTS
# ============================================================

test_dice = (
    total_dice
    / len(test_dataset)
)

test_iou = (
    total_iou
    / len(test_dataset)
)

print("\n" + "=" * 60)
print("FINAL TEST RESULTS")
print("=" * 60)

print(
    f"Test Dice Score: "
    f"{test_dice:.4f}"
)

print(
    f"Test IoU Score: "
    f"{test_iou:.4f}"
)

print(
    f"Test Dice (%): "
    f"{test_dice * 100:.2f}%"
)

print(
    f"Test IoU (%): "
    f"{test_iou * 100:.2f}%"
)

print("\nEvaluation complete.")