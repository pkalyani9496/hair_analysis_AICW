import os
import time
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
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

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "hair_segmentation_unet.pth"
)

IMAGE_SIZE = 224
BATCH_SIZE = 4
NUM_EPOCHS = 15
LEARNING_RATE = 0.001
NUM_WORKERS = 0

os.makedirs(MODEL_DIR, exist_ok=True)

# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("HAIR SEGMENTATION MODEL TRAINING")
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
            f"image/mask pairs in "
            f"{root_dir}"
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

        # Resize image
        image = image.resize(
            (self.image_size, self.image_size),
            Image.Resampling.BILINEAR
        )

        # Resize mask using nearest-neighbor
        mask = mask.resize(
            (self.image_size, self.image_size),
            Image.Resampling.NEAREST
        )

        image = transforms.ToTensor()(image)

        # Normalize image
        image = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )(image)

        mask = transforms.ToTensor()(mask)

        # Convert mask to binary
        mask = (mask > 0.5).float()

        return image, mask


# ============================================================
# U-NET BUILDING BLOCK
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


# ============================================================
# U-NET
# ============================================================

class UNet(nn.Module):

    def __init__(self):

        super().__init__()

        # Encoder
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
            kernel_size=2,
            stride=2
        )

        # Bottleneck
        self.bottleneck = DoubleConv(
            256,
            512
        )

        # Decoder
        self.up4 = nn.ConvTranspose2d(
            512,
            256,
            kernel_size=2,
            stride=2
        )

        self.dec4 = DoubleConv(
            512,
            256
        )

        self.up3 = nn.ConvTranspose2d(
            256,
            128,
            kernel_size=2,
            stride=2
        )

        self.dec3 = DoubleConv(
            256,
            128
        )

        self.up2 = nn.ConvTranspose2d(
            128,
            64,
            kernel_size=2,
            stride=2
        )

        self.dec2 = DoubleConv(
            128,
            64
        )

        self.up1 = nn.ConvTranspose2d(
            64,
            32,
            kernel_size=2,
            stride=2
        )

        self.dec1 = DoubleConv(
            64,
            32
        )

        self.output = nn.Conv2d(
            32,
            1,
            kernel_size=1
        )

    def forward(self, x):

        # Encoder
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

        # Bottleneck
        b = self.bottleneck(
            self.pool(e4)
        )

        # Decoder
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
# DICE LOSS
# ============================================================

class DiceLoss(nn.Module):

    def __init__(self):

        super().__init__()

    def forward(
        self,
        predictions,
        targets
    ):

        predictions = torch.sigmoid(
            predictions
        )

        predictions = predictions.view(
            -1
        )

        targets = targets.view(
            -1
        )

        intersection = (
            predictions * targets
        ).sum()

        dice = (
            2.0 * intersection + 1.0
        ) / (
            predictions.sum()
            + targets.sum()
            + 1.0
        )

        return 1.0 - dice


# ============================================================
# DICE SCORE
# ============================================================

def dice_score(
    predictions,
    targets
):

    predictions = (
        torch.sigmoid(predictions)
        > 0.5
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


# ============================================================
# IOU SCORE
# ============================================================

def iou_score(
    predictions,
    targets
):

    predictions = (
        torch.sigmoid(predictions)
        > 0.5
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
# LOAD DATA
# ============================================================

print("\nLoading datasets...")

train_dataset = HairSegmentationDataset(
    os.path.join(
        DATA_DIR,
        "train"
    ),
    IMAGE_SIZE
)

val_dataset = HairSegmentationDataset(
    os.path.join(
        DATA_DIR,
        "val"
    ),
    IMAGE_SIZE
)

test_dataset = HairSegmentationDataset(
    os.path.join(
        DATA_DIR,
        "test"
    ),
    IMAGE_SIZE
)

print("\nDataset sizes:")

print(
    f"Train: {len(train_dataset)}"
)

print(
    f"Validation: {len(val_dataset)}"
)

print(
    f"Test: {len(test_dataset)}"
)

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

print("\nCreating U-Net...")

model = UNet().to(device)

print(
    f"Model parameters: "
    f"{sum(p.numel() for p in model.parameters()):,}"
)

# ============================================================
# LOSS AND OPTIMIZER
# ============================================================

bce_loss = nn.BCEWithLogitsLoss()

dice_loss = DiceLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

# ============================================================
# TRAINING
# ============================================================

best_val_dice = 0.0

start_time = time.time()

for epoch in range(NUM_EPOCHS):

    epoch_start = time.time()

    model.train()

    train_loss = 0.0

    for batch_idx, (
        images,
        masks
    ) in enumerate(train_loader):

        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss_bce = bce_loss(
            outputs,
            masks
        )

        loss_dice = dice_loss(
            outputs,
            masks
        )

        loss = (
            0.5 * loss_bce
            +
            0.5 * loss_dice
        )

        loss.backward()

        optimizer.step()

        train_loss += (
            loss.item()
            *
            images.size(0)
        )

        if (batch_idx + 1) % 50 == 0:

            print(
                f"Epoch "
                f"{epoch + 1}/{NUM_EPOCHS} "
                f"- Batch "
                f"{batch_idx + 1}/"
                f"{len(train_loader)}"
            )

    train_loss /= len(
        train_dataset
    )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    val_loss = 0.0
    val_dice = 0.0
    val_iou = 0.0

    with torch.no_grad():

        for images, masks in val_loader:

            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)

            loss_bce = bce_loss(
                outputs,
                masks
            )

            loss_dice = dice_loss(
                outputs,
                masks
            )

            loss = (
                0.5 * loss_bce
                +
                0.5 * loss_dice
            )

            val_loss += (
                loss.item()
                *
                images.size(0)
            )

            val_dice += (
                dice_score(
                    outputs,
                    masks
                )
                *
                images.size(0)
            )

            val_iou += (
                iou_score(
                    outputs,
                    masks
                )
                *
                images.size(0)
            )

    val_loss /= len(
        val_dataset
    )

    val_dice /= len(
        val_dataset
    )

    val_iou /= len(
        val_dataset
    )

    epoch_time = (
        time.time()
        - epoch_start
    )

    print("\n" + "-" * 60)

    print(
        f"Epoch {epoch + 1}/{NUM_EPOCHS}"
    )

    print(
        f"Train Loss: {train_loss:.4f}"
    )

    print(
        f"Val Loss: {val_loss:.4f}"
    )

    print(
        f"Val Dice: {val_dice:.4f}"
    )

    print(
        f"Val IoU: {val_iou:.4f}"
    )

    print(
        f"Epoch Time: {epoch_time / 60:.2f} minutes"
    )

    print("-" * 60)

    # --------------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------------

    if val_dice > best_val_dice:

        best_val_dice = val_dice

        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),

                "image_size":
                    IMAGE_SIZE,

                "best_val_dice":
                    best_val_dice
            },
            MODEL_PATH
        )

        print(
            f"Best model saved!"
        )

        print(
            f"Validation Dice: "
            f"{best_val_dice:.4f}"
        )

# ============================================================
# TRAINING COMPLETE
# ============================================================

total_time = (
    time.time()
    - start_time
)

print("\n" + "=" * 60)

print("TRAINING COMPLETE")

print("=" * 60)

print(
    f"Best Validation Dice: "
    f"{best_val_dice:.4f}"
)

print(
    f"Total Training Time: "
    f"{total_time / 60:.2f} minutes"
)

print(
    f"Model saved to:\n"
    f"{MODEL_PATH}"
)

# ============================================================
# LOAD BEST MODEL
# ============================================================

print("\nLoading best model...")

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

# ============================================================
# FINAL TEST
# ============================================================

print("\nRunning final test evaluation...")

test_dice = 0.0
test_iou = 0.0

with torch.no_grad():

    for images, masks in test_loader:

        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)

        test_dice += (
            dice_score(
                outputs,
                masks
            )
            *
            images.size(0)
        )

        test_iou += (
            iou_score(
                outputs,
                masks
            )
            *
            images.size(0)
        )

test_dice /= len(
    test_dataset
)

test_iou /= len(
    test_dataset
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

print("\nSegmentation training and evaluation complete.")