import os
import random

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = r"C:\Users\kalya\OneDrive\Desktop\hair_analysis"

DATA_DIR = os.path.join(
    BASE_DIR,
    "processed_datasets",
    "hair_segmentation",
    "test",
    "images"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "hair_segmentation_unet.pth"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "models",
    "segmentation_visualizations"
)

IMAGE_SIZE = 224

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("=" * 60)
print("SEGMENTATION VISUALIZATION")
print("=" * 60)

print(f"Device: {device}")


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
                3,
                padding=1
            ),

            nn.BatchNorm2d(
                out_channels
            ),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                3,
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
# LOAD MODEL
# ============================================================

print("\nLoading model...")

model = UNet().to(device)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("Model loaded.")


# ============================================================
# IMAGE TRANSFORM
# ============================================================

transform = transforms.Compose([
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
# FIND TEST IMAGES
# ============================================================

images = [
    f for f in os.listdir(DATA_DIR)
    if f.lower().endswith(
        (".jpg", ".jpeg", ".png")
    )
]

print(
    f"\nFound {len(images)} test images."
)


# ============================================================
# SELECT 10 IMAGES
# ============================================================

random.seed(42)

selected_images = random.sample(
    images,
    min(10, len(images))
)


# ============================================================
# GENERATE VISUALIZATIONS
# ============================================================

for index, filename in enumerate(
    selected_images
):

    image_path = os.path.join(
        DATA_DIR,
        filename
    )

    original = Image.open(
        image_path
    ).convert("RGB")

    resized = original.resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    )

    tensor = transform(
        original
    ).unsqueeze(0).to(device)

    with torch.no_grad():

        output = model(tensor)

        probability = torch.sigmoid(
            output
        )

        mask = (
            probability[0, 0]
            .cpu()
            .numpy()
            > 0.5
        )


    # --------------------------------------------------------
    # CREATE MASK IMAGE
    # --------------------------------------------------------

    mask_image = (
        mask.astype(np.uint8)
        * 255
    )

    mask_image = Image.fromarray(
        mask_image
    )


    # --------------------------------------------------------
    # CREATE EXTRACTED HAIR
    # --------------------------------------------------------

    original_array = np.array(
        resized
    )

    extracted = (
        original_array
        * mask[:, :, None]
    )

    extracted = extracted.astype(
        np.uint8
    )

    extracted_image = Image.fromarray(
        extracted
    )


    # --------------------------------------------------------
    # SAVE FILES
    # --------------------------------------------------------

    base = os.path.splitext(
        filename
    )[0]

    original_path = os.path.join(
        OUTPUT_DIR,
        f"{index + 1}_{base}_original.png"
    )

    mask_path = os.path.join(
        OUTPUT_DIR,
        f"{index + 1}_{base}_mask.png"
    )

    extracted_path = os.path.join(
        OUTPUT_DIR,
        f"{index + 1}_{base}_hair.png"
    )

    resized.save(
        original_path
    )

    mask_image.save(
        mask_path
    )

    extracted_image.save(
        extracted_path
    )

    print(
        f"Generated example "
        f"{index + 1}/"
        f"{len(selected_images)}"
    )


print("\n" + "=" * 60)

print("VISUALIZATION COMPLETE")

print("=" * 60)

print(
    f"\nResults saved to:\n"
    f"{OUTPUT_DIR}"
)