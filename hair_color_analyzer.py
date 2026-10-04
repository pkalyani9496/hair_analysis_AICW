import os
import json
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = r"C:\Users\kalya\OneDrive\Desktop\hair_analysis"

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "hair_segmentation_unet.pth"
)

IMAGE_SIZE = 224

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# U-NET
# ============================================================

class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):

    def __init__(self):
        super().__init__()

        # Encoder
        self.enc1 = DoubleConv(3, 32)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = DoubleConv(64, 128)
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = DoubleConv(128, 256)
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(256, 512)

        # Decoder
        self.up4 = nn.ConvTranspose2d(
            512,
            256,
            kernel_size=2,
            stride=2
        )

        self.dec4 = DoubleConv(512, 256)

        self.up3 = nn.ConvTranspose2d(
            256,
            128,
            kernel_size=2,
            stride=2
        )

        self.dec3 = DoubleConv(256, 128)

        self.up2 = nn.ConvTranspose2d(
            128,
            64,
            kernel_size=2,
            stride=2
        )

        self.dec2 = DoubleConv(128, 64)

        self.up1 = nn.ConvTranspose2d(
            64,
            32,
            kernel_size=2,
            stride=2
        )

        self.dec1 = DoubleConv(64, 32)

        # IMPORTANT:
        # The trained model uses "output", not "final".
        self.output = nn.Conv2d(
            32,
            1,
            kernel_size=1
        )

    def forward(self, x):

        # Encoder
        e1 = self.enc1(x)

        e2 = self.enc2(
            self.pool1(e1)
        )

        e3 = self.enc3(
            self.pool2(e2)
        )

        e4 = self.enc4(
            self.pool3(e3)
        )

        # Bottleneck
        b = self.bottleneck(
            self.pool4(e4)
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
# LOAD SEGMENTATION MODEL
# ============================================================

def load_segmentation_model():

    print("=" * 60)
    print("HAIR COLOR ANALYZER")
    print("=" * 60)

    print()
    print(f"Device: {DEVICE}")

    print()
    print("Loading segmentation model...")

    if not os.path.exists(MODEL_PATH):

        raise FileNotFoundError(
            f"Model not found:\n{MODEL_PATH}"
        )

    model = UNet()

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    # Support both checkpoint formats:
    #
    # 1. {"model_state_dict": ...}
    # 2. Direct state_dict

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:

            state_dict = checkpoint[
                "model_state_dict"
            ]

        else:

            state_dict = checkpoint

    else:

        state_dict = checkpoint

    model.load_state_dict(
        state_dict,
        strict=True
    )

    model.to(DEVICE)

    model.eval()

    print("Segmentation model loaded.")

    return model


# ============================================================
# SEGMENT HAIR
# ============================================================

def segment_hair(model, image):

    original_height, original_width = image.shape[:2]

    # OpenCV image is BGR.
    # Convert to RGB for torchvision.
    rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    transform = transforms.Compose([
        transforms.ToPILImage(),

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

    tensor = transform(rgb)

    tensor = tensor.unsqueeze(0)

    tensor = tensor.to(DEVICE)

    with torch.no_grad():

        prediction = model(tensor)

        prediction = torch.sigmoid(
            prediction
        )

    mask = prediction.squeeze().cpu().numpy()

    # Threshold
    mask = (
        mask >= 0.5
    ).astype(np.uint8) * 255

    # Resize mask back to original image
    mask = cv2.resize(
        mask,
        (
            original_width,
            original_height
        ),
        interpolation=cv2.INTER_NEAREST
    )

    return mask


# ============================================================
# CLEAN MASK
# ============================================================

def clean_mask(mask):

    kernel = np.ones(
        (5, 5),
        np.uint8
    )

    # Remove small noise
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    # Fill small holes
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    return mask


# ============================================================
# CREATE SCALP / MIDDLE / ENDS REGIONS
# ============================================================

def create_hair_regions(mask):

    ys, xs = np.where(
        mask > 0
    )

    if len(ys) == 0:

        raise ValueError(
            "No hair was detected in the image."
        )

    min_y = int(
        np.min(ys)
    )

    max_y = int(
        np.max(ys)
    )

    height = (
        max_y -
        min_y +
        1
    )

    # Divide the detected hair vertically
    scalp_end = (
        min_y +
        int(height * 0.33)
    )

    middle_end = (
        min_y +
        int(height * 0.66)
    )

    # Create empty masks
    scalp_mask = np.zeros_like(mask)
    middle_mask = np.zeros_like(mask)
    ends_mask = np.zeros_like(mask)

    y_coordinates = np.indices(
        mask.shape
    )[0]

    hair_pixels = (
        mask > 0
    )

    # Scalp
    scalp_region = (
        hair_pixels &
        (y_coordinates <= scalp_end)
    )

    # Middle
    middle_region = (
        hair_pixels &
        (y_coordinates > scalp_end) &
        (y_coordinates <= middle_end)
    )

    # Ends
    ends_region = (
        hair_pixels &
        (y_coordinates > middle_end)
    )

    scalp_mask[
        scalp_region
    ] = 255

    middle_mask[
        middle_region
    ] = 255

    ends_mask[
        ends_region
    ] = 255

    return {
        "Scalp": scalp_mask,
        "Middle": middle_mask,
        "Ends": ends_mask
    }


# ============================================================
# DOMINANT COLOR
# ============================================================

def get_dominant_color(
    image,
    region_mask
):

    pixels = image[
        region_mask > 0
    ]

    if len(pixels) < 20:

        return None

    # OpenCV BGR → RGB
    pixels_rgb = pixels[:, ::-1]

    # Calculate brightness
    brightness = np.mean(
        pixels_rgb,
        axis=1
    )

    # Remove extremely bright pixels
    valid = pixels_rgb[
        brightness < 245
    ]

    if len(valid) < 20:

        valid = pixels_rgb

    data = np.float32(
        valid
    )

    # Number of color clusters
    k = min(
        3,
        len(data)
    )

    criteria = (
        cv2.TERM_CRITERIA_EPS +
        cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.2
    )

    _, labels, centers = cv2.kmeans(
        data,
        k,
        None,
        criteria,
        10,
        cv2.KMEANS_PP_CENTERS
    )

    counts = np.bincount(
        labels.flatten()
    )

    dominant_index = np.argmax(
        counts
    )

    color = centers[
        dominant_index
    ]

    return tuple(
        int(x)
        for x in color
    )


# ============================================================
# BASIC HAIR COLOR CLASSIFICATION
# ============================================================

def classify_hair_color(rgb):

    if rgb is None:

        return "Unknown"

    r, g, b = rgb

    pixel = np.uint8([
        [rgb]
    ])

    hsv_pixel = cv2.cvtColor(
        pixel,
        cv2.COLOR_RGB2HSV
    )[0][0]

    h, s, v = hsv_pixel

    # Black
    if v < 45:

        return "Black"

    # Dark brown / black
    if v < 85:

        if r > b + 15:

            return "Dark Brown"

        return "Dark Brown / Black"

    # Brown
    if (
        r > b + 15
        and r > g + 5
    ):

        if v < 140:

            return "Medium Brown"

        return "Light Brown"

    # Red / Burgundy
    if (
        r > g * 1.25
        and r > b * 1.25
    ):

        return "Red / Burgundy"

    # Blonde / Golden
    if (
        r > 140
        and g > 120
        and b < 110
    ):

        return "Blonde / Golden"

    # Grey
    if (
        abs(int(r) - int(g)) < 15
        and abs(int(g) - int(b)) < 15
    ):

        if v > 120:

            return "Grey"

    return "Mixed / Other"


# ============================================================
# COLOR RECOMMENDATIONS
# ============================================================

def recommend_colors(
    current_color
):

    recommendations = {

        "Black": [
            "Dark Brown",
            "Chocolate Brown",
            "Burgundy"
        ],

        "Dark Brown": [
            "Chocolate Brown",
            "Warm Brown",
            "Burgundy"
        ],

        "Dark Brown / Black": [
            "Chocolate Brown",
            "Burgundy",
            "Deep Brown"
        ],

        "Medium Brown": [
            "Warm Brown",
            "Caramel Brown",
            "Burgundy"
        ],

        "Light Brown": [
            "Honey Brown",
            "Caramel",
            "Golden Brown"
        ],

        "Red / Burgundy": [
            "Deep Burgundy",
            "Dark Red",
            "Warm Brown"
        ],

        "Blonde / Golden": [
            "Honey Blonde",
            "Golden Brown",
            "Warm Caramel"
        ],

        "Grey": [
            "Silver",
            "Ash Brown",
            "Dark Grey"
        ],

        "Mixed / Other": [
            "Natural Brown",
            "Dark Brown",
            "Burgundy"
        ],

        "Unknown": [
            "Natural Brown",
            "Dark Brown",
            "Chocolate Brown"
        ]
    }

    return recommendations.get(
        current_color,
        recommendations["Unknown"]
    )


# ============================================================
# SAVE VISUALIZATIONS
# ============================================================

def save_visualizations(
    image,
    full_mask,
    regions,
    output_dir
):

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    # Original
    cv2.imwrite(
        os.path.join(
            output_dir,
            "original.png"
        ),
        image
    )

    # Full hair mask
    cv2.imwrite(
        os.path.join(
            output_dir,
            "hair_mask.png"
        ),
        full_mask
    )

    # Individual regions
    for name, region_mask in regions.items():

        # Region mask
        cv2.imwrite(
            os.path.join(
                output_dir,
                f"{name.lower()}_mask.png"
            ),
            region_mask
        )

        # Extract region
        extracted = cv2.bitwise_and(
            image,
            image,
            mask=region_mask
        )

        cv2.imwrite(
            os.path.join(
                output_dir,
                f"{name.lower()}_hair.png"
            ),
            extracted
        )


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = load_segmentation_model()

    # --------------------------------------------------------
    # Ask for image
    # --------------------------------------------------------

    print()

    image_path = input(
        "Enter the full path of a hair image: "
    ).strip().strip('"')

    if not os.path.exists(
        image_path
    ):

        print()
        print(
            "ERROR: Image file not found."
        )

        return

    # --------------------------------------------------------
    # Read image
    # --------------------------------------------------------

    image = cv2.imread(
        image_path
    )

    if image is None:

        print()
        print(
            "ERROR: Could not read the image."
        )

        return

    print()

    # --------------------------------------------------------
    # Segmentation
    # --------------------------------------------------------

    print(
        "Segmenting hair..."
    )

    mask = segment_hair(
        model,
        image
    )

    mask = clean_mask(
        mask
    )

    print(
        "Hair segmentation completed."
    )

    # --------------------------------------------------------
    # Create regions
    # --------------------------------------------------------

    print()

    print(
        "Creating hair regions..."
    )

    try:

        regions = create_hair_regions(
            mask
        )

    except ValueError as error:

        print()
        print(
            f"ERROR: {error}"
        )

        return

    print(
        "Regions created:"
    )

    print(
        "  1. Scalp"
    )

    print(
        "  2. Middle"
    )

    print(
        "  3. Ends"
    )

    # --------------------------------------------------------
    # Output folder
    # --------------------------------------------------------

    output_dir = os.path.join(
        BASE_DIR,
        "models",
        "hair_color_analysis"
    )

    # --------------------------------------------------------
    # Save visualizations
    # --------------------------------------------------------

    save_visualizations(
        image,
        mask,
        regions,
        output_dir
    )

    # --------------------------------------------------------
    # Analyze colors
    # --------------------------------------------------------

    print()

    print("=" * 60)
    print(
        "REGION COLOR ANALYSIS"
    )
    print("=" * 60)

    results = {}

    for name, region_mask in regions.items():

        rgb = get_dominant_color(
            image,
            region_mask
        )

        color_name = classify_hair_color(
            rgb
        )

        recommendations = recommend_colors(
            color_name
        )

        results[name] = {
            "rgb": rgb,
            "detected_color": color_name,
            "recommendations": recommendations
        }

        print()

        print(
            name.upper()
        )

        print(
            "-" * 30
        )

        print(
            f"Detected color: {color_name}"
        )

        if rgb is not None:

            print(
                f"RGB: "
                f"({rgb[0]}, "
                f"{rgb[1]}, "
                f"{rgb[2]})"
            )

        print(
            "Recommendations:"
        )

        for recommendation in recommendations:

            print(
                f"  - {recommendation}"
            )

    # --------------------------------------------------------
    # Save JSON results
    # --------------------------------------------------------

    results_path = os.path.join(
        output_dir,
        "color_analysis.json"
    )

    with open(
        results_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            results,
            file,
            indent=4
        )

    # --------------------------------------------------------
    # Finished
    # --------------------------------------------------------

    print()

    print("=" * 60)
    print(
        "COLOR ANALYSIS COMPLETE"
    )
    print("=" * 60)

    print()

    print(
        "Results saved to:"
    )

    print(
        output_dir
    )


# ============================================================
# PROGRAM START
# ============================================================

if __name__ == "__main__":

    main()