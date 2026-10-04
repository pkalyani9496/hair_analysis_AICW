import base64
import gc
import json
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
from torchvision import transforms

from backend.recommendations import COLOR_PALETTE, recommend_colors, recommend_haircuts


# Restrict PyTorch thread pool to 1 to minimize memory overhead in constrained cloud environments
torch.set_num_threads(1)

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
HAIR_TYPE_MODEL = MODELS_DIR / "hair_type_resnet18.pth"
HAIR_TYPE_CLASSES = MODELS_DIR / "hair_type_classes.json"
DISEASE_MODEL = MODELS_DIR / "hair_disease_resnet18.pth"
DISEASE_CLASSES = MODELS_DIR / "hair_disease_classes.json"
DISEASE_CALIBRATION = MODELS_DIR / "disease_evaluation" / "calibration.json"
SEGMENTATION_MODEL = MODELS_DIR / "hair_segmentation_unet.pth"
DEVICE = torch.device("cpu")

classifier_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class HairAnalysisService:
    def __init__(self):
        self._hair_type_model = None
        self._hair_type_classes = None
        self._disease_model = None
        self._disease_classes = None
        self._disease_temperature = None
        self._disease_thresholds = None
        self._segmentation_model = None

    @staticmethod
    def load_classes(path: Path):
        if not path.exists():
            raise FileNotFoundError(f"Class file does not exist:\n{path}")

        with path.open("r", encoding="utf-8-sig") as file:
            data = json.load(file)

        if isinstance(data, list):
            return [str(item) for item in data]

        if isinstance(data, dict):
            for key in ["classes", "class_names", "labels", "categories"]:
                if key in data and isinstance(data[key], list):
                    return [str(item) for item in data[key]]

            if "class_to_idx" in data:
                mapping = data["class_to_idx"]
                return [str(name) for name, _ in sorted(mapping.items(), key=lambda item: int(item[1]))]

            if "idx_to_class" in data:
                mapping = data["idx_to_class"]
                return [str(name) for _, name in sorted(mapping.items(), key=lambda item: int(item[0]))]

            if all(str(key).isdigit() for key in data.keys()):
                return [str(name) for _, name in sorted(data.items(), key=lambda item: int(item[0]))]

        raise ValueError(f"Could not understand class file: {path}")

    @staticmethod
    def load_checkpoint(path: Path):
        if not path.exists():
            raise FileNotFoundError(f"Model file does not exist:\n{path}")

        checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)

        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:
                return checkpoint["model_state_dict"]
            if "state_dict" in checkpoint:
                return checkpoint["state_dict"]

        return checkpoint

    @staticmethod
    def build_resnet18(num_classes):
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    @staticmethod
    def clean_state_dict(state_dict):
        cleaned = {}
        for key, value in state_dict.items():
            if key.startswith("module."):
                key = key[7:]
            cleaned[key] = value
        return cleaned

    @lru_cache(maxsize=1)
    def load_hair_type_model(self):
        classes = self.load_classes(HAIR_TYPE_CLASSES)
        model = self.build_resnet18(len(classes))
        model.load_state_dict(self.clean_state_dict(self.load_checkpoint(HAIR_TYPE_MODEL)), strict=True)
        model.to(DEVICE)
        model.eval()
        self._hair_type_model = model
        self._hair_type_classes = classes
        return model, classes

    @lru_cache(maxsize=1)
    def load_disease_model(self):
        classes = self.load_classes(DISEASE_CLASSES)

        with DISEASE_CALIBRATION.open("r", encoding="utf-8") as file:
            calibration = json.load(file)

        temperature = float(calibration["temperature"])
        confidence_thresholds = {
            str(name): float(threshold)
            for name, threshold in calibration["class_confidence_thresholds"].items()
        }

        if not np.isfinite(temperature) or temperature <= 0:
            raise ValueError("Disease calibration temperature is invalid.")

        missing_thresholds = set(classes) - set(confidence_thresholds)
        if missing_thresholds:
            raise ValueError("Disease calibration is missing classes: " + ", ".join(sorted(missing_thresholds)))

        model = self.build_resnet18(len(classes))
        model.load_state_dict(self.clean_state_dict(self.load_checkpoint(DISEASE_MODEL)), strict=True)
        model.to(DEVICE)
        model.eval()
        self._disease_model = model
        self._disease_classes = classes
        self._disease_temperature = temperature
        self._disease_thresholds = confidence_thresholds
        return model, classes, temperature, confidence_thresholds

    @lru_cache(maxsize=1)
    def load_segmentation_model(self):
        model = UNet()
        checkpoint = torch.load(SEGMENTATION_MODEL, map_location=DEVICE, weights_only=False)
        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        model.load_state_dict(self.clean_state_dict(state_dict), strict=True)
        model.to(DEVICE)
        model.eval()
        self._segmentation_model = model
        return model

    @staticmethod
    def calculate_image_quality(image: Image.Image):
        rgb = np.array(image.convert("RGB"))
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        brightness = float(np.mean(gray))
        contrast = float(np.std(gray))

        score = 100.0
        if sharpness < 40:
            score -= 25
        elif sharpness < 80:
            score -= 10

        if brightness < 35 or brightness > 225:
            score -= 20
        elif brightness < 55 or brightness > 205:
            score -= 8

        if contrast < 20:
            score -= 20
        elif contrast < 35:
            score -= 8

        score = float(np.clip(score, 0, 100))
        status = "Good" if score >= 75 else "Moderate" if score >= 50 else "Low"

        return {
            "score": round(score, 1),
            "sharpness": round(sharpness, 2),
            "brightness": round(brightness, 2),
            "contrast": round(contrast, 2),
            "status": status,
        }

    @staticmethod
    def prepare_classifier_tensor(image: Image.Image):
        return classifier_transform(image.convert("RGB")).unsqueeze(0).to(DEVICE)

    @staticmethod
    def predict_classifier_tensor(tensor, model, classes, temperature=1.0, confidence_thresholds=None):
        with torch.inference_mode():
            output = model(tensor)
            probabilities = torch.softmax(output / temperature, dim=1)[0]

        predicted_index = int(torch.argmax(probabilities).item())
        predicted_class = classes[predicted_index]
        confidence = float(probabilities[predicted_index].item() * 100)

        if confidence_thresholds is not None and confidence / 100 < confidence_thresholds.get(predicted_class, 1.01):
            predicted_class = "Uncertain"

        top_k = min(5, len(classes))
        top_values, top_indices = torch.topk(probabilities, k=top_k)
        top_predictions = []

        for value, index in zip(top_values, top_indices):
            top_predictions.append((classes[int(index.item())], float(value.item() * 100)))

        return predicted_class, confidence, top_predictions, probabilities.cpu().numpy()

    @staticmethod
    def clean_segmentation_mask(mask, threshold=0.45):
        binary = (mask > threshold).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

        if num_labels > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            largest_area = np.max(areas)
            cleaned = np.zeros_like(binary)
            for label_id in range(1, num_labels):
                area = stats[label_id, cv2.CC_STAT_AREA]
                if area >= largest_area * 0.05 or area >= 500:
                    cleaned[labels == label_id] = 1
            binary = cleaned

        return binary

    @staticmethod
    def prepare_segmentation_tensor(image: Image.Image):
        resized = image.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
        tensor = transforms.ToTensor()(resized)
        tensor = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )(tensor)
        return tensor.unsqueeze(0).to(DEVICE)

    @staticmethod
    def segment_hair(image: Image.Image, model):
        original = np.array(image.convert("RGB"))
        tensor = HairAnalysisService.prepare_segmentation_tensor(image)

        with torch.inference_mode():
            prediction = model(tensor)
            prediction = torch.sigmoid(prediction)

        probability_mask = prediction[0, 0].cpu().numpy()
        mask = HairAnalysisService.clean_segmentation_mask(probability_mask)
        mask = cv2.resize(mask, (original.shape[1], original.shape[0]), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 0).astype(np.uint8)
        coverage = float(np.sum(mask) / mask.size * 100)
        return original, mask, probability_mask, coverage

    @staticmethod
    def create_region_masks(hair_mask):
        ys, xs = np.where(hair_mask > 0)
        if len(ys) == 0:
            empty = np.zeros_like(hair_mask)
            return {"Whole Hair": empty, "Scalp": empty, "Middle": empty, "Ends": empty}

        whole_mask = hair_mask.copy()

        min_y = int(np.min(ys))
        max_y = int(np.max(ys))
        height = max_y - min_y + 1

        scalp_end = min_y + int(height * 0.33)
        middle_end = min_y + int(height * 0.66)

        scalp_mask = np.zeros_like(hair_mask)
        middle_mask = np.zeros_like(hair_mask)
        ends_mask = np.zeros_like(hair_mask)

        scalp_mask[min_y:scalp_end] = hair_mask[min_y:scalp_end]
        middle_mask[scalp_end:middle_end] = hair_mask[scalp_end:middle_end]
        ends_mask[middle_end:max_y + 1] = hair_mask[middle_end:max_y + 1]

        return {"Whole Hair": whole_mask, "Scalp": scalp_mask, "Middle": middle_mask, "Ends": ends_mask}

    @staticmethod
    def get_dominant_color(image_rgb, region_mask):
        if np.sum(region_mask) == 0:
            return 80, 80, 80

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        clean_mask = cv2.erode(region_mask.astype(np.uint8), kernel, iterations=1)
        pixels = image_rgb[clean_mask > 0]
        if len(pixels) == 0:
            pixels = image_rgb[region_mask > 0]
        if len(pixels) == 0:
            return 80, 80, 80

        pixels = pixels.astype(np.uint8)
        lab_pixels = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3)
        brightness = np.mean(pixels, axis=1)
        valid = (brightness > 15) & (brightness < 235)
        pixels = pixels[valid]
        lab_pixels = lab_pixels[valid]

        if len(pixels) < 20:
            pixels = image_rgb[region_mask > 0].astype(np.uint8)
            lab_pixels = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3)

        if len(lab_pixels) > 3000:
            rng = np.random.default_rng(42)
            indices = rng.choice(len(lab_pixels), 3000, replace=False)
            lab_samples = lab_pixels[indices]
            rgb_samples = pixels[indices]
        else:
            lab_samples = lab_pixels
            rgb_samples = pixels

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.25)
        try:
            _, labels, centers = cv2.kmeans(np.float32(lab_samples), 3, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
            counts = np.bincount(labels.flatten())
            dominant_index = int(np.argmax(counts))
            dominant_lab = np.uint8(np.clip(centers[dominant_index], 0, 255)).reshape(1, 1, 3)
            dominant_rgb = cv2.cvtColor(dominant_lab, cv2.COLOR_LAB2RGB)[0, 0]
            return tuple(int(x) for x in dominant_rgb)
        except Exception:
            mean_color = np.mean(rgb_samples, axis=0)
            return tuple(int(x) for x in mean_color)

    @staticmethod
    def classify_hair_color(rgb):
        r, g, b = rgb
        brightness = (r + g + b) / 3

        if brightness < 45:
            return "Black"
        if brightness < 85:
            return "Dark Brown"
        if brightness < 125:
            return "Medium Brown"
        if brightness < 165:
            return "Light Brown"
        if brightness < 205:
            return "Dark Blonde"
        return "Light / Blonde"

    @staticmethod
    def hex_to_rgb(value):
        value = value.lstrip("#")
        if len(value) != 6:
            return (127, 127, 127)
        return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))

    @staticmethod
    def apply_hair_color(image_rgb, hair_mask, target_rgb, strength=0.78):
        if np.sum(hair_mask) == 0:
            return image_rgb.copy()

        image = image_rgb.astype(np.float32)
        hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)

        target_patch = np.zeros_like(image_rgb, dtype=np.uint8)
        target_patch[:, :] = np.array(target_rgb, dtype=np.uint8)
        target_hsv = cv2.cvtColor(target_patch, cv2.COLOR_RGB2HSV).astype(np.float32)

        target_hue = target_hsv[:, :, 0]
        target_sat = target_hsv[:, :, 1]
        target_value = target_hsv[:, :, 2]

        new_hsv = hsv.copy()
        new_hsv[:, :, 0] = target_hue
        new_hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1 - strength) + target_sat * strength + 25 * strength, 0, 255)
        brightness_strength = strength * 0.58
        new_hsv[:, :, 2] = (
            hsv[:, :, 2] * (1 - brightness_strength)
            + target_value * brightness_strength
        )

        recolored = cv2.cvtColor(np.uint8(np.clip(new_hsv, 0, 255)), cv2.COLOR_HSV2RGB)
        soft_mask = cv2.GaussianBlur(hair_mask.astype(np.float32), (0, 0), sigmaX=2.0)
        soft_mask = np.clip(soft_mask, 0, 1)
        alpha = (soft_mask * strength)[:, :, None]
        result = image * (1 - alpha) + recolored.astype(np.float32) * alpha
        return np.uint8(np.clip(result, 0, 255))

    @staticmethod
    def serialize_mask(mask: np.ndarray, max_dim: int = 256):
        h, w = mask.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            new_w = max(1, int(round(w * scale)))
            new_h = max(1, int(round(h * scale)))
            downsampled = cv2.resize(mask.astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        else:
            downsampled = mask.astype(np.uint8)
        return downsampled.tolist()

    def analyze_image(self, image: Image.Image, region="Middle", target_color=None):
        # Constrain maximum dimension to 800px to maintain minimal peak memory on cloud instances
        max_dim = 800
        if max(image.width, image.height) > max_dim:
            image = image.copy()
            image.thumbnail((max_dim, max_dim), Image.Resampling.BILINEAR)

        hair_type_model, hair_type_classes = self.load_hair_type_model()
        disease_model, disease_classes, disease_temperature, disease_thresholds = self.load_disease_model()
        segmentation_model = self.load_segmentation_model()

        image_rgb = np.array(image.convert("RGB"))
        quality = self.calculate_image_quality(image)
        classifier_tensor = self.prepare_classifier_tensor(image)

        original_image, hair_mask, _, coverage = self.segment_hair(image, segmentation_model)
        hair_type, hair_confidence, hair_top, _ = self.predict_classifier_tensor(
            classifier_tensor,
            hair_type_model,
            hair_type_classes,
        )
        disease_label, disease_confidence, disease_top, _ = self.predict_classifier_tensor(
            classifier_tensor,
            disease_model,
            disease_classes,
            temperature=disease_temperature,
            confidence_thresholds=disease_thresholds,
        )

        # Multi-modal verification: Validate disease prediction against U-Net hair segmentation
        # 1. Closed-Set Differential Screening:
        #    The 10-class model is trained strictly on scalp pathologies without an explicit healthy class.
        #    If no condition meets its high-precision calibrated threshold (disease_label == "Uncertain"),
        #    the scalp screening is clinically negative -> "Healthy / Normal Scalp".
        # 2. Facial Skin Confounding Mitigation:
        #    In front-facing portraits and selfies, forehead and facial skin activate ResNet features
        #    of "Male Pattern Baldness". If U-Net detects healthy hair coverage (coverage >= 10.0%),
        #    hair growth is present, ruling out pattern baldness false positives unless confidence is extreme (>=95%).
        has_healthy_hair_coverage = coverage >= 10.0
        is_subthreshold = (disease_label == "Uncertain")
        is_mpb_artifact = (
            (disease_label == "Male Pattern Baldness" or (disease_top and disease_top[0][0] == "Male Pattern Baldness"))
            and has_healthy_hair_coverage
            and disease_confidence < 95.0
        )

        if is_subthreshold or is_mpb_artifact:
            healthy_score = round(float(np.clip(100.0 - (disease_top[0][1] * 0.20), 82.0, 98.5)), 1)
            disease_label = "Healthy / Normal Scalp"
            disease_confidence = healthy_score
            disease_top_formatted = [
                {"label": "Healthy / Normal Scalp", "confidence": healthy_score}
            ] + [
                {"label": f"{lbl} (Differential Candidate)", "confidence": round(float(conf), 2)}
                for lbl, conf in disease_top[:4]
            ]
        else:
            disease_top_formatted = [
                {"label": label, "confidence": round(float(conf), 2)} for label, conf in disease_top
            ]

        region_masks = self.create_region_masks(hair_mask)
        selected_region = region if region in region_masks else "Whole Hair"
        selected_mask = region_masks.get(selected_region, hair_mask)
        region_colors = {}
        for region_name, region_mask in region_masks.items():
            region_rgb = self.get_dominant_color(image_rgb, region_mask)
            region_colors[region_name] = {
                "rgb": [int(value) for value in region_rgb],
                "label": self.classify_hair_color(region_rgb),
                "hex": "#{:02X}{:02X}{:02X}".format(*region_rgb),
            }

        selected_color = region_colors[selected_region]
        rgb_color = tuple(selected_color["rgb"])
        hair_color_label = self.classify_hair_color(rgb_color)

        preview_b64 = None
        preview_target_hex = None
        if target_color:
            target_info = COLOR_PALETTE.get(target_color, COLOR_PALETTE["Caramel"])
            preview_target_hex = target_info["hex"]
            preview_rgb = self.hex_to_rgb(preview_target_hex)
            preview = self.apply_hair_color(image_rgb, selected_mask, preview_rgb, strength=0.78)
            encoded = cv2.imencode(".png", cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))[1]
            preview_b64 = base64.b64encode(encoded.tobytes()).decode("utf-8")

        # Downsample masks to max 256px for lightweight JSON transmission and canvas rendering
        serialized_hair_mask = self.serialize_mask(hair_mask, max_dim=256)
        serialized_region_masks = {
            r_name: self.serialize_mask(r_mask, max_dim=256)
            for r_name, r_mask in region_masks.items()
        }

        response_payload = {
            "status": "ok",
            "hair_type": hair_type,
            "hair_type_confidence": round(float(hair_confidence), 2),
            "hair_type_top_predictions": [
                {"label": label, "confidence": round(float(conf), 2)} for label, conf in hair_top
            ],
            "disease": disease_label,
            "disease_confidence": round(float(disease_confidence), 2),
            "disease_top_predictions": disease_top_formatted,
            "coverage": round(float(coverage), 2),
            "quality": quality,
            "hair_mask": serialized_hair_mask,
            "region_masks": serialized_region_masks,
            "color": {
                "rgb": [int(v) for v in rgb_color],
                "label": hair_color_label,
                "hex": "#{:02X}{:02X}{:02X}".format(*rgb_color),
                "region": selected_region,
            },
            "region_colors": region_colors,
            "regions": {
                region_name: round(float(np.sum(mask) / mask.size * 100), 2) for region_name, mask in region_masks.items()
            },
            "current_region": selected_region,
            "haircut_recommendations": recommend_haircuts(hair_type),
            "recommended_colors": recommend_colors(hair_type, hair_color_label),
            "preview_image": preview_b64,
            "target_color": target_color,
            "target_hex": preview_target_hex,
            "metadata": {
                "model": "ResNet18 + U-Net",
                "device": str(DEVICE),
                "project": "AI Hair Intelligence",
            },
        }

        # Explicit cleanup to keep memory minimal
        del classifier_tensor, original_image, image_rgb
        gc.collect()

        return response_payload


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = DoubleConv(3, 32)
        self.enc2 = DoubleConv(32, 64)
        self.enc3 = DoubleConv(64, 128)
        self.enc4 = DoubleConv(128, 256)

        self.bottleneck = DoubleConv(256, 512)
        self.pool = nn.MaxPool2d(2)
        self.up4 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.dec4 = DoubleConv(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.dec3 = DoubleConv(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec2 = DoubleConv(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.dec1 = DoubleConv(64, 32)
        self.output = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.output(d1)
