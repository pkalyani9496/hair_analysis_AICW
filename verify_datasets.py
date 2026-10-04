import os
import hashlib
from pathlib import Path
from collections import Counter, defaultdict
from PIL import Image

# ============================================================
# AI HAIR INTELLIGENCE PROJECT
# DATASET VERIFICATION SCRIPT
#
# READ-ONLY
# This script DOES NOT:
# - delete files
# - move files
# - rename files
# - resize images
# - modify datasets
# ============================================================

PROJECT_ROOT = Path(
    r"C:\Users\kalya\OneDrive\Desktop\hair_analysis"
)

DATASETS = {
    "hair_disease": PROJECT_ROOT / "hair_disease",
    "hair_segmentation": PROJECT_ROOT / "hair_segmentation",
    "hair_type": PROJECT_ROOT / "hair_type",
}

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".gif",
    ".tif",
    ".tiff",
    ".webp",
}

MASK_EXTENSIONS = {
    ".pbm",
    ".pgm",
    ".png",
    ".jpg",
    ".jpeg",
}

REPORT_PATH = PROJECT_ROOT / "dataset_verification_report.txt"


# ============================================================
# GENERAL FUNCTIONS
# ============================================================

def separator(char="=", length=75):
    print(char * length)


def get_files(folder):
    """Return all files recursively."""
    if not folder.exists():
        return []

    return [
        p
        for p in folder.rglob("*")
        if p.is_file()
    ]


def get_image_files(folder, include_masks=False):
    """Return valid image files while ignoring __MACOSX."""

    extensions = IMAGE_EXTENSIONS.copy()

    if include_masks:
        extensions.update(MASK_EXTENSIONS)

    files = []

    if not folder.exists():
        return files

    for path in folder.rglob("*"):

        if not path.is_file():
            continue

        # Ignore macOS metadata
        if "__MACOSX" in path.parts:
            continue

        # Ignore hidden files
        if path.name.startswith("."):
            continue

        if path.suffix.lower() in extensions:
            files.append(path)

    return files


def relative(path):
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def sha256(path):
    """Calculate file hash without modifying the file."""

    h = hashlib.sha256()

    try:
        with open(path, "rb") as f:

            while True:

                chunk = f.read(1024 * 1024)

                if not chunk:
                    break

                h.update(chunk)

        return h.hexdigest()

    except Exception:
        return None


def check_image(path):
    """
    Verify whether an image can be opened.
    """

    try:

        with Image.open(path) as img:

            width, height = img.size
            image_format = img.format
            mode = img.mode

            # Force image verification
            img.verify()

        return {
            "valid": True,
            "width": width,
            "height": height,
            "format": image_format,
            "mode": mode,
            "error": None,
        }

    except Exception as e:

        return {
            "valid": False,
            "width": None,
            "height": None,
            "format": None,
            "mode": None,
            "error": str(e),
        }


# ============================================================
# BASIC DATASET VERIFICATION
# ============================================================

def verify_basic_dataset(name, folder, report):

    separator()
    print(f"DATASET: {name}")
    separator()

    report.append("=" * 75)
    report.append(f"DATASET: {name}")
    report.append("=" * 75)

    if not folder.exists():

        print("❌ DATASET FOLDER NOT FOUND")

        report.append("DATASET FOLDER NOT FOUND")
        return

    print(f"Path: {folder}")

    report.append(f"Path: {folder}")

    all_files = get_files(folder)
    image_files = get_image_files(folder)

    print(f"Total files: {len(all_files)}")
    print(f"Image files: {len(image_files)}")

    report.append(f"Total files: {len(all_files)}")
    report.append(f"Image files: {len(image_files)}")

    # --------------------------------------------------------
    # Extensions
    # --------------------------------------------------------

    extension_counts = Counter()

    for path in all_files:

        extension = path.suffix.lower()

        if extension:
            extension_counts[extension] += 1
        else:
            extension_counts["[NO EXTENSION]"] += 1

    print()
    print("File types:")

    report.append("")
    report.append("File types:")

    for extension, count in sorted(
        extension_counts.items()
    ):

        print(f"  {extension:<15} {count}")

        report.append(
            f"  {extension:<15} {count}"
        )

    # --------------------------------------------------------
    # Image verification
    # --------------------------------------------------------

    print()
    print("Checking image integrity...")

    report.append("")
    report.append("IMAGE INTEGRITY")

    corrupted = []
    dimensions = Counter()
    formats = Counter()
    modes = Counter()
    small_images = []

    for image in image_files:

        result = check_image(image)

        if not result["valid"]:

            corrupted.append(
                (
                    image,
                    result["error"]
                )
            )

            continue

        dimensions[
            (
                result["width"],
                result["height"]
            )
        ] += 1

        formats[result["format"]] += 1
        modes[result["mode"]] += 1

        if (
            result["width"] < 100
            or result["height"] < 100
        ):

            small_images.append(
                (
                    image,
                    result["width"],
                    result["height"]
                )
            )

    print(
        f"Corrupted/unreadable images: "
        f"{len(corrupted)}"
    )

    report.append(
        f"Corrupted/unreadable images: "
        f"{len(corrupted)}"
    )

    if corrupted:

        print("\nCorrupted examples:")

        report.append("")
        report.append("Corrupted examples:")

        for image, error in corrupted[:20]:

            print(f"  {relative(image)}")
            print(f"    Error: {error}")

            report.append(
                f"  {relative(image)}"
            )

            report.append(
                f"    Error: {error}"
            )

    # --------------------------------------------------------
    # Formats
    # --------------------------------------------------------

    print()
    print("Image formats:")

    report.append("")
    report.append("Image formats:")

    for fmt, count in formats.items():

        print(f"  {fmt}: {count}")
        report.append(f"  {fmt}: {count}")

    # --------------------------------------------------------
    # Dimensions
    # --------------------------------------------------------

    print()
    print("Most common dimensions:")

    report.append("")
    report.append("Most common dimensions:")

    for dimension, count in dimensions.most_common(15):

        print(
            f"  {dimension[0]} x {dimension[1]}"
            f" : {count}"
        )

        report.append(
            f"  {dimension[0]} x {dimension[1]}"
            f" : {count}"
        )

    # --------------------------------------------------------
    # Modes
    # --------------------------------------------------------

    print()
    print("Image modes:")

    report.append("")
    report.append("Image modes:")

    for mode, count in modes.items():

        print(f"  {mode}: {count}")
        report.append(f"  {mode}: {count}")

    # --------------------------------------------------------
    # Small images
    # --------------------------------------------------------

    print()
    print(
        f"Very small images (<100x100): "
        f"{len(small_images)}"
    )

    report.append("")
    report.append(
        f"Very small images (<100x100): "
        f"{len(small_images)}"
    )

    if small_images:

        print("\nExamples:")

        report.append("")
        report.append("Examples:")

        for image, width, height in small_images[:20]:

            print(
                f"  {relative(image)}"
                f" -> {width}x{height}"
            )

            report.append(
                f"  {relative(image)}"
                f" -> {width}x{height}"
            )


# ============================================================
# DUPLICATE CHECK
# ============================================================

def check_duplicates(name, folder, report):

    print()
    separator("-")
    print(f"DUPLICATE CHECK: {name}")
    separator("-")

    report.append("")
    report.append("-" * 75)
    report.append(f"DUPLICATE CHECK: {name}")
    report.append("-" * 75)

    images = get_image_files(folder)

    hashes = defaultdict(list)

    print("Calculating SHA256 hashes...")

    for image in images:

        file_hash = sha256(image)

        if file_hash:

            hashes[file_hash].append(image)

    duplicate_groups = [
        paths
        for paths in hashes.values()
        if len(paths) > 1
    ]

    print(
        f"Exact duplicate groups: "
        f"{len(duplicate_groups)}"
    )

    report.append(
        f"Exact duplicate groups: "
        f"{len(duplicate_groups)}"
    )

    duplicate_files = sum(
        len(group)
        for group in duplicate_groups
    )

    print(
        f"Files belonging to duplicate groups: "
        f"{duplicate_files}"
    )

    report.append(
        f"Files belonging to duplicate groups: "
        f"{duplicate_files}"
    )

    if duplicate_groups:

        print()
        print("Duplicate examples:")

        report.append("")
        report.append("Duplicate examples:")

        for number, group in enumerate(
            duplicate_groups[:20],
            start=1
        ):

            print(f"\nGroup {number}:")

            report.append(
                f"\nGroup {number}:"
            )

            for path in group:

                print(
                    f"  {relative(path)}"
                )

                report.append(
                    f"  {relative(path)}"
                )


# ============================================================
# HAIR DISEASE VERIFICATION
# ============================================================

def verify_hair_disease(report):

    folder = DATASETS["hair_disease"]

    separator()
    print("HAIR DISEASE DATASET")
    separator()

    report.append("")
    report.append("=" * 75)
    report.append("HAIR DISEASE DATASET")
    report.append("=" * 75)

    if not folder.exists():

        print("❌ Folder not found.")
        return

    # Find train / val / test
    split_counts = {}

    for split in ["train", "val", "test"]:

        split_folder = folder / "Hair Diseases - Final" / split

        if not split_folder.exists():

            # fallback: search case-insensitively
            candidates = [
                p
                for p in folder.rglob("*")
                if p.is_dir()
                and p.name.lower() == split.lower()
            ]

            if candidates:
                split_folder = candidates[0]

        if split_folder.exists():

            images = get_image_files(split_folder)

            class_counts = Counter()

            for image in images:

                parts = image.relative_to(
                    split_folder
                ).parts

                if parts:

                    class_counts[parts[0]] += 1

            split_counts[split] = class_counts

    for split in ["train", "val", "test"]:

        print()
        print(split.upper())

        report.append("")
        report.append(split.upper())

        if split not in split_counts:

            print("  ❌ Split not found")
            report.append("  ❌ Split not found")
            continue

        total = 0

        for class_name, count in sorted(
            split_counts[split].items()
        ):

            print(
                f"  {class_name:<30} {count}"
            )

            report.append(
                f"  {class_name:<30} {count}"
            )

            total += count

        print(f"  TOTAL: {total}")

        report.append(
            f"  TOTAL: {total}"
        )

    # --------------------------------------------------------
    # Check class consistency
    # --------------------------------------------------------

    print()
    print("Checking class consistency...")

    report.append("")
    report.append("CLASS CONSISTENCY")

    class_sets = {}

    for split, counts in split_counts.items():

        class_sets[split] = set(counts.keys())

    all_classes = set()

    for classes in class_sets.values():
        all_classes.update(classes)

    consistent = True

    for split, classes in class_sets.items():

        missing = all_classes - classes

        if missing:

            consistent = False

            print(
                f"  ⚠ {split} missing: "
                f"{sorted(missing)}"
            )

            report.append(
                f"  ⚠ {split} missing: "
                f"{sorted(missing)}"
            )

    if consistent:

        print("  ✓ All splits contain the same classes.")
        report.append(
            "  ✓ All splits contain the same classes."
        )


# ============================================================
# HAIR TYPE VERIFICATION
# ============================================================

def verify_hair_type(report):

    folder = DATASETS["hair_type"]

    separator()
    print("HAIR TYPE DATASET")
    separator()

    report.append("")
    report.append("=" * 75)
    report.append("HAIR TYPE DATASET")
    report.append("=" * 75)

    if not folder.exists():

        print("❌ Folder not found.")
        return

    images = get_image_files(folder)

    classes = Counter()

    for image in images:

        relative_path = image.relative_to(folder)

        if len(relative_path.parts) >= 2:

            class_name = relative_path.parts[0]

            classes[class_name] += 1

    print(f"Total images: {len(images)}")

    report.append(
        f"Total images: {len(images)}"
    )

    print()
    print("Class distribution:")

    report.append("")
    report.append("Class distribution:")

    for class_name, count in sorted(classes.items()):

        print(
            f"  {class_name:<25} {count}"
        )

        report.append(
            f"  {class_name:<25} {count}"
        )

    # Balance calculation
    if classes:

        values = list(classes.values())

        minimum = min(values)
        maximum = max(values)

        ratio = minimum / maximum

        print()
        print(
            f"Smallest class: {minimum}"
        )

        print(
            f"Largest class: {maximum}"
        )

        print(
            f"Minimum/Maximum ratio: "
            f"{ratio:.3f}"
        )

        report.append("")
        report.append(
            f"Smallest class: {minimum}"
        )

        report.append(
            f"Largest class: {maximum}"
        )

        report.append(
            f"Minimum/Maximum ratio: "
            f"{ratio:.3f}"
        )


# ============================================================
# SEGMENTATION VERIFICATION
# ============================================================

def find_folder(parent, name):

    if not parent.exists():
        return None

    for item in parent.iterdir():

        if (
            item.is_dir()
            and item.name.lower() == name.lower()
        ):

            return item

    return None


def verify_segmentation(report):

    folder = DATASETS["hair_segmentation"]

    separator()
    print("HAIR SEGMENTATION DATASET")
    separator()

    report.append("")
    report.append("=" * 75)
    report.append("HAIR SEGMENTATION DATASET")
    report.append("=" * 75)

    if not folder.exists():

        print("❌ Folder not found.")
        return

    # --------------------------------------------------------
    # MACOSX
    # --------------------------------------------------------

    mac_files = []

    for path in folder.rglob("*"):

        if "__MACOSX" in path.parts and path.is_file():

            mac_files.append(path)

    print(
        f"__MACOSX files found: "
        f"{len(mac_files)}"
    )

    report.append(
        f"__MACOSX files found: "
        f"{len(mac_files)}"
    )

    # --------------------------------------------------------
    # FIGARO
    # --------------------------------------------------------

    figaro = find_folder(folder, "Figaro1k")

    if figaro is None:

        print("❌ Figaro1k not found.")
        report.append("❌ Figaro1k not found.")

    else:

        nested = find_folder(figaro, "Figaro1k")

        if nested:
            figaro = nested

        print()
        print("FIGARO1K")

        report.append("")
        report.append("FIGARO1K")

        original = find_folder(
            figaro,
            "Original"
        )

        gt = find_folder(
            figaro,
            "GT"
        )

        if original:

            original_images = get_image_files(
                original
            )

            print(
                f"Original images: "
                f"{len(original_images)}"
            )

            report.append(
                f"Original images: "
                f"{len(original_images)}"
            )

        else:

            original_images = []

            print("Original folder not found.")
            report.append(
                "Original folder not found."
            )

        if gt:

            mask_files = []

            for path in gt.rglob("*"):

                if not path.is_file():
                    continue

                if "__MACOSX" in path.parts:
                    continue

                if path.name.startswith("."):
                    continue

                if path.suffix.lower() in MASK_EXTENSIONS:

                    mask_files.append(path)

            print(
                f"Mask files: "
                f"{len(mask_files)}"
            )

            report.append(
                f"Mask files: "
                f"{len(mask_files)}"
            )

        else:

            mask_files = []

            print("GT folder not found.")
            report.append(
                "GT folder not found."
            )

        # ----------------------------------------------------
        # Split counts
        # ----------------------------------------------------

        print()
        print("Figaro split counts:")

        report.append("")
        report.append("Figaro split counts:")

        for split in ["Training", "Testing"]:

            if original:

                split_folder = find_folder(
                    original,
                    split
                )

                if split_folder:

                    count = len(
                        get_image_files(
                            split_folder
                        )
                    )

                    print(
                        f"  Original {split}: "
                        f"{count}"
                    )

                    report.append(
                        f"  Original {split}: "
                        f"{count}"
                    )

            if gt:

                split_folder = find_folder(
                    gt,
                    split
                )

                if split_folder:

                    count = 0

                    for path in split_folder.rglob("*"):

                        if not path.is_file():
                            continue

                        if "__MACOSX" in path.parts:
                            continue

                        if path.name.startswith("."):
                            continue

                        if (
                            path.suffix.lower()
                            in MASK_EXTENSIONS
                        ):

                            count += 1

                    print(
                        f"  GT {split}: "
                        f"{count}"
                    )

                    report.append(
                        f"  GT {split}: "
                        f"{count}"
                    )

    # --------------------------------------------------------
    # PATCH1K
    # --------------------------------------------------------

    patch = find_folder(folder, "Patch1k")

    print()
    print("PATCH1K")

    report.append("")
    report.append("PATCH1K")

    if patch is None:

        print("❌ Patch1k not found.")
        report.append("❌ Patch1k not found.")

    else:

        nested = find_folder(
            patch,
            "Patch1k"
        )

        if nested:
            patch = nested

        for class_name in ["Hair", "NonHair"]:

            class_folder = find_folder(
                patch,
                class_name
            )

            if class_folder:

                images = get_image_files(
                    class_folder
                )

                print(
                    f"{class_name}: "
                    f"{len(images)}"
                )

                report.append(
                    f"{class_name}: "
                    f"{len(images)}"
                )

                for split in ["Training", "Testing"]:

                    split_folder = find_folder(
                        class_folder,
                        split
                    )

                    if split_folder:

                        count = len(
                            get_image_files(
                                split_folder
                            )
                        )

                        print(
                            f"  {split}: "
                            f"{count}"
                        )

                        report.append(
                            f"  {split}: "
                            f"{count}"
                        )

            else:

                print(
                    f"{class_name}: folder not found"
                )

                report.append(
                    f"{class_name}: folder not found"
                )


# ============================================================
# FINAL SUMMARY
# ============================================================

def main():

    report = []

    separator("=")
    print("AI HAIR INTELLIGENCE PROJECT")
    print("DATASET VERIFICATION")
    separator("=")

    print()
    print(f"Project: {PROJECT_ROOT}")

    report.append(
        "AI HAIR INTELLIGENCE PROJECT"
    )

    report.append(
        "DATASET VERIFICATION"
    )

    report.append(
        f"Project: {PROJECT_ROOT}"
    )

    # --------------------------------------------------------
    # Check dataset folders
    # --------------------------------------------------------

    print()
    print("DATASET FOLDER STATUS")

    report.append("")
    report.append("DATASET FOLDER STATUS")

    for name, path in DATASETS.items():

        if path.exists():

            print(f"  ✓ {name}")

            report.append(
                f"  ✓ {name}"
            )

        else:

            print(f"  ❌ {name} NOT FOUND")

            report.append(
                f"  ❌ {name} NOT FOUND"
            )

    # --------------------------------------------------------
    # Verify datasets
    # --------------------------------------------------------

    for name, folder in DATASETS.items():

        verify_basic_dataset(
            name,
            folder,
            report
        )

    # --------------------------------------------------------
    # Specific verification
    # --------------------------------------------------------

    verify_hair_disease(report)

    verify_hair_type(report)

    verify_segmentation(report)

    # --------------------------------------------------------
    # Duplicate checks
    # --------------------------------------------------------

    check_duplicates(
        "hair_disease",
        DATASETS["hair_disease"],
        report
    )

    check_duplicates(
        "hair_type",
        DATASETS["hair_type"],
        report
    )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    separator("=")
    print("VERIFICATION COMPLETED")
    separator("=")

    report.append("")
    report.append("=" * 75)
    report.append("VERIFICATION COMPLETED")
    report.append("=" * 75)

    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "\n".join(report)
        )

    print()
    print("Report saved to:")

    print(REPORT_PATH)

    print()
    print("IMPORTANT:")
    print("No dataset files were modified.")
    print()
    print(
        "Send me the COMPLETE terminal output "
        "before we clean anything."
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()