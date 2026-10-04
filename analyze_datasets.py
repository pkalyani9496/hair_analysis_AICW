from pathlib import Path
from collections import Counter
from PIL import Image

# ============================================================
# AI HAIR INTELLIGENCE PROJECT
# DATASET ANALYSIS
# ============================================================

BASE_DIR = Path(
    r"C:\Users\kalya\OneDrive\Desktop\hair_analysis"
)

CLEANED_DIR = BASE_DIR / "cleaned_datasets"

DISEASE_DIR = CLEANED_DIR / "hair_disease"
TYPE_DIR = CLEANED_DIR / "hair_type"
SEGMENTATION_DIR = CLEANED_DIR / "hair_segmentation"

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".gif",
    ".webp",
    ".tif",
    ".tiff",
}

SEGMENTATION_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".pbm",
    ".pgm",
    ".ppm",
    ".tif",
    ".tiff",
}


# ============================================================
# HELPER
# ============================================================

def image_files(root, extensions=None):

    if not root.exists():
        return []

    if extensions is None:
        extensions = IMAGE_EXTENSIONS

    return [
        p for p in root.rglob("*")
        if p.is_file()
        and p.suffix.lower() in extensions
    ]


def print_distribution(counter):

    total = sum(counter.values())

    for name, count in sorted(counter.items()):

        percentage = (
            count / total * 100
            if total > 0
            else 0
        )

        print(
            f"  {name:<35} "
            f"{count:>6} "
            f"({percentage:>6.2f}%)"
        )


# ============================================================
# HAIR DISEASE ANALYSIS
# ============================================================

def analyze_disease():

    print("\n" + "=" * 70)
    print("HAIR DISEASE DATASET ANALYSIS")
    print("=" * 70)

    files = image_files(DISEASE_DIR)

    print(f"\nTotal images: {len(files)}")

    split_counter = Counter()
    class_counter = Counter()
    split_class_counter = Counter()

    for file in files:

        relative = file.relative_to(DISEASE_DIR)

        parts = relative.parts

        if len(parts) >= 2:

            split = parts[0]
            disease_class = parts[1]

            split_counter[split] += 1
            class_counter[disease_class] += 1

            split_class_counter[
                (split, disease_class)
            ] += 1

    print("\nDataset splits:")
    print_distribution(split_counter)

    print("\nDisease classes:")
    print_distribution(class_counter)

    print("\nImages per class and split:")

    splits = sorted(split_counter.keys())
    classes = sorted(class_counter.keys())

    for disease_class in classes:

        values = []

        for split in splits:

            count = split_class_counter[
                (split, disease_class)
            ]

            values.append(
                f"{split}: {count}"
            )

        print(
            f"  {disease_class:<35} "
            + " | ".join(values)
        )


# ============================================================
# HAIR TYPE ANALYSIS
# ============================================================

def analyze_type():

    print("\n" + "=" * 70)
    print("HAIR TYPE DATASET ANALYSIS")
    print("=" * 70)

    files = image_files(TYPE_DIR)

    print(f"\nTotal images: {len(files)}")

    class_counter = Counter()

    for file in files:

        relative = file.relative_to(TYPE_DIR)

        if len(relative.parts) >= 2:

            hair_class = relative.parts[0]

            class_counter[hair_class] += 1

    print("\nHair type classes:")
    print_distribution(class_counter)


# ============================================================
# SEGMENTATION ANALYSIS
# ============================================================

def analyze_segmentation():

    print("\n" + "=" * 70)
    print("HAIR SEGMENTATION DATASET ANALYSIS")
    print("=" * 70)

    files = image_files(
        SEGMENTATION_DIR,
        SEGMENTATION_EXTENSIONS
    )

    extension_counter = Counter()

    for file in files:
        extension_counter[
            file.suffix.lower()
        ] += 1

    print(f"\nTotal files: {len(files)}")

    print("\nFile types:")

    for extension, count in sorted(
        extension_counter.items()
    ):
        print(
            f"  {extension:<10} {count:>6}"
        )


# ============================================================
# IMAGE DIMENSION ANALYSIS
# ============================================================

def analyze_dimensions():

    print("\n" + "=" * 70)
    print("IMAGE DIMENSION ANALYSIS")
    print("=" * 70)

    datasets = {
        "hair_disease": DISEASE_DIR,
        "hair_type": TYPE_DIR,
    }

    for dataset_name, root in datasets.items():

        files = image_files(root)

        width_counter = Counter()
        height_counter = Counter()

        checked = 0

        for file in files:

            try:

                with Image.open(file) as img:

                    width, height = img.size

                width_counter[width] += 1
                height_counter[height] += 1

                checked += 1

            except Exception:
                pass

        print(f"\n{dataset_name}")

        print(
            f"  Images checked: {checked}"
        )

        print(
            "  Most common widths:"
        )

        for width, count in width_counter.most_common(10):

            print(
                f"    {width}px: {count}"
            )

        print(
            "  Most common heights:"
        )

        for height, count in height_counter.most_common(10):

            print(
                f"    {height}px: {count}"
            )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("AI HAIR INTELLIGENCE PROJECT")
    print("CLEANED DATASET ANALYSIS")
    print("=" * 70)

    print("\nDataset location:")
    print(CLEANED_DIR)

    analyze_disease()

    analyze_type()

    analyze_segmentation()

    analyze_dimensions()

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()