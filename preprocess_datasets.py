from pathlib import Path
from collections import Counter
from PIL import Image
import random
import shutil

# ============================================================
# AI HAIR INTELLIGENCE PROJECT
# DATASET PREPROCESSING
# ============================================================

BASE_DIR = Path(
    r"C:\Users\kalya\OneDrive\Desktop\hair_analysis"
)

CLEANED_DIR = BASE_DIR / "cleaned_datasets"
PROCESSED_DIR = BASE_DIR / "processed_datasets"

DISEASE_DIR = CLEANED_DIR / "hair_disease"
TYPE_DIR = CLEANED_DIR / "hair_type"
SEGMENTATION_DIR = CLEANED_DIR / "hair_segmentation"

DISEASE_OUTPUT = PROCESSED_DIR / "hair_disease"
TYPE_OUTPUT = PROCESSED_DIR / "hair_type"
SEGMENTATION_OUTPUT = PROCESSED_DIR / "hair_segmentation"

# Standard image size for model training
IMAGE_SIZE = (224, 224)

# Hair-type split
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

RANDOM_SEED = 42

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
# HELPER FUNCTIONS
# ============================================================

def get_image_files(root, extensions):

    if not root.exists():
        return []

    return sorted(
        [
            p
            for p in root.rglob("*")
            if p.is_file()
            and p.suffix.lower() in extensions
        ]
    )


def recreate_directory(directory):

    if directory.exists():
        shutil.rmtree(directory)

    directory.mkdir(
        parents=True,
        exist_ok=True
    )


def save_resized_image(source, destination):

    destination.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with Image.open(source) as img:

        # Convert everything to RGB for model compatibility
        img = img.convert("RGB")

        # Resize to standard model input size
        img = img.resize(
            IMAGE_SIZE,
            Image.Resampling.LANCZOS
        )

        img.save(
            destination,
            format="JPEG",
            quality=95
        )


# ============================================================
# HAIR DISEASE PREPROCESSING
# ============================================================

def preprocess_disease():

    print("\n" + "=" * 70)
    print("PREPROCESSING HAIR DISEASE DATASET")
    print("=" * 70)

    recreate_directory(DISEASE_OUTPUT)

    files = get_image_files(
        DISEASE_DIR,
        IMAGE_EXTENSIONS
    )

    processed = 0
    failed = 0

    for source in files:

        try:

            relative = source.relative_to(
                DISEASE_DIR
            )

            destination = (
                DISEASE_OUTPUT
                / relative.with_suffix(".jpg")
            )

            save_resized_image(
                source,
                destination
            )

            processed += 1

        except Exception as error:

            failed += 1

            print(
                f"[FAILED] {source}"
            )

            print(
                f"         {error}"
            )

    print(f"\nOriginal images: {len(files)}")
    print(f"Processed images: {processed}")
    print(f"Failed images: {failed}")
    print(f"Output size: {IMAGE_SIZE}")


# ============================================================
# HAIR TYPE PREPROCESSING
# ============================================================

def preprocess_hair_type():

    print("\n" + "=" * 70)
    print("PREPROCESSING HAIR TYPE DATASET")
    print("=" * 70)

    recreate_directory(TYPE_OUTPUT)

    random.seed(RANDOM_SEED)

    class_directories = [
        p
        for p in TYPE_DIR.iterdir()
        if p.is_dir()
    ]

    total_processed = 0

    for class_directory in sorted(
        class_directories
    ):

        hair_class = class_directory.name

        files = get_image_files(
            class_directory,
            IMAGE_EXTENSIONS
        )

        random.shuffle(files)

        total = len(files)

        train_count = int(
            total * TRAIN_RATIO
        )

        val_count = int(
            total * VAL_RATIO
        )

        train_files = files[
            :train_count
        ]

        val_files = files[
            train_count:
            train_count + val_count
        ]

        test_files = files[
            train_count + val_count:
        ]

        print(
            f"\n{hair_class}"
        )

        print(
            f"  Total: {total}"
        )

        print(
            f"  Train: {len(train_files)}"
        )

        print(
            f"  Validation: {len(val_files)}"
        )

        print(
            f"  Test: {len(test_files)}"
        )

        splits = {
            "train": train_files,
            "val": val_files,
            "test": test_files,
        }

        for split_name, split_files in splits.items():

            for index, source in enumerate(
                split_files
            ):

                try:

                    destination = (
                        TYPE_OUTPUT
                        / split_name
                        / hair_class
                        / f"{index:05d}.jpg"
                    )

                    save_resized_image(
                        source,
                        destination
                    )

                    total_processed += 1

                except Exception as error:

                    print(
                        f"[FAILED] {source}"
                    )

                    print(
                        f"         {error}"
                    )

    print(
        f"\nTotal processed hair-type images: "
        f"{total_processed}"
    )

    print(
        f"Output size: {IMAGE_SIZE}"
    )


# ============================================================
# SEGMENTATION PAIR ANALYSIS
# ============================================================

def analyze_segmentation():

    print("\n" + "=" * 70)
    print("ANALYZING SEGMENTATION DATASET")
    print("=" * 70)

    image_files = [
        p
        for p in SEGMENTATION_DIR.rglob("*")
        if p.is_file()
        and p.suffix.lower() in {
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".webp",
        }
    ]

    mask_files = [
        p
        for p in SEGMENTATION_DIR.rglob("*")
        if p.is_file()
        and p.suffix.lower() in {
            ".pbm",
            ".pgm",
            ".ppm",
        }
    ]

    print(
        f"\nImage files: {len(image_files)}"
    )

    print(
        f"Mask files: {len(mask_files)}"
    )

    image_names = {
        p.stem
        for p in image_files
    }

    mask_names = {
        p.stem
        for p in mask_files
    }

    paired = image_names & mask_names

    images_without_masks = (
        image_names - mask_names
    )

    masks_without_images = (
        mask_names - image_names
    )

    print(
        f"Potential pairs: {len(paired)}"
    )

    print(
        f"Images without matching mask: "
        f"{len(images_without_masks)}"
    )

    print(
        f"Masks without matching image: "
        f"{len(masks_without_images)}"
    )

    if images_without_masks:

        print("\nExample images without masks:")

        for name in sorted(
            images_without_masks
        )[:10]:

            print(
                f"  {name}"
            )

    if masks_without_images:

        print("\nExample masks without images:")

        for name in sorted(
            masks_without_images
        )[:10]:

            print(
                f"  {name}"
            )

    return (
        image_files,
        mask_files
    )


# ============================================================
# SEGMENTATION IMAGE DIMENSIONS
# ============================================================

def inspect_segmentation_dimensions(
    image_files,
    mask_files
):

    print("\n" + "=" * 70)
    print("SEGMENTATION DIMENSION CHECK")
    print("=" * 70)

    image_sizes = Counter()
    mask_sizes = Counter()

    for file in image_files:

        try:

            with Image.open(file) as img:

                image_sizes[
                    img.size
                ] += 1

        except Exception:
            pass

    for file in mask_files:

        try:

            with Image.open(file) as img:

                mask_sizes[
                    img.size
                ] += 1

        except Exception:
            pass

    print("\nMost common image sizes:")

    for size, count in image_sizes.most_common(10):

        print(
            f"  {size}: {count}"
        )

    print("\nMost common mask sizes:")

    for size, count in mask_sizes.most_common(10):

        print(
            f"  {size}: {count}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("AI HAIR INTELLIGENCE PROJECT")
    print("DATASET PREPROCESSING")
    print("=" * 70)

    print(
        f"\nInput dataset:"
    )

    print(
        CLEANED_DIR
    )

    print(
        f"\nOutput dataset:"
    )

    print(
        PROCESSED_DIR
    )

    print(
        f"\nTarget image size:"
        f" {IMAGE_SIZE}"
    )

    # --------------------------------------------------------
    # Disease
    # --------------------------------------------------------

    preprocess_disease()

    # --------------------------------------------------------
    # Hair type
    # --------------------------------------------------------

    preprocess_hair_type()

    # --------------------------------------------------------
    # Segmentation
    # --------------------------------------------------------

    image_files, mask_files = (
        analyze_segmentation()
    )

    inspect_segmentation_dimensions(
        image_files,
        mask_files
    )

    # --------------------------------------------------------
    # Completion
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("PREPROCESSING ANALYSIS COMPLETED")
    print("=" * 70)

    print(
        f"\nProcessed datasets are located at:"
    )

    print(
        PROCESSED_DIR
    )

    print(
        "\nYour original cleaned datasets "
        "were not modified."
    )


if __name__ == "__main__":
    main()