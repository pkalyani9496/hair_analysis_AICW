from pathlib import Path
from collections import Counter
from PIL import Image
import random
import shutil

# ============================================================
# AI HAIR INTELLIGENCE PROJECT
# SEGMENTATION DATASET PREPROCESSING
# ============================================================

BASE_DIR = Path(
    r"C:\Users\kalya\OneDrive\Desktop\hair_analysis"
)

CLEANED_DIR = (
    BASE_DIR / "cleaned_datasets" / "hair_segmentation"
)

OUTPUT_DIR = (
    BASE_DIR / "processed_datasets" / "hair_segmentation"
)

IMAGE_SIZE = (224, 224)

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

RANDOM_SEED = 42


# ============================================================
# FIND IMAGE / MASK PAIRS
# ============================================================

def find_pairs():

    print("\n" + "=" * 70)
    print("FINDING IMAGE / MASK PAIRS")
    print("=" * 70)

    image_files = sorted(
        CLEANED_DIR.rglob("*-org.jpg")
    )

    mask_files = sorted(
        CLEANED_DIR.rglob("*-gt.pbm")
    )

    print(
        f"\nImages found: {len(image_files)}"
    )

    print(
        f"Masks found: {len(mask_files)}"
    )

    # --------------------------------------------------------
    # Build mask lookup
    # --------------------------------------------------------

    mask_lookup = {}

    for mask in mask_files:

        frame_id = mask.stem.replace(
            "-gt",
            ""
        )

        # Include dataset + frame ID to avoid collisions
        relative = mask.relative_to(
            CLEANED_DIR
        )

        dataset = (
            relative.parts[0]
            if relative.parts
            else "unknown"
        )

        key = (
            dataset,
            frame_id
        )

        mask_lookup[key] = mask

    pairs = []
    unmatched_images = []

    # --------------------------------------------------------
    # Match images
    # --------------------------------------------------------

    for image in image_files:

        frame_id = image.stem.replace(
            "-org",
            ""
        )

        relative = image.relative_to(
            CLEANED_DIR
        )

        dataset = (
            relative.parts[0]
            if relative.parts
            else "unknown"
        )

        key = (
            dataset,
            frame_id
        )

        mask = mask_lookup.get(key)

        if mask is not None:

            pairs.append(
                (
                    image,
                    mask,
                    dataset,
                    frame_id
                )
            )

        else:

            unmatched_images.append(
                image
            )

    matched_mask_keys = {
        (
            dataset,
            frame_id
        )
        for _, _, dataset, frame_id in pairs
    }

    unmatched_masks = []

    for mask in mask_files:

        frame_id = mask.stem.replace(
            "-gt",
            ""
        )

        relative = mask.relative_to(
            CLEANED_DIR
        )

        dataset = (
            relative.parts[0]
            if relative.parts
            else "unknown"
        )

        key = (
            dataset,
            frame_id
        )

        if key not in matched_mask_keys:

            unmatched_masks.append(
                mask
            )

    print(
        f"\nMatched pairs: {len(pairs)}"
    )

    print(
        f"Images without masks: "
        f"{len(unmatched_images)}"
    )

    print(
        f"Masks without images: "
        f"{len(unmatched_masks)}"
    )

    return (
        pairs,
        unmatched_images,
        unmatched_masks
    )


# ============================================================
# CHECK IMAGE / MASK DIMENSIONS
# ============================================================

def check_dimensions(pairs):

    print("\n" + "=" * 70)
    print("CHECKING IMAGE / MASK DIMENSIONS")
    print("=" * 70)

    valid_pairs = []
    dimension_mismatches = []

    for image, mask, dataset, frame_id in pairs:

        try:

            with Image.open(image) as img:

                image_size = img.size

            with Image.open(mask) as msk:

                mask_size = msk.size

            if image_size == mask_size:

                valid_pairs.append(
                    (
                        image,
                        mask,
                        dataset,
                        frame_id
                    )
                )

            else:

                dimension_mismatches.append(
                    (
                        image,
                        mask,
                        image_size,
                        mask_size
                    )
                )

        except Exception as error:

            print(
                f"[FAILED CHECK] {image}"
            )

            print(
                f"                {error}"
            )

    print(
        f"\nValid dimension pairs: "
        f"{len(valid_pairs)}"
    )

    print(
        f"Dimension mismatches: "
        f"{len(dimension_mismatches)}"
    )

    if dimension_mismatches:

        print(
            "\nExamples of mismatches:"
        )

        for (
            image,
            mask,
            image_size,
            mask_size
        ) in dimension_mismatches[:10]:

            print(
                f"  {image.name}: "
                f"{image_size} vs {mask_size}"
            )

    return valid_pairs


# ============================================================
# SAVE IMAGE
# ============================================================

def save_image(
    source,
    destination
):

    destination.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with Image.open(source) as img:

        img = img.convert("RGB")

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
# SAVE MASK
# ============================================================

def save_mask(
    source,
    destination
):

    destination.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with Image.open(source) as mask:

        # Convert mask to grayscale
        mask = mask.convert("L")

        # IMPORTANT:
        # Nearest-neighbor preserves segmentation labels.
        mask = mask.resize(
            IMAGE_SIZE,
            Image.Resampling.NEAREST
        )

        # Save as PNG so the mask remains lossless.
        mask.save(
            destination,
            format="PNG"
        )


# ============================================================
# SPLIT PAIRS
# ============================================================

def split_pairs(pairs):

    print("\n" + "=" * 70)
    print("CREATING DATASET SPLITS")
    print("=" * 70)

    random.seed(RANDOM_SEED)

    # Keep datasets separate.
    datasets = {}

    for pair in pairs:

        dataset = pair[2]

        if dataset not in datasets:

            datasets[dataset] = []

        datasets[dataset].append(pair)

    split_data = {}

    for dataset, dataset_pairs in sorted(
        datasets.items()
    ):

        random.shuffle(dataset_pairs)

        total = len(dataset_pairs)

        train_count = int(
            total * TRAIN_RATIO
        )

        val_count = int(
            total * VAL_RATIO
        )

        train = dataset_pairs[
            :train_count
        ]

        val = dataset_pairs[
            train_count:
            train_count + val_count
        ]

        test = dataset_pairs[
            train_count + val_count:
        ]

        split_data[dataset] = {
            "train": train,
            "val": val,
            "test": test
        }

        print(
            f"\n{dataset}"
        )

        print(
            f"  Total: {total}"
        )

        print(
            f"  Train: {len(train)}"
        )

        print(
            f"  Validation: {len(val)}"
        )

        print(
            f"  Test: {len(test)}"
        )

    return split_data


# ============================================================
# PROCESS SPLITS
# ============================================================

def process_splits(split_data):

    print("\n" + "=" * 70)
    print("PROCESSING SEGMENTATION IMAGES AND MASKS")
    print("=" * 70)

    processed = 0
    failed = 0

    for dataset, splits in split_data.items():

        for split_name, pairs in splits.items():

            for index, (
                image,
                mask,
                _,
                frame_id
            ) in enumerate(pairs):

                try:

                    base_name = (
                        f"{dataset}_{frame_id}"
                    )

                    image_destination = (
                        OUTPUT_DIR
                        / split_name
                        / "images"
                        / f"{base_name}.jpg"
                    )

                    mask_destination = (
                        OUTPUT_DIR
                        / split_name
                        / "masks"
                        / f"{base_name}.png"
                    )

                    save_image(
                        image,
                        image_destination
                    )

                    save_mask(
                        mask,
                        mask_destination
                    )

                    processed += 1

                except Exception as error:

                    failed += 1

                    print(
                        f"\n[FAILED]"
                    )

                    print(
                        f"Image: {image}"
                    )

                    print(
                        f"Mask: {mask}"
                    )

                    print(
                        f"Error: {error}"
                    )

    print(
        f"\nProcessed pairs: {processed}"
    )

    print(
        f"Failed pairs: {failed}"
    )

    return processed, failed


# ============================================================
# VERIFY OUTPUT
# ============================================================

def verify_output():

    print("\n" + "=" * 70)
    print("VERIFYING PROCESSED SEGMENTATION DATASET")
    print("=" * 70)

    for split in [
        "train",
        "val",
        "test"
    ]:

        image_dir = (
            OUTPUT_DIR
            / split
            / "images"
        )

        mask_dir = (
            OUTPUT_DIR
            / split
            / "masks"
        )

        images = sorted(
            image_dir.glob("*.jpg")
        )

        masks = sorted(
            mask_dir.glob("*.png")
        )

        image_names = {
            p.stem
            for p in images
        }

        mask_names = {
            p.stem
            for p in masks
        }

        matched = (
            image_names & mask_names
        )

        print(
            f"\n{split}"
        )

        print(
            f"  Images: {len(images)}"
        )

        print(
            f"  Masks: {len(masks)}"
        )

        print(
            f"  Matched: {len(matched)}"
        )

        print(
            f"  Image/mask difference: "
            f"{len(image_names ^ mask_names)}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("AI HAIR INTELLIGENCE PROJECT")
    print("SEGMENTATION DATASET PREPROCESSING")
    print("=" * 70)

    print(
        f"\nInput:"
    )

    print(
        CLEANED_DIR
    )

    print(
        f"\nOutput:"
    )

    print(
        OUTPUT_DIR
    )

    print(
        f"\nTarget size:"
        f" {IMAGE_SIZE}"
    )

    # --------------------------------------------------------
    # Recreate output directory
    # --------------------------------------------------------

    if OUTPUT_DIR.exists():

        print(
            "\nRemoving previous segmentation "
            "processed output..."
        )

        shutil.rmtree(
            OUTPUT_DIR
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Find pairs
    # --------------------------------------------------------

    (
        pairs,
        unmatched_images,
        unmatched_masks
    ) = find_pairs()

    # --------------------------------------------------------
    # Dimension validation
    # --------------------------------------------------------

    valid_pairs = check_dimensions(
        pairs
    )

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    split_data = split_pairs(
        valid_pairs
    )

    # --------------------------------------------------------
    # Process
    # --------------------------------------------------------

    processed, failed = process_splits(
        split_data
    )

    # --------------------------------------------------------
    # Verify
    # --------------------------------------------------------

    verify_output()

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("SEGMENTATION PREPROCESSING COMPLETED")
    print("=" * 70)

    print(
        f"\nValid pairs used: "
        f"{len(valid_pairs)}"
    )

    print(
        f"Processed pairs: "
        f"{processed}"
    )

    print(
        f"Failed pairs: "
        f"{failed}"
    )

    print(
        f"\nOutput location:"
    )

    print(
        OUTPUT_DIR
    )

    print(
        "\nOriginal cleaned dataset was not modified."
    )


if __name__ == "__main__":
    main()