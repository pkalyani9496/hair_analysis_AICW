from pathlib import Path
from collections import Counter
from PIL import Image

BASE_DIR = Path(
    r"C:\Users\kalya\OneDrive\Desktop\hair_analysis"
)

SEGMENTATION_DIR = (
    BASE_DIR
    / "cleaned_datasets"
    / "hair_segmentation"
)

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}

MASK_EXTENSIONS = {
    ".pbm",
    ".pgm",
    ".ppm",
}


def get_files(extensions):

    return sorted(
        [
            p
            for p in SEGMENTATION_DIR.rglob("*")
            if p.is_file()
            and p.suffix.lower() in extensions
        ]
    )


def show_examples(title, files, count=20):

    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

    for file in files[:count]:

        relative = file.relative_to(
            SEGMENTATION_DIR
        )

        print(relative)


def inspect_folders():

    print("\n" + "=" * 70)
    print("FOLDER STRUCTURE")
    print("=" * 70)

    folders = set()

    for file in SEGMENTATION_DIR.rglob("*"):

        if file.is_file():

            relative = file.relative_to(
                SEGMENTATION_DIR
            )

            if len(relative.parts) > 1:

                folders.add(
                    relative.parts[0]
                )

    for folder in sorted(folders):

        print(folder)


def inspect_extensions(files):

    counter = Counter(
        file.suffix.lower()
        for file in files
    )

    print("\n" + "=" * 70)
    print("FILE EXTENSIONS")
    print("=" * 70)

    for extension, count in sorted(
        counter.items()
    ):

        print(
            f"{extension:<10} {count}"
        )


def inspect_dimensions(
    title,
    files
):

    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

    dimensions = Counter()

    for file in files:

        try:

            with Image.open(file) as img:

                dimensions[
                    img.size
                ] += 1

        except Exception:
            pass

    for size, count in dimensions.most_common(20):

        print(
            f"{size}: {count}"
        )


def main():

    print("=" * 70)
    print("SEGMENTATION DATASET INVESTIGATION")
    print("=" * 70)

    print(
        "\nDataset:"
    )

    print(
        SEGMENTATION_DIR
    )

    images = get_files(
        IMAGE_EXTENSIONS
    )

    masks = get_files(
        MASK_EXTENSIONS
    )

    print(
        f"\nImages: {len(images)}"
    )

    print(
        f"Masks: {len(masks)}"
    )

    inspect_folders()

    inspect_extensions(
        images + masks
    )

    show_examples(
        "IMAGE FILE EXAMPLES",
        images
    )

    show_examples(
        "MASK FILE EXAMPLES",
        masks
    )

    inspect_dimensions(
        "IMAGE DIMENSIONS",
        images
    )

    inspect_dimensions(
        "MASK DIMENSIONS",
        masks
    )

    print("\n" + "=" * 70)
    print("INVESTIGATION COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()