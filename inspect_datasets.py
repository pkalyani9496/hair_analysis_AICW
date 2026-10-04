import os
from pathlib import Path
from collections import Counter

# ============================================================
# AI HAIR ANALYSIS PROJECT
# DATASET INSPECTION SCRIPT
# ============================================================

PROJECT_DIR = Path(__file__).parent

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"
}

CSV_EXTENSIONS = {".csv"}


def print_line():
    print("=" * 70)


def inspect_folder(folder_path):
    print_line()
    print(f"DATASET: {folder_path.name}")
    print(f"PATH   : {folder_path}")
    print_line()

    if not folder_path.exists():
        print("❌ Folder not found!")
        return

    # --------------------------------------------------------
    # Find all files
    # --------------------------------------------------------
    all_files = [
        p for p in folder_path.rglob("*")
        if p.is_file()
    ]

    print(f"Total files: {len(all_files)}")

    # --------------------------------------------------------
    # File extensions
    # --------------------------------------------------------
    extension_counter = Counter(
        p.suffix.lower() for p in all_files
    )

    print("\nFile types:")
    for extension, count in sorted(extension_counter.items()):
        extension_name = extension if extension else "[no extension]"
        print(f"  {extension_name:<10} : {count}")

    # --------------------------------------------------------
    # Image files
    # --------------------------------------------------------
    image_files = [
        p for p in all_files
        if p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    print(f"\nTotal image files: {len(image_files)}")

    # --------------------------------------------------------
    # CSV files
    # --------------------------------------------------------
    csv_files = [
        p for p in all_files
        if p.suffix.lower() in CSV_EXTENSIONS
    ]

    print(f"Total CSV files: {len(csv_files)}")

    if csv_files:
        print("\nCSV files found:")
        for csv in csv_files:
            print(f"  - {csv.relative_to(folder_path)}")

    # --------------------------------------------------------
    # Folder structure
    # --------------------------------------------------------
    directories = [
        p for p in folder_path.rglob("*")
        if p.is_dir()
    ]

    print(f"\nSubfolders: {len(directories)}")

    if directories:
        print("\nFolder structure:")

        for directory in sorted(directories):
            relative_path = directory.relative_to(folder_path)

            image_count = sum(
                1
                for p in directory.iterdir()
                if p.is_file()
                and p.suffix.lower() in IMAGE_EXTENSIONS
            )

            print(
                f"  {relative_path} "
                f"→ {image_count} image(s)"
            )

    # --------------------------------------------------------
    # Show sample image filenames
    # --------------------------------------------------------
    if image_files:
        print("\nSample image files:")

        for image in image_files[:10]:
            print(f"  - {image.relative_to(folder_path)}")

        if len(image_files) > 10:
            print(f"  ... and {len(image_files) - 10} more")

    print()


# ============================================================
# FIND DATASET FOLDERS
# ============================================================

print("\n")
print_line()
print("        AI HAIR INTELLIGENCE PROJECT")
print("             DATASET INSPECTION")
print_line()

print(f"\nProject folder:")
print(PROJECT_DIR)

print("\nDatasets found directly inside project folder:")

dataset_folders = [
    p for p in PROJECT_DIR.iterdir()
    if p.is_dir()
]

if not dataset_folders:
    print("❌ No folders found!")
else:
    for folder in sorted(dataset_folders):
        print(f"  ✓ {folder.name}")


# ============================================================
# INSPECT EVERY DATASET
# ============================================================

for dataset_folder in sorted(dataset_folders):
    inspect_folder(dataset_folder)


# ============================================================
# FINISHED
# ============================================================

print_line()
print("INSPECTION COMPLETED")
print_line()

print("\nNext step:")
print("Send me the COMPLETE terminal output.")
print("DO NOT delete or modify any dataset yet.")
print()