from pathlib import Path
import shutil
import hashlib
from collections import defaultdict
from PIL import Image

# ============================================================
# AI HAIR INTELLIGENCE PROJECT
# DATASET CLEANING
# ============================================================

print("=" * 70)
print("AI HAIR INTELLIGENCE PROJECT")
print("DATASET CLEANING")
print("=" * 70)

print("\nIMPORTANT:")
print("Original datasets will NOT be modified.\n")

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(r"C:\Users\kalya\OneDrive\Desktop\hair_analysis")

DISEASE_SOURCE = BASE_DIR / "hair_disease"
TYPE_SOURCE = BASE_DIR / "hair_type"
SEGMENTATION_SOURCE = BASE_DIR / "hair_segmentation"

OUTPUT_DIR = BASE_DIR / "cleaned_datasets"

DISEASE_OUTPUT = OUTPUT_DIR / "hair_disease"
TYPE_OUTPUT = OUTPUT_DIR / "hair_type"
SEGMENTATION_OUTPUT = OUTPUT_DIR / "hair_segmentation"

QUARANTINE_DIR = OUTPUT_DIR / "_quarantine"

# ============================================================
# SETTINGS
# ============================================================

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

MIN_WIDTH = 32
MIN_HEIGHT = 32

# Windows path safety.
# Keep generated destination paths comfortably below the
# traditional Windows MAX_PATH boundary.
MAX_DESTINATION_LENGTH = 230


# ============================================================
# GENERAL HELPERS
# ============================================================

def sha256_file(path):
    """Calculate SHA256 hash of a file."""
    sha256 = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)

            if not chunk:
                break

            sha256.update(chunk)

    return sha256.hexdigest()


def is_image_file(path):
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def safe_image_check(path):
    """
    Validate an image using Pillow.
    Returns:
        True, None
        False, reason
    """

    try:
        with Image.open(path) as img:
            img.verify()

        with Image.open(path) as img:
            width, height = img.size

        if width < MIN_WIDTH or height < MIN_HEIGHT:
            return False, f"Small image: {width}x{height}"

        return True, None

    except Exception as e:
        return False, str(e)


def make_safe_filename(source, destination_root, relative_path):
    """
    Generate a safe destination path.

    Normally the original relative path is preserved.

    If the resulting Windows path is too long, shorten only
    the filename while preserving the extension.
    """

    destination = destination_root / relative_path

    if len(str(destination)) <= MAX_DESTINATION_LENGTH:
        return destination

    parent = destination.parent
    original_name = destination.name
    suffix = destination.suffix

    stem = destination.stem

    # Preserve a readable portion of the original filename.
    shortened_stem = stem[:80]

    # Add a hash so shortened filenames remain unique.
    name_hash = hashlib.sha1(
        str(relative_path).encode("utf-8")
    ).hexdigest()[:12]

    safe_name = f"{shortened_stem}_{name_hash}{suffix}"

    safe_destination = parent / safe_name

    # In the extremely unlikely case that this is still too long,
    # shorten further.
    if len(str(safe_destination)) > MAX_DESTINATION_LENGTH:
        safe_name = f"image_{name_hash}{suffix}"
        safe_destination = parent / safe_name

    return safe_destination


def copy_file_safe(source, destination_root, relative_path):
    """
    Copy a file safely.

    Automatically:
    - creates destination folders
    - shortens excessively long filenames
    """

    destination = make_safe_filename(
        source,
        destination_root,
        relative_path
    )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    shutil.copy2(source, destination)

    return destination


def quarantine_file(source, quarantine_root, relative_path, reason):
    """
    Copy problematic files to quarantine.

    Original source remains untouched.
    """

    quarantine_path = (
        quarantine_root /
        reason /
        relative_path
    )

    quarantine_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    try:
        shutil.copy2(source, quarantine_path)
    except Exception:
        pass


def find_files(root):
    """Recursively find all files."""
    if not root.exists():
        return []

    return [
        p for p in root.rglob("*")
        if p.is_file()
    ]


# ============================================================
# SOURCE CHECK
# ============================================================

print("=" * 70)
print("SOURCE DATASET CHECK")
print("=" * 70)

for name, path in [
    ("hair_disease", DISEASE_SOURCE),
    ("hair_type", TYPE_SOURCE),
    ("hair_segmentation", SEGMENTATION_SOURCE),
]:
    status = "FOUND" if path.exists() else "NOT FOUND"

    print(f"\n{name}: {status}")

    if path.exists():
        print(f"  {path}")


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DISEASE_OUTPUT.mkdir(parents=True, exist_ok=True)
TYPE_OUTPUT.mkdir(parents=True, exist_ok=True)
SEGMENTATION_OUTPUT.mkdir(parents=True, exist_ok=True)
QUARANTINE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# HAIR DISEASE DATASET
# ============================================================

print("\n" + "=" * 70)
print("HAIR DISEASE DATASET")
print("=" * 70)


def find_disease_root(source):
    """
    Locate the actual 'Hair Diseases - Final' folder.
    """

    candidates = []

    for p in source.rglob("*"):
        if p.is_dir() and p.name.lower() == "hair diseases - final":
            candidates.append(p)

    if not candidates:
        return source

    # Prefer one containing train/test/validation.
    for candidate in candidates:
        child_names = {
            x.name.lower()
            for x in candidate.iterdir()
            if x.is_dir()
        }

        if (
            "train" in child_names
            or "test" in child_names
            or "validation" in child_names
            or "valid" in child_names
        ):
            return candidate

    return candidates[0]


def clean_disease_dataset():

    disease_root = find_disease_root(DISEASE_SOURCE)

    print("\nDetected disease dataset root:")
    print(f"  {disease_root}")

    files = [
        p for p in disease_root.rglob("*")
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
        and "__MACOSX" not in p.parts
        and p.name != ".DS_Store"
    ]

    print(f"\nDisease image files found: {len(files)}")

    copied = 0
    invalid = 0
    copy_failures = 0
    metadata = 0

    total = len(files)

    for index, source in enumerate(files, start=1):

        relative_path = source.relative_to(disease_root)

        valid, reason = safe_image_check(source)

        if not valid:

            if reason and reason.startswith("Small image"):
                quarantine_file(
                    source,
                    QUARANTINE_DIR,
                    relative_path,
                    "disease_small"
                )
            else:
                quarantine_file(
                    source,
                    QUARANTINE_DIR,
                    relative_path,
                    "disease_invalid"
                )

            invalid += 1
            continue

        try:

            copy_file_safe(
                source,
                DISEASE_OUTPUT,
                relative_path
            )

            copied += 1

        except Exception as e:

            copy_failures += 1

            print("\n[ERROR] COPY FAILED")
            print(f"  SOURCE:      {source}")
            print(f"  ERROR:       {e}")

    print(f"\nCopying disease images: {copied}/{total}")

    print("\nHair disease cleaning summary:")
    print(f"  Source root:                 {disease_root}")
    print(f"  Images found:                {total}")
    print(f"  Copied:                      {copied}")
    print(f"  Invalid images:              {invalid}")
    print(f"  Copy failures:               {copy_failures}")
    print(f"  Metadata files ignored:      {metadata}")
    print(f"  Original files deleted:      0")

    return {
        "found": total,
        "copied": copied,
        "invalid": invalid,
        "copy_failures": copy_failures,
    }


# ============================================================
# HAIR TYPE DATASET
# ============================================================

def clean_type_dataset():

    print("\n" + "=" * 70)
    print("HAIR TYPE DATASET")
    print("=" * 70)

    files = [
        p for p in TYPE_SOURCE.rglob("*")
        if is_image_file(p)
        and "__MACOSX" not in p.parts
        and p.name != ".DS_Store"
    ]

    print(f"Original image files: {len(files)}")

    # --------------------------------------------------------
    # HASH FILES
    # --------------------------------------------------------

    print(
        f"\nCalculating SHA256 hashes for {len(files)} files..."
    )

    hashes = defaultdict(list)

    for index, source in enumerate(files, start=1):

        if index == 1 or index % 100 == 0 or index == len(files):
            print(
                f"Hashing: {index}/{len(files)}",
                end="\r"
            )

        try:
            file_hash = sha256_file(source)
            hashes[file_hash].append(source)

        except Exception as e:

            print(
                f"\n[HASH ERROR] {source}"
            )
            print(f"  ERROR: {e}")

    print()

    duplicate_groups = [
        group
        for group in hashes.values()
        if len(group) > 1
    ]

    print(
        f"\nExact duplicate groups found: "
        f"{len(duplicate_groups)}"
    )

    # --------------------------------------------------------
    # DUPLICATE DECISIONS
    # --------------------------------------------------------

    skip_files = set()

    same_class_duplicates = 0
    cross_class_duplicates = 0

    for group in duplicate_groups:

        class_names = {
            source.parent.name
            for source in group
        }

        # ----------------------------------------------------
        # CROSS-CLASS DUPLICATE
        # ----------------------------------------------------

        if len(class_names) > 1:

            cross_class_duplicates += len(group)

            print("\n[CROSS-CLASS DUPLICATE]")
            print(f"SHA256: {sha256_file(group[0])}")

            for source in group:
                print(
                    f"  {source.relative_to(TYPE_SOURCE)}"
                )

            # Do not keep ambiguous duplicate copies.
            # Keep the first and quarantine the others.
            keep = sorted(
                group,
                key=lambda p: str(p).lower()
            )[0]

            for source in group:

                if source != keep:
                    skip_files.add(source)

                    quarantine_file(
                        source,
                        QUARANTINE_DIR,
                        source.relative_to(TYPE_SOURCE),
                        "type_cross_class_duplicate"
                    )

            continue

        # ----------------------------------------------------
        # SAME-CLASS DUPLICATE
        # ----------------------------------------------------

        same_class_duplicates += len(group) - 1

        sorted_group = sorted(
            group,
            key=lambda p: str(p).lower()
        )

        # Prefer a filename without (1), (2), (3), etc.
        preferred = None

        for source in sorted_group:

            stem_lower = source.stem.lower()

            if not (
                stem_lower.endswith("(1)")
                or stem_lower.endswith("(2)")
                or stem_lower.endswith("(3)")
            ):
                preferred = source
                break

        if preferred is None:
            preferred = sorted_group[0]

        print("\n[DUPLICATE]")
        print(
            f"  KEEP: "
            f"{preferred.relative_to(TYPE_SOURCE)}"
        )

        for source in sorted_group:

            if source != preferred:

                print(
                    f"  SKIP: "
                    f"{source.relative_to(TYPE_SOURCE)}"
                )

                skip_files.add(source)

                quarantine_file(
                    source,
                    QUARANTINE_DIR,
                    source.relative_to(TYPE_SOURCE),
                    "type_duplicate"
                )

    # --------------------------------------------------------
    # COPY FILES
    # --------------------------------------------------------

    copied = 0
    small_images = 0
    invalid_images = 0
    copy_failures = 0

    for source in files:

        if source in skip_files:
            continue

        relative_path = source.relative_to(TYPE_SOURCE)

        valid, reason = safe_image_check(source)

        if not valid:

            if reason and reason.startswith("Small image"):

                print("\n[SMALL IMAGE]")
                print(f"  File: {relative_path}")
                print(
                    f"  {reason.replace('Small image: ', '')}"
                )

                quarantine_file(
                    source,
                    QUARANTINE_DIR,
                    relative_path,
                    "type_small"
                )

                small_images += 1

            else:

                print("\n[INVALID IMAGE]")
                print(f"  File: {relative_path}")
                print(f"  Reason: {reason}")

                quarantine_file(
                    source,
                    QUARANTINE_DIR,
                    relative_path,
                    "type_invalid"
                )

                invalid_images += 1

            continue

        try:

            destination = copy_file_safe(
                source,
                TYPE_OUTPUT,
                relative_path
            )

            copied += 1

            # Tell user when long filename was shortened.
            normal_destination = TYPE_OUTPUT / relative_path

            if destination != normal_destination:

                print("\n[LONG PATH FIXED]")
                print(f"  Original: {relative_path}")
                print(
                    f"  Cleaned:  "
                    f"{destination.relative_to(TYPE_OUTPUT)}"
                )

        except Exception as e:

            copy_failures += 1

            print("\n[ERROR] COPY FAILED")
            print(f"  SOURCE:      {source}")
            print(
                f"  DESTINATION: "
                f"{TYPE_OUTPUT / relative_path}"
            )
            print(f"  ERROR:       {e}")

    print("\nHair type cleaning summary:")
    print(f"  Original images:             {len(files)}")
    print(f"  Copied to clean dataset:     {copied}")
    print(f"  Same-class duplicates:       {same_class_duplicates}")
    print(f"  Cross-class duplicates:      {cross_class_duplicates}")
    print(f"  Small images:                {small_images}")
    print(f"  Invalid images:              {invalid_images}")
    print(f"  Copy failures:               {copy_failures}")

    print(
        "\nIMPORTANT: Cross-class duplicate images were "
        "quarantined rather than automatically assigned "
        "to a class."
    )

    return {
        "found": len(files),
        "copied": copied,
        "same_class_duplicates": same_class_duplicates,
        "cross_class_duplicates": cross_class_duplicates,
        "small_images": small_images,
        "invalid": invalid_images,
        "copy_failures": copy_failures,
    }


# ============================================================
# HAIR SEGMENTATION DATASET
# ============================================================

def clean_segmentation_dataset():

    print("\n" + "=" * 70)
    print("HAIR SEGMENTATION DATASET")
    print("=" * 70)

    files = find_files(SEGMENTATION_SOURCE)

    total_source = len(files)

    supported = 0
    macosx_ignored = 0
    extensionless = 0
    unsupported = 0
    invalid = 0
    copy_failures = 0

    for source in files:

        relative_path = source.relative_to(SEGMENTATION_SOURCE)

        # ----------------------------------------------------
        # Ignore macOS metadata directory
        # ----------------------------------------------------

        if "__MACOSX" in source.parts:

            macosx_ignored += 1
            continue

        # ----------------------------------------------------
        # .DS_Store
        # ----------------------------------------------------

        if source.name == ".DS_Store":

            extensionless += 1

            print("\n[EXTENSIONLESS FILE QUARANTINED]")
            print(f"{relative_path}")

            quarantine_file(
                source,
                QUARANTINE_DIR,
                relative_path,
                "segmentation_metadata"
            )

            continue

        # ----------------------------------------------------
        # Extensionless files
        # ----------------------------------------------------

        if source.suffix == "":

            extensionless += 1

            print("\n[EXTENSIONLESS FILE QUARANTINED]")
            print(f"{relative_path}")

            quarantine_file(
                source,
                QUARANTINE_DIR,
                relative_path,
                "segmentation_extensionless"
            )

            continue

        # ----------------------------------------------------
        # Unsupported extensions
        # ----------------------------------------------------

        if source.suffix.lower() not in SEGMENTATION_EXTENSIONS:

            unsupported += 1

            print("\n[UNSUPPORTED FILE QUARANTINED]")
            print(f"  File: {relative_path}")

            quarantine_file(
                source,
                QUARANTINE_DIR,
                relative_path,
                "segmentation_unsupported"
            )

            continue

        # ----------------------------------------------------
        # PBM / PGM / PPM
        # ----------------------------------------------------

        if source.suffix.lower() in {
            ".pbm",
            ".pgm",
            ".ppm",
        }:

            try:

                copy_file_safe(
                    source,
                    SEGMENTATION_OUTPUT,
                    relative_path
                )

                supported += 1

            except Exception as e:

                copy_failures += 1

                print("\n[ERROR] COPY FAILED")
                print(f"  SOURCE: {source}")
                print(f"  ERROR: {e}")

            continue

        # ----------------------------------------------------
        # Normal images
        # ----------------------------------------------------

        valid, reason = safe_image_check(source)

        if not valid:

            invalid += 1

            print("\n[INVALID IMAGE]")
            print(f"  File: {relative_path}")
            print(f"  Reason: {reason}")

            quarantine_file(
                source,
                QUARANTINE_DIR,
                relative_path,
                "segmentation_invalid"
            )

            continue

        try:

            copy_file_safe(
                source,
                SEGMENTATION_OUTPUT,
                relative_path
            )

            supported += 1

        except Exception as e:

            copy_failures += 1

            print("\n[ERROR] COPY FAILED")
            print(f"  SOURCE: {source}")
            print(f"  ERROR: {e}")

    print("\nHair segmentation cleaning summary:")
    print(f"  Total source files:           {total_source}")
    print(f"  Supported files copied:       {supported}")
    print(f"  __MACOSX files ignored:       {macosx_ignored}")
    print(f"  Extensionless quarantined:    {extensionless}")
    print(f"  Unsupported quarantined:      {unsupported}")
    print(f"  Invalid images quarantined:   {invalid}")
    print(f"  Copy failures:                {copy_failures}")

    print(
        "\nPBM files were preserved because they may be "
        "legitimate segmentation masks."
    )

    return {
        "found": total_source,
        "copied": supported,
        "macosx": macosx_ignored,
        "extensionless": extensionless,
        "unsupported": unsupported,
        "invalid": invalid,
        "copy_failures": copy_failures,
    }


# ============================================================
# VERIFICATION
# ============================================================

def verify_dataset(root, name):

    print(f"\n{name}:")

    if not root.exists():
        print("  Files: 0")
        return

    files = [
        p for p in root.rglob("*")
        if p.is_file()
    ]

    extension_counts = defaultdict(int)

    for file in files:
        extension_counts[file.suffix.lower()] += 1

    print(f"  Files: {len(files)}")

    for extension in sorted(extension_counts):
        print(
            f"  {extension or '[no extension]'}: "
            f"{extension_counts[extension]}"
        )


def verify_cleaned_datasets():

    print("\n" + "=" * 70)
    print("CLEANED DATASET VERIFICATION")
    print("=" * 70)

    verify_dataset(
        DISEASE_OUTPUT,
        "hair_disease"
    )

    verify_dataset(
        TYPE_OUTPUT,
        "hair_type"
    )

    verify_dataset(
        SEGMENTATION_OUTPUT,
        "hair_segmentation"
    )


# ============================================================
# REPORT
# ============================================================

def write_report(
    disease_stats,
    type_stats,
    segmentation_stats
):

    report_path = OUTPUT_DIR / "cleaning_report.txt"

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "AI HAIR INTELLIGENCE PROJECT\n"
            "DATASET CLEANING REPORT\n"
            "=" * 60 +
            "\n\n"
        )

        f.write(
            "Original datasets were NOT modified.\n\n"
        )

        f.write(
            "HAIR DISEASE\n"
            "------------\n"
        )

        for key, value in disease_stats.items():
            f.write(f"{key}: {value}\n")

        f.write("\n")

        f.write(
            "HAIR TYPE\n"
            "---------\n"
        )

        for key, value in type_stats.items():
            f.write(f"{key}: {value}\n")

        f.write("\n")

        f.write(
            "HAIR SEGMENTATION\n"
            "-----------------\n"
        )

        for key, value in segmentation_stats.items():
            f.write(f"{key}: {value}\n")

        f.write("\n")

        f.write(
            "OUTPUT LOCATION\n"
            "---------------\n"
        )

        f.write(str(OUTPUT_DIR))

    return report_path


# ============================================================
# MAIN
# ============================================================

def main():

    print("\nSource:")
    print(f"  {BASE_DIR}")

    print("\nOutput:")
    print(f"  {OUTPUT_DIR}")

    disease_stats = clean_disease_dataset()

    type_stats = clean_type_dataset()

    segmentation_stats = clean_segmentation_dataset()

    verify_cleaned_datasets()

    report_path = write_report(
        disease_stats,
        type_stats,
        segmentation_stats
    )

    print("\n" + "=" * 70)
    print("CLEANING COMPLETED")
    print("=" * 70)

    print("\nOriginal datasets were NOT modified.")

    print("\nCleaned dataset location:")
    print(f"  {OUTPUT_DIR}")

    print("\nCleaning report:")
    print(f"  {report_path}")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)

    print("\nCleaned datasets:")
    print(OUTPUT_DIR)

    print("\nReport:")
    print(report_path)


if __name__ == "__main__":
    main()