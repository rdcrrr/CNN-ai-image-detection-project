"""
reset_database.py
-----------------
Deletes all images from the selected training/test folders.
Keeps the folders themselves intact.

Place at:  imageMetadata/reset_database.py
Run with:  python reset_database.py
"""

import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

FOLDERS_TO_WIPE = [
    PROJECT_ROOT / "images" / "aiImagesLarge",
    PROJECT_ROOT / "images" / "real_imagesLarge",
    PROJECT_ROOT / "images" / "ai_imagesRandom",
    PROJECT_ROOT / "images" / "newData",
    PROJECT_ROOT / "images" / "testImages",
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def wipe_folder(folder: Path):
    if not folder.exists():
        print(f"  [SKIP] Folder does not exist: {folder.name}")
        return

    # Count first
    all_files = list(folder.rglob("*"))
    images    = [f for f in all_files if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS]
    others    = [f for f in all_files if f.is_file() and f.suffix.lower() not in IMAGE_EXTENSIONS]

    print(f"\n  {folder.name}/")
    print(f"    Images to delete : {len(images)}")
    if others:
        print(f"    Non-image files  : {len(others)} (will be kept)")

    if not images:
        print(f"    Nothing to delete.")
        return

    # Delete all images
    deleted = 0
    for f in images:
        try:
            f.unlink()
            deleted += 1
        except Exception as e:
            print(f"    [ERROR] Could not delete {f.name}: {e}")

    # Remove empty subfolders
    for subfolder in sorted(folder.rglob("*"), reverse=True):
        if subfolder.is_dir():
            try:
                subfolder.rmdir()   # only removes if empty
            except OSError:
                pass                # not empty, leave it

    print(f"    Deleted {deleted} images ✓")


if __name__ == "__main__":
    print("=" * 50)
    print("DATABASE RESET")
    print("=" * 50)
    print("\nFolders to wipe:")
    for f in FOLDERS_TO_WIPE:
        print(f"  - {f.name}")

    confirm = input("\nType YES to confirm: ").strip()
    if confirm != "YES":
        print("Aborted.")
        exit(0)

    for folder in FOLDERS_TO_WIPE:
        wipe_folder(folder)

    print("\n" + "=" * 50)
    print("Done. All image folders are now empty.")
    print("=" * 50)