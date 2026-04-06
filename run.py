import os
import shutil
from pathlib import Path
import gdown

# ---------------- CONFIG ----------------
DRIVE_FOLDER_LINK = "https://drive.google.com/drive/folders/1I8AwWnpu6CBlfElDUPVFGXMHOG3cl48T?usp=drive_link"

LOCAL_RESULTS_DIR = Path("backend/results")
TEMP_DIR = Path("temp_results")

# ---------------- FUNCTION ----------------
def install_results():

    # Step 1: Clean temp if exists
    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR)

    print("⬇️ Downloading folder from Google Drive...")
    
    # Download entire folder
    gdown.download_folder(
    url=DRIVE_FOLDER_LINK,
    output=str(TEMP_DIR),
    quiet=False,
    use_cookies=False,
    remaining_ok=True   # IMPORTANT FIX
)
    if not TEMP_DIR.exists():
        print("❌ Download failed!")
        return

    # Step 2: Find correct root folder
    # (Drive sometimes nests folders)
    subfolders = list(TEMP_DIR.iterdir())
    source_dir = subfolders[0] if len(subfolders) == 1 else TEMP_DIR

    print(f"📂 Using source: {source_dir}")

    # Step 3: Copy required folders
    for folder in ["runs", "plots", "metrics"]:
        src = source_dir / folder
        dst = LOCAL_RESULTS_DIR / folder

        if src.exists():
            shutil.copytree(src, dst, dirs_exist_ok=True)
            print(f"✅ Updated {folder}")
        else:
            print(f"⚠️ {folder} not found in Drive folder")

    # Step 4: Cleanup
    shutil.rmtree(TEMP_DIR)

    print("\n🚀 Directory updated successfully! Refresh Streamlit UI.")

# ---------------- RUN ----------------
if __name__ == "__main__":
    install_results()