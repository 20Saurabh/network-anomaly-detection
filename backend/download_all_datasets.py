"""
Automated dataset downloader using the Kaggle API.
Requires a pre-configured ~/.kaggle/kaggle.json file.
"""
import os
import subprocess
from pathlib import Path

# Mapping of dataset names to their Kaggle IDs
DATASETS = {
    "unsw_nb15": "mrwellsdavid/unsw-nb15",
    "edge_iiot": "mohamedamineferrag/edgeiiotset-cyber-security-dataset",
    "cicids2017": "cicdataset/cicids2017",
}

def main():
    print("=" * 60)
    print("  NETGUARD DATASET DOWNLOADER")
    print("=" * 60)

    # Check if kaggle is installed
    try:
        import kaggle
    except ImportError:
        print("[!] The 'kaggle' pip package is not installed.")
        print("[!] Please run: pip install kaggle")
        return

    base_raw_dir = Path(__file__).resolve().parent / "data" / "raw"
    base_raw_dir.mkdir(parents=True, exist_ok=True)

    for name, kaggle_id in DATASETS.items():
        download_dir = base_raw_dir / name
        download_dir.mkdir(exist_ok=True)

        print(f"\n[*] Downloading {name} ({kaggle_id})...")
        print(f"    Target: {download_dir}")
        print("    This may take a while depending on your network.")

        cmd = [
            "kaggle", "datasets", "download", 
            "-d", kaggle_id, 
            "-p", str(download_dir), 
            "--unzip"
        ]

        try:
            subprocess.run(cmd, check=True)
            print(f"[*] {name} downloaded and unzipped successfully.")
        except subprocess.CalledProcessError as e:
            print(f"[!] Error downloading {name}. Ensure your kaggle key is valid.")
            print(f"[!] Details: {e}")

    print("\n" + "=" * 60)
    print("[*] Dataset Downloads Complete!")
    print("[*] Note: For CICIoT2023, please download manually from UNB due to filesize constraints.")
    print("=" * 60)

if __name__ == "__main__":
    main()
