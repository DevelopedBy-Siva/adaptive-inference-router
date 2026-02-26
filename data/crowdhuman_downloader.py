import os
import zipfile
from huggingface_hub import snapshot_download

BASE_DIR = "./crowdhuman"

ZIP_MAP = {
    "CrowdHuman_train01.zip": "train01",
    "CrowdHuman_train02.zip": "train02",
    "CrowdHuman_train03.zip": "train03",
    "CrowdHuman_val.zip": "val",
    "CrowdHuman_test.zip": "test",
}


def download_dataset():
    os.makedirs(BASE_DIR, exist_ok=True)

    snapshot_download(
        repo_id="sshao0516/CrowdHuman",
        repo_type="dataset",
        local_dir=BASE_DIR,
        local_dir_use_symlinks=False,
        resume_download=True,
    )

    print("Download complete.")


def extract_and_organize():
    for zip_name, folder_name in ZIP_MAP.items():
        zip_path = os.path.join(BASE_DIR, zip_name)

        if not os.path.exists(zip_path):
            print(f"⚠ {zip_name} not found, skipping.")
            continue

        extract_path = os.path.join(BASE_DIR, folder_name)
        os.makedirs(extract_path, exist_ok=True)

        print(f"Extracting {zip_name} → {extract_path}")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_path)

        os.remove(zip_path)
        print(f"Removed {zip_name}")

    print("Extraction complete.")


if __name__ == "__main__":
    download_dataset()
    extract_and_organize()

    print("\n🎯 CrowdHuman ready at ./crowdhuman/")