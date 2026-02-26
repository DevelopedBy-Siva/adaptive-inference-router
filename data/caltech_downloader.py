import os
import requests
import zipfile

URL = "https://data.caltech.edu/records/f6rph-90m20/files/data_and_labels.zip"
BASE_DIR = "./caltech"
ZIP_PATH = os.path.join(BASE_DIR, "data_and_labels.zip")


def download_file(url, output_path, chunk_size=1024 * 1024):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print("Downloading Caltech dataset...")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(output_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
    print("Download complete.")


def extract_zip(zip_path, extract_to):
    print("Extracting dataset...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print("Extraction complete.")


def main():
    os.makedirs(BASE_DIR, exist_ok=True)

    download_file(URL, ZIP_PATH)

    extract_zip(ZIP_PATH, BASE_DIR)

    os.remove(ZIP_PATH)

    print("\nCaltech dataset ready at ./caltech/")


if __name__ == "__main__":
    main()