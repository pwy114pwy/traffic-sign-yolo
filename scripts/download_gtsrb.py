# scripts/download_gtsrb.py
import os
import urllib.request
import zipfile

def download_and_extract(url, dest):
    os.makedirs(dest, exist_ok=True)
    filename = os.path.join(dest, "gtsrb.zip")
    print("Downloading GTSRB...")
    urllib.request.urlretrieve(url, filename)
    print("Extracting...")
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall(dest)
    os.remove(filename)
    print("Done.")

if __name__ == "__main__":
    download_and_extract(
        "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB-Complete.zip",
        "datasets/"
    )