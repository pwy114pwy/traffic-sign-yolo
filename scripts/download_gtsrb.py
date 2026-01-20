# scripts/download_gtsrb.py
import os
import urllib.request
import zipfile
import argparse
from pathlib import Path

def download_and_extract(url, dest):
    """
    下载并解压 GTSRB 数据集
    
    Args:
        url: 数据集下载 URL
        dest: 解压目标目录
    """
    os.makedirs(dest, exist_ok=True)
    filename = os.path.join(dest, "gtsrb.zip")
    print("Downloading GTSRB dataset...")
    urllib.request.urlretrieve(url, filename)
    print("Extracting dataset...")
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall(dest)
    os.remove(filename)
    print("✅ Download and extraction completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download and extract GTSRB dataset')
    parser.add_argument('--url', type=str, 
                      default="https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB-Complete.zip",
                      help='GTSRB dataset download URL')
    parser.add_argument('--output_dir', type=str, default='../datasets',
                      help='Output directory for downloaded dataset')
    
    args = parser.parse_args()
    
    # 获取项目根目录
    PROJECT_ROOT = Path(__file__).parent.parent
    
    # 构建路径
    OUTPUT_DIR = Path(args.output_dir)
    if not OUTPUT_DIR.is_absolute():
        OUTPUT_DIR = PROJECT_ROOT / OUTPUT_DIR
    
    download_and_extract(args.url, str(OUTPUT_DIR))
    print(f"📁 Dataset saved to: {OUTPUT_DIR}")