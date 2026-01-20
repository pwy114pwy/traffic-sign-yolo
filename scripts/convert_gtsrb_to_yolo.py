# scripts/convert_gtsrb_to_yolo.py
import os
import pandas as pd
from PIL import Image
import argparse
from pathlib import Path

def convert_csv_to_yolo(csv_path, img_base_dir, label_out_dir, image_out_dir):
    """
    将 GTSRB 数据集的 CSV 标注转换为 YOLO 格式
    
    Args:
        csv_path: CSV 文件路径
        img_base_dir: 图像基础目录
        label_out_dir: 标签输出目录
        image_out_dir: 图像输出目录
    """
    df = pd.read_csv(csv_path)
    os.makedirs(label_out_dir, exist_ok=True)
    os.makedirs(image_out_dir, exist_ok=True)

    for _, row in df.iterrows():
        # 获取信息
        class_id = int(row['ClassId'])
        x1, y1 = int(row['Roi.X1']), int(row['Roi.Y1'])
        x2, y2 = int(row['Roi.X2']), int(row['Roi.Y2'])
        rel_path = row['Path']  # 如 "Train/20/00020_00000_00000.png"
        
        full_img_path = os.path.join(img_base_dir, rel_path)
        if not os.path.exists(full_img_path):
            print(f"Warning: {full_img_path} not found")
            continue

        # 读取图像
        img = Image.open(full_img_path)
        w, h = img.size

        # 计算 YOLO 格式（归一化）
        x_center = (x1 + x2) / 2 / w
        y_center = (y1 + y2) / 2 / h
        box_w = (x2 - x1) / w
        box_h = (y2 - y1) / h

        # 生成输出文件名
        # 处理不同的 Path 格式
        if '/' in rel_path:
            # Train/20/00020_00000_00000.png
            base_name = os.path.splitext(os.path.basename(rel_path))[0]
        else:
            # 12624.png
            base_name = os.path.splitext(rel_path)[0]
            
        txt_path = os.path.join(label_out_dir, base_name + '.txt')
        jpg_path = os.path.join(image_out_dir, base_name + '.jpg')

        # 写入标签
        with open(txt_path, 'w') as f:
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}\n")

        # 保存为 JPG（提高兼容性）
        img.convert("RGB").save(jpg_path, "JPEG")

def main():
    parser = argparse.ArgumentParser(description='Convert GTSRB dataset to YOLO format')
    parser.add_argument('--input_dir', type=str, default='../datasets/GTSRB', 
                      help='Input directory containing GTSRB dataset')
    parser.add_argument('--output_dir', type=str, default='../datasets/gtsrb', 
                      help='Output directory for YOLO format dataset')
    
    args = parser.parse_args()
    
    # 获取项目根目录
    PROJECT_ROOT = Path(__file__).parent.parent
    
    # 构建路径
    INPUT_DIR = Path(args.input_dir)
    if not INPUT_DIR.is_absolute():
        INPUT_DIR = PROJECT_ROOT / INPUT_DIR
    
    OUTPUT_DIR = Path(args.output_dir)
    if not OUTPUT_DIR.is_absolute():
        OUTPUT_DIR = PROJECT_ROOT / OUTPUT_DIR
    
    # 训练集转换
    print("Converting train set...")
    convert_csv_to_yolo(
        csv_path=str(INPUT_DIR / "Train.csv"),
        img_base_dir=str(INPUT_DIR),
        label_out_dir=str(OUTPUT_DIR / "labels/train"),
        image_out_dir=str(OUTPUT_DIR / "images/train")
    )
    
    # 测试集转换
    print("Converting test set...")
    convert_csv_to_yolo(
        csv_path=str(INPUT_DIR / "Test.csv"),
        img_base_dir=str(INPUT_DIR),
        label_out_dir=str(OUTPUT_DIR / "labels/val"),
        image_out_dir=str(OUTPUT_DIR / "images/val")
    )
    
    print("✅ Conversion completed!")
    print(f"📁 Output directory: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()