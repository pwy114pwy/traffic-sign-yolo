# scripts/convert_gtsrb_to_yolo.py
import os
import pandas as pd
from PIL import Image

def convert_csv_to_yolo(csv_path, img_base_dir, label_out_dir, image_out_dir):
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
        base_name = os.path.splitext(os.path.basename(rel_path))[0]
        txt_path = os.path.join(label_out_dir, base_name + '.txt')
        jpg_path = os.path.join(image_out_dir, base_name + '.jpg')

        # 写入标签
        with open(txt_path, 'w') as f:
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}\n")

        # 保存为 JPG（提高兼容性）
        img.convert("RGB").save(jpg_path, "JPEG")

if __name__ == "__main__":
    # 训练集：Train.csv 中的 Path 是 "Train/xx/xxx.png"
    convert_csv_to_yolo(

        # 本地电脑运行时训练集
        csv_path="datasets/GTSRB/Train.csv",
        img_base_dir="datasets/GTSRB",          

        # googleColab运行时训练集
        # csv_path="/kaggle/input/gtsrb-german-traffic-sign/Test.csv",
        # img_base_dir="/kaggle/input/gtsrb-german-traffic-sign",

        label_out_dir="datasets/gtsrb/labels/train",
        image_out_dir="datasets/gtsrb/images/train"
    )
 
    # 测试集：Test.csv 中的 Path 是 "Test/12624.png"
    convert_csv_to_yolo(
        # 本地电脑运行时测试集
        csv_path="datasets/GTSRB/Test.csv",
        img_base_dir="datasets/GTSRB",      
        
        # googleColab运行时测试集
        # csv_path="/kaggle/input/gtsrb-german-traffic-sign/Test.csv",
        # img_base_dir="/kaggle/input/gtsrb-german-traffic-sign",    
        
        label_out_dir="datasets/gtsrb/labels/val",
        image_out_dir="datasets/gtsrb/images/val"
    )
    print("✅ Conversion completed!")