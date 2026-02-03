"""
创建合成多目标交通标志数据集

功能:
1. 从训练集中随机选择2-5个交通标志
2. 将它们合成到同一张背景图片中
3. 调整大小、位置、角度
4. 生成新的YOLO格式标注
5. 增加训练集的多样性,提高多目标检测能力
"""

import os
import cv2
import numpy as np
import random
import argparse
from pathlib import Path
from tqdm import tqdm
import shutil


def parse_args():
    parser = argparse.ArgumentParser(description='创建合成多目标交通标志数据集')
    parser.add_argument('--input-dir', type=str, default='datasets/gtsrb/images/train',
                        help='输入图片目录')
    parser.add_argument('--label-dir', type=str, default='datasets/gtsrb/labels/train',
                        help='输入标签目录')
    parser.add_argument('--output-dir', type=str, default='datasets/gtsrb_multi/images/train',
                        help='输出图片目录')
    parser.add_argument('--output-label-dir', type=str, default='datasets/gtsrb_multi/labels/train',
                        help='输出标签目录')
    parser.add_argument('--num-images', type=int, default=1000,
                        help='生成的合成图片数量')
    parser.add_argument('--min-signs', type=int, default=2,
                        help='每张图片最少标志数量')
    parser.add_argument('--max-signs', type=int, default=5,
                        help='每张图片最多标志数量')
    parser.add_argument('--bg-width', type=int, default=1280,
                        help='背景图片宽度')
    parser.add_argument('--bg-height', type=int, default=720,
                        help='背景图片高度')
    parser.add_argument('--min-scale', type=float, default=0.3,
                        help='标志最小缩放比例')
    parser.add_argument('--max-scale', type=float, default=1.0,
                        help='标志最大缩放比例')
    parser.add_argument('--rotation-range', type=int, default=15,
                        help='旋转角度范围 (+/- degrees)')
    return parser.parse_args()


def load_yolo_label(label_path):
    """加载YOLO格式标签"""
    if not os.path.exists(label_path):
        return []
    
    labels = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                cls, x, y, w, h = map(float, parts)
                labels.append({
                    'class': int(cls),
                    'x': x,
                    'y': y,
                    'w': w,
                    'h': h
                })
    return labels


def extract_sign_from_image(img, label):
    """从图片中提取单个标志"""
    h, w = img.shape[:2]
    
    # 将YOLO格式转换为像素坐标
    x_center = int(label['x'] * w)
    y_center = int(label['y'] * h)
    box_w = int(label['w'] * w)
    box_h = int(label['h'] * h)
    
    # 计算边界框
    x1 = max(0, x_center - box_w // 2)
    y1 = max(0, y_center - box_h // 2)
    x2 = min(w, x_center + box_w // 2)
    y2 = min(h, y_center + box_h // 2)
    
    # 提取标志
    sign = img[y1:y2, x1:x2]
    
    return sign, label['class']


def rotate_image(image, angle):
    """旋转图片"""
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    
    # 获取旋转矩阵
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # 计算旋转后的边界框
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    
    # 调整旋转矩阵
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]
    
    # 执行旋转
    rotated = cv2.warpAffine(image, M, (new_w, new_h), 
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=(114, 114, 114))
    
    return rotated


def create_synthetic_image(sign_data_list, bg_width, bg_height, 
                          min_scale, max_scale, rotation_range):
    """创建合成图片"""
    # 创建背景 (灰色)
    background = np.ones((bg_height, bg_width, 3), dtype=np.uint8) * 114
    
    # 添加一些噪声使背景更真实
    noise = np.random.randint(-20, 20, (bg_height, bg_width, 3), dtype=np.int16)
    background = np.clip(background.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    labels = []
    
    for sign_img, sign_class in sign_data_list:
        # 随机缩放
        scale = random.uniform(min_scale, max_scale)
        new_h = int(sign_img.shape[0] * scale)
        new_w = int(sign_img.shape[1] * scale)
        
        # 确保标志不会太小
        if new_h < 30 or new_w < 30:
            continue
            
        sign_resized = cv2.resize(sign_img, (new_w, new_h))
        
        # 随机旋转
        if rotation_range > 0:
            angle = random.uniform(-rotation_range, rotation_range)
            sign_resized = rotate_image(sign_resized, angle)
        
        # 更新尺寸
        new_h, new_w = sign_resized.shape[:2]
        
        # 随机位置 (确保完全在图片内)
        max_x = bg_width - new_w
        max_y = bg_height - new_h
        
        if max_x <= 0 or max_y <= 0:
            continue
            
        x = random.randint(0, max_x)
        y = random.randint(0, max_y)
        
        # 检查是否与已有标志重叠过多
        overlap = False
        for label in labels:
            # 简单的重叠检测
            label_x = int((label['x'] - label['w'] / 2) * bg_width)
            label_y = int((label['y'] - label['h'] / 2) * bg_height)
            label_w = int(label['w'] * bg_width)
            label_h = int(label['h'] * bg_height)
            
            # 计算IoU
            x_overlap = max(0, min(x + new_w, label_x + label_w) - max(x, label_x))
            y_overlap = max(0, min(y + new_h, label_y + label_h) - max(y, label_y))
            overlap_area = x_overlap * y_overlap
            
            if overlap_area > 0.3 * (new_w * new_h):
                overlap = True
                break
        
        if overlap:
            continue
        
        # 粘贴到背景
        try:
            background[y:y+new_h, x:x+new_w] = sign_resized
        except:
            continue
        
        # 创建YOLO格式标签
        x_center = (x + new_w / 2) / bg_width
        y_center = (y + new_h / 2) / bg_height
        w_norm = new_w / bg_width
        h_norm = new_h / bg_height
        
        labels.append({
            'class': sign_class,
            'x': x_center,
            'y': y_center,
            'w': w_norm,
            'h': h_norm
        })
    
    return background, labels


def main():
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.output_label_dir, exist_ok=True)
    
    # 获取所有训练图片
    input_dir = Path(args.input_dir)
    label_dir = Path(args.label_dir)
    
    image_files = list(input_dir.glob('*.jpg')) + list(input_dir.glob('*.png'))
    
    if len(image_files) == 0:
        print(f"❌ 在 {args.input_dir} 中未找到图片")
        return
    
    print(f"📁 找到 {len(image_files)} 张训练图片")
    
    # 加载所有标志数据
    sign_pool = []
    
    print("📦 加载标志数据...")
    for img_path in tqdm(image_files):
        # 读取图片
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        # 读取标签
        label_path = label_dir / (img_path.stem + '.txt')
        labels = load_yolo_label(str(label_path))
        
        # 提取每个标志
        for label in labels:
            try:
                sign, sign_class = extract_sign_from_image(img, label)
                if sign.shape[0] > 10 and sign.shape[1] > 10:
                    sign_pool.append((sign, sign_class))
            except:
                continue
    
    print(f"✅ 加载了 {len(sign_pool)} 个标志")
    
    if len(sign_pool) < args.max_signs:
        print(f"❌ 标志数量不足,至少需要 {args.max_signs} 个")
        return
    
    # 生成合成图片
    print(f"🎨 生成 {args.num_images} 张合成图片...")
    
    for i in tqdm(range(args.num_images)):
        # 随机选择标志数量
        num_signs = random.randint(args.min_signs, args.max_signs)
        
        # 随机选择标志
        selected_signs = random.sample(sign_pool, num_signs)
        
        # 创建合成图片
        synthetic_img, labels = create_synthetic_image(
            selected_signs,
            args.bg_width,
            args.bg_height,
            args.min_scale,
            args.max_scale,
            args.rotation_range
        )
        
        # 保存图片
        img_filename = f'synthetic_{i:05d}.jpg'
        img_path = os.path.join(args.output_dir, img_filename)
        cv2.imwrite(img_path, synthetic_img)
        
        # 保存标签
        label_filename = f'synthetic_{i:05d}.txt'
        label_path = os.path.join(args.output_label_dir, label_filename)
        
        with open(label_path, 'w') as f:
            for label in labels:
                f.write(f"{label['class']} {label['x']:.6f} {label['y']:.6f} "
                       f"{label['w']:.6f} {label['h']:.6f}\n")
    
    print(f"✅ 完成! 生成了 {args.num_images} 张合成图片")
    print(f"📁 图片保存在: {args.output_dir}")
    print(f"📁 标签保存在: {args.output_label_dir}")
    
    # 打印统计信息
    print("\n📊 统计信息:")
    print(f"  - 每张图片标志数量: {args.min_signs}-{args.max_signs}")
    print(f"  - 图片尺寸: {args.bg_width}x{args.bg_height}")
    print(f"  - 缩放范围: {args.min_scale}-{args.max_scale}")
    print(f"  - 旋转范围: ±{args.rotation_range}°")


if __name__ == '__main__':
    main()
