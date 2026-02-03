"""
测试多目标检测性能

评估模型在多目标场景下的检测能力
"""

import os
import cv2
import torch
import numpy as np
from pathlib import Path
import sys
import argparse
from tqdm import tqdm
import json

# 添加 yolov5 到系统路径
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
YOLOV5_ROOT = ROOT / 'yolov5'
sys.path.append(str(YOLOV5_ROOT))

from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_boxes
from utils.augmentations import letterbox


def parse_args():
    parser = argparse.ArgumentParser(description='测试多目标检测性能')
    parser.add_argument('--weights', type=str, required=True,
                        help='模型权重文件路径')
    parser.add_argument('--test-dir', type=str, default='datasets/gtsrb_multi/images/train',
                        help='测试图片目录')
    parser.add_argument('--label-dir', type=str, default='datasets/gtsrb_multi/labels/train',
                        help='测试标签目录')
    parser.add_argument('--img-size', type=int, default=640,
                        help='输入图片尺寸')
    parser.add_argument('--conf-thres', type=float, default=0.25,
                        help='置信度阈值')
    parser.add_argument('--iou-thres', type=float, default=0.45,
                        help='NMS IoU阈值')
    parser.add_argument('--device', type=str, default='',
                        help='设备 (cuda/cpu)')
    parser.add_argument('--save-results', action='store_true',
                        help='保存检测结果图片')
    parser.add_argument('--output-dir', type=str, default='test_results',
                        help='结果输出目录')
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


def calculate_iou(box1, box2):
    """计算两个框的IoU"""
    # box格式: [x_center, y_center, width, height] (归一化)
    
    # 转换为 [x1, y1, x2, y2]
    box1_x1 = box1[0] - box1[2] / 2
    box1_y1 = box1[1] - box1[3] / 2
    box1_x2 = box1[0] + box1[2] / 2
    box1_y2 = box1[1] + box1[3] / 2
    
    box2_x1 = box2[0] - box2[2] / 2
    box2_y1 = box2[1] - box2[3] / 2
    box2_x2 = box2[0] + box2[2] / 2
    box2_y2 = box2[1] + box2[3] / 2
    
    # 计算交集
    x1 = max(box1_x1, box2_x1)
    y1 = max(box1_y1, box2_y1)
    x2 = min(box1_x2, box2_x2)
    y2 = min(box1_y2, box2_y2)
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    # 计算并集
    box1_area = box1[2] * box1[3]
    box2_area = box2[2] * box2[3]
    union = box1_area + box2_area - intersection
    
    return intersection / union if union > 0 else 0


def evaluate_detection(gt_labels, pred_labels, iou_threshold=0.5):
    """评估检测结果"""
    if len(gt_labels) == 0:
        return {
            'tp': 0,
            'fp': len(pred_labels),
            'fn': 0,
            'precision': 0,
            'recall': 0,
            'f1': 0
        }
    
    if len(pred_labels) == 0:
        return {
            'tp': 0,
            'fp': 0,
            'fn': len(gt_labels),
            'precision': 0,
            'recall': 0,
            'f1': 0
        }
    
    # 匹配预测和真实标签
    matched_gt = set()
    matched_pred = set()
    
    for i, pred in enumerate(pred_labels):
        best_iou = 0
        best_gt_idx = -1
        
        for j, gt in enumerate(gt_labels):
            if j in matched_gt:
                continue
            
            # 只匹配相同类别
            if pred['class'] != gt['class']:
                continue
            
            iou = calculate_iou(
                [pred['x'], pred['y'], pred['w'], pred['h']],
                [gt['x'], gt['y'], gt['w'], gt['h']]
            )
            
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = j
        
        if best_iou >= iou_threshold and best_gt_idx != -1:
            matched_gt.add(best_gt_idx)
            matched_pred.add(i)
    
    tp = len(matched_pred)
    fp = len(pred_labels) - tp
    fn = len(gt_labels) - len(matched_gt)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def main():
    args = parse_args()
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() and args.device != 'cpu' else 'cpu')
    print(f"🖥️  使用设备: {device}")
    
    # 加载模型
    print(f"⏳ 加载模型: {args.weights}")
    model = attempt_load(args.weights, device=device)
    stride = int(model.stride.max())
    names = model.module.names if hasattr(model, 'module') else model.names
    print(f"✅ 模型加载成功! 类别数: {len(names)}")
    
    # 获取测试图片
    test_dir = Path(args.test_dir)
    label_dir = Path(args.label_dir)
    
    image_files = list(test_dir.glob('*.jpg')) + list(test_dir.glob('*.png'))
    
    if len(image_files) == 0:
        print(f"❌ 在 {args.test_dir} 中未找到图片")
        return
    
    print(f"📁 找到 {len(image_files)} 张测试图片")
    
    # 创建输出目录
    if args.save_results:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # 统计信息
    total_stats = {
        'tp': 0,
        'fp': 0,
        'fn': 0,
        'total_gt': 0,
        'total_pred': 0,
        'images_with_multi_targets': 0,
        'detection_by_num_targets': {}  # 按目标数量分组的统计
    }
    
    # 逐图片测试
    print("🧪 开始测试...")
    
    for img_path in tqdm(image_files):
        # 读取图片
        img0 = cv2.imread(str(img_path))
        if img0 is None:
            continue
        
        # 读取真实标签
        label_path = label_dir / (img_path.stem + '.txt')
        gt_labels = load_yolo_label(str(label_path))
        
        num_targets = len(gt_labels)
        total_stats['total_gt'] += num_targets
        
        if num_targets >= 2:
            total_stats['images_with_multi_targets'] += 1
        
        # 预处理
        img = letterbox(img0, args.img_size, stride=stride)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
        img = torch.from_numpy(img.copy()).to(device).float() / 255.0
        img = img.unsqueeze(0)
        
        # 推理
        with torch.no_grad():
            pred = model(img, augment=True)[0]
            pred = non_max_suppression(pred, 
                                      conf_thres=args.conf_thres, 
                                      iou_thres=args.iou_thres)
        
        # 处理预测结果
        pred_labels = []
        
        for det in pred:
            if len(det):
                # 缩放到原始图片尺寸
                det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], img0.shape).round()
                
                # 转换为YOLO格式
                h, w = img0.shape[:2]
                for *xyxy, conf, cls in det:
                    x1, y1, x2, y2 = xyxy
                    x_center = ((x1 + x2) / 2 / w).item()
                    y_center = ((y1 + y2) / 2 / h).item()
                    width = ((x2 - x1) / w).item()
                    height = ((y2 - y1) / h).item()
                    
                    pred_labels.append({
                        'class': int(cls.item()),
                        'x': x_center,
                        'y': y_center,
                        'w': width,
                        'h': height,
                        'conf': conf.item()
                    })
        
        total_stats['total_pred'] += len(pred_labels)
        
        # 评估
        result = evaluate_detection(gt_labels, pred_labels)
        
        total_stats['tp'] += result['tp']
        total_stats['fp'] += result['fp']
        total_stats['fn'] += result['fn']
        
        # 按目标数量分组统计
        if num_targets not in total_stats['detection_by_num_targets']:
            total_stats['detection_by_num_targets'][num_targets] = {
                'count': 0,
                'tp': 0,
                'fp': 0,
                'fn': 0
            }
        
        total_stats['detection_by_num_targets'][num_targets]['count'] += 1
        total_stats['detection_by_num_targets'][num_targets]['tp'] += result['tp']
        total_stats['detection_by_num_targets'][num_targets]['fp'] += result['fp']
        total_stats['detection_by_num_targets'][num_targets]['fn'] += result['fn']
    
    # 计算总体指标
    tp = total_stats['tp']
    fp = total_stats['fp']
    fn = total_stats['fn']
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # 打印结果
    print("\n" + "="*60)
    print("📊 多目标检测性能评估结果")
    print("="*60)
    print(f"\n总体统计:")
    print(f"  - 测试图片数: {len(image_files)}")
    print(f"  - 多目标图片数: {total_stats['images_with_multi_targets']}")
    print(f"  - 真实目标总数: {total_stats['total_gt']}")
    print(f"  - 检测目标总数: {total_stats['total_pred']}")
    print(f"\n检测性能:")
    print(f"  - True Positives (TP): {tp}")
    print(f"  - False Positives (FP): {fp}")
    print(f"  - False Negatives (FN): {fn}")
    print(f"  - Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"  - Recall: {recall:.4f} ({recall*100:.2f}%)")
    print(f"  - F1 Score: {f1:.4f} ({f1*100:.2f}%)")
    
    print(f"\n按目标数量分组的性能:")
    for num_targets in sorted(total_stats['detection_by_num_targets'].keys()):
        stats = total_stats['detection_by_num_targets'][num_targets]
        tp_group = stats['tp']
        fp_group = stats['fp']
        fn_group = stats['fn']
        
        prec = tp_group / (tp_group + fp_group) if (tp_group + fp_group) > 0 else 0
        rec = tp_group / (tp_group + fn_group) if (tp_group + fn_group) > 0 else 0
        f1_group = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        
        print(f"\n  {num_targets} 个目标 ({stats['count']} 张图片):")
        print(f"    - Precision: {prec:.4f} ({prec*100:.2f}%)")
        print(f"    - Recall: {rec:.4f} ({rec*100:.2f}%)")
        print(f"    - F1 Score: {f1_group:.4f} ({f1_group*100:.2f}%)")
    
    # 保存结果到JSON
    results = {
        'overall': {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'total_images': len(image_files),
            'multi_target_images': total_stats['images_with_multi_targets'],
            'total_gt': total_stats['total_gt'],
            'total_pred': total_stats['total_pred']
        },
        'by_num_targets': {}
    }
    
    for num_targets, stats in total_stats['detection_by_num_targets'].items():
        tp_group = stats['tp']
        fp_group = stats['fp']
        fn_group = stats['fn']
        
        prec = tp_group / (tp_group + fp_group) if (tp_group + fp_group) > 0 else 0
        rec = tp_group / (tp_group + fn_group) if (tp_group + fn_group) > 0 else 0
        f1_group = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        
        results['by_num_targets'][str(num_targets)] = {
            'count': stats['count'],
            'precision': prec,
            'recall': rec,
            'f1': f1_group,
            'tp': tp_group,
            'fp': fp_group,
            'fn': fn_group
        }
    
    result_file = os.path.join(args.output_dir, 'multi_target_results.json')
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 结果已保存到: {result_file}")
    print("="*60)


if __name__ == '__main__':
    main()
