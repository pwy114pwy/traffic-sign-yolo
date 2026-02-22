# scripts/train_yolov8.py
"""
使用 YOLOv8 训练交通标志检测模型。

运行方式（在项目根目录执行）：
    python scripts/train_yolov8.py

训练完成后，权重文件位于：
    runs/train/yolov8_gtsrb/weights/best.pt

将 .env 中的 WEIGHTS_PATH 改为上述路径即可在 Web 服务中使用。
"""

import os
import argparse
from pathlib import Path
from ultralytics import YOLO

# ======================================================
# 配置区（也可通过命令行参数覆盖）
# ======================================================
ROOT = Path(__file__).resolve().parents[1]  # 项目根目录

DEFAULT_DATA    = str(ROOT / "data" / "gtsrb.yaml")  # 数据集配置文件
DEFAULT_MODEL   = "yolov8s.pt"  # 基础预训练权重（首次运行会自动下载）
DEFAULT_EPOCHS  = 100           # 训练轮数（建议不低于 50，有条件可设 150+）
DEFAULT_IMGSZ   = 640           # 输入图像尺寸
DEFAULT_BATCH   = 16            # 批大小（显存不足时调小，如 8）
DEFAULT_WORKERS = 4             # 数据加载线程数（Windows 上如有报错可改为 0）
DEFAULT_PROJECT = str(ROOT / "runs" / "train")
DEFAULT_NAME    = "yolov8_gtsrb"
DEFAULT_DEVICE  = ""            # "" 表示自动选择（GPU 优先），可手动指定 "0" 或 "cpu"


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv8 交通标志训练脚本")
    parser.add_argument("--data",    default=DEFAULT_DATA,    help="数据集 yaml 路径")
    parser.add_argument("--model",   default=DEFAULT_MODEL,   help="基础模型权重（.pt）")
    parser.add_argument("--epochs",  default=DEFAULT_EPOCHS,  type=int,   help="训练轮数")
    parser.add_argument("--imgsz",   default=DEFAULT_IMGSZ,   type=int,   help="输入尺寸")
    parser.add_argument("--batch",   default=DEFAULT_BATCH,   type=int,   help="批大小")
    parser.add_argument("--workers", default=DEFAULT_WORKERS, type=int,   help="数据加载线程数")
    parser.add_argument("--project", default=DEFAULT_PROJECT, help="输出目录")
    parser.add_argument("--name",    default=DEFAULT_NAME,    help="实验名称")
    parser.add_argument("--device",  default=DEFAULT_DEVICE,  help="设备 ('' / '0' / 'cpu')")
    parser.add_argument("--resume",  action="store_true",     help="从上次中断处继续训练")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("  YOLOv8 交通标志检测训练")
    print("=" * 60)
    print(f"  数据集配置  : {args.data}")
    print(f"  基础模型    : {args.model}")
    print(f"  训练轮数    : {args.epochs}")
    print(f"  图像尺寸    : {args.imgsz}")
    print(f"  批大小      : {args.batch}")
    print(f"  输出目录    : {args.project}/{args.name}")
    print("=" * 60)

    # 加载模型
    model = YOLO(args.model)

    # 开始训练
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        project=args.project,
        name=args.name,
        device=args.device if args.device else None,
        resume=args.resume,
        # 数据增强（YOLOv8 默认已开启，可按需调整）
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        flipud=0.0,      # 交通标志不需要上下翻转
        fliplr=0.0,      # 交通标志不需要左右翻转（有方向性）
        mosaic=1.0,
        mixup=0.1,
    )

    # 输出最终权重路径
    best_weights = Path(args.project) / args.name / "weights" / "best.pt"
    print("\n" + "=" * 60)
    print("  ✅ 训练完成！")
    print(f"  最佳权重: {best_weights}")
    print(f"\n  请将 .env 中的 WEIGHTS_PATH 更新为：")
    print(f"  WEIGHTS_PATH={best_weights}")
    print("=" * 60)


if __name__ == "__main__":
    main()
