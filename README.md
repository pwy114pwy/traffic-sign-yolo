# 交通标志识别系统（基于 YOLOv5）

## 简介

本项目使用 YOLOv5s 在 GTSRB 数据集上训练交通标志检测模型，支持：

- 图像检测
- 视频/摄像头实时检测
- Web 在线演示

## 环境

- Python 3.8+
- PyTorch 1.8+
- OpenCV, Flask

## 使用步骤

1. `pip install -r requirements.txt`
2. 准备数据集（见 scripts/）
3. `python yolov5/train.py --data data/gtsrb.yaml --weights yolov5s.pt --epochs 50`
4. `python demo/webcam_demo.py` 查看实时效果

## 复试亮点

- 自建数据处理 pipeline
- 对比 YOLOv5s/v5m 性能
- 实现实时检测与 Web 部署
