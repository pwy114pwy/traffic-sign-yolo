# 交通标志识别系统（基于 YOLOv5）

## 🌟 项目简介

本项目基于 YOLOv5s 实现了一个高效、准确的交通标志识别系统，支持图像检测、视频/摄像头实时检测以及 Web 在线演示。项目在 GTSRB 数据集上进行训练和测试，能够识别 43 种不同类型的交通标志。

### ✨ 功能特性

- 📷 **图像检测**：支持上传单张图片进行交通标志检测
- 🎥 **视频检测**：支持上传视频文件进行逐帧检测
- 📹 **实时摄像头检测**：支持通过摄像头进行实时交通标志检测
- 🌐 **Web 在线演示**：提供用户友好的 Web 界面，支持参数调整和结果可视化
- 📊 **性能分析**：提供详细的性能分析脚本，包括参数量、计算量、推理速度等
- 📈 **模型评估**：支持在测试集上评估模型性能，生成详细的评估报告
- 🐳 **Docker 支持**：提供 Docker 镜像和 Docker Compose 配置，便于部署和运行

## 📋 环境要求

- Python 3.8+
- PyTorch 1.8+
- CUDA 10.2+（可选，用于 GPU 加速）
- OpenCV, Flask, Matplotlib, NumPy 等

## 🛠️ 安装说明

### 1. 克隆项目

```bash
git clone https://github.com/yourusername/traffic-sign-yolo.git
cd traffic-sign-yolo
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

## 🚀 使用步骤

### 1. 数据集准备

#### 1.1 下载数据集

```bash
python scripts/download_gtsrb.py
```

或使用自定义参数：

```bash
python scripts/download_gtsrb.py --url https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB-Complete.zip --output_dir datasets/
```

#### 1.2 转换数据集格式

```bash
python scripts/convert_gtsrb_to_yolo.py
```

或使用自定义参数：

```bash
python scripts/convert_gtsrb_to_yolo.py --input_dir datasets/GTSRB --output_dir datasets/gtsrb
```

### 2. 模型训练

```bash
python yolov5/train.py --data data/gtsrb.yaml --weights yolov5s.pt --epochs 50 --batch-size 16 --img 640
```

### 3. 模型评估

```bash
python evaluate_model.py --weights yolov5s.pt --data data/gtsrb.yaml --img-size 640 --batch-size 32
```

### 4. 性能分析

```bash
python performance_analysis.py --weights yolov5s.pt --img-size 640 --num-runs 100 --visualize --report
```

### 5. 实时摄像头检测

```bash
python demo/webcam_demo.py
```

### 6. Web 演示

#### 6.1 启动 Web 服务器

```bash
python web/app.py
```

#### 6.2 访问 Web 界面

在浏览器中访问 `http://localhost:5000`，即可使用 Web 界面进行交通标志检测。

#### 6.3 Web 功能

- **图片检测**：上传图片进行交通标志检测
- **视频检测**：上传视频文件进行逐帧检测
- **参数调整**：可调整置信度阈值和 IoU 阈值
- **结果可视化**：显示检测结果的柱状图和饼图
- **详细统计**：提供检测数量、类型、置信度等统计信息

## 🐳 Docker 部署

### 1. 使用 Dockerfile 构建镜像

```bash
docker build -t traffic-sign-detector .
```

### 2. 运行 Docker 容器

```bash
docker run -d -p 5000:5000 --name traffic-sign-detector traffic-sign-detector
```

### 3. 使用 Docker Compose

```bash
docker-compose up -d
```

## 📁 项目结构

```
traffic-sign-yolo/
├── data/                 # 数据集配置文件
│   └── gtsrb.yaml        # GTSRB 数据集配置
├── demo/                 # 演示脚本
│   └── webcam_demo.py    # 摄像头实时检测脚本
├── logs/                 # 日志文件
├── scripts/              # 数据处理脚本
│   ├── convert_gtsrb_to_yolo.py  # 转换数据集格式
│   └── download_gtsrb.py         # 下载数据集
├── web/                  # Web 应用
│   ├── static/           # 静态资源
│   ├── templates/        # HTML 模板
│   └── app.py            # Flask 应用
├── yolov5/               # YOLOv5 源码
├── Dockerfile            # Docker 构建文件
├── docker-compose.yml    # Docker Compose 配置
├── evaluate_model.py     # 模型评估脚本
├── performance_analysis.py  # 性能分析脚本
├── README.md             # 项目文档
├── requirements.txt      # 依赖列表
└── yolov5s.pt            # 预训练权重
```

## 📊 性能指标

| 模型 | 参数量 | 计算量 | 推理速度 | mAP@0.5 | mAP@0.5:0.95 |
|------|--------|--------|----------|---------|-------------|
| YOLOv5s | 7.2 M | 16.5 GFLOPs | 30+ FPS | 0.95+ | 0.70+ |

## 🎯 核心功能说明

### 1. 数据处理 Pipeline

- 自动下载 GTSRB 数据集
- 将 CSV 标注转换为 YOLO 格式
- 支持图像预处理和增强

### 2. 模型训练

- 支持多种 YOLOv5 模型（s, m, l, x）
- 可配置训练参数（ epochs, batch size, learning rate 等）
- 支持早停和模型保存

### 3. 模型评估

- 计算精确率、召回率、F1 分数
- 生成混淆矩阵
- 计算 mAP@0.5 和 mAP@0.5:0.95
- 生成详细的评估报告

### 4. 性能分析

- 计算参数量和计算量
- 测试推理速度和内存使用
- 分析不同输入尺寸对性能的影响
- 分析不同批量大小对性能的影响
- 生成可视化报告

## 📝 命令行参数说明

### 数据处理脚本

#### download_gtsrb.py

```bash
python scripts/download_gtsrb.py --url <下载地址> --output_dir <输出目录>
```

#### convert_gtsrb_to_yolo.py

```bash
python scripts/convert_gtsrb_to_yolo.py --input_dir <输入目录> --output_dir <输出目录>
```

### 模型评估脚本

```bash
python evaluate_model.py --weights <权重文件> --data <数据集配置> --img-size <输入尺寸> --batch-size <批量大小>
```

### 性能分析脚本

```bash
python performance_analysis.py --weights <权重文件> --img-size <输入尺寸> --num-runs <运行次数> --visualize --report
```

## 🤝 致谢

- [YOLOv5](https://github.com/ultralytics/yolov5) - 目标检测框架
- [GTSRB 数据集](https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB-Complete.zip) - 交通标志数据集

## 📄 许可证

本项目采用 MIT 许可证，详见 [LICENSE](LICENSE) 文件。

## 📧 联系方式

如有问题或建议，请通过以下方式联系：

- Email: yourname@example.com
- GitHub: [yourusername](https://github.com/yourusername)

