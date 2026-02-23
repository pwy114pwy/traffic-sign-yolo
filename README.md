# 交通标志识别系统（基于 YOLOv8）

## 🌟 项目简介

本项目基于 YOLOv8s 实现了一个高效、准确的交通标志识别系统，支持图像检测、视频/摄像头实时检测以及 Web 在线演示。项目在 GTSRB 数据集上进行训练和测试，能够识别 43 种不同类型的交通标志。相比之前的版本，YOLOv8 提供了更强的特征提取能力、更快的推理速度以及更简便的操作接口。

### ✨ 功能特性

- 📷 **图像检测**：支持上传单张图片进行交通标志检测
- 🎥 **视频检测**：支持上传视频文件进行逐帧检测
- 📹 **实时摄像头检测**：支持通过摄像头进行实时交通标志检测
- 🌐 **Web 在线演示**：提供用户友好的 Web 界面，支持参数调整和结果可视化
- 📊 **性能分析**：提供详细的性能分析脚本，包括参数量、计算量、推理速度等
- 📈 **模型评估**：支持在测试集上评估模型性能，生成详细的评估报告
- 🐳 **Docker 支持**：提供 Docker 镜像和 Docker Compose 配置，便于部署和运行

## 📋 环境要求

- Python 3.9+
- PyTorch 2.0+
- CUDA 11.x+（可选，用于 GPU 加速）
- ultralytics (YOLOv8 核心库)
- OpenCV, Flask, Matplotlib, NumPy 等

## 🛠️ 安装说明

### 1. 克隆项目

```bash
git clone https://github.com/pwy114pwy/traffic-sign-yolo.git
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

使用 YOLOv8 命令行接口进行训练：

```bash
yolo task=detect mode=train model=yolov8s.pt data=data/gtsrb.yaml epochs=50 imgsz=640 batch=16
```

### 3. 模型评估

在验证集上评估训练好的模型：

```bash
yolo task=detect mode=val model=runs/detect/train/weights/best.pt data=data/gtsrb.yaml imgsz=640
```

### 4. 性能分析

运行性能分析脚本，评估模型在特定硬件上的推理速度：

```bash
python performance_analysis.py --weights yolov8s.pt --img-size 640 --num-runs 100 --visualize --report
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
├── Dockerfile            # Docker 构建文件
├── docker-compose.yml    # Docker Compose 配置
├── performance_analysis.py  # 性能分析脚本
├── README.md             # 项目文档
├── requirements.txt      # 依赖列表
└── yolov8s.pt            # 预训练权重
```

## 📊 性能指标

| 模型 | 参数量 | 计算量 | 推理速度 | mAP@0.5 | mAP@0.5:0.95 |
|------|--------|--------|----------|---------|-------------|
| YOLOv8s | 11.2 M | 28.6 GFLOPs | 30+ FPS | 0.96+ | 0.72+ |

## 🎯 核心功能说明

### 1. 数据处理 Pipeline

- 自动下载 GTSRB 数据集
- 将 CSV 标注转换为 YOLO 格式
- 支持图像预处理和增强

### 2. 模型训练

- 支持完整的 YOLOv8 架构（n, s, m, l, x）
- 集成 Ultralytics 训练生态，支持多种增强配置
- 自动生成训练可视化报告（PR 曲线、混淆矩阵等）

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

### 模型评估

```bash
yolo task=detect mode=val model=<权重文件> data=<数据集配置> imgsz=<输入尺寸>
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

## 🔧 故障排除

### 问题1：模型文件未找到

**错误信息：**
```
❌ 模型文件未找到: runs/detect/train/weights/best.pt
```

**解决方案：**

1. **如果还没有训练模型**，请先训练：
   ```bash
   yolo task=detect mode=train model=yolov8s.pt data=data/gtsrb.yaml
   ```

2. **或者使用预训练权重作为临时方案**：
   - 创建 `.env` 文件（复制 `.env.example`）
   - 设置：`WEIGHTS_PATH=yolov8s.pt`
   - 注意：`ultralytics` 会自动下载该模型，但未针对交通标志优化。

### 问题2：依赖安装失败

**错误信息：**
```
ERROR: Could not find a version that satisfies the requirement...
```

**解决方案：**

1. **升级 pip**：
   ```bash
   python -m pip install --upgrade pip
   ```

2. **分步安装依赖**：
   ```bash
   # 先安装 PyTorch（根据您的 CUDA 版本选择）
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   
   # 再安装其他依赖
   pip install -r requirements.txt
   ```

3. **使用虚拟环境**（推荐）：
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   # source venv/bin/activate  # Linux/Mac
   pip install -r requirements.txt
   ```

### 问题3：CUDA 不可用

**错误信息：**
```
CUDA available: False
```

**解决方案：**

1. **检查 CUDA 安装**：
   ```bash
   nvidia-smi
   ```

2. **安装对应版本的 PyTorch**：
   - 访问 https://pytorch.org/
   - 选择您的 CUDA 版本
   - 按照说明安装

3. **如果没有 GPU**，使用 CPU 版本（会较慢）：
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
   ```

### 问题4：Web 应用无法启动

**错误信息：**
```
Address already in use
```

**解决方案：**

1. **更改端口**：
   - 在 `.env` 文件中设置：`FLASK_PORT=5001`

2. **或者停止占用端口的进程**：
   ```bash
   # Windows
   netstat -ano | findstr :5000
   taskkill /PID <进程ID> /F
   
   # Linux/Mac
   lsof -i :5000
   kill -9 <进程ID>
   ```

### 问题5：图片/视频检测失败

**可能原因：**
- 文件格式不支持
- 文件已损坏
- 内存不足

**解决方案：**

1. **检查文件格式**：
   - 支持的图片格式：PNG, JPG, JPEG, BMP, WEBP
   - 支持的视频格式：MP4, AVI, MOV, WMV, MKV

2. **检查日志文件**：
   ```bash
   cat logs/app.log  # Linux/Mac
   type logs\app.log  # Windows
   ```

3. **减小文件大小**：
   - 压缩图片/视频
   - 降低分辨率

### 问题6：检测结果不准确

**解决方案：**

1. **调整检测参数**：
   - 降低置信度阈值（默认 0.25）
   - 调整 IoU 阈值（默认 0.45）

2. **使用训练好的模型**：
   - 预训练的 `yolov5s.pt` 是在 COCO 数据集上训练的
   - 需要使用在 GTSRB 数据集上训练的模型才能获得最佳效果

3. **重新训练模型**：
   ```bash
   yolo task=detect mode=train model=yolov8s.pt data=data/gtsrb.yaml epochs=100
   ```

