# 交通标志识别系统 - 快速开始指南

## 📋 前置要求

- Python 3.8 或更高版本
- （可选）NVIDIA GPU + CUDA 用于加速

## 🚀 快速开始（5分钟）

### 步骤 1: 克隆项目

```bash
cd d:\traffic-sign-yolo
```

### 步骤 2: 创建虚拟环境（推荐）

```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac
```

### 步骤 3: 安装依赖

```bash
# 升级 pip
python -m pip install --upgrade pip

# 安装依赖
pip install -r requirements.txt
```

> **注意**：如果您有 NVIDIA GPU，请先安装对应的 PyTorch CUDA 版本：
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
> ```

### 步骤 4: 配置环境变量

```bash
# 复制环境变量示例文件
copy .env.example .env  # Windows
# cp .env.example .env  # Linux/Mac
```

编辑 `.env` 文件，设置模型路径：

```bash
# 使用预训练权重（临时方案，可立即运行）
WEIGHTS_PATH=yolov5s.pt

# 或者使用训练好的模型（需要先训练）
# WEIGHTS_PATH=yolov5/runs/train/exp3/weights/best.pt
```

### 步骤 5: 启动 Web 应用

```bash
python web/app.py
```

您应该看到类似以下的输出：

```
2026-02-03 22:59:52,123 - __main__ - INFO - 📁 PROJECT_ROOT: d:\traffic-sign-yolo
2026-02-03 22:59:52,124 - __main__ - INFO - 📦 WEIGHTS_PATH: d:\traffic-sign-yolo\yolov5s.pt
2026-02-03 22:59:52,125 - __main__ - INFO - 📤 UPLOAD_FOLDER: d:\traffic-sign-yolo\web\static\uploads
2026-02-03 22:59:52,126 - __main__ - INFO - 📥 RESULT_FOLDER: d:\traffic-sign-yolo\web\static\results
2026-02-03 22:59:52,127 - __main__ - INFO - 🖥️  使用设备: cuda
2026-02-03 22:59:52,128 - __main__ - INFO - ⏳ 正在加载模型: d:\traffic-sign-yolo\yolov5s.pt
2026-02-03 22:59:55,456 - __main__ - INFO - ✅ 模型加载成功！类别数: 80
2026-02-03 22:59:55,457 - __main__ - INFO - 启动Flask服务器: 0.0.0.0:5000, debug=False
 * Running on all addresses (0.0.0.0)
 * Running on http://127.0.0.1:5000
 * Running on http://192.168.1.100:5000
```

### 步骤 6: 访问 Web 界面

在浏览器中打开：**http://localhost:5000**

## 🎯 使用说明

### 图片检测

1. 点击"图片检测"标签
2. 点击"选择图片文件"上传图片
3. 调整检测参数（可选）：
   - **置信度阈值**：降低可检测更多目标，但可能增加误检
   - **IoU阈值**：用于过滤重叠的检测框
4. 点击"开始检测"
5. 查看检测结果和统计信息

### 视频检测

1. 点击"视频检测"标签
2. 点击"选择视频文件"上传视频
3. 调整检测参数（可选）
4. 点击"开始检测"
5. 等待处理完成（处理时间取决于视频长度）
6. 查看处理后的视频和统计信息

## ⚙️ 高级配置

### 环境变量说明

编辑 `.env` 文件可以自定义以下配置：

```bash
# 模型权重路径
WEIGHTS_PATH=yolov5s.pt

# 文件存储目录
UPLOAD_FOLDER=web/static/uploads
RESULT_FOLDER=web/static/results

# 检测参数
ANOMALY_CONF_THRES=0.5  # 异常检测阈值
SCALE_THRESHOLD=200      # 图片缩放阈值

# Flask 配置
FLASK_HOST=0.0.0.0
FLASK_PORT=5000
FLASK_DEBUG=False

# 日志配置
LOG_LEVEL=INFO
LOG_FILE=logs/app.log
```

### 训练自己的模型

如果您想获得最佳的交通标志检测效果，需要训练专门的模型：

#### 1. 准备数据集

```bash
# 下载 GTSRB 数据集
python scripts/download_gtsrb.py

# 转换为 YOLO 格式
python scripts/convert_gtsrb_to_yolo.py
```

#### 2. 训练模型

```bash
python yolov5/train.py --data data/gtsrb.yaml --weights yolov5s.pt --epochs 50 --batch-size 16 --img 640
```

训练参数说明：
- `--data`: 数据集配置文件
- `--weights`: 预训练权重（迁移学习）
- `--epochs`: 训练轮数（建议 50-100）
- `--batch-size`: 批量大小（根据 GPU 内存调整）
- `--img`: 输入图像尺寸

#### 3. 使用训练好的模型

训练完成后，修改 `.env` 文件：

```bash
WEIGHTS_PATH=yolov5/runs/train/exp/weights/best.pt
```

## 🔍 日志查看

应用运行时会生成日志文件，可以用于调试：

```bash
# Windows
type logs\app.log

# Linux/Mac
cat logs/app.log

# 实时查看日志
tail -f logs/app.log  # Linux/Mac
Get-Content logs\app.log -Wait  # PowerShell
```

## 📊 性能优化建议

### 1. 使用 GPU 加速

确保安装了 CUDA 版本的 PyTorch：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 2. 调整批量大小

对于视频处理，可以修改代码实现批量推理以提高速度。

### 3. 使用更小的模型

如果速度是首要考虑，可以使用 YOLOv5n（nano）：

```bash
python yolov5/train.py --data data/gtsrb.yaml --weights yolov5n.pt --epochs 50
```

## ❓ 常见问题

### Q: 为什么检测结果不准确？

**A:** 如果使用的是预训练的 `yolov5s.pt`，它是在 COCO 数据集（80个通用类别）上训练的，不包含交通标志类别。需要使用在 GTSRB 数据集上训练的模型才能准确检测交通标志。

### Q: 如何提高检测速度？

**A:** 
1. 使用 GPU（CUDA）
2. 降低输入图像分辨率
3. 使用更小的模型（如 YOLOv5n）
4. 关闭 augment 参数

### Q: 支持哪些文件格式？

**A:**
- **图片**：PNG, JPG, JPEG, BMP, WEBP
- **视频**：MP4, AVI, MOV, WMV, MKV

### Q: 如何在生产环境部署？

**A:** 
1. 使用 Docker（推荐）：
   ```bash
   docker-compose up -d
   ```

2. 或使用生产级 WSGI 服务器（如 Gunicorn）：
   ```bash
   pip install gunicorn
   gunicorn -w 4 -b 0.0.0.0:5000 web.app:app
   ```

## 📚 更多资源

- [YOLOv5 官方文档](https://github.com/ultralytics/yolov5)
- [GTSRB 数据集](https://benchmark.ini.rub.de/gtsrb_news.html)
- [项目 README](README.md)
- [故障排除指南](README.md#故障排除)

## 🎉 开始使用

现在您已经准备好使用交通标志识别系统了！如果遇到任何问题，请查看[故障排除部分](README.md#故障排除)或查看日志文件。
