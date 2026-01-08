# web/app.py
import os
import cv2
import torch
from flask import Flask, request, render_template
from pathlib import Path
import sys

# 添加 yolov5 到系统路径
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # 项目根目录 (traffic-sign-yolo)
YOLOV5_ROOT = ROOT / 'yolov5'
sys.path.append(str(YOLOV5_ROOT))

from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_boxes
from utils.plots import Annotator, colors
from utils.augmentations import letterbox

# ======================
# 配置区（只需改这里！）
# ======================
PROJECT_ROOT = r"D:\traffic-sign-yolo"  # 你的项目根目录
WEIGHTS_PATH = os.path.join(PROJECT_ROOT, "yolov5", "runs", "train", "exp3", "weights", "best.pt")
UPLOAD_FOLDER = os.path.join(PROJECT_ROOT, "web", "static", "uploads")
RESULT_FOLDER = os.path.join(PROJECT_ROOT, "web", "static", "results")

# 创建文件夹
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# 全局加载模型（启动时只加载一次）
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Loading model from: {WEIGHTS_PATH}")
model = attempt_load(WEIGHTS_PATH, device=DEVICE)
stride = int(model.stride.max())  # 获取模型步长
names = model.module.names if hasattr(model, 'module') else model.names
print(f"Model loaded. Classes: {len(names)}")

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    if not file or not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        return "请上传有效的图片文件！", 400

    # 保存上传文件
    input_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(input_path)

    # 读取原图
    img0 = cv2.imread(input_path)
    if img0 is None:
        return "无法读取图片，请检查格式！", 400

    # 使用letterbox预处理，保持宽高比
    img = letterbox(img0, 640, stride=stride)[0]  # 使用模型的stride
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
    img = torch.from_numpy(img.copy()).to(DEVICE).float() / 255.0
    img = img.unsqueeze(0)

    # 推理
    with torch.no_grad():
        pred = model(img, augment=True)[0]  # 启用augment进行多尺度测试
        pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)  # 降低阈值

    # 画检测框
    annotator = Annotator(img0, line_width=2, example=str(names))
    for det in pred:
        if len(det):
            # 使用正确的缩放函数
            det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], img0.shape).round()
            for *xyxy, conf, cls in reversed(det):
                c = int(cls)
                label = f'{names[c]} {conf:.2f}'
                annotator.box_label(xyxy, label, color=colors(c, True))

    # 为结果图片生成不同的文件名
    output_path = os.path.join(RESULT_FOLDER, file.filename)
    cv2.imwrite(output_path, annotator.result())

    # 返回结果页面，显示处理后的图片
    return render_template('result.html', img_path=file.filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)