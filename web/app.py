# web/app.py
import os
import cv2
import torch
import time
from flask import Flask, request, render_template, jsonify
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

    # 检查图片是否过小，如果是则放大
    original_shape = img0.shape
    min_dim = min(original_shape[0], original_shape[1])
    SCALE_THRESHOLD = 200  # 最小尺寸阈值，提高到200像素以确保标签有足够空间
    scale_factor = 1.0
    
    if min_dim < SCALE_THRESHOLD:
        # 计算放大比例
        scale_factor = SCALE_THRESHOLD / min_dim
        new_width = int(original_shape[1] * scale_factor)
        new_height = int(original_shape[0] * scale_factor)
        img0 = cv2.resize(img0, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    
    # 动态计算线条宽度，根据图片大小调整
    min_dim = min(img0.shape[0], img0.shape[1])
    line_width = max(1, int(min_dim / 300))  # 每300像素对应1个像素宽度，最小为1
    line_width = min(line_width, 4)  # 最大宽度为4
    
    # 画检测框
    annotator = Annotator(img0, line_width=line_width, example=str(names))
    
    # 异常检测设置：置信度阈值
    ANOMALY_CONF_THRES = 0.5  # 低于此阈值的检测结果视为异常
    
    for det in pred:
        if len(det):
            # 使用正确的缩放函数
            det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], original_shape).round()
            
            # 如果图片被放大，检测框坐标也需要相应放大
            if scale_factor != 1.0:
                det[:, :4] *= scale_factor
                det[:, :4] = det[:, :4].round()
                
            for *xyxy, conf, cls in reversed(det):
                if conf < ANOMALY_CONF_THRES:
                    # 未知交通标志（异常）
                    label = f'Unknown Sign {conf:.2f}'
                    annotator.box_label(xyxy, label, color=(255, 0, 0))  # 红色框
                else:
                    # 已知交通标志
                    c = int(cls)
                    label = f'{names[c]} {conf:.2f}'
                    annotator.box_label(xyxy, label, color=colors(c, True))

    output_path = os.path.join(RESULT_FOLDER, file.filename)
    cv2.imwrite(output_path, annotator.result())

    # 计算检测统计信息
    detection_count = 0
    unique_classes = set()
    total_confidence = 0.0
    detections = []
    anomaly_count = 0  # 异常检测计数
    
    # 异常检测设置：置信度阈值
    ANOMALY_CONF_THRES = 0.5  # 低于此阈值的检测结果视为异常
    
    for det in pred:
        if len(det):
            detection_count += len(det)
            for *xyxy, conf, cls in reversed(det):
                if conf < ANOMALY_CONF_THRES:
                    # 未知交通标志（异常）
                    unique_classes.add('Unknown Sign')
                    total_confidence += conf.item()
                    detections.append({
                        'name': 'Unknown Sign',
                        'confidence': conf.item() * 100
                    })
                    anomaly_count += 1
                else:
                    # 已知交通标志
                    c = int(cls)
                    unique_classes.add(names[c])
                    total_confidence += conf.item()
                    detections.append({
                        'name': names[c],
                        'confidence': conf.item() * 100
                    })
    
    unique_classes_count = len(unique_classes)
    avg_confidence = (total_confidence / detection_count * 100) if detection_count > 0 else 0.0
    
    # 返回结果页面，显示处理后的图片和统计信息
    return render_template('result.html', 
                         img_path=file.filename,
                         detection_count=detection_count,
                         unique_classes_count=unique_classes_count,
                         avg_confidence=avg_confidence,
                         detections=detections,
                         anomaly_count=anomaly_count)

@app.route('/predict_video', methods=['POST'])
def predict_video():
    file = request.files['video']
    if not file or not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.wmv')):
        return "请上传有效的视频文件！", 400
    
    # 保存上传文件
    input_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(input_path)
    
    # 打开视频
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        return "无法打开视频文件！", 400
    
    # 获取视频信息
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 创建视频编写器
    output_path = os.path.join(RESULT_FOLDER, file.filename)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用mp4格式
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # 检测统计信息
    detection_count = 0
    unique_classes = set()
    total_confidence = 0.0
    frame_detections = []
    anomaly_count = 0  # 异常检测计数
    
    # 逐帧处理
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # 预处理
        img = letterbox(frame, 640, stride=stride)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
        img = torch.from_numpy(img.copy()).to(DEVICE).float() / 255.0
        img = img.unsqueeze(0)
        
        # 推理
        with torch.no_grad():
            pred = model(img, augment=True)[0]
            pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)
        
        # 动态计算线条宽度，根据图片大小调整
        min_dim = min(frame.shape[0], frame.shape[1])
        line_width = max(1, int(min_dim / 300))  # 每300像素对应1个像素宽度，最小为1
        line_width = min(line_width, 4)  # 最大宽度为4
        
        # 画检测框
        annotator = Annotator(frame, line_width=line_width, example=str(names))
        
        # 异常检测设置：置信度阈值
        ANOMALY_CONF_THRES = 0.5  # 低于此阈值的检测结果视为异常
        
        for det in pred:
            if len(det):
                # 使用正确的缩放函数
                det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], frame.shape).round()
                detection_count += len(det)
                for *xyxy, conf, cls in reversed(det):
                    if conf < ANOMALY_CONF_THRES:
                        # 未知交通标志（异常）
                        unique_classes.add('Unknown Sign')
                        total_confidence += conf.item()
                        label = f'Unknown Sign {conf:.2f}'
                        annotator.box_label(xyxy, label, color=(255, 0, 0))  # 红色框
                        anomaly_count += 1
                    else:
                        # 已知交通标志
                        c = int(cls)
                        unique_classes.add(names[c])
                        total_confidence += conf.item()
                        label = f'{names[c]} {conf:.2f}'
                        annotator.box_label(xyxy, label, color=colors(c, True))
        
        # 写入处理后的帧
        out.write(annotator.result())
    
    # 释放资源
    cap.release()
    out.release()
    
    unique_classes_count = len(unique_classes)
    avg_confidence = (total_confidence / detection_count * 100) if detection_count > 0 else 0.0
    
    # 返回结果页面，显示处理后的视频和统计信息
    return render_template('result.html', 
                         video_path=file.filename,
                         detection_count=detection_count,
                         unique_classes_count=unique_classes_count,
                         avg_confidence=avg_confidence,
                         anomaly_count=anomaly_count)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)