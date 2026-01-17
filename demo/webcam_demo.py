# demo/webcam_demo.py
import cv2
import torch
import sys
import os
from pathlib import Path
import time

# 添加 yolov5 到系统路径
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # 项目根目录 (traffic-sign-yolo)
YOLOV5_ROOT = ROOT / 'yolov5'
sys.path.append(str(YOLOV5_ROOT))

from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_boxes
from utils.plots import Annotator, colors
from utils.augmentations import letterbox

class TrafficSignDetector:
    def __init__(self, weights_path, conf_thres=0.4, iou_thres=0.45, img_size=640):
        # 设备设置
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 模型加载
        self.model = attempt_load(weights_path, device=self.device)
        self.model.eval()  # 设置为评估模式
        
        # 参数设置
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.img_size = img_size
        self.stride = int(self.model.stride.max())
        
        # 类别名称
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        
        # FPS 计算
        self.prev_time = time.time()
    
    def preprocess(self, frame):
        """图像预处理"""
        # 使用 letterbox 保持宽高比
        img = letterbox(frame, self.img_size, stride=self.stride)[0]
        
        # 转换格式 BGR -> RGB, HWC -> CHW
        img = img[:, :, ::-1].transpose(2, 0, 1)
        
        # 转换为张量并归一化
        img = torch.from_numpy(img.copy()).to(self.device).float() / 255.0
        
        # 添加批次维度
        img = img.unsqueeze(0)
        
        return img
    
    def inference(self, img):
        """模型推理"""
        with torch.no_grad():
            pred = self.model(img, augment=False)[0]
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres)
        return pred
    
    def draw_results(self, frame, pred, img):
        """绘制检测结果"""
        annotator = Annotator(frame, line_width=2, example=str(self.names))
        
        # 异常检测设置：置信度阈值
        ANOMALY_CONF_THRES = 0.5  # 低于此阈值的检测结果视为异常
        
        for det in pred:
            if len(det):
                # 缩放检测框坐标到原始图像尺寸
                det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], frame.shape).round()
                
                # 绘制检测框和标签
                for *xyxy, conf, cls in reversed(det):
                    if conf < ANOMALY_CONF_THRES:
                        # 未知交通标志（异常）
                        label = f'Unknown Sign {conf:.2f}'
                        annotator.box_label(xyxy, label, color=(255, 0, 0))  # 红色框
                    else:
                        # 已知交通标志
                        c = int(cls)
                        label = f'{self.names[c]} {conf:.2f}'
                        annotator.box_label(xyxy, label, color=colors(c, True))
        
        return annotator.result()
    
    def calculate_fps(self):
        """计算并显示FPS"""
        curr_time = time.time()
        fps = 1 / (curr_time - self.prev_time)
        self.prev_time = curr_time
        return fps
    
    def detect_frame(self, frame):
        """处理单帧图像"""
        # 预处理
        img = self.preprocess(frame)
        
        # 推理
        pred = self.inference(img)
        
        # 绘制结果
        result_frame = self.draw_results(frame, pred, img)
        
        # 计算FPS
        fps = self.calculate_fps()
        
        # 在图像上显示FPS
        cv2.putText(result_frame, f'FPS: {fps:.1f}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return result_frame

def main():
    # 配置参数
    WEIGHTS_PATH = r"D:\traffic-sign-yolo\yolov5\runs\train\exp3\weights\best.pt"
    CONF_THRES = 0.4
    IOU_THRES = 0.45
    IMG_SIZE = 640
    
    # 创建检测器实例
    detector = TrafficSignDetector(
        weights_path=WEIGHTS_PATH,
        conf_thres=CONF_THRES,
        iou_thres=IOU_THRES,
        img_size=IMG_SIZE
    )
    
    # 初始化摄像头
    cap = cv2.VideoCapture(0)
    
    # 设置摄像头参数（可选）
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("无法读取摄像头画面")
                break
            
            # 处理单帧
            result_frame = detector.detect_frame(frame)
            
            # 显示结果
            cv2.imshow('Traffic Sign Detection', result_frame)
            
            # 检测退出键
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("程序被用户中断")
    finally:
        # 清理资源
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()