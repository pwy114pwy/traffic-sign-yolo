# 性能分析脚本
import torch
import time
import numpy as np
import cv2
import os
import sys

# 添加 yolov5 到系统路径
FILE = os.path.abspath(__file__)
ROOT = os.path.dirname(FILE)
YOLOV5_ROOT = os.path.join(ROOT, 'yolov5')
sys.path.append(str(YOLOV5_ROOT))

from fvcore.nn import FlopCountAnalysis, parameter_count
from models.experimental import attempt_load
from utils.general import non_max_suppression
from utils.augmentations import letterbox

# 加载模型
def load_model(weights_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = attempt_load(weights_path, device=device)
    model.eval()
    return model, device

# 计算参数量
def calculate_parameters(model):
    param_count = parameter_count(model)
    total_params = param_count['']  # 总参数量
    return total_params

# 计算计算量 (FLOPs)
def calculate_flops(model, input_shape=(3, 640, 640)):
    # 创建随机输入张量
    input_tensor = torch.randn(1, *input_shape).to(next(model.parameters()).device)
    
    # 计算FLOPs
    flops = FlopCountAnalysis(model, input_tensor)
    total_flops = flops.total()  # 总FLOPs
    
    return total_flops

# 测试推理速度
def test_inference_speed(model, device, num_runs=100, input_shape=(640, 640)):
    # 准备测试图像
    img = np.random.randint(0, 255, (input_shape[0], input_shape[1], 3), dtype=np.uint8)
    
    # 预处理
    img = letterbox(img, input_shape[0], stride=int(model.stride.max()))[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
    img = torch.from_numpy(img.copy()).to(device).float() / 255.0
    img = img.unsqueeze(0)
    
    # 预热
    for _ in range(10):
        with torch.no_grad():
            model(img)
    
    # 测试推理速度
    start_time = time.time()
    for _ in range(num_runs):
        with torch.no_grad():
            pred = model(img)[0]
            pred = non_max_suppression(pred, 0.25, 0.45)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_runs
    fps = 1 / avg_time
    
    return avg_time, fps

# 测试端到端速度（包括预处理和后处理）
def test_end_to_end_speed(model, device, num_runs=100, input_shape=(640, 640)):
    # 准备测试图像
    img = np.random.randint(0, 255, (input_shape[0], input_shape[1], 3), dtype=np.uint8)
    
    # 测试端到端速度
    start_time = time.time()
    for _ in range(num_runs):
        # 预处理
        img_processed = letterbox(img, input_shape[0], stride=int(model.stride.max()))[0]
        img_processed = img_processed[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
        img_processed = torch.from_numpy(img_processed.copy()).to(device).float() / 255.0
        img_processed = img_processed.unsqueeze(0)
        
        # 推理
        with torch.no_grad():
            pred = model(img_processed)[0]
            pred = non_max_suppression(pred, 0.25, 0.45)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_runs
    fps = 1 / avg_time
    
    return avg_time, fps

# 主函数
if __name__ == "__main__":
    
    # 模型路径
    weights_path = os.path.join(ROOT, 'yolov5s.pt')
    
    print("===== 性能分析 ====")
    
    # 1. 加载模型
    print("1. 加载模型...")
    model, device = load_model(weights_path)
    print(f"   设备: {device}")
    
    # 2. 计算参数量
    print("2. 计算参数量...")
    total_params = calculate_parameters(model)
    print(f"   总参数量: {total_params:,} (约 {total_params / 1e6:.2f} M)")
    
    # 3. 计算计算量
    print("3. 计算计算量 (FLOPs)...")
    total_flops = calculate_flops(model)
    print(f"   总计算量: {total_flops:,} (约 {total_flops / 1e9:.2f} GFLOPs)")
    
    # 4. 测试推理速度
    print("4. 测试推理速度...")
    avg_inference_time, inference_fps = test_inference_speed(model, device)
    print(f"   平均推理时间: {avg_inference_time * 1000:.2f} ms")
    print(f"   推理FPS: {inference_fps:.2f}")
    
    # 5. 测试端到端速度
    print("5. 测试端到端速度 (包含预处理和后处理)...")
    avg_e2e_time, e2e_fps = test_end_to_end_speed(model, device)
    print(f"   平均端到端时间: {avg_e2e_time * 1000:.2f} ms")
    print(f"   端到端FPS: {e2e_fps:.2f}")
    
    # 6. 模型信息
    print("\n===== 模型信息 ====")
    print(f"   模型名称: YOLOv5s")
    print(f"   输入尺寸: 640x640")
    print(f"   设备: {device}")
    print(f"   参数量: {total_params / 1e6:.2f} M")
    print(f"   计算量: {total_flops / 1e9:.2f} GFLOPs")
    print(f"   推理速度: {inference_fps:.2f} FPS")
    print(f"   端到端速度: {e2e_fps:.2f} FPS")
    
    print("\n===== 分析完成 ====")
