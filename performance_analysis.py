# 性能分析脚本
import torch
import time
import numpy as np
import cv2
import os
import sys
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import psutil

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

# 测试内存使用
def test_memory_usage(model, device, input_shape=(3, 640, 640)):
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # 创建输入张量
        input_tensor = torch.randn(1, *input_shape).to(device)
        
        # 推理
        with torch.no_grad():
            model(input_tensor)
        
        peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
        return peak_memory
    else:
        # CPU内存使用
        process = psutil.Process(os.getpid())
        before_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # 创建输入张量
        input_tensor = torch.randn(1, *input_shape).to(device)
        
        # 推理
        with torch.no_grad():
            model(input_tensor)
        
        after_memory = process.memory_info().rss / 1024 / 1024  # MB
        return after_memory - before_memory

# 测试推理速度
def test_inference_speed(model, device, num_runs=100, input_shape=(640, 640), batch_size=1):
    # 准备测试图像
    img = np.random.randint(0, 255, (input_shape[0], input_shape[1], 3), dtype=np.uint8)
    
    # 预处理
    img = letterbox(img, input_shape[0], stride=int(model.stride.max()))[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
    img = torch.from_numpy(img.copy()).to(device).float() / 255.0
    img = img.unsqueeze(0)
    
    # 批量处理
    img_batch = img.repeat(batch_size, 1, 1, 1)
    
    # 预热
    for _ in range(10):
        with torch.no_grad():
            model(img_batch)
    
    # 测试推理速度
    start_time = time.time()
    times = []
    for _ in range(num_runs):
        iter_start = time.time()
        with torch.no_grad():
            pred = model(img_batch)[0]
            pred = non_max_suppression(pred, 0.25, 0.45)
        iter_end = time.time()
        times.append(iter_end - iter_start)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_runs
    fps = batch_size / avg_time
    
    return avg_time, fps, times

# 测试端到端速度（包括预处理和后处理）
def test_end_to_end_speed(model, device, num_runs=100, input_shape=(640, 640), batch_size=1):
    # 准备测试图像
    img = np.random.randint(0, 255, (input_shape[0], input_shape[1], 3), dtype=np.uint8)
    
    # 测试端到端速度
    start_time = time.time()
    times = []
    for _ in range(num_runs):
        iter_start = time.time()
        
        # 批量预处理
        img_processed_list = []
        for _ in range(batch_size):
            img_processed = letterbox(img, input_shape[0], stride=int(model.stride.max()))[0]
            img_processed = img_processed[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
            img_processed = torch.from_numpy(img_processed.copy()).to(device).float() / 255.0
            img_processed_list.append(img_processed)
        img_batch = torch.stack(img_processed_list)
        
        # 推理
        with torch.no_grad():
            pred = model(img_batch)[0]
            pred = non_max_suppression(pred, 0.25, 0.45)
        
        iter_end = time.time()
        times.append(iter_end - iter_start)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_runs
    fps = batch_size / avg_time
    
    return avg_time, fps, times

# 测试不同输入尺寸下的性能
def test_different_input_sizes(model, device, sizes=[320, 480, 640, 800, 960], num_runs=50):
    results = []
    
    for size in sizes:
        print(f"\nTesting input size: {size}x{size}")
        
        # 计算FLOPs
        flops = calculate_flops(model, input_shape=(3, size, size))
        
        # 测试推理速度
        avg_inf_time, inf_fps, _ = test_inference_speed(model, device, num_runs=num_runs, input_shape=(size, size))
        
        # 测试端到端速度
        avg_e2e_time, e2e_fps, _ = test_end_to_end_speed(model, device, num_runs=num_runs, input_shape=(size, size))
        
        # 测试内存使用
        memory = test_memory_usage(model, device, input_shape=(3, size, size))
        
        results.append({
            'size': size,
            'flops': flops,
            'inf_time': avg_inf_time,
            'inf_fps': inf_fps,
            'e2e_time': avg_e2e_time,
            'e2e_fps': e2e_fps,
            'memory': memory
        })
    
    return results

# 测试不同批量大小下的性能
def test_different_batch_sizes(model, device, batch_sizes=[1, 2, 4, 8, 16], input_shape=(640, 640), num_runs=50):
    results = []
    
    for batch_size in batch_sizes:
        print(f"\nTesting batch size: {batch_size}")
        
        # 测试推理速度
        avg_inf_time, inf_fps, _ = test_inference_speed(model, device, num_runs=num_runs, 
                                                       input_shape=input_shape, batch_size=batch_size)
        
        # 测试端到端速度
        avg_e2e_time, e2e_fps, _ = test_end_to_end_speed(model, device, num_runs=num_runs, 
                                                       input_shape=input_shape, batch_size=batch_size)
        
        results.append({
            'batch_size': batch_size,
            'inf_time': avg_inf_time,
            'inf_fps': inf_fps,
            'e2e_time': avg_e2e_time,
            'e2e_fps': e2e_fps
        })
    
    return results

# 可视化性能结果
def visualize_results(base_name, input_size_results, batch_size_results):
    # 创建输出目录
    output_dir = os.path.join(ROOT, 'performance_results')
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 输入尺寸对性能的影响
    plt.figure(figsize=(12, 10))
    
    # 1.1 FPS vs 输入尺寸
    plt.subplot(2, 2, 1)
    sizes = [r['size'] for r in input_size_results]
    inf_fps = [r['inf_fps'] for r in input_size_results]
    e2e_fps = [r['e2e_fps'] for r in input_size_results]
    plt.plot(sizes, inf_fps, 'o-', label='推理FPS')
    plt.plot(sizes, e2e_fps, 's-', label='端到端FPS')
    plt.xlabel('输入尺寸 (像素)')
    plt.ylabel('FPS')
    plt.title('输入尺寸对FPS的影响')
    plt.legend()
    plt.grid(True)
    
    # 1.2 延迟 vs 输入尺寸
    plt.subplot(2, 2, 2)
    inf_time = [r['inf_time'] * 1000 for r in input_size_results]  # 转换为ms
    e2e_time = [r['e2e_time'] * 1000 for r in input_size_results]  # 转换为ms
    plt.plot(sizes, inf_time, 'o-', label='推理延迟 (ms)')
    plt.plot(sizes, e2e_time, 's-', label='端到端延迟 (ms)')
    plt.xlabel('输入尺寸 (像素)')
    plt.ylabel('延迟 (ms)')
    plt.title('输入尺寸对延迟的影响')
    plt.legend()
    plt.grid(True)
    
    # 1.3 FLOPs vs 输入尺寸
    plt.subplot(2, 2, 3)
    flops = [r['flops'] / 1e9 for r in input_size_results]  # 转换为GFLOPs
    plt.plot(sizes, flops, 'o-', color='orange')
    plt.xlabel('输入尺寸 (像素)')
    plt.ylabel('GFLOPs')
    plt.title('输入尺寸对计算量的影响')
    plt.grid(True)
    
    # 1.4 内存使用 vs 输入尺寸
    plt.subplot(2, 2, 4)
    memory = [r['memory'] for r in input_size_results]
    plt.plot(sizes, memory, 'o-', color='green')
    plt.xlabel('输入尺寸 (像素)')
    plt.ylabel('内存使用 (MB)')
    plt.title('输入尺寸对内存使用的影响')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{base_name}_input_size_performance.png'), dpi=300, bbox_inches='tight')
    
    # 2. 批量大小对性能的影响
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    batch_sizes = [r['batch_size'] for r in batch_size_results]
    inf_fps = [r['inf_fps'] for r in batch_size_results]
    e2e_fps = [r['e2e_fps'] for r in batch_size_results]
    plt.plot(batch_sizes, inf_fps, 'o-', label='推理FPS')
    plt.plot(batch_sizes, e2e_fps, 's-', label='端到端FPS')
    plt.xlabel('批量大小')
    plt.ylabel('FPS')
    plt.title('批量大小对FPS的影响')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    inf_time = [r['inf_time'] * 1000 for r in batch_size_results]  # 转换为ms
    e2e_time = [r['e2e_time'] * 1000 for r in batch_size_results]  # 转换为ms
    plt.plot(batch_sizes, inf_time, 'o-', label='推理延迟 (ms)')
    plt.plot(batch_sizes, e2e_time, 's-', label='端到端延迟 (ms)')
    plt.xlabel('批量大小')
    plt.ylabel('延迟 (ms)')
    plt.title('批量大小对延迟的影响')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{base_name}_batch_size_performance.png'), dpi=300, bbox_inches='tight')
    
    plt.close('all')
    
    print(f"\n可视化结果已保存到: {output_dir}")

# 生成性能报告
def generate_report(base_name, model, device, params, flops, 
                   inf_time, inf_fps, e2e_time, e2e_fps, memory, 
                   input_size_results, batch_size_results):
    # 创建输出目录
    output_dir = os.path.join(ROOT, 'performance_results')
    os.makedirs(output_dir, exist_ok=True)
    
    report_path = os.path.join(output_dir, f'{base_name}_performance_report.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("===== 性能分析报告 =====\n\n")
        f.write(f"模型名称: {base_name}\n")
        f.write(f"设备: {device}\n\n")
        
        f.write("===== 基本性能指标 =====\n")
        f.write(f"参数量: {params:,} ({params / 1e6:.2f} M)\n")
        f.write(f"计算量: {flops:,} ({flops / 1e9:.2f} GFLOPs)\n")
        f.write(f"推理时间: {inf_time * 1000:.2f} ms\n")
        f.write(f"推理FPS: {inf_fps:.2f}\n")
        f.write(f"端到端时间: {e2e_time * 1000:.2f} ms\n")
        f.write(f"端到端FPS: {e2e_fps:.2f}\n")
        f.write(f"内存使用: {memory:.2f} MB\n\n")
        
        f.write("===== 不同输入尺寸性能 =====\n")
        f.write(f"{'尺寸':<10} {'GFLOPs':<10} {'推理FPS':<12} {'推理延迟(ms)':<15} {'端到端FPS':<12} {'端到端延迟(ms)':<15} {'内存(MB)':<10}\n")
        f.write("-" * 80 + "\n")
        for r in input_size_results:
            f.write(f"{r['size']:^10} {r['flops']/1e9:<10.2f} {r['inf_fps']:<12.2f} {r['inf_time']*1000:<15.2f} {r['e2e_fps']:<12.2f} {r['e2e_time']*1000:<15.2f} {r['memory']:<10.2f}\n")
        f.write("\n")
        
        f.write("===== 不同批量大小性能 =====\n")
        f.write(f"{'批量大小':<12} {'推理FPS':<12} {'推理延迟(ms)':<15} {'端到端FPS':<12} {'端到端延迟(ms)':<15}\n")
        f.write("-" * 70 + "\n")
        for r in batch_size_results:
            f.write(f"{r['batch_size']:^12} {r['inf_fps']:<12.2f} {r['inf_time']*1000:<15.2f} {r['e2e_fps']:<12.2f} {r['e2e_time']*1000:<15.2f}\n")
    
    print(f"\n性能报告已保存到: {report_path}")

# 主函数
def main():
    parser = argparse.ArgumentParser(description='YOLOv5 性能分析脚本')
    parser.add_argument('--weights', type=str, default=os.path.join(ROOT, 'yolov5s.pt'),
                      help='模型权重文件路径')
    parser.add_argument('--num_runs', type=int, default=100,
                      help='测试运行次数')
    parser.add_argument('--input_shape', type=int, default=640,
                      help='默认输入尺寸')
    parser.add_argument('--visualize', action='store_true', default=True,
                      help='生成可视化结果')
    parser.add_argument('--report', action='store_true', default=True,
                      help='生成性能报告')
    
    args = parser.parse_args()
    
    print("===== 性能分析 ====\n")
    
    # 1. 加载模型
    print("1. 加载模型...")
    model, device = load_model(args.weights)
    print(f"   设备: {device}")
    
    # 2. 计算参数量
    print("2. 计算参数量...")
    total_params = calculate_parameters(model)
    print(f"   总参数量: {total_params:,} (约 {total_params / 1e6:.2f} M)")
    
    # 3. 计算计算量
    print("3. 计算计算量 (FLOPs)...")
    total_flops = calculate_flops(model, input_shape=(3, args.input_shape, args.input_shape))
    print(f"   总计算量: {total_flops:,} (约 {total_flops / 1e9:.2f} GFLOPs)")
    
    # 4. 测试内存使用
    print("4. 测试内存使用...")
    memory = test_memory_usage(model, device, input_shape=(3, args.input_shape, args.input_shape))
    print(f"   内存使用: {memory:.2f} MB")
    
    # 5. 测试推理速度
    print("5. 测试推理速度...")
    avg_inference_time, inference_fps, _ = test_inference_speed(model, device, 
                                                              num_runs=args.num_runs, 
                                                              input_shape=(args.input_shape, args.input_shape))
    print(f"   平均推理时间: {avg_inference_time * 1000:.2f} ms")
    print(f"   推理FPS: {inference_fps:.2f}")
    
    # 6. 测试端到端速度
    print("6. 测试端到端速度 (包含预处理和后处理)...")
    avg_e2e_time, e2e_fps, _ = test_end_to_end_speed(model, device, 
                                                    num_runs=args.num_runs, 
                                                    input_shape=(args.input_shape, args.input_shape))
    print(f"   平均端到端时间: {avg_e2e_time * 1000:.2f} ms")
    print(f"   端到端FPS: {e2e_fps:.2f}")
    
    # 7. 测试不同输入尺寸下的性能
    print("\n7. 测试不同输入尺寸下的性能...")
    input_size_results = test_different_input_sizes(model, device, num_runs=args.num_runs // 2)
    
    # 8. 测试不同批量大小下的性能
    print("\n8. 测试不同批量大小下的性能...")
    batch_size_results = test_different_batch_sizes(model, device, num_runs=args.num_runs // 2)
    
    # 9. 生成可视化结果
    if args.visualize:
        print("\n9. 生成可视化结果...")
        base_name = os.path.splitext(os.path.basename(args.weights))[0]
        visualize_results(base_name, input_size_results, batch_size_results)
    
    # 10. 生成性能报告
    if args.report:
        print("\n10. 生成性能报告...")
        base_name = os.path.splitext(os.path.basename(args.weights))[0]
        generate_report(base_name, model, device, total_params, total_flops, 
                       avg_inference_time, inference_fps, avg_e2e_time, e2e_fps, memory, 
                       input_size_results, batch_size_results)
    
    # 11. 模型信息
    print("\n===== 模型信息 ====")
    print(f"   模型名称: {os.path.splitext(os.path.basename(args.weights))[0]}")
    print(f"   输入尺寸: {args.input_shape}x{args.input_shape}")
    print(f"   设备: {device}")
    print(f"   参数量: {total_params / 1e6:.2f} M")
    print(f"   计算量: {total_flops / 1e9:.2f} GFLOPs")
    print(f"   推理速度: {inference_fps:.2f} FPS")
    print(f"   端到端速度: {e2e_fps:.2f} FPS")
    print(f"   内存使用: {memory:.2f} MB")
    
    print("\n===== 分析完成 ====")

if __name__ == "__main__":
    main()
