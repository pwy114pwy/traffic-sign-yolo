# web/app.py
import os
import cv2
import torch
import time
import logging
from flask import Flask, request, render_template, jsonify
from pathlib import Path
import sys
import dotenv
from typing import Tuple, List, Dict, Any

# 加载环境变量
dotenv.load_dotenv()

# ======================
# 日志配置
# ======================
log_level = os.getenv("LOG_LEVEL", "INFO")
log_file = os.getenv("LOG_FILE", "logs/app.log")

# 确保日志目录存在
os.makedirs(os.path.dirname(log_file) if os.path.dirname(log_file) else "logs", exist_ok=True)

# 配置日志
logging.basicConfig(
    level=getattr(logging, log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

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
# 配置区
# ======================
# 从环境变量获取配置，或使用默认值
PROJECT_ROOT = os.getenv("PROJECT_ROOT", str(ROOT))
WEIGHTS_PATH = os.getenv("WEIGHTS_PATH", 
                       os.path.join(PROJECT_ROOT, "yolov5", "runs", "train", "exp3", "weights", "best.pt"))
UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", 
                         os.path.join(PROJECT_ROOT, "web", "static", "uploads"))
RESULT_FOLDER = os.getenv("RESULT_FOLDER", 
                         os.path.join(PROJECT_ROOT, "web", "static", "results"))

# 创建文件夹
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# 打印配置信息
logger.info(f"📁 PROJECT_ROOT: {PROJECT_ROOT}")
logger.info(f"📦 WEIGHTS_PATH: {WEIGHTS_PATH}")
logger.info(f"📤 UPLOAD_FOLDER: {UPLOAD_FOLDER}")
logger.info(f"📥 RESULT_FOLDER: {RESULT_FOLDER}")

# ======================
# 模型加载
# ======================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"🖥️  使用设备: {DEVICE}")

# 检查模型文件是否存在
if not os.path.exists(WEIGHTS_PATH):
    error_msg = f"""
    ❌ 模型文件未找到: {WEIGHTS_PATH}
    
    请选择以下解决方案之一：
    
    1. 如果您已有训练好的模型：
       - 确保模型文件路径正确
       - 或在 .env 文件中设置正确的 WEIGHTS_PATH
    
    2. 如果还未训练模型：
       方案A：训练新模型（推荐）
       python yolov5/train.py --data data/gtsrb.yaml --weights yolov5s.pt --epochs 50
       
       方案B：使用预训练权重（临时方案）
       在 .env 文件中设置: WEIGHTS_PATH=yolov5s.pt
    """
    logger.error(error_msg)
    print(error_msg)
    sys.exit(1)

try:
    logger.info(f"⏳ 正在加载模型: {WEIGHTS_PATH}")
    model = attempt_load(WEIGHTS_PATH, device=DEVICE)
    stride = int(model.stride.max())  # 获取模型步长
    names = model.module.names if hasattr(model, 'module') else model.names
    logger.info(f"✅ 模型加载成功！类别数: {len(names)}")
    logger.info(f"📋 类别列表: {names}")
except Exception as e:
    logger.error(f"❌ 模型加载失败: {str(e)}", exc_info=True)
    print(f"\n❌ 模型加载失败: {str(e)}")
    print("请检查模型文件是否完整或尝试重新下载/训练模型")
    sys.exit(1)

# 从环境变量获取检测参数
ANOMALY_CONF_THRES = float(os.getenv("ANOMALY_CONF_THRES", "0.5"))
SCALE_THRESHOLD = int(os.getenv("SCALE_THRESHOLD", "200"))

app = Flask(__name__)
app.logger.setLevel(getattr(logging, log_level))

# ======================
# 辅助函数
# ======================

def preprocess_image(img: Any, img_size: int = 640) -> torch.Tensor:
    """预处理图像用于模型推理
    
    Args:
        img: 输入图像（numpy array）
        img_size: 目标图像尺寸
        
    Returns:
        预处理后的张量
    """
    img = letterbox(img, img_size, stride=stride)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
    img = torch.from_numpy(img.copy()).to(DEVICE).float() / 255.0
    return img.unsqueeze(0)


def detect_objects(img: torch.Tensor, conf_thres: float = 0.25, iou_thres: float = 0.45) -> List:
    """执行目标检测
    
    Args:
        img: 预处理后的图像张量
        conf_thres: 置信度阈值
        iou_thres: IoU阈值
        
    Returns:
        检测结果列表
    """
    with torch.no_grad():
        pred = model(img, augment=True)[0]
        pred = non_max_suppression(pred, conf_thres=conf_thres, iou_thres=iou_thres)
    return pred


def annotate_detection(annotator: Annotator, xyxy: List, conf: float, cls: int, 
                       anomaly_thres: float = ANOMALY_CONF_THRES) -> str:
    """标注单个检测结果
    
    Args:
        annotator: 标注器对象
        xyxy: 边界框坐标
        conf: 置信度
        cls: 类别ID
        anomaly_thres: 异常检测阈值
        
    Returns:
        类别名称
    """
    if conf < anomaly_thres:
        class_name = 'Unknown Sign'
        label = f'Unknown Sign {conf:.2f}'
        annotator.box_label(xyxy, label, color=(255, 0, 0))  # 红色框
    else:
        c = int(cls)
        class_name = names[c]
        label = f'{class_name} {conf:.2f}'
        annotator.box_label(xyxy, label, color=colors(c, True))
    return class_name


def calculate_line_width(img_shape: Tuple[int, int]) -> int:
    """根据图像尺寸计算合适的线条宽度
    
    Args:
        img_shape: 图像形状 (height, width)
        
    Returns:
        线条宽度
    """
    min_dim = min(img_shape[0], img_shape[1])
    line_width = max(1, int(min_dim / 300))  # 每300像素对应1个像素宽度
    return min(line_width, 4)  # 最大宽度为4


# ======================
# 路由处理
# ======================

@app.route('/')
def index():
    """主页"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """图片检测接口"""
    try:
        # 验证文件上传
        if 'image' not in request.files:
            logger.warning("未上传文件")
            return jsonify({'error': '未上传文件'}), 400
        
        file = request.files['image']
        if not file or not file.filename:
            logger.warning("文件为空")
            return jsonify({'error': '文件为空'}), 400
        
        # 验证文件类型
        allowed_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
        if not file.filename.lower().endswith(allowed_extensions):
            logger.warning(f"不支持的文件格式: {file.filename}")
            ext_list = ', '.join(allowed_extensions)
            return jsonify({'error': f'不支持的文件格式，请上传 {ext_list} 图片'}), 400

        # 获取检测参数
        conf_thres = float(request.form.get('conf_thres', 0.25))
        iou_thres = float(request.form.get('iou_thres', 0.45))

        # 保存上传文件
        input_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(input_path)
        logger.info(f"文件已保存: {input_path}")

        # 读取原图
        img0 = cv2.imread(input_path)
        if img0 is None:
            logger.error(f"无法读取图片: {input_path}")
            # 删除无效文件
            if os.path.exists(input_path):
                os.remove(input_path)
            return jsonify({'error': '无法读取图片，文件可能已损坏'}), 400

        # 开始计时
        start_time = time.time()

        # 预处理图像
        img = preprocess_image(img0, 640)

        # 推理
        pred = detect_objects(img, conf_thres=conf_thres, iou_thres=iou_thres)

        # 计算推理时间
        inference_time = time.time() - start_time
        logger.info(f"推理耗时: {inference_time:.3f}秒")

        # 检查图片是否过小，如果是则放大
        original_shape = img0.shape
        min_dim = min(original_shape[0], original_shape[1])
        scale_factor = 1.0
        
        if min_dim < SCALE_THRESHOLD:
            # 计算放大比例
            scale_factor = SCALE_THRESHOLD / min_dim
            new_width = int(original_shape[1] * scale_factor)
            new_height = int(original_shape[0] * scale_factor)
            img0 = cv2.resize(img0, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            logger.info(f"图片已放大: {original_shape[1]}x{original_shape[0]} -> {new_width}x{new_height}")
        
        # 动态计算线条宽度
        line_width = calculate_line_width(img0.shape)
        
        # 画检测框
        annotator = Annotator(img0, line_width=line_width, example=str(names))
        
        # 检测统计信息
        detection_count = 0
        unique_classes = set()
        total_confidence = 0.0
        class_counts = {}
        anomaly_count = 0  # 异常检测计数
        
        for det in pred:
            if len(det):
                # 使用正确的缩放函数
                det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], original_shape).round()
                
                # 如果图片被放大，检测框坐标也需要相应放大
                if scale_factor != 1.0:
                    det[:, :4] *= scale_factor
                    det[:, :4] = det[:, :4].round()
                    
                detection_count += len(det)
                for *xyxy, conf, cls in reversed(det):
                    class_name = annotate_detection(annotator, xyxy, conf.item(), cls.item())
                    unique_classes.add(class_name)
                    total_confidence += conf.item()
                    
                    # 更新类别计数
                    if class_name not in class_counts:
                        class_counts[class_name] = 0
                    class_counts[class_name] += 1
                    
                    # 统计异常检测
                    if conf < ANOMALY_CONF_THRES:
                        anomaly_count += 1

        output_path = os.path.join(RESULT_FOLDER, file.filename)
        cv2.imwrite(output_path, annotator.result())
        logger.info(f"检测完成: {detection_count} 个目标, {len(unique_classes)} 个类别")

        # 准备检测详情列表
        detections = []
        for class_name, count in class_counts.items():
            detections.append({
                'name': class_name,
                'count': count,
                'percentage': (count / detection_count * 100) if detection_count > 0 else 0.0
            })
        
        # 按检测数量排序
        detections.sort(key=lambda x: x['count'], reverse=True)
        
        unique_classes_count = len(unique_classes)
        avg_confidence = (total_confidence / detection_count * 100) if detection_count > 0 else 0.0
        
        logger.info(f"平均置信度: {avg_confidence:.2f}%, 异常检测: {anomaly_count}")
        
        # 返回结果页面，显示处理后的图片和统计信息
        return render_template('result.html', 
                             img_path=file.filename,
                             detection_count=detection_count,
                             unique_classes_count=unique_classes_count,
                             avg_confidence=avg_confidence,
                             detections=detections,
                             anomaly_count=anomaly_count,
                             inference_time=inference_time,
                             class_counts=class_counts,
                             conf_thres=conf_thres,
                             iou_thres=iou_thres)
    
    except Exception as e:
        logger.error(f"图片检测失败: {str(e)}", exc_info=True)
        return jsonify({'error': f'检测失败: {str(e)}'}), 500

@app.route('/predict_video', methods=['POST'])
def predict_video():
    """视频检测接口"""
    cap = None
    out = None
    
    try:
        # 验证文件上传
        if 'video' not in request.files:
            logger.warning("未上传视频文件")
            return jsonify({'error': '未上传视频文件'}), 400
        
        file = request.files['video']
        if not file or not file.filename:
            logger.warning("视频文件为空")
            return jsonify({'error': '视频文件为空'}), 400
        
        # 验证文件类型
        allowed_extensions = ('.mp4', '.avi', '.mov', '.wmv', '.mkv')
        if not file.filename.lower().endswith(allowed_extensions):
            logger.warning(f"不支持的视频格式: {file.filename}")
            ext_list = ', '.join(allowed_extensions)
            return jsonify({'error': f'不支持的视频格式，请上传 {ext_list} 视频'}), 400
    
        # 获取检测参数
        conf_thres = float(request.form.get('conf_thres', 0.25))
        iou_thres = float(request.form.get('iou_thres', 0.45))
        
        # 保存上传文件
        input_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(input_path)
        logger.info(f"视频文件已保存: {input_path}")
        
        # 打开视频
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            logger.error(f"无法打开视频: {input_path}")
            if os.path.exists(input_path):
                os.remove(input_path)
            return jsonify({'error': '无法打开视频文件，文件可能已损坏'}), 400
        
        # 获取视频信息
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info(f"视频信息: {width}x{height}, {fps}fps, {frame_count}帧")
        
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
        class_counts = {}
        
        # 开始计时
        start_time = time.time()
        processed_frames = 0
        
        # 逐帧处理
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                processed_frames += 1
                
                # 预处理
                img = preprocess_image(frame, 640)
                
                # 推理
                pred = detect_objects(img, conf_thres=conf_thres, iou_thres=iou_thres)
                
                # 动态计算线条宽度
                line_width = calculate_line_width(frame.shape)
                
                # 画检测框
                annotator = Annotator(frame, line_width=line_width, example=str(names))
                
                frame_det_count = 0
                for det in pred:
                    if len(det):
                        # 使用正确的缩放函数
                        det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], frame.shape).round()
                        frame_det_count += len(det)
                        detection_count += len(det)
                        for *xyxy, conf, cls in reversed(det):
                            class_name = annotate_detection(annotator, xyxy, conf.item(), cls.item())
                            unique_classes.add(class_name)
                            total_confidence += conf.item()
                            
                            # 更新类别计数
                            if class_name not in class_counts:
                                class_counts[class_name] = 0
                            class_counts[class_name] += 1
                            
                            # 统计异常检测
                            if conf < ANOMALY_CONF_THRES:
                                anomaly_count += 1
                
                # 计算实时FPS
                current_time = time.time()
                elapsed_time = current_time - start_time
                realtime_fps = processed_frames / elapsed_time if elapsed_time > 0 else 0
                
                # 在视频上显示FPS和检测数量
                cv2.putText(frame, f'FPS: {realtime_fps:.1f}', (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f'Detections: {frame_det_count}', (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # 写入处理后的帧
                out.write(annotator.result())
                
                # 记录每帧检测数量
                frame_detections.append(frame_det_count)
                
                # 每处理100帧记录一次进度
                if processed_frames % 100 == 0:
                    logger.info(f"已处理 {processed_frames}/{frame_count} 帧")
        
        finally:
            # 确保资源释放
            if cap is not None:
                cap.release()
            if out is not None:
                out.release()
            logger.info("视频资源已释放")
        
        # 计算总处理时间
        total_time = time.time() - start_time
        avg_fps = processed_frames / total_time if total_time > 0 else 0
        logger.info(f"视频处理完成: {processed_frames}帧, 耗时{total_time:.2f}秒, 平均FPS: {avg_fps:.2f}")
        
        unique_classes_count = len(unique_classes)
        avg_confidence = (total_confidence / detection_count * 100) if detection_count > 0 else 0.0
        
        # 准备检测详情列表
        detections = []
        for class_name, count in class_counts.items():
            detections.append({
                'name': class_name,
                'count': count,
                'percentage': (count / detection_count * 100) if detection_count > 0 else 0.0
            })
        
        # 按检测数量排序
        detections.sort(key=lambda x: x['count'], reverse=True)
        
        logger.info(f"总检测: {detection_count}, 平均置信度: {avg_confidence:.2f}%, 异常: {anomaly_count}")
        
        # 返回结果页面，显示处理后的视频和统计信息
        return render_template('result.html', 
                             video_path=file.filename,
                             detection_count=detection_count,
                             unique_classes_count=unique_classes_count,
                             avg_confidence=avg_confidence,
                             anomaly_count=anomaly_count,
                             avg_fps=avg_fps,
                             total_time=total_time,
                             detections=detections,
                             class_counts=class_counts)
    
    except Exception as e:
        logger.error(f"视频检测失败: {str(e)}", exc_info=True)
        # 确保资源释放
        if cap is not None:
            cap.release()
        if out is not None:
            out.release()
        return jsonify({'error': f'视频检测失败: {str(e)}'}), 500


if __name__ == '__main__':
    # 从环境变量获取Flask配置
    flask_host = os.getenv("FLASK_HOST", "0.0.0.0")
    flask_port = int(os.getenv("FLASK_PORT", "5000"))
    flask_debug = os.getenv("FLASK_DEBUG", "False").lower() == "true"
    
    logger.info(f"启动Flask服务器: {flask_host}:{flask_port}, debug={flask_debug}")
    app.run(host=flask_host, port=flask_port, debug=flask_debug)