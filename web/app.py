# web/app.py
import os
import cv2
import time
import logging
import requests
from io import BytesIO
import numpy as np
from flask import Flask, request, render_template, jsonify
from pathlib import Path
import sys
import dotenv
from typing import Tuple, List, Dict, Any
import platform
import pathlib

plt = platform.system()
if plt == 'Windows':
    pathlib.PosixPath = pathlib.WindowsPath

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

# ======================
# 引入 YOLOv8 (ultralytics)
# ======================
from ultralytics import YOLO

# ======================
# 配置区
# ======================
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # 项目根目录

PROJECT_ROOT  = os.getenv("PROJECT_ROOT", str(ROOT))
WEIGHTS_PATH  = os.getenv("WEIGHTS_PATH",
                           os.path.join(PROJECT_ROOT, "runs", "train", "yolov8_gtsrb", "weights", "best.pt"))
UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER",
                           os.path.join(PROJECT_ROOT, "web", "static", "uploads"))
RESULT_FOLDER = os.getenv("RESULT_FOLDER",
                           os.path.join(PROJECT_ROOT, "web", "static", "results"))

# 创建文件夹
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

logger.info(f"📁 PROJECT_ROOT: {PROJECT_ROOT}")
logger.info(f"📦 WEIGHTS_PATH: {WEIGHTS_PATH}")
logger.info(f"📤 UPLOAD_FOLDER: {UPLOAD_FOLDER}")
logger.info(f"📥 RESULT_FOLDER: {RESULT_FOLDER}")

# ======================
# 模型加载
# ======================
# 判断是否为 ultralytics 内置模型名（如 yolov8s.pt）
# 内置模型名不含路径分隔符，ultralytics 会在加载时自动下载
_is_builtin_model = os.sep not in WEIGHTS_PATH and '/' not in WEIGHTS_PATH

# 只有用户指定了完整/相对路径时才检查文件是否存在
if not _is_builtin_model and not os.path.exists(WEIGHTS_PATH):
    error_msg = f"""
    ❌ 模型文件未找到: {WEIGHTS_PATH}
    
    请选择以下解决方案之一：
    
    1. 使用预训练 YOLOv8 权重（快速测试，无需训练）：
       在 .env 中设置: WEIGHTS_PATH=yolov8s.pt
       首次运行时会自动下载（约 22 MB）。
    
    2. 训练专属交通标志模型（推荐）：
       python scripts/train_yolov8.py
       训练完成后在 .env 中设置:
       WEIGHTS_PATH=runs/train/yolov8_gtsrb/weights/best.pt
    """
    logger.error(error_msg)
    print(error_msg)
    sys.exit(1)

if _is_builtin_model:
    logger.info(f"📡 使用 ultralytics 内置模型: {WEIGHTS_PATH}（首次运行将自动下载）")

try:
    logger.info(f"⏳ 正在加载 YOLOv8 模型: {WEIGHTS_PATH}")
    model = YOLO(WEIGHTS_PATH)
    names = model.names  # dict: {0: 'class_name', ...}
    logger.info(f"✅ 模型加载成功！类别数: {len(names)}")
    logger.info(f"📋 类别列表: {names}")
except Exception as e:
    logger.error(f"❌ 模型加载失败: {str(e)}", exc_info=True)
    print(f"\n❌ 模型加载失败: {str(e)}")
    print("请检查模型文件是否完整或尝试重新下载/训练模型")
    sys.exit(1)

# 从环境变量获取检测参数
ANOMALY_CONF_THRES = float(os.getenv("ANOMALY_CONF_THRES", "0.5"))
SCALE_THRESHOLD    = int(os.getenv("SCALE_THRESHOLD", "200"))

app = Flask(__name__)
app.logger.setLevel(getattr(logging, log_level))

# ======================
# 辅助函数
# ======================

def detect_objects(img: np.ndarray, conf_thres: float = 0.25, iou_thres: float = 0.45):
    """使用 YOLOv8 对单张图像执行目标检测。

    Args:
        img: BGR 格式的 numpy 图像（原始分辨率）
        conf_thres: 置信度阈值
        iou_thres: IoU（NMS）阈值

    Returns:
        ultralytics Results 对象（单张图片）
    """
    results = model.predict(
        source=img,
        conf=conf_thres,
        iou=iou_thres,
        verbose=False,
        device=None,  # 自动选择设备
    )
    return results[0]  # 单张图片只有一个结果


def upscale_if_small(img: np.ndarray) -> Tuple[np.ndarray, float]:
    """若图片过小则放大，返回（放大后图片, 缩放比例）。"""
    min_dim = min(img.shape[0], img.shape[1])
    if min_dim < SCALE_THRESHOLD:
        scale = SCALE_THRESHOLD / min_dim
        new_w = int(img.shape[1] * scale)
        new_h = int(img.shape[0] * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        return img, scale
    return img, 1.0


def draw_detections(img: np.ndarray, result, scale_factor: float = 1.0) -> Tuple[np.ndarray, int, set, float, Dict, int]:
    """在图像上绘制检测结果并收集统计信息。

    Args:
        img: 目标图像（用于绘制）
        result: YOLOv8 Results 对象
        scale_factor: 图像放大比例（用于坐标变换）

    Returns:
        (annotated_img, detection_count, unique_classes, total_confidence, class_counts, anomaly_count)
    """
    detection_count = 0
    unique_classes  = set()
    total_confidence = 0.0
    class_counts    = {}
    anomaly_count   = 0

    boxes = result.boxes
    if boxes is not None and len(boxes):
        for box in boxes:
            conf = float(box.conf[0])
            cls  = int(box.cls[0])
            # 原始坐标（相对于传入 predict 的原图）；若放大则需乘以 scale_factor
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1 = int(x1 * scale_factor)
            y1 = int(y1 * scale_factor)
            x2 = int(x2 * scale_factor)
            y2 = int(y2 * scale_factor)

            if conf < ANOMALY_CONF_THRES:
                class_name = 'Unknown Sign'
                color = (0, 0, 255)  # 红色
            else:
                class_name = names[cls]
                # 为不同类别生成不同颜色
                hue = int(cls * 180 / max(len(names), 1)) % 180
                bgr = cv2.cvtColor(np.uint8([[[hue, 200, 200]]]), cv2.COLOR_HSV2BGR)[0][0]
                color = (int(bgr[0]), int(bgr[1]), int(bgr[2]))
                anomaly_count_delta = 0

            label = f'{class_name} {conf:.2f}'

            # 绘制边框
            lw = max(1, int(min(img.shape[:2]) / 300))
            lw = min(lw, 4)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, lw)

            # 绘制标签背景和文字
            font_scale = max(0.4, lw * 0.4)
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, lw)
            cv2.rectangle(img, (x1, y1 - th - baseline - 4), (x1 + tw, y1), color, -1)
            cv2.putText(img, label, (x1, y1 - baseline - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), max(1, lw - 1))

            # 统计
            detection_count += 1
            unique_classes.add(class_name)
            total_confidence += conf
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            if conf < ANOMALY_CONF_THRES:
                anomaly_count += 1

    return img, detection_count, unique_classes, total_confidence, class_counts, anomaly_count


def build_detections_list(class_counts: Dict, detection_count: int) -> List[Dict]:
    """将类别计数转换为排序后的检测详情列表。"""
    detections = [
        {
            'name': name,
            'count': count,
            'percentage': (count / detection_count * 100) if detection_count > 0 else 0.0
        }
        for name, count in class_counts.items()
    ]
    detections.sort(key=lambda x: x['count'], reverse=True)
    return detections


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
        iou_thres  = float(request.form.get('iou_thres', 0.45))

        # 保存上传文件
        input_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(input_path)
        logger.info(f"文件已保存: {input_path}")

        # 读取原图
        img0 = cv2.imread(input_path)
        if img0 is None:
            logger.error(f"无法读取图片: {input_path}")
            if os.path.exists(input_path):
                os.remove(input_path)
            return jsonify({'error': '无法读取图片，文件可能已损坏'}), 400

        # 开始计时
        start_time = time.time()

        # 推理（传入原始图像，YOLOv8 内部自动处理预处理）
        result = detect_objects(img0, conf_thres=conf_thres, iou_thres=iou_thres)

        inference_time = time.time() - start_time
        logger.info(f"推理耗时: {inference_time:.3f}秒")

        # 若图片过小则放大
        img_draw, scale_factor = upscale_if_small(img0.copy())

        # 绘制检测结果
        img_draw, detection_count, unique_classes, total_confidence, class_counts, anomaly_count = \
            draw_detections(img_draw, result, scale_factor)

        # 保存结果图
        output_path = os.path.join(RESULT_FOLDER, file.filename)
        cv2.imwrite(output_path, img_draw)
        logger.info(f"检测完成: {detection_count} 个目标, {len(unique_classes)} 个类别")

        detections       = build_detections_list(class_counts, detection_count)
        avg_confidence   = (total_confidence / detection_count * 100) if detection_count > 0 else 0.0
        logger.info(f"平均置信度: {avg_confidence:.2f}%, 异常检测: {anomaly_count}")

        return render_template('result.html',
                               img_path=file.filename,
                               detection_count=detection_count,
                               unique_classes_count=len(unique_classes),
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
        iou_thres  = float(request.form.get('iou_thres', 0.45))

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
        fps         = int(cap.get(cv2.CAP_PROP_FPS))
        width       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info(f"视频信息: {width}x{height}, {fps}fps, {frame_count}帧")

        # 创建视频编写器
        output_path = os.path.join(RESULT_FOLDER, file.filename)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out    = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # 检测统计信息
        detection_count  = 0
        unique_classes   = set()
        total_confidence = 0.0
        frame_detections = []
        anomaly_count    = 0
        class_counts     = {}

        # 开始计时
        start_time      = time.time()
        processed_frames = 0

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                processed_frames += 1

                # YOLOv8 推理
                result = detect_objects(frame, conf_thres=conf_thres, iou_thres=iou_thres)

                # 绘制检测结果（视频帧不做 upscale）
                annotated_frame, fd_count, fc_classes, fc_conf, fc_class_counts, fc_anomaly = \
                    draw_detections(frame.copy(), result, scale_factor=1.0)

                detection_count  += fd_count
                unique_classes   |= fc_classes
                total_confidence += fc_conf
                anomaly_count    += fc_anomaly
                for name, cnt in fc_class_counts.items():
                    class_counts[name] = class_counts.get(name, 0) + cnt

                # 计算实时 FPS
                elapsed      = time.time() - start_time
                realtime_fps = processed_frames / elapsed if elapsed > 0 else 0

                # 在视频帧上叠加 FPS 和检测数量
                cv2.putText(annotated_frame, f'FPS: {realtime_fps:.1f}', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(annotated_frame, f'Detections: {fd_count}', (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                out.write(annotated_frame)
                frame_detections.append(fd_count)

                if processed_frames % 100 == 0:
                    logger.info(f"已处理 {processed_frames}/{frame_count} 帧")

        finally:
            if cap is not None:
                cap.release()
            if out is not None:
                out.release()
            logger.info("视频资源已释放")

        total_time   = time.time() - start_time
        avg_fps      = processed_frames / total_time if total_time > 0 else 0
        avg_confidence = (total_confidence / detection_count * 100) if detection_count > 0 else 0.0
        logger.info(f"视频处理完成: {processed_frames}帧, 耗时{total_time:.2f}秒, 平均FPS: {avg_fps:.2f}")

        detections = build_detections_list(class_counts, detection_count)

        return render_template('result.html',
                               video_path=file.filename,
                               detection_count=detection_count,
                               unique_classes_count=len(unique_classes),
                               avg_confidence=avg_confidence,
                               anomaly_count=anomaly_count,
                               avg_fps=avg_fps,
                               total_time=total_time,
                               detections=detections,
                               class_counts=class_counts)

    except Exception as e:
        logger.error(f"视频检测失败: {str(e)}", exc_info=True)
        if cap is not None:
            cap.release()
        if out is not None:
            out.release()
        return jsonify({'error': f'视频检测失败: {str(e)}'}), 500


@app.route('/predict_url', methods=['POST'])
def predict_url():
    """通过图片 URL 进行检测"""
    try:
        image_url = request.form.get('image_url', '').strip()
        if not image_url:
            return jsonify({'error': '未提供图片网址'}), 400

        if not image_url.startswith(('http://', 'https://')):
            return jsonify({'error': '请输入有效的 http/https 图片网址'}), 400

        # 获取检测参数
        conf_thres = float(request.form.get('conf_thres', 0.25))
        iou_thres  = float(request.form.get('iou_thres', 0.45))

        # 下载图片
        logger.info(f"正在下载图片: {image_url}")
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        try:
            resp = requests.get(image_url, timeout=15, headers=headers)
            resp.raise_for_status()
        except requests.exceptions.SSLError:
            logger.warning("SSL 验证失败，尝试关闭 SSL 验证重新下载")
            try:
                import urllib3
                urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
                resp = requests.get(image_url, timeout=15, headers=headers, verify=False)
                resp.raise_for_status()
            except requests.exceptions.RequestException as e:
                return jsonify({'error': f'下载图片失败（SSL）: {str(e)}'}), 400
        except requests.exceptions.ProxyError as e:
            logger.warning(f"代理连接失败，尝试绕过代理直连: {e}")
            try:
                resp = requests.get(image_url, timeout=15, headers=headers,
                                    verify=False, proxies={'http': None, 'https': None})
                resp.raise_for_status()
            except requests.exceptions.RequestException as e2:
                return jsonify({'error': f'下载图片失败（代理）: {str(e2)}'}), 400
        except requests.exceptions.Timeout:
            return jsonify({'error': '下载图片超时，请检查网址或网络'}), 400
        except requests.exceptions.RequestException as e:
            return jsonify({'error': f'下载图片失败: {str(e)}'}), 400

        # 解码图片
        img_array = np.frombuffer(resp.content, dtype=np.uint8)
        img0 = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img0 is None:
            return jsonify({'error': '无法解码图片，请确认链接指向有效的图片文件'}), 400

        # 从 URL 提取文件名
        from urllib.parse import urlparse
        url_path = urlparse(image_url).path
        filename = os.path.basename(url_path) or 'url_image.jpg'
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
            filename += '.jpg'

        # 保存原图
        input_path = os.path.join(UPLOAD_FOLDER, filename)
        cv2.imwrite(input_path, img0)
        logger.info(f"URL 图片已保存: {input_path}")

        # 开始计时
        start_time = time.time()

        # 推理
        result = detect_objects(img0, conf_thres=conf_thres, iou_thres=iou_thres)

        inference_time = time.time() - start_time
        logger.info(f"推理耗时: {inference_time:.3f}秒")

        # 若图片过小则放大
        img_draw, scale_factor = upscale_if_small(img0.copy())

        # 绘制检测结果
        img_draw, detection_count, unique_classes, total_confidence, class_counts, anomaly_count = \
            draw_detections(img_draw, result, scale_factor)

        # 保存结果图
        output_path = os.path.join(RESULT_FOLDER, filename)
        cv2.imwrite(output_path, img_draw)
        logger.info(f"URL 图片检测完成: {detection_count} 个目标")

        detections     = build_detections_list(class_counts, detection_count)
        avg_confidence = (total_confidence / detection_count * 100) if detection_count > 0 else 0.0

        return render_template('result.html',
                               img_path=filename,
                               detection_count=detection_count,
                               unique_classes_count=len(unique_classes),
                               avg_confidence=avg_confidence,
                               detections=detections,
                               anomaly_count=anomaly_count,
                               inference_time=inference_time,
                               class_counts=class_counts,
                               conf_thres=conf_thres,
                               iou_thres=iou_thres)

    except Exception as e:
        logger.error(f"URL 图片检测失败: {str(e)}", exc_info=True)
        return jsonify({'error': f'检测失败: {str(e)}'}), 500


if __name__ == '__main__':
    flask_host  = os.getenv("FLASK_HOST", "0.0.0.0")
    flask_port  = int(os.getenv("FLASK_PORT", "5000"))
    flask_debug = os.getenv("FLASK_DEBUG", "False").lower() == "true"

    logger.info(f"启动Flask服务器: {flask_host}:{flask_port}, debug={flask_debug}")
    app.run(host=flask_host, port=flask_port, debug=flask_debug)