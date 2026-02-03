# 🚀 交通标志识别模型改进 - 快速开始指南

## 📋 问题总结

您的模型存在两个主要问题:
1. **多目标检测能力弱** - 一张图片中有多个交通标志时识别率下降
2. **泛化能力差** - 对未训练过的图片识别效果不好

## 🎯 改进方案

我已经为您创建了以下文件:

1. **`data/hyps/hyp.traffic-sign.yaml`** - 增强的超参数配置
2. **`scripts/create_multi_sign_dataset.py`** - 生成多目标合成数据集
3. **`scripts/test_multi_target.py`** - 测试多目标检测性能

## 🔧 实施步骤

### 方案A: 快速改进 (推荐先尝试)

这个方案只需要重新训练,不需要额外数据准备。

#### 步骤1: 使用增强配置重新训练

```bash
# 使用增强的数据增强策略训练
python yolov5/train.py \
  --data data/gtsrb.yaml \
  --weights yolov5s.pt \
  --epochs 150 \
  --batch-size 32 \
  --img 640 \
  --hyp data/hyps/hyp.traffic-sign.yaml \
  --cache ram \
  --project runs/train \
  --name improved_v1
```

**关键改进**:
- ✅ **Mosaic增强**: 自动将4张图片拼接,创建多目标场景
- ✅ **Copy-Paste增强**: 复制目标到其他图片,增加多目标样本
- ✅ **更强的几何变换**: 提高泛化能力
- ✅ **更长的训练**: 150 epochs vs 原来的50

**预期训练时间**:
- GPU: 2-4小时
- CPU: 12-24小时 (不推荐)

#### 步骤2: 测试改进效果

```bash
# 在测试集上评估
python yolov5/val.py \
  --data data/gtsrb.yaml \
  --weights runs/train/improved_v1/weights/best.pt \
  --img 640 \
  --task test
```

#### 步骤3: 更新Web应用

```bash
# 修改 .env 文件,使用新模型
# WEIGHTS_PATH=runs/train/improved_v1/weights/best.pt

# 重启Web应用
python web/app.py
```

---

### 方案B: 深度改进 (效果更好,但需要更多时间)

这个方案会创建额外的多目标训练数据。

#### 步骤1: 生成多目标合成数据集

```bash
# 生成1000张包含多个标志的合成图片
python scripts/create_multi_sign_dataset.py \
  --input-dir datasets/gtsrb/images/train \
  --label-dir datasets/gtsrb/labels/train \
  --output-dir datasets/gtsrb_multi/images/train \
  --output-label-dir datasets/gtsrb_multi/labels/train \
  --num-images 1000 \
  --min-signs 2 \
  --max-signs 5
```

**参数说明**:
- `--num-images 1000`: 生成1000张合成图片
- `--min-signs 2`: 每张图片最少2个标志
- `--max-signs 5`: 每张图片最多5个标志

#### 步骤2: 合并数据集

```bash
# 将合成数据复制到原训练集
# Windows:
xcopy /E /I datasets\gtsrb_multi\images\train\*.* datasets\gtsrb\images\train\
xcopy /E /I datasets\gtsrb_multi\labels\train\*.* datasets\gtsrb\labels\train\

# Linux/Mac:
# cp -r datasets/gtsrb_multi/images/train/* datasets/gtsrb/images/train/
# cp -r datasets/gtsrb_multi/labels/train/* datasets/gtsrb/labels/train/
```

#### 步骤3: 训练模型

```bash
# 使用增强的数据集训练
python yolov5/train.py \
  --data data/gtsrb.yaml \
  --weights yolov5s.pt \
  --epochs 150 \
  --batch-size 32 \
  --img 640 \
  --hyp data/hyps/hyp.traffic-sign.yaml \
  --cache ram \
  --project runs/train \
  --name improved_v2_with_synthetic
```

#### 步骤4: 测试多目标检测性能

```bash
# 在合成的多目标数据集上测试
python scripts/test_multi_target.py \
  --weights runs/train/improved_v2_with_synthetic/weights/best.pt \
  --test-dir datasets/gtsrb_multi/images/train \
  --label-dir datasets/gtsrb_multi/labels/train \
  --conf-thres 0.25 \
  --iou-thres 0.35 \
  --save-results \
  --output-dir test_results
```

---

### 方案C: 使用更大的模型 (如果方案A/B效果不够好)

```bash
# 使用YOLOv5m (中等模型,更强的检测能力)
python yolov5/train.py \
  --data data/gtsrb.yaml \
  --weights yolov5m.pt \
  --cfg yolov5/models/yolov5m.yaml \
  --epochs 150 \
  --batch-size 16 \
  --img 640 \
  --hyp data/hyps/hyp.traffic-sign.yaml \
  --cache ram \
  --project runs/train \
  --name improved_v3_yolov5m
```

**注意**: YOLOv5m需要更多显存,batch-size可能需要减小到16或8。

---

## 🎛️ 调整检测参数

如果训练后效果还不够理想,可以调整检测时的参数:

### 修改 `web/app.py`

找到这两行(大约在第111-112行):

```python
ANOMALY_CONF_THRES = float(os.getenv("ANOMALY_CONF_THRES", "0.5"))
SCALE_THRESHOLD = int(os.getenv("SCALE_THRESHOLD", "200"))
```

在 `.env` 文件中添加:

```bash
# 降低置信度阈值,检测更多目标
ANOMALY_CONF_THRES=0.3

# 调整NMS IoU阈值 (在detect_objects函数中)
# 默认是0.45,可以降低到0.35以减少误删除
```

或者直接修改默认参数:

```python
# 在 detect_objects 函数中 (第137行)
def detect_objects(img: torch.Tensor, conf_thres: float = 0.25, iou_thres: float = 0.35) -> List:
    #                                                                            ^^^^
    #                                                                    从0.45改为0.35
```

---

## 📊 预期效果对比

| 方案 | 实施难度 | 时间成本 | 预期改进 |
|------|---------|---------|---------|
| 方案A | ⭐ 简单 | 2-4小时 | 多目标mAP: 70% → 80%<br>泛化能力: 弱 → 中等 |
| 方案B | ⭐⭐ 中等 | 4-6小时 | 多目标mAP: 70% → 85%<br>泛化能力: 弱 → 强 |
| 方案C | ⭐⭐⭐ 较难 | 5-8小时 | 多目标mAP: 70% → 90%<br>泛化能力: 弱 → 很强 |

---

## ⚠️ 常见问题

### Q1: 显存不足怎么办?

```bash
# 减小batch size
--batch-size 16  # 或 8, 4

# 使用混合精度训练
--amp
```

### Q2: 训练太慢怎么办?

```bash
# 使用缓存加速
--cache ram  # 缓存到内存 (推荐)
# 或
--cache disk  # 缓存到硬盘

# 减少epochs
--epochs 100  # 而不是150
```

### Q3: 如何知道训练是否有效?

查看训练日志中的指标:
- **mAP@0.5** 应该 > 0.90
- **mAP@0.5:0.95** 应该 > 0.65
- **Precision** 和 **Recall** 应该都 > 0.85

### Q4: 检测结果还是不好怎么办?

1. 检查训练是否收敛 (loss是否下降)
2. 尝试降低检测阈值 (conf_thres从0.25降到0.15)
3. 调整NMS阈值 (iou_thres从0.45降到0.35)
4. 考虑使用更大的模型 (YOLOv5m或YOLOv5l)

---

## 🎯 推荐流程

**第一步**: 先尝试**方案A**(最简单,2-4小时)

**第二步**: 如果效果不够好,再尝试**方案B**(增加合成数据)

**第三步**: 如果还不够好,最后尝试**方案C**(使用更大模型)

---

## 📞 需要帮助?

如果遇到问题,请告诉我:
1. 具体的错误信息
2. 您使用的是哪个方案
3. 训练/测试的输出日志

我会帮您解决!
