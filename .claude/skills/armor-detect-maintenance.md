# RMArmorDetect 维护指南

## 依赖关系原则

**单向依赖**：`armor_detect` → `ultralytics` ✓

```
armor_detect/
  ├── trainers/      → 可以导入 ultralytics
  ├── validators/    → 可以导入 ultralytics
  ├── losses/        → 可以导入 ultralytics.utils
  ├── models/        → 可以导入 ultralytics.nn
  └── data/          → 可以导入 ultralytics.data

ultralytics/
  └── 不能导入任何 armor_detect 模块 ❌
```

## 架构概览

```
train.py
  └── ArmorTrainer (armor_detect/trainers/)
        └── ArmorPoseModel (armor_detect/models/)
              ├── DetectionModel (ultralytics) ← 单向依赖
              ├── ArmorPoseHead (armor_detect/models/heads/)
              └── ArmorPoseLoss (armor_detect/losses/)
```

## 关键文件清单

| 文件 | 职责 |
|------|------|
| `armor_detect/models/armor_pose_model.py` | 模型入口，继承 DetectionModel |
| `armor_detect/models/heads/armor_pose_head.py` | 三分支 Head (num + color + size) |
| `armor_detect/losses/armor_pose_loss.py` | 三分支 Loss |
| `armor_detect/trainers/armor_trainer.py` | 训练器 |
| `armor_detect/validators/armor_validator.py` | 验证器 |
| `armor_detect/configs/armor_yolo11.yaml` | 模型配置 |

## 添加新功能的正确方法

### 添加新的 Loss 组件

1. 在 `armor_detect/losses/` 中创建新文件
2. 在 `ArmorPoseLoss` 中集成
3. **不要**修改 `ultralytics/utils/loss.py`

### 添加新的 Head 分支

1. 修改 `armor_detect/models/heads/armor_pose_head.py`
2. 更新 `ArmorPoseLoss` 以处理新输出
3. 更新 `ArmorValidator` 以评估新指标
4. **不要**修改 `ultralytics/nn/modules/head.py`

### 添加新的数据增强

1. 继承 ultralytics 的增强类
2. 在 `armor_detect/data/` 中扩展
3. 确保传递所有自定义字段 (color, size 等)

## 常见错误

### 错误1: 在 ultralytics 中导入 armor_detect

```python
# ❌ 错误
# ultralytics/nn/tasks.py
from armor_detect.losses import ArmorPoseLoss

# ✓ 正确
# armor_detect/models/armor_pose_model.py
from armor_detect.losses import ArmorPoseLoss
```

### 错误2: 训练时使用 PoseModel

```python
# ❌ 错误
from ultralytics.nn.tasks import PoseModel
model = PoseModel(cfg)

# ✓ 正确
from armor_detect.models import ArmorPoseModel
model = ArmorPoseModel(cfg)
```

### 错误3: 添加字段时遗漏数据管道

添加新标签字段时，必须检查整个数据管道：

- `ultralytics/data/dataset.py` → `YOLODataset.update_labels_info`
- `ultralytics/data/augment.py` → `Format.__call__`
- `ultralytics/data/augment.py` → `Mosaic._mix_transform`
- `ultralytics/data/augment.py` → `MixUp._mix_transform`
- `ultralytics/data/augment.py` → `CutMix._mix_transform`
- `ultralytics/data/augment.py` → `RandomPerspective.__call__`
- `ultralytics/data/augment.py` → `Albumentations.__call__`
- `ultralytics/data/dataset.py` → `collate_fn`

搜索现有字段（如 `"color"`）的处理位置，在每个位置添加新字段的处理。

## 验证依赖关系

```bash
# 检查是否有反向依赖（应该没有结果）
grep -r "from armor_detect" ultralytics/
grep -r "import armor_detect" ultralytics/
```

## 测试训练

```bash
python train.py
```

如果出现 `NotImplementedError: PoseModel.init_criterion() is deprecated`，
说明代码错误地使用了 `ultralytics.PoseModel` 而不是 `ArmorPoseModel`。

## 历史记录

### 2026-01-17: 依赖解耦

**问题**: `ultralytics/nn/tasks.py` 中的 `PoseModel.init_criterion()` 导入了 `armor_detect`，
导致反向依赖。

**解决**:
1. 创建 `ArmorPoseModel` 类在 `armor_detect/models/` 中
2. 修改 `ArmorTrainer` 使用 `ArmorPoseModel`
3. 将 `PoseModel.init_criterion()` 改为抛出 `NotImplementedError`

**涉及文件**:
- `armor_detect/models/armor_pose_model.py` (新建)
- `armor_detect/trainers/armor_trainer.py` (修改)
- `armor_detect/models/__init__.py` (修改)
- `ultralytics/nn/tasks.py` (修改)
