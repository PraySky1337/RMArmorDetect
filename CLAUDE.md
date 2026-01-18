# RMArmorDetect 项目须知

## 架构说明

本项目基于 Ultralytics YOLO 框架，但有自定义的装甲板检测逻辑。

**依赖关系原则**：`armor_detect` 单向依赖 `ultralytics`（armor_detect → ultralytics ✓，ultralytics → armor_detect ✗）

### 核心组件位置

| 组件 | 位置 | 说明 |
|------|------|------|
| **Model** | `armor_detect/models/armor_pose_model.py` → `ArmorPoseModel` | 模型入口 |
| **Pose Head** | `armor_detect/models/heads/armor_pose_head.py` → `ArmorPoseHead` | 三分支分类头 (num + color + size) |
| **Loss** | `armor_detect/losses/armor_pose_loss.py` → `ArmorPoseLoss` | 三分支损失函数 |
| **Trainer** | `armor_detect/trainers/armor_trainer.py` → `ArmorTrainer` | 训练器 |
| **Validator** | `armor_detect/validators/armor_validator.py` → `ArmorValidator` | 验证器 |
| **Dataset** | `armor_detect/data/armor_dataset.py` | 数据加载 |

### 关键调用链

```
train.py
  → ArmorTrainer (armor_detect/trainers/)
    → ArmorPoseModel (armor_detect/models/)
      → DetectionModel (ultralytics) ← 单向依赖
      → ArmorPoseHead (armor_detect/models/heads/)
      → init_criterion() → ArmorPoseLoss (armor_detect/losses/)
```

## 重要警告

### 1. 使用正确的模型类

训练时必须使用 `ArmorPoseModel`，**不要**使用 `ultralytics.PoseModel`：

```python
# ✓ 正确
from armor_detect.models import ArmorPoseModel
model = ArmorPoseModel(cfg)

# ❌ 错误 - 会抛出 NotImplementedError
from ultralytics.nn.tasks import PoseModel
model = PoseModel(cfg)
```

### 2. Head 输出格式

`ArmorPoseHead` 输出 **4 个值**（三分支）：
```python
return feats, (cls_num, cls_color, cls_size, kpt)
```

所有消费这个输出的代码必须解包 4 个值，不是 3 个。

### 3. 配置文件参数

`armor_detect/configs/armor_yolo11.yaml` 中 Pose head 参数：
```yaml
- [[15, 18, 21], 1, Pose, [nc_num, nc_color, nc_size, kpt_shape]]
# 例如: [8, 4, 2, [4, 2]]
```

参数顺序：`nc_num`, `nc_color`, `nc_size`, `kpt_shape`, `ch`（ch 由 parse_model 自动追加）

### 4. 禁止反向依赖

**ultralytics 目录下的代码不能导入 armor_detect 中的任何模块**。

验证方法：
```bash
grep -r "from armor_detect" ultralytics/
grep -r "import armor_detect" ultralytics/
# 应该没有结果
```

## 历史教训

### 2026-01-17: 依赖解耦

**问题**: `ultralytics/nn/tasks.py` 中的 `PoseModel.init_criterion()` 导入了 `armor_detect`，
导致反向依赖，违反了单向依赖原则。

**解决**:
1. 创建 `ArmorPoseModel` 类在 `armor_detect/models/` 中
2. 修改 `ArmorTrainer` 使用 `ArmorPoseModel`
3. 将 `PoseModel.init_criterion()` 改为抛出 `NotImplementedError`

**涉及文件**:
- `armor_detect/models/armor_pose_model.py` (新建)
- `armor_detect/trainers/armor_trainer.py` (修改)
- `armor_detect/models/__init__.py` (修改)
- `ultralytics/nn/tasks.py` (修改)

### 2026-01-05: 双分支→三分支升级

**问题**: 将 Pose head 从双分支 (num + color) 升级为三分支 (num + color + size) 后报错：
```
TypeError: __init__() takes from 1 to 5 positional arguments but 6 were given
ValueError: too many values to unpack (expected 3)
```

**原因**:
1. `Pose.__init__` 签名没更新（少了 `nc_size` 参数）
2. `v8PoseLoss` 解包时只取 3 个值，但 head 输出 4 个

**解决**:
1. 更新 `Pose` head 签名和实现（添加 `nc_size` 和 `cv_size` 分支）
2. 改用 `ArmorPoseLoss`（已支持三分支）
3. 删除过时的 `v8PoseLoss`

**涉及文件**:
- `ultralytics/nn/modules/head.py` - Pose 类
- `ultralytics/nn/tasks.py` - PoseModel.init_criterion()
- `ultralytics/utils/loss.py` - 删除 v8PoseLoss

### 2026-01-05: 添加 size 字段时 collate_fn 报错

**问题**: 添加 `size` 分类后训练时报错：
```
IndexError: list index out of range
```
发生在 `ultralytics/data/dataset.py` 的 `collate_fn`。

**原因**:
数据管道中多处只处理了 `color` 但没处理 `size`：
1. `YOLODataset.update_labels_info` - 只处理 color，没处理 size
2. `Format.__call__` (augment.py) - 只 pop 和处理 color
3. `Mosaic._mix_transform` - 只拼接 color
4. `MixUp._mix_transform` - 只拼接 color
5. `CutMix._mix_transform` - 只拼接 color
6. `RandomPerspective.__call__` - 只过滤 color
7. `Albumentations.__call__` - 只过滤 color
8. `collate_fn` - 已处理，但前序步骤缺失导致 key 不一致

**解决**:
在所有处理 `color` 的地方添加相同的 `size` 处理逻辑。

**涉及文件**:
- `ultralytics/data/dataset.py`:
  - `update_labels_info` - 添加 size 处理
  - `collate_fn` - 添加 size 到处理集合
- `ultralytics/data/augment.py`:
  - `Format.__call__` - 添加 size pop 和输出
  - `Mosaic._mix_transform` - 添加 size 拼接
  - `MixUp._mix_transform` - 添加 size 拼接
  - `CutMix._mix_transform` - 添加 size 拼接
  - `RandomPerspective.__call__` - 添加 size 过滤
  - `Albumentations.__call__` - 添加 size 过滤

**教训**: 添加新的标签字段时，必须检查整个数据管道，确保所有 transform 都正确传递和处理该字段。搜索现有字段（如 `"color"`）的处理位置，在每个位置添加新字段的处理。

