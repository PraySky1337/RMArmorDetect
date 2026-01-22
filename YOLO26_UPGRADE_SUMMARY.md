# YOLO26 升级总结

## 升级信息

| 项目 | 值 |
|------|-----|
| 原版本 | ultralytics 8.3.233 (YOLO11) |
| 新版本 | ultralytics 8.4.7 (YOLO26) |
| 升级日期 | 2026-01-22 |
| 分支 | `upgrade-to-yolo26` |
| 备份分支 | `backup-before-yolo26-upgrade` |

## 新增的 YOLO26 特性

### 1. 新模型类
- **Pose26**: 支持 RLE 损失的姿态估计模型
- **OBB26**: YOLO26 旋转边界框检测
- **Segment26**: YOLO26 分割模型
- **YOLOESegment26**: YOLOE 分割模型

### 2. 新优化器
- **MuSGD**: 混合优化器（SGD + Muon），适合大规模训练
- **AdamW**: 新增的优化器选项
- **自动选择**: `optimizer: auto` 会根据训练迭代数自动选择

### 3. 新损失函数
- **PoseLoss26**: 带 RLE 支持的姿态损失
- **RLELoss**: 残差对数似然损失
- **MultiChannelDiceLoss**: 多通道 Dice 损失
- **BCEDiceLoss**: BCE + Dice 组合损失

### 4. 配置参数变化
- `multi_scale`: 布尔值 → 浮点数 (如 0.5 表示 50% 尺寸范围)
- `rle`: RLE 损失权重 (默认 1.0)
- `angle`: OBB 角度损失权重 (默认 1.0)

### 5. 新增配置文件
```
ultralytics/cfg/models/26/
├── yolo26.yaml
├── yolo26-cls.yaml
├── yolo26-pose.yaml
├── yolo26-seg.yaml
├── yolo26-obb.yaml
├── yolo26-p2.yaml
├── yolo26-p6.yaml
├── yoloe-26.yaml
└── yoloe-26-seg.yaml
```

## 保留的装甲板检测特性

### 1. 自定义 Pose 类
- 三分支分类架构 (num + color + size)
- 4 关键点装甲板检测
- 无边界框的关键点检测

### 2. 自定义损失函数
- **WingLoss**: 关键点回归损失
- **ArmorPoseLoss**: 三分支损失函数

### 3. 自定义参数
```yaml
color: 1.0          # 颜色分类损失权重
size: 1.0           # 尺寸分类损失权重
focal_gamma: 1.5    # Focal Loss gamma
wing_omega: 10.0    # WingLoss omega
wing_epsilon: 2.0   # WingLoss epsilon
bg_weight: 0.3      # 背景权重
```

### 4. 数据管道
- 保留 color/size 字段在数据增强中的同步
- 保留 Albumentations 增强的特殊处理

## 解决的冲突文件

| 文件 | 修改内容 |
|------|----------|
| `ultralytics/cfg/default.yaml` | 添加 YOLO26 参数，保留装甲板检测参数 |
| `ultralytics/cfg/models/11/yolo11-pose.yaml` | 保留装甲板检测配置 |
| `ultralytics/data/augment.py` | 保留 color/size 同步，添加 sem_masks |
| `ultralytics/models/yolo/pose/predict.py` | 保留装甲板检测推理逻辑 |
| `ultralytics/models/yolo/pose/train.py` | 保留装甲板检测 loss_names |
| `ultralytics/models/yolo/pose/val.py` | 保留装甲板检测属性 |
| `ultralytics/nn/modules/head.py` | 保留装甲板 Pose 类，添加 YOLO26 类 |
| `ultralytics/nn/tasks.py` | 更新 PoseModel 支持 YOLO26 |
| `ultralytics/utils/loss.py` | 恢复 v8PoseLoss，添加 Pose26/RLE 损失 |

## 使用新特性的建议

### 1. MuSGD 优化器
```yaml
# 在训练配置中设置
optimizer: auto  # 自动选择 (>10000 iterations 用 MuSGD)
# 或
optimizer: MuSGD
```

### 2. AdamW 优化器
```yaml
optimizer: AdamW
lr0: 0.001  # AdamW 通常需要较低的学习率
```

### 3. 多尺度训练
```yaml
multi_scale: 0.5  # 50% 尺寸范围 (如 640 → [320, 960])
```

### 4. End-to-End 模式 (NMS-Free)
注意：装甲板检测使用关键点，可能不需要此模式。如需测试：
```yaml
# 在模型配置中
end2end: True
```

## 验证结果

```
✓ 版本: 8.3.233 → 8.4.7
✓ YOLO26 配置文件: 9 个
✓ MuSGD 优化器支持
✓ RLE 损失函数
✓ Pose26, OBB26, Segment26 类
✓ multi_scale 浮点参数
✓ 装甲板检测特性全部保留
✓ 无冲突标记
✓ Python 语法检查通过
```

## 下一步操作

1. **测试训练**: 在 PyTorch 环境中测试装甲板检测训练
2. **性能对比**: 对比 YOLO11 和 YOLO26 的训练效果
3. **决定是否合并**: 如果测试通过，合并到主分支

## 回滚方案

如需回滚到升级前：
```bash
git checkout main
git merge backup-before-yolo26-upgrade
```
