# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is an **Ultralytics YOLO26** repository with **custom Pose26 modifications** for multi-attribute pose estimation. The standard Ultralytics codebase has been extended to support a custom Pose26 head that outputs three independent attribute branches (color, size, object type) instead of a single 64-dimensional classification head.

### Key Modification: Pose26 Three-Attribute Branches

The Pose26 model (`ultralytics/nn/modules/head.py:650-850`) has been modified to split classification into three separate branches:

- **color_head**: 4-dimensional (B, G, R, P)
- **size_head**: 2-dimensional (s, b)
- **obj_head**: 8-dimensional (G, 1, 2, 3, 4, 5, O, B)

Data mapping: `cls_id = color_id * 16 + size_id * 8 + obj_id` (flattened 64-dim label)

**Loss indices** in PoseLoss26:
| Index | Loss |
|-------|------|
| 0 | box_loss |
| 1 | kpt_loc_loss |
| 2 | kpt_vis_loss |
| 3 | cls_loss (obj) |
| 4 | dfl_loss |
| 5 | rle_loss (optional) |
| 6 | **color_loss** |
| 7 | **size_loss** |
| 8 | **obj_loss** |

## Common Commands

### Training
```bash
# Run pose training with custom Pose26 model
python3 train.py

# Or use YOLO CLI directly
yolo pose train model=ultralytics/cfg/models/26/yolo26-pose.yaml data=data.yaml epochs=100
```

### Model Testing/Validation
```python
from ultralytics import YOLO
model = YOLO("ultralytics/cfg/models/26/yolo26-pose.yaml")
model.val()
```

### Running Tests
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_python.py

# Run with coverage
pytest --cov
```

## Architecture

### Model Parsing (`ultralytics/nn/tasks.py`)
- `parse_model(d, ch, verbose=True)` parses YAML configs and builds models
- **Important**: Variables like `num_color`, `num_size`, `num_obj` must be added to parse_model for Pose26 (line 1537-1539)
- `reg_max` and `end2end` are **auto-added** by parse_model - do NOT include them in YAML Pose26 args

### Head Modules (`ultralytics/nn/modules/head.py`)
- `Detect`: Standard YOLO detection head (box + cls)
- `Pose`: Standard pose head (box + cls + kpts)
- `Pose26`: **Custom** pose head with three attribute branches (color, size, obj)
  - Overrides: `__init__`, `one2many`, `one2one`, `forward_head`, `_inference`, `bias_init`, `fuse`
  - Key: Uses obj branch for target assignment, shares results with color/size branches

### Loss Functions (`ultralytics/utils/loss.py`)
- `v8DetectionLoss`: Base detection loss (box, cls, dfl)
- `v8PoseLoss`: Standard pose loss (extends detection + kpt_loc, kpt_vis, rle)
- `PoseLoss26`: **Custom** pose loss for three-attribute branches
  - Decodes 64-dim cls into color_id, size_id, obj_id
  - Uses obj branch for TAL (Task Aligned Learning) assignment
  - Computes separate BCE losses for color, size, obj

### Task Aligned Learning (`ultralytics/utils/tal.py`)
- `TaskAlignedAssigner`: Assigns ground truth to anchor points
- **Critical fix** (line 195): Index order must be `pd_scores[ind[0], ind[1], :]` NOT `pd_scores[ind[0], :, ind[1]]`
  - Correctly selects anchor dimension: `[batch, class, anchor]` → `[batch, anchor, :]`

### Training Flow
1. `model.train()` → `trainer.py` → `PoseTrainer`
2. Forward pass: backbone → neck → Pose26 head → outputs `{boxes, color, size, obj, kpts, kpts_sigma, feats}`
3. Loss computation: `PoseLoss26.loss()` → decodes cls, computes BCE for each branch
4. Backward pass and optimization

### Model Configs (`ultralytics/cfg/models/26/`)
- `yolo26-pose.yaml`: **Modified** with three-attribute parameters
- `yolo26-pose_dwconv.yaml`: Depthwise separable conv variant

## File Structure

```
ultralytics/
├── cfg/
│   ├── models/26/          # YOLO26 model configs
│   └── datasets/           # Dataset configs (coco8.yaml, coco-pose.yaml, etc.)
├── engine/
│   ├── trainer.py          # BaseTrainer for all tasks
│   ├── validator.py        # Validation logic
│   └── predictor.py        # Inference logic
├── models/
│   └── yolo/
│       ├── classify/       # Image classification
│       ├── detect/         # Object detection
│       ├── pose/           # **Custom Pose26 trainer**
│       ├── segment/        # Instance segmentation
│       └── obb/            # Oriented bounding boxes
├── nn/
│   ├── modules/
│   │   ├── head.py         # **Pose26 class (modified)**
│   │   ├── block.py        # Building blocks (C3k2, C2f, etc.)
│   │   └── conv.py         # Conv layers (Conv, DWConv, etc.)
│   └── tasks.py            # Model parsing (parse_model - modified)
└── utils/
    ├── loss.py             # **PoseLoss26 (modified)**
    └── tal.py              # TaskAlignedAssigner (fixed)
```

## Important Notes

### Pose26 Customization Details

1. **YAML Config** (`yolo26-pose.yaml`):
   - `nc`: obj classes (not total flattened classes)
   - `num_color`, `num_size`, `num_obj`: attribute dimensions
   - Pose26 args: `[nc, kpt_shape, num_color, num_size, num_obj]` (NO reg_max, NO end2end)

2. **Data Format**:
   - Training: 64-dim cls labels → decoded to (color_id, size_id, obj_id)
   - Inference output: `[boxes(4), color(4), size(2), obj(8), kpts(8)]` = 26 dims

3. **TAL Assignment**:
   - Uses obj branch (8 classes) for target assignment
   - `self.nc` and `assigner.num_classes` must be set to `num_obj` in PoseLoss26.__init__

4. **Tensor Shapes**:
   - Preds: `{boxes: (B, 4*reg_max, A), color/size/obj: (B, num_classes, A), kpts: (B, nk, A)}`
   - A = total anchors (sum of H*W across P3/P4/P5)
   - target_scores after assigner: `(B, A, num_classes)` - note the permute!

### Debugging Common Issues

- **Shape mismatch in TAL**: Check `pd_scores[ind[0], ind[1], :]` indexing
- **Wrong loss dimensions**: Ensure PoseLoss26 has `self.nc = self.num_obj`
- **Parse errors**: Verify YAML doesn't include auto-added params (reg_max, end2end)
- **Bias initialization**: Pose26 has custom `bias_init()` for three branches

## Development

### Adding New Model Variants
1. Create YAML in `ultralytics/cfg/models/26/`
2. Add new parameters to `parse_model()` if needed
3. Extend appropriate head class (Detect, Pose, Pose26, etc.)

### Modifying Loss Functions
- Detection tasks: `v8DetectionLoss`
- Pose tasks: `v8PoseLoss` or `PoseLoss26` (for custom attributes)
- Return format: `(loss * batch_size, loss.detach())`

### Testing
- Tests located in `tests/`
- Use `pytest` for running tests
- Key files: `test_python.py`, `test_engine.py`
