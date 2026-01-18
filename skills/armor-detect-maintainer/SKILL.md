---
name: armor-detect-maintainer
description: Maintain RMArmorDetect armor_detect models and configs; use when modifying ArmorPoseModel, backbone/neck/head definitions, adding custom modules in armor_detect/modules (e.g., MobileNetV4 or ShuffleNetV2), checking params/GFLOPs for edge deployment, or ensuring TensorBoard metric logging during training.
---

# RMArmorDetect maintenance

## Follow project guardrails
- Keep the dependency one-way: `armor_detect` may import `ultralytics`, never the reverse.
- Instantiate `ArmorPoseModel` (not `ultralytics.PoseModel`) for training/inference.
- Preserve the ArmorPoseHead output contract: `return feats, (cls_num, cls_color, cls_size, kpt)`.
- Keep Pose args order in YAML: `nc_num, nc_color, nc_size, kpt_shape`.

## Identify backbone, neck, head
- Open `armor_detect/configs/armor_yolo11.yaml` (or a variant) and read `backbone` and `head` lists.
- Treat the backbone as the stack that produces P3/P4/P5 feature maps.
- Treat the neck as the FPN/PAN blocks inside `head` before the final Pose layer.
- Treat the head as the final `Pose` entry that consumes `[P3, P4, P5]`.
- When replacing modules, update layer indices so P3/P4/P5 remain strides 8/16/32 for ArmorPoseHead.

## Check params and GFLOPs
- Use `ArmorPoseModel(...).info(imgsz=...)` to log Params and GFLOPs at the training image size.
- Example:
  ```python
  from armor_detect.models import ArmorPoseModel
  model = ArmorPoseModel("armor_detect/configs/armor_yolo11.yaml", verbose=True)
  model.info(imgsz=416)
  ```
- If GFLOPs show 0, install `thop` or use `ultralytics.utils.torch_utils.get_flops_with_torch_profiler`.

## Keep TensorBoard logging reliable
- Enable settings via `train.py` (`SETTINGS.update({"tensorboard": True})`) or `yolo settings tensorboard=True`.
- Verify `ultralytics/utils/callbacks/tensorboard.py` is active and logs:
  `train/pose_loss`, `train/kobj_loss`, `train/cls_loss`, `train/color_loss`, `train/size_loss`, `metrics/*`, and `lr/*`.
- Check the run directory in `train_config.yaml` and open TensorBoard with `tensorboard --logdir <runs_dir>`.

## Add or swap backbones safely
- Implement new backbones under `armor_detect/modules/` and keep them independent of `ultralytics` internals.
- If a backbone returns multiple feature maps, use `Index` layers in YAML to select P3/P4/P5.
- Inject custom backbone classes in `ArmorPoseModel._from_yaml` before calling `parse_model`, then restore originals.
- Add new model YAMLs per backbone to keep switching simple (update `train_config.yaml` or training args only).
