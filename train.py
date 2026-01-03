"""Quick-start script to train a local pose model on datasets/images + datasets/labels."""

from __future__ import annotations

import random
from datetime import datetime
from pathlib import Path

import yaml
import numpy as np
from ultralytics import YOLO
from ultralytics.utils import SETTINGS

SETTINGS.update({
    "datasets_dir": "/home/rry/RMArmorDetect/datasets",
    "weights_dir": "/home/rry/RMArmorDetect/weights",
    "runs_dir": "/home/rry/RMArmorDetect/runs",
})

ROOT = Path(__file__).resolve().parent
IMG_DIR = ROOT / "datasets" / "images"
LBL_DIR = ROOT / "datasets" / "labels"
SPLIT_DIR = ROOT / "datasets" / "splits"
DATA_YAML = ROOT / "datasets" / "rmarmor-pose.yaml"
TRAIN_CFG = ROOT / "train_config.yaml"
 
COLOR_NAMES = ["B","R","N","P"]
CLASS_NAMES = [
    "Gs",  
    "1b",  
    "2s",  
    "3s",  
    "4s",  
    "Os",  
    "Bs", 
    "Bb",  
] 

def load_train_cfg(path: Path = TRAIN_CFG) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Training config not found: {path}")
    cfg = yaml.safe_load(path.read_text()) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid config structure in {path}, expected a mapping.")
    return cfg

def infer_dataset_info(labels_dir: Path) -> tuple[set[int], tuple[int, int], set[int]]:
    """Infer classes, keypoints shape, and color classes from label files."""
    label_files = sorted(labels_dir.glob("*.txt"))
    if not label_files:
        raise FileNotFoundError(f"No label files found under {labels_dir}")

    classes_present: set[int] = set()
    colors_present: set[int] = set()
    num_kpts, ndim = None, None

    for lf in label_files:
        for line in lf.read_text().splitlines():
            parts = line.strip().split()
            if not parts:
                continue
            color = int(float(parts[0]))  
            cls = int(float(parts[1]))
            classes_present.add(cls)
            colors_present.add(color)

            coords = len(parts) - 2
            if coords <= 0:
                continue
            if ndim is None:
                if coords % 3 == 0:
                    ndim = 3
                elif coords % 2 == 0:
                    ndim = 2
                else:
                    raise ValueError(f"Inconsistent keypoint dims in {lf}: {coords} coords")
                num_kpts = coords // ndim
            else:
                if coords % ndim != 0:
                    raise ValueError(f"Inconsistent keypoint dims in {lf}: {coords} coords")
                kpts_here = coords // ndim
                if kpts_here != num_kpts:
                    raise ValueError(f"Inconsistent keypoint count in {lf}: {kpts_here} vs {num_kpts}")

    return classes_present, (num_kpts, ndim), colors_present

def make_splits(img_dir: Path, train_ratio: float = 0.9, seed: int = 0) -> tuple[Path, Path]:
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    images = [p.resolve() for p in img_dir.iterdir() if p.suffix.lower() in valid_exts]
    if not images:
        raise FileNotFoundError(f"No images found under {img_dir}")

    rng = random.Random(seed)
    rng.shuffle(images)

    n_train = max(1, int(len(images) * train_ratio))
    if n_train == len(images):
        n_train = len(images) - 1
    train_imgs = images[:n_train]
    val_imgs = images[n_train:]

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    train_txt = SPLIT_DIR / "train.txt"
    val_txt = SPLIT_DIR / "val.txt"
    train_txt.write_text("\n".join(map(str, train_imgs)), encoding="utf-8")
    val_txt.write_text("\n".join(map(str, val_imgs)), encoding="utf-8")
    return train_txt, val_txt

def write_data_yaml(names: dict[int, str], kpt_shape: tuple[int, int], train_txt: Path, val_txt: Path) -> Path:
    """Write pose + color data.yaml"""
    data = {
        "path": str(ROOT),
        "train": str(train_txt),
        "val": str(val_txt),
        "kpt_shape": list(kpt_shape),
        "names": {int(k): v for k, v in names.items()},
        "color_names": COLOR_NAMES,  
    }
    DATA_YAML.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return DATA_YAML

def main() -> None:
    cfg = load_train_cfg()

    if cfg.get("tensorboard", True):
        SETTINGS.update({"tensorboard": True})

    # Generate unique run name with timestamp
    base_name = cfg.get("name", "rmarmor-pose")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{base_name}_{timestamp}"

    classes_present, kpt_shape, colors_present = infer_dataset_info(LBL_DIR)
    missing = [i for i in range(len(CLASS_NAMES)) if i not in classes_present]
    if missing:
        print(f"Note: {len(missing)} classes not present in labels (kept anyway): {missing[:10]}{'...' if len(missing) > 10 else ''}")

    names = {i: name for i, name in enumerate(CLASS_NAMES)}

    train_txt, val_txt = make_splits(
        IMG_DIR,
        train_ratio=float(cfg.get("train_ratio", 0.9)),
        seed=int(cfg.get("seed", 0)),
    )
    data_yaml = write_data_yaml(names, kpt_shape, train_txt, val_txt)

   
    model = YOLO(cfg.get("model", "ultralytics/cfg/models/11/yolo11-pose.yaml"))

    model.train(
        data=str(data_yaml),
        epochs=int(cfg.get("epochs", 50)),
        imgsz=int(cfg.get("imgsz", 640)),
        batch=int(cfg.get("batch", 16)),
        device=cfg.get("device", 0),
        workers=int(cfg.get("workers", 8)),
        cache=cfg.get("cache", False),

        optimizer=cfg.get("optimizer", "AdamW"),
        lr0=float(cfg.get("lr0", 1e-3)),
        lrf=float(cfg.get("lrf", 0.01)),
        momentum=float(cfg.get("momentum", 0.937)),
        weight_decay=float(cfg.get("weight_decay", 5e-4)),
        warmup_epochs=float(cfg.get("warmup_epochs", 3.0)),
        warmup_momentum=float(cfg.get("warmup_momentum", 0.8)),
        warmup_bias_lr=float(cfg.get("warmup_bias_lr", 0.1)),
        cos_lr=bool(cfg.get("cos_lr", True)),

        # loss weights (keypoint-only mode, no bbox)
        pose=float(cfg.get("pose", 12.0)),
        kobj=float(cfg.get("kobj", 2.0)),
        cls=float(cfg.get("cls", 0.5)),
        color=float(cfg.get("color", 1.0)),

        # data augmentation
        hsv_h=float(cfg.get("hsv_h", 0.015)),
        hsv_s=float(cfg.get("hsv_s", 0.7)),
        hsv_v=float(cfg.get("hsv_v", 0.4)),
        degrees=float(cfg.get("degrees", 0.0)),
        translate=float(cfg.get("translate", 0.1)),
        scale=float(cfg.get("scale", 0.5)),
        shear=float(cfg.get("shear", 0.0)),
        perspective=float(cfg.get("perspective", 0.0)),
        flipud=float(cfg.get("flipud", 0.0)),
        fliplr=float(cfg.get("fliplr", 0.5)),
        mosaic=float(cfg.get("mosaic", 1.0)),
        mixup=float(cfg.get("mixup", 0.0)),
        copy_paste=float(cfg.get("copy_paste", 0.0)),
        erasing=float(cfg.get("erasing", 0.4)),
        close_mosaic=int(cfg.get("close_mosaic", 10)),

        patience=int(cfg.get("patience", 50)),
        resume=cfg.get("resume", False),
        pretrained=cfg.get("pretrained", True),
        amp=cfg.get("amp", True),
        deterministic=cfg.get("deterministic", False),
        seed=int(cfg.get("seed", 0)),
        val=cfg.get("val", True),

        project=str(cfg.get("project", ROOT / "runs" / "pose")),
        name=run_name,
        exist_ok=False,  # Always create new folder
        save=cfg.get("save", True),
        save_period=int(cfg.get("save_period", -1)),
        plots=cfg.get("plots", True),
    )


if __name__ == "__main__":
    main()


# from __future__ import annotations

# import random
# from pathlib import Path

# import yaml
# from ultralytics import YOLO
# from ultralytics.utils import SETTINGS

# SETTINGS.update({
#     "datasets_dir": "/home/rry/RMArmorPose/datasets",
#     "weights_dir": "/home/rry/RMArmorPose/weights",
#     "runs_dir": "/home/ljy/RMArmorPose/runs",
# })

# ROOT = Path(__file__).resolve().parent
# IMG_DIR = ROOT / "datasets" / "images"
# LBL_DIR = ROOT / "datasets" / "labels"
# SPLIT_DIR = ROOT / "datasets" / "splits"
# DATA_YAML = ROOT / "datasets" / "rmarmor-pose.yaml"
# TRAIN_CFG = ROOT / "train_config.yaml"
# # CLASS_NAMES = [
# #     "BGs",
# #     "B1b",
# #     "B2s",
# #     "B3s",
# #     "B4s",
# #     "B5s",
# #     "BO",
# #     "BBs",
# #     "BBb",
# #     "BGb",
# #     "B3b",
# #     "B4b",
# #     "B5b",
# #     "RGs",
# #     "R1b",
# #     "R2s",
# #     "R3s",
# #     "R4s",
# #     "R5s",
# #     "RO",
# #     "RBs",
# #     "RBb",
# #     "RGb",
# #     "R3b",
# #     "R4b",
# #     "R5b",
# #     "NGs",
# #     "N1s",
# #     "N2s",
# #     "N3s",
# #     "N4s",
# #     "N5s",
# #     "NO",
# #     "NBs",
# #     "NBb",
# #     "NGb",
# #     "N3b",
# #     "N4b",
# #     "N5b",
# #     "PGs",
# #     "P1s",
# #     "P2s",
# #     "P3s",
# #     "P4s",
# #     "P5s",
# #     "PO",
# #     "PBs",
# #     "PBb",
# #     "PGb",
# #     "P3b",
# #     "P4b",
# #     "P5b",
# # ]

# COLOR_NAMES = ["B","R","N","P"]
# CLASS_NAMES = [
#     "Gs",
#     "1b",
#     "2s",
#     "3s",
#     "4s",
#     "5s",
#     "O",
#     "Bs",
#     "Bb",
#     "Gb",
#     "3b",
#     "4b",
#     "5b",
# ]

# def load_train_cfg(path: Path = TRAIN_CFG) -> dict:
#     """Load training configuration YAML."""
#     if not path.exists():
#         raise FileNotFoundError(f"Training config not found: {path}")
#     cfg = yaml.safe_load(path.read_text()) or {}
#     if not isinstance(cfg, dict):
#         raise ValueError(f"Invalid config structure in {path}, expected a mapping.")
#     return cfg


# def infer_dataset_info(labels_dir: Path) -> tuple[set[int], tuple[int, int]]:
#     """Infer present classes and keypoint shape from the raw label files (cls kpt...)."""
#     label_files = sorted(labels_dir.glob("*.txt"))
#     if not label_files:
#         raise FileNotFoundError(f"No label files found under {labels_dir}")

#     classes_present: set[int] = set()
#     num_kpts, ndim = None, None
#     for lf in label_files:
#         for line in lf.read_text().splitlines():
#             parts = line.strip().split()
#             if not parts:
#                 continue
#             cls = int(float(parts[0]))
# #             classes_present.add(cls)
#             coords = len(parts) - 1
#             if coords <= 0:
#                 continue
#             # Resolve dims from the first line, then enforce consistency.
#             if ndim is None:
#                 if coords % 3 == 0:
#                     ndim = 3
#                 elif coords % 2 == 0:
#                     ndim = 2
#                 else:
#                     raise ValueError(f"Inconsistent keypoint dims in {lf}: {coords} coords")
#                 num_kpts = coords // ndim
#             else:
#                 if coords % ndim != 0:
#                     raise ValueError(f"Inconsistent keypoint dims in {lf}: {coords} coords")
#                 kpts_here = coords // ndim
#                 if kpts_here != num_kpts:
#                     raise ValueError(f"Inconsistent keypoint count in {lf}: {kpts_here} vs {num_kpts}")

#     if not classes_present:
#         raise RuntimeError("No non-empty labels found to infer classes.")
#     max_cls = max(classes_present)
#     if max_cls >= len(CLASS_NAMES):
#         raise ValueError(
#             f"Label class {max_cls} exceeds provided names ({len(CLASS_NAMES)}). Update CLASS_NAMES to match."
#         )
#     if num_kpts is None or ndim is None:
#         raise RuntimeError("Failed to infer dataset info (classes/keypoints).")

#     return classes_present, (num_kpts, ndim) ,


# def make_splits(img_dir: Path, train_ratio: float = 0.9, seed: int = 0) -> tuple[Path, Path]:
#     """Create deterministic train/val splits (by writing txt lists) without moving files."""
#     valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
#     images = [p.resolve() for p in img_dir.iterdir() if p.suffix.lower() in valid_exts]
#     if not images:
#         raise FileNotFoundError(f"No images found under {img_dir}")

#     rng = random.Random(seed)
#     rng.shuffle(images)

#     if len(images) < 2:
#         train_imgs = val_imgs = images
#     else:
#         n_train = max(1, int(len(images) * train_ratio))
#         if n_train == len(images):  # ensure at least one val image
#             n_train = len(images) - 1
#         train_imgs = images[:n_train]
#         val_imgs = images[n_train:]

#     SPLIT_DIR.mkdir(parents=True, exist_ok=True)
#     train_txt = SPLIT_DIR / "train.txt"
#     val_txt = SPLIT_DIR / "val.txt"
#     train_txt.write_text("\n".join(map(str, train_imgs)), encoding="utf-8")
#     val_txt.write_text("\n".join(map(str, val_imgs)), encoding="utf-8")
#     return train_txt, val_txt


# def write_data_yaml(names: dict[int, str], kpt_shape: tuple[int, int], train_txt: Path, val_txt: Path) -> Path:
#     """Materialize a minimal pose data.yaml pointing to the generated split files."""
#     data = {
#         "path": str(ROOT),
#         "train": str(train_txt),
#         "val": str(val_txt),
#         "kpt_shape": list(kpt_shape),
#         "names": {int(k): v for k, v in names.items()},
#     }
#     DATA_YAML.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
#     return DATA_YAML

# def main() -> None:
#     cfg = load_train_cfg()

#     # 是否启用 TensorBoard
#     if cfg.get("tensorboard", True):
#         SETTINGS.update({"tensorboard": True})

#     # 从标签自动推断类别与关键点形状
#     classes_present, kpt_shape = infer_dataset_info(LBL_DIR)
#     missing = [i for i in range(len(CLASS_NAMES)) if i not in classes_present]
#     if missing:
#         print(
#             f"Note: {len(missing)} classes not present in labels (kept anyway): "
#             f"{missing[:10]}{'...' if len(missing) > 10 else ''}"
#         )
#     names = {i: name for i, name in enumerate(CLASS_NAMES)}

#     # 生成 train/val 列表
#     train_txt, val_txt = make_splits(
#         IMG_DIR,
#         train_ratio=float(cfg.get("train_ratio", 0.9)),
#         seed=int(cfg.get("seed", 0)),
#     )
#     data_yaml = write_data_yaml(names, kpt_shape, train_txt, val_txt)

#     # 模型，可换成 .pt 微调
#     model = YOLO(cfg.get("model", "ultralytics/cfg/models/11/yolo11-pose.yaml"))

#     # ----------------- 训练参数大集合 -----------------
#     model.train(
#         # 基本数据设置
#         data=str(data_yaml),
#         epochs=int(cfg.get("epochs", 50)),
#         imgsz=int(cfg.get("imgsz", 640)),
#         batch=int(cfg.get("batch", 16)),
#         device=cfg.get("device", 0),
#         workers=int(cfg.get("workers", 8)),
#         cache=cfg.get("cache", False),

#         # 优化器与学习率相关
#         optimizer=cfg.get("optimizer", "AdamW"),   # 'SGD', 'Adam', 'AdamW'
#         lr0=float(cfg.get("lr0", 1e-3)),          # 初始学习率
#         lrf=float(cfg.get("lrf", 0.01)),          # 最终学习率比例
#         momentum=float(cfg.get("momentum", 0.937)),
#         weight_decay=float(cfg.get("weight_decay", 5e-4)),
#         warmup_epochs=float(cfg.get("warmup_epochs", 3.0)),
#         warmup_momentum=float(cfg.get("warmup_momentum", 0.8)),
#         warmup_bias_lr=float(cfg.get("warmup_bias_lr", 0.1)),
#         cos_lr=bool(cfg.get("cos_lr", True)),     # 余弦退火

#         # pose 相关 loss 权重（官方推荐：pose=12.0, kobj=2.0）
#         box=float(cfg.get("box", 7.5)),
#         cls=float(cfg.get("cls", 0.5)),
#         dfl=float(cfg.get("dfl", 1.5)),
#         pose=float(cfg.get("pose", 12.0)),
#         kobj=float(cfg.get("kobj", 2.0)),

#         # 数据增强超参数（可按需要精调）
#         hsv_h=float(cfg.get("hsv_h", 0.015)),
#         hsv_s=float(cfg.get("hsv_s", 0.7)),
#         hsv_v=float(cfg.get("hsv_v", 0.4)),
#         degrees=float(cfg.get("degrees", 0.0)),
#         translate=float(cfg.get("translate", 0.1)),
#         scale=float(cfg.get("scale", 0.5)),
#         shear=float(cfg.get("shear", 0.0)),
#         perspective=float(cfg.get("perspective", 0.0)),
#         flipud=float(cfg.get("flipud", 0.0)),
#         fliplr=float(cfg.get("fliplr", 0.5)),
#         mosaic=float(cfg.get("mosaic", 1.0)),
#         mixup=float(cfg.get("mixup", 0.0)),
#         copy_paste=float(cfg.get("copy_paste", 0.0)),
#         erasing=float(cfg.get("erasing", 0.4)),
#         close_mosaic=int(cfg.get("close_mosaic", 10)),  # 最后 N 个 epoch 关闭 mosaic

#         # 训练流程控制
#         patience=int(cfg.get("patience", 50)),     # 早停
#         resume=cfg.get("resume", False),
#         pretrained=cfg.get("pretrained", True),
#         amp=cfg.get("amp", True),                  # 混合精度
#         deterministic=cfg.get("deterministic", False),
#         seed=int(cfg.get("seed", 0)),
#         val=cfg.get("val", True),                  # 训练过程中是否做 val

#         # 日志 & 权重保存
#         project=str(cfg.get("project", ROOT / "runs" / "pose")),
#         name=str(cfg.get("name", "rmarmor-pose")),
#         exist_ok=cfg.get("exist_ok", False),
#         save=cfg.get("save", True),
#         save_period=int(cfg.get("save_period", -1)),   # 每多少 epoch 额外存一次
#         plots=cfg.get("plots", True),                  # 保存训练曲线图片
#     )


# if __name__ == "__main__":
#     main()
