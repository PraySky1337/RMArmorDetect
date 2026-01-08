"""Armor Detection Training Script using armor_detect framework."""

from __future__ import annotations

import random
from datetime import datetime
from pathlib import Path

import yaml
from ultralytics.utils import SETTINGS

from armor_detect.trainers import ArmorTrainer

SETTINGS.update({
    "datasets_dir": "/home/rry/RMArmorDetect/datasets",
    "weights_dir": "/home/rry/RMArmorDetect/weights",
    "runs_dir": "/home/rry/RMArmorDetect/runs",
})

ROOT = Path(__file__).resolve().parent
IMG_DIR = ROOT / "datasets" / "images"
LBL_DIR = ROOT / "datasets" / "label"
SPLIT_DIR = ROOT / "datasets" / "splits"
DATA_YAML = ROOT / "datasets" / "rmarmor-pose.yaml"
TRAIN_CFG = ROOT / "train_config.yaml"
MODEL_CFG = ROOT / "armor_detect" / "configs" / "armor_yolo11.yaml"

COLOR_NAMES = ["B", "R", "N", "P"]
CLASS_NAMES = [
    "G",
    "1",
    "2",
    "3",
    "4",
    "5",
    "O",
    "B",
    
]
SIZE_NAMES = ["s","b"]


def load_train_cfg(path: Path = TRAIN_CFG) -> dict:
    """Load training configuration YAML."""
    if not path.exists():
        raise FileNotFoundError(f"Training config not found: {path}")
    cfg = yaml.safe_load(path.read_text()) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid config structure in {path}, expected a mapping.")
    return cfg


def infer_dataset_info(labels_dir: Path) -> tuple[set[int], tuple[int, int], set[int], set[int]]:
    """Infer classes, keypoints shape, color classes, and size classes from label files.

    Label format: color size cls x1 y1 x2 y2 x3 y3 x4 y4
    """
    label_files = sorted(labels_dir.glob("*.txt"))
    if not label_files:
        raise FileNotFoundError(f"No label files found under {labels_dir}")

    classes_present: set[int] = set()
    colors_present: set[int] = set()
    sizes_present: set[int] = set()
    num_kpts, ndim = None, None

    for lf in label_files:
        for line in lf.read_text().splitlines():
            parts = line.strip().split()
            if not parts:
                continue
            color = int(float(parts[0]))  # Column 0: color
            size = int(float(parts[1]))   # Column 1: size
            cls = int(float(parts[2]))    # Column 2: cls (armor number)
            classes_present.add(cls)
            colors_present.add(color)
            sizes_present.add(size)

            coords = len(parts) - 3  # Subtract color, size, cls
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
                kpts_here= coords // ndim
                if kpts_here != num_kpts:
                    raise ValueError(f"Inconsistent keypoint count in {lf}: {kpts_here} vs {num_kpts}")

    return classes_present, (num_kpts, ndim), colors_present, sizes_present


def make_splits(img_dir: Path, train_ratio: float = 0.9, seed: int = 0) -> tuple[Path, Path]:
    """Create deterministic train/val splits without moving files."""
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
    """Write pose + color + size data.yaml."""
    data = {
        "path": str(ROOT),
        "train": str(train_txt),
        "val": str(val_txt),
        "kpt_shape": list(kpt_shape),
        "names": {int(k): v for k, v in names.items()},
        "color_names": COLOR_NAMES,
        "size_names": SIZE_NAMES,
    }
    DATA_YAML.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return DATA_YAML


def main() -> None:
    """Main training entry point using ArmorTrainer."""
    cfg = load_train_cfg()

    if cfg.get("tensorboard", True):
        SETTINGS.update({"tensorboard": True})

    # Generate unique run name with timestamp
    base_name = cfg.get("name", "armor")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{base_name}_{timestamp}"

    # Infer dataset info from labels
    classes_present, kpt_shape, _colors_present, _sizes_present = infer_dataset_info(LBL_DIR)
    missing = [i for i in range(len(CLASS_NAMES)) if i not in classes_present]
    if missing:
        print(f"Note: {len(missing)} classes not present in labels: {missing[:10]}{'...' if len(missing) > 10 else ''}")

    names = {i: name for i, name in enumerate(CLASS_NAMES)}

    # Create train/val splits
    train_txt, val_txt = make_splits(
        IMG_DIR,
        train_ratio=float(cfg.get("train_ratio", 0.91)),
        seed=int(cfg.get("seed", 0)),
    )
    data_yaml = write_data_yaml(names, kpt_shape, train_txt, val_txt)

    # Build training overrides from config
    overrides = {
        # Model and data
        "model": str(MODEL_CFG),
        "data": str(data_yaml),
        # Basic training settings
        "epochs": int(cfg.get("epochs", 200)),
        "imgsz": int(cfg.get("imgsz", 416)),
        "batch": int(cfg.get("batch", 256)),
        "device": cfg.get("device", "0,1"),
        "workers": int(cfg.get("workers", 8)),
        "cache": cfg.get("cache", False),
        # Optimizer
        "optimizer": cfg.get("optimizer", "AdamW"),
        "lr0": float(cfg.get("lr0", 0.00025)),
        "lrf": float(cfg.get("lrf", 0.01)),
        "momentum": float(cfg.get("momentum", 0.937)),
        "weight_decay": float(cfg.get("weight_decay", 0.0008)),
        "warmup_epochs": float(cfg.get("warmup_epochs", 5)),
        "warmup_momentum": float(cfg.get("warmup_momentum", 0.8)),
        "warmup_bias_lr": float(cfg.get("warmup_bias_lr", 0.1)),
        "cos_lr": bool(cfg.get("cos_lr", True)),
        # Data augmentation
        "hsv_h": float(cfg.get("hsv_h", 0.08)),
        "hsv_s": float(cfg.get("hsv_s", 0.9)),
        "hsv_v": float(cfg.get("hsv_v", 0.9)),
        "degrees": float(cfg.get("degrees", 15.0)),
        "translate": float(cfg.get("translate", 0.2)),
        "scale": float(cfg.get("scale", 0.7)),
        "shear": float(cfg.get("shear", 0.0)),
        "perspective": float(cfg.get("perspective", 0.0)),
        "flipud": float(cfg.get("flipud", 0.0)),
        "fliplr": float(cfg.get("fliplr", 0.0)),
        "mosaic": float(cfg.get("mosaic", 1.0)),
        "mixup": float(cfg.get("mixup", 0.1)),
        "copy_paste": float(cfg.get("copy_paste", 0.0)),
        "erasing": float(cfg.get("erasing", 0.5)),
        "close_mosaic": int(cfg.get("close_mosaic", 150)),
        # Training control
        "patience": int(cfg.get("patience", 50)),
        "resume": cfg.get("resume", False),
        "pretrained": cfg.get("pretrained", True),
        "amp": cfg.get("amp", True),
        "deterministic": cfg.get("deterministic", False),
        "seed": int(cfg.get("seed", 0)),
        "val": cfg.get("val", True),
        # Logging and saving
        "project": str(cfg.get("project", ROOT / "runs")),
        "name": run_name,
        "exist_ok": False,
        "save": cfg.get("save", True),
        "save_period": int(cfg.get("save_period", -1)),
        "plots": cfg.get("plots", True),
    }

    # Create trainer and start training
    trainer = ArmorTrainer(overrides=overrides)
    trainer.train()


if __name__ == "__main__":
    main()
