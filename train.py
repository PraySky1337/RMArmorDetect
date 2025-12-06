"""Quick-start script to train a local pose model on datasets/images + datasets/labels."""

from __future__ import annotations

import random
from pathlib import Path

import yaml
from ultralytics import YOLO
from ultralytics.utils import SETTINGS


ROOT = Path(__file__).resolve().parent
IMG_DIR = ROOT / "datasets" / "images"
LBL_DIR = ROOT / "datasets" / "labels"
SPLIT_DIR = ROOT / "datasets" / "splits"
DATA_YAML = ROOT / "datasets" / "rmarmor-pose.yaml"
TRAIN_CFG = ROOT / "train_config.yaml"
CLASS_NAMES = [
    "BGs",
    "B1b",
    "B2s",
    "B3s",
    "B4s",
    "B5s",
    "BO",
    "BBs",
    "BBb",
    "BGb",
    "B3b",
    "B4b",
    "B5b",
    "RGs",
    "R1b",
    "R2s",
    "R3s",
    "R4s",
    "R5s",
    "RO",
    "RBs",
    "RBb",
    "RGb",
    "R3b",
    "R4b",
    "R5b",
    "NGs",
    "N1s",
    "N2s",
    "N3s",
    "N4s",
    "N5s",
    "NO",
    "NBs",
    "NBb",
    "NGb",
    "N3b",
    "N4b",
    "N5b",
    "PGs",
    "P1s",
    "P2s",
    "P3s",
    "P4s",
    "P5s",
    "PO",
    "PBs",
    "PBb",
    "PGb",
    "P3b",
    "P4b",
    "P5b",
]

def load_train_cfg(path: Path = TRAIN_CFG) -> dict:
    """Load training configuration YAML."""
    if not path.exists():
        raise FileNotFoundError(f"Training config not found: {path}")
    cfg = yaml.safe_load(path.read_text()) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid config structure in {path}, expected a mapping.")
    return cfg


def infer_dataset_info(labels_dir: Path) -> tuple[set[int], tuple[int, int]]:
    """Infer present classes and keypoint shape from the raw label files (cls kpt...)."""
    label_files = sorted(labels_dir.glob("*.txt"))
    if not label_files:
        raise FileNotFoundError(f"No label files found under {labels_dir}")

    classes_present: set[int] = set()
    num_kpts, ndim = None, None
    for lf in label_files:
        for line in lf.read_text().splitlines():
            parts = line.strip().split()
            if not parts:
                continue
            cls = int(float(parts[0]))
            classes_present.add(cls)
            coords = len(parts) - 1
            if coords <= 0:
                continue
            # Resolve dims from the first line, then enforce consistency.
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

    if not classes_present:
        raise RuntimeError("No non-empty labels found to infer classes.")
    max_cls = max(classes_present)
    if max_cls >= len(CLASS_NAMES):
        raise ValueError(
            f"Label class {max_cls} exceeds provided names ({len(CLASS_NAMES)}). Update CLASS_NAMES to match."
        )
    if num_kpts is None or ndim is None:
        raise RuntimeError("Failed to infer dataset info (classes/keypoints).")

    return classes_present, (num_kpts, ndim)


def make_splits(img_dir: Path, train_ratio: float = 0.9, seed: int = 0) -> tuple[Path, Path]:
    """Create deterministic train/val splits (by writing txt lists) without moving files."""
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    images = [p.resolve() for p in img_dir.iterdir() if p.suffix.lower() in valid_exts]
    if not images:
        raise FileNotFoundError(f"No images found under {img_dir}")

    rng = random.Random(seed)
    rng.shuffle(images)

    if len(images) < 2:
        train_imgs = val_imgs = images
    else:
        n_train = max(1, int(len(images) * train_ratio))
        if n_train == len(images):  # ensure at least one val image
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
    """Materialize a minimal pose data.yaml pointing to the generated split files."""
    data = {
        "path": str(ROOT),
        "train": str(train_txt),
        "val": str(val_txt),
        "kpt_shape": list(kpt_shape),
        "names": {int(k): v for k, v in names.items()},
    }
    DATA_YAML.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return DATA_YAML


def main() -> None:
    cfg = load_train_cfg()
    if cfg.get("tensorboard", True):
        SETTINGS.update({"tensorboard": True})

    classes_present, kpt_shape = infer_dataset_info(LBL_DIR)
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

    # Uses a local model config to avoid downloading weights; swap to a .pt to finetune instead.
    model = YOLO(cfg.get("model", "ultralytics/cfg/models/11/yolo11-pose.yaml"))
    model.train(
        data=str(data_yaml),
        epochs=int(cfg.get("epochs", 50)),
        imgsz=cfg.get("imgsz", 640),
        batch=cfg.get("batch", 16),
        device=cfg.get("device", 0),  # change to 'cpu' or other device index/list as needed
    )


if __name__ == "__main__":
    main()
