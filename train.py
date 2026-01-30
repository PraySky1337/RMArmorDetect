from ultralytics import YOLO


def main():
    model = YOLO(
        "/home/rry/ultralytics/ultralytics/cfg/models/26/yolo26n-pose.yaml"
    )  # load a pretrained model (recommended for training)

    model.train(
    data="/home/rry/ultralytics/data.yaml",
    task="pose",
    epochs=200,
    batch=128,
    imgsz=640,
    device="0,1",
    workers=32,
    pretrained=True,
    seed=0,
    deterministic=True,

    # ========== 几何增强：不扭形状，但制造尺度/遮挡 ==========
    degrees=3.0,          # 小角度，避免数字语义被破坏
    translate=0.2,
    scale=0.5,           
    shear=0.0,
    perspective=0.0,
    fliplr=0.0,         
    flipud=0.0,

    mosaic=1.0,
    close_mosaic=30,
    mixup=0.1,
    cutmix=0.2,          
    multi_scale=0.25,

    # ========== 颜色增强：保护 R/B/Grey/Purple ==========
    hsv_h=0.01, 
    hsv_s=0.25,
    hsv_v=0.25,
    bgr=0.0,

    # ========== loss ==========
    box=7.5,
    cls=0.5,
    dfl=1.5,
    pose=12.0,
    kobj=1.0,

    # ========== 其它 ==========
    optimizer="auto",
    lr0=0.0025,
    lrf=0.01,
    cos_lr=True,
    patience=100,
    amp=True,
    val=True,
    plots=True,
    project="/home/rry/ultralytics/runs",
    name="train_pose_strong_aug",
    )


    model.val()


if __name__ == "__main__":
    main()
