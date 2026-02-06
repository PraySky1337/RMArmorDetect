from ultralytics import YOLO
import albumentations as A

def main():
    model = YOLO(
        "/home/praysky/ultralytics/ultralytics/cfg/models/26/yolo26n-pose.yaml"
    )  # load a pretrained model (recommended for training)

    # ========== 自定义 Albumentations 增强 ==========
    # 注意：对于 pose 任务，只能使用非空间变换（不影响关键点坐标的变换）
    # Ultralytics 会自动处理 A.Compose() 构建
    custom_augmentations = [
        # 噪声与模糊类 - 提升对图像质量的鲁棒性
        A.GaussianBlur(blur_limit=(3, 7), p=0.2),
        A.MotionBlur(blur_limit=(3, 7), p=0.2),
        A.GaussNoise(var_limit=(10, 50), p=0.2),

        # 天气与光照类 - 提升对环境的适应性
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.CLAHE(clip_limit=4.0, p=0.3),
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),

        # 压缩与质量类 - 提升对不同编码格式的鲁棒性
        A.ImageCompression(quality_lower=70, quality_upper=100, p=0.3),
    ]


    model.train(
    data="/home/praysky/ultralytics/data.yaml",
    task="pose",
    epochs=10,
    batch=4,
    imgsz=640,
    device="0",
    workers=4,
    pretrained=True,
    seed=0,
    deterministic=True,

    # ========== 几何增强：提升泛化能力 ==========
    degrees=15.0,         # 增加旋转角度范围，提升方向鲁棒性
    translate=0.2,        # 降低平移幅度，避免物体过度移出画面
    scale=0.5,            # 增加缩放范围，模拟不同距离
    shear=0.0,            # 添加轻微错切，提升视角变化
    perspective=0.0,   # 添加轻微透视变换，模拟3D效果
    fliplr=0.0,           # 破坏语义
    flipud=0.0,           # 破坏语义

    # ========== 镶嵌增强：核心泛化技术 ==========
    mosaic=1.0,           # 保持马赛克，这是YOLO的核心增强
    close_mosaic=15,      # 最后15个epoch关闭mosaic，稳定训练
    mixup=0.5,           # 增加mixup比例，引入标签噪声提升泛化
    cutmix=0.5,          # 增加cutmix，创建遮挡场景
    multi_scale=0.5,      # 增加多尺度范围

    # ========== 颜色增强：提升光照鲁棒性 ==========
    hsv_h=0.03,           # 增加色调变化，适应不同环境光
    hsv_s=0.34,            # 增加饱和度变化，适应不同场景
    hsv_v=0.35,            # 保持明度变化，适应不同亮度
    bgr=0.0,              # 保持关闭（除非数据有通道排序问题）

    augmentations=custom_augmentations,

    # ========== loss ==========
    box=7.5,
    cls=0.5,
    dfl=1.5,
    pose=12.0,
    kobj=1.0,

    # ========== 其它 ==========
    optimizer="auto",
    lr0=0.00025,
    lrf=0.01,
    cos_lr=True,
    patience=20,
    amp=True,
    val=True,
    plots=True,
    project="/home/praysky/ultralytics/runs",
    name="train_pose_strong_aug",
    )


    model.val()


if __name__ == "__main__":
    main()
