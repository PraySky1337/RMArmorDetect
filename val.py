from ultralytics import YOLO

model = YOLO(
    "/home/rry/ultralytics/runs/train_pose_strong_aug2/weights/best.pt"
)

metrics = model.val(
    data="/home/rry/ultralytics/data.yaml",
    task="pose",
    imgsz=640,
    device="0,1",
)
