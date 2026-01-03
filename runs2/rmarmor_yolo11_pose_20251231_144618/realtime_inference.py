#!/usr/bin/env python3
"""实时摄像头推理脚本 - YOLO Pose 模型"""

import cv2
import numpy as np
import time
from ultralytics import YOLO


def draw_results(img, results, names):
    """绘制检测结果"""
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
        (0, 0, 128), (128, 128, 0)
    ]
    kpt_colors = [(0, 255, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0)]

    for r in results:
        boxes = r.boxes
        keypoints = r.keypoints

        if boxes is None:
            continue

        for i, box in enumerate(boxes):
            # 边界框
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            color = colors[cls % len(colors)]

            # 绘制边界框
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            # 绘制标签
            label = f'{names[cls]} {conf:.2f}'
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x1, y1 - h - 4), (x1 + w, y1), color, -1)
            cv2.putText(img, label, (x1, y1 - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # 绘制关键点
            if keypoints is not None and len(keypoints) > i:
                kpts = keypoints[i].data[0].cpu().numpy()  # shape: (num_kpts, 2 or 3)

                visible_pts = []
                for j, kpt in enumerate(kpts):
                    x, y = int(kpt[0]), int(kpt[1])
                    # 检查是否有置信度（dim=3）或只有坐标（dim=2）
                    if len(kpt) == 3:
                        c = kpt[2]
                        if c < 0.5:
                            continue
                    if x > 0 and y > 0:
                        kpt_color = kpt_colors[j % len(kpt_colors)]
                        cv2.circle(img, (x, y), 4, kpt_color, -1)
                        cv2.putText(img, str(j), (x + 5, y - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, kpt_color, 1)
                        visible_pts.append((x, y))

                # 连接关键点（四边形: 0-1-2-3-0）
                if len(visible_pts) >= 2:
                    pts = np.array(visible_pts, dtype=np.int32)
                    for k in range(len(pts)):
                        cv2.line(img, tuple(pts[k]), tuple(pts[(k + 1) % len(pts)]), (0, 255, 0), 2)

    return img


def main():
    # 配置
    model_path = "weights/best.pt"
    imgsz = 640
    conf_thres = 0.25
    camera_id = 0

    # 加载模型
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)
    names = model.names
    print(f"Classes: {names}")

    # 打开摄像头
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Error: Cannot open camera {camera_id}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("Press 'q' to quit")

    fps = 0
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Cannot read frame")
            break

        # 推理（内部已包含 letterbox 预处理）
        results = model.predict(frame, imgsz=imgsz, conf=conf_thres, verbose=False)

        # 绘制结果
        frame = draw_results(frame, results, names)

        # 计算 FPS
        curr_time = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / (curr_time - prev_time + 1e-6))
        prev_time = curr_time

        # 显示信息
        cv2.putText(frame, f"FPS: {fps:.1f} | Press 'q' to quit", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow('YOLO Pose Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
