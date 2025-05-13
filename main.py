import os
import time
import cv2
from ultralytics import YOLO
from src.config import *
from src.screen_capture import capture_screen
from src.aim_logic import aim, shoot
from src.detection import get_centroids, select_target

model = YOLO(MODEL_PATH)

if __name__ == '__main__':
    os.makedirs(DETECTIONS_DIR, exist_ok=True)
    screen_center = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
    current_target = None

    while True:
        img, offset_left, offset_top, width, height = capture_screen()
        screen_center = (width // 2, height // 2)

        results = model(img, conf=CONFIDENCE)
        detections = results[0].boxes.xyxy.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()
        centroids = get_centroids(detections, classes)

        target = select_target(centroids, screen_center)
        if current_target and current_target in centroids:
            target = current_target
        else:
            current_target = target

        if target:
            local_x, local_y = target[0], target[1]
            absolute_x = offset_left + local_x
            absolute_y = offset_top + local_y
            print(f"[+] Aiming at local: ({local_x},{local_y}), absoluto: ({absolute_x},{absolute_y})")
            aim(absolute_x, absolute_y)
            shoot()
            cv2.line(img, screen_center, (local_x, local_y), (0, 0, 255), 2)

        annotated_frame = results[0].plot()
        combined_frame = annotated_frame.copy()
        if target:
            cv2.line(combined_frame, screen_center, (local_x, local_y), (0, 0, 255), 2)

        result_path = os.path.join(DETECTIONS_DIR, f"result_{int(time.time())}.jpg")
        cv2.imwrite(result_path, combined_frame)
        print(f"Frame salvo em: {result_path}")