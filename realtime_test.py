import time
import cv2
from ultralytics import YOLO

MODEL_PATH = r"E:\Course\大三下\机器视觉理论与应用\YOLO11\yolo11_local\pt\best.pt"
CAMERA_ID = 1
IMGSZ = 640
CONF = 0.25

model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(CAMERA_ID)
if not cap.isOpened():
    raise RuntimeError(f"无法打开摄像头 {CAMERA_ID}")

# 可选：设置采集分辨率
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

prev = time.time()
fps = 0.0

while True:
    ok, frame = cap.read()
    if not ok:
        print("读取摄像头失败")
        break

    results = model.predict(
        source=frame,
        imgsz=IMGSZ,
        conf=CONF,
        verbose=False
    )

    annotated = results[0].plot()

    now = time.time()
    cur_fps = 1.0 / max(now - prev, 1e-6)
    prev = now
    fps = cur_fps if fps == 0 else (0.9 * fps + 0.1 * cur_fps)

    cv2.putText(
        annotated,
        f"FPS: {fps:.2f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )

    cv2.imshow("YOLO11 Realtime", annotated)

    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()