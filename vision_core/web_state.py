import threading
import time

import cv2

status_lock = threading.Lock()
current_status = {
    "occupied_shelves": [],
    "empty_shelves": [],
    "fps": 0.0,
    "timestamp": "",
    "plc_connected": False,
    "outbound_done": False,
    "inbound_done": False,
    # 完成提示采用后端保持：检测到完成后一直保留，直到下次点击入库/出库
    "done_message": "",
    "done_action": "",
    "done_station_no": None,
    "done_seq": 0,
}

command_state = {
    "pending_action": None,
    "pending_station_no": None,
    "awaiting_done": False,
    "done_seen_low": {"outbound": True, "inbound": True},
    "done_seq": 0,
}

plc_writer_global = None

video_lock = threading.Lock()
latest_jpeg = None


def set_plc_writer(writer):
    global plc_writer_global
    plc_writer_global = writer


def update_web_status(occupied_list, empty_list, fps, plc_connected, outbound_done=None, inbound_done=None):
    with status_lock:
        current_status["occupied_shelves"] = list(map(int, occupied_list))
        current_status["empty_shelves"] = list(map(int, empty_list))
        current_status["fps"] = round(float(fps), 2)
        current_status["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
        current_status["plc_connected"] = bool(plc_connected)

        if outbound_done is not None:
            current_status["outbound_done"] = bool(outbound_done)

        if inbound_done is not None:
            current_status["inbound_done"] = bool(inbound_done)


def update_done_latch(outbound_done, inbound_done):
    """
    将 PLC 的完成反馈转换为前端可保持显示的完成提示。
    规则：
      1. 前端点击入库/出库后，后端开始等待对应完成位。
      2. 如果点击时完成位已经是 True，则必须等它先变 False，再变 True，避免旧完成信号误判。
      3. 检测到完成后，done_message 一直保留，直到下一次点击入库/出库。
    """
    raw = {
        "outbound": bool(outbound_done),
        "inbound": bool(inbound_done),
    }

    with status_lock:
        current_status["outbound_done"] = raw["outbound"]
        current_status["inbound_done"] = raw["inbound"]

        action = command_state.get("pending_action")
        if action not in {"outbound", "inbound"}:
            return

        if not raw[action]:
            command_state["done_seen_low"][action] = True
            return

        if command_state.get("awaiting_done") and command_state["done_seen_low"].get(action, False) and raw[action]:
            station_no = command_state.get("pending_station_no")
            action_cn = "出库" if action == "outbound" else "入库"

            command_state["done_seq"] += 1
            command_state["awaiting_done"] = False
            command_state["pending_action"] = None
            command_state["pending_station_no"] = None

            current_status["done_action"] = action
            current_status["done_station_no"] = station_no
            current_status["done_message"] = f"{action_cn}完成：{station_no}号工位"
            current_status["done_seq"] = command_state["done_seq"]


def update_latest_frame(vis_frame):
    """把检测后的 OpenCV 画面编码为 JPEG，供前端 /video_feed 显示。"""
    global latest_jpeg

    if vis_frame is None:
        return

    ok, buffer = cv2.imencode(".jpg", vis_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if not ok:
        return

    with video_lock:
        latest_jpeg = buffer.tobytes()


def mjpeg_generator():
    """MJPEG 视频流生成器。"""
    while True:
        with video_lock:
            frame = latest_jpeg

        if frame is None:
            time.sleep(0.05)
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        )
        time.sleep(0.03)
