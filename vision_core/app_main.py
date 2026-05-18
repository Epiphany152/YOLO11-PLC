import threading
import time

import cv2

from . import web_state
from .config import (
    CAMERA_HEIGHT,
    CAMERA_ID,
    CAMERA_WIDTH,
    MODEL_PATH,
    OUTPUT_DIR,
    PLC_DB_NUMBER,
    PLC_ENABLE,
    PLC_IP,
    PLC_RACK,
    PLC_SEND_INTERVAL,
    PLC_SLOT,
    PRINT_INTERVAL,
    SHOW_LOCAL_WINDOW,
    WEB_ENABLE,
    WEB_PORT,
)
from .detector import (
    get_status_lists,
    load_yolo_model,
    match_detections_to_slots,
    print_shelf_status,
    run_yolo_detection,
    update_smooth_status,
)
from .plc_client import PLCWriter
from .reference_slots import build_all_slots, check_files, load_reference_slots, scale_slot_to_frame
from .result_saver import save_current_result
from .visualizer import draw_results
from .web_server import run_web_server


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    check_files()

    print("正在加载参考货位...")
    reference_slots = load_reference_slots()
    all_raw_slots = build_all_slots(reference_slots)

    print("\n统一货架编号结果：")
    for slot in all_raw_slots:
        print(
            f"  货架 {int(slot.slot_id):02d}: "
            f"layer={slot.layer_id}, order={slot.order_in_layer}, "
            f"box=({slot.x1:.1f}, {slot.y1:.1f}, {slot.x2:.1f}, {slot.y2:.1f})"
        )

    print("\n正在加载 YOLO 模型...")
    model = load_yolo_model(MODEL_PATH)

    print("\n正在打开摄像头...")
    cap = cv2.VideoCapture(CAMERA_ID)

    if not cap.isOpened():
        raise RuntimeError(f"无法打开摄像头 {CAMERA_ID}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    prev_time = time.time()
    fps = 0.0
    last_print_time = 0.0
    last_plc_send_time = 0.0
    last_plc_done_read_time = 0.0
    outbound_done = False
    inbound_done = False

    plc_writer = None
    if PLC_ENABLE:
        plc_writer = PLCWriter(
            ip=PLC_IP,
            rack=PLC_RACK,
            slot=PLC_SLOT,
            db_number=PLC_DB_NUMBER,
        )
        web_state.set_plc_writer(plc_writer)

    if WEB_ENABLE:
        web_thread = threading.Thread(target=run_web_server, daemon=True)
        web_thread.start()
        print(f"\n前端地址：http://127.0.0.1:{WEB_PORT}")

    status_history = {}

    print("\n程序已启动：")
    print("实时输出 1~20 号货架状态")
    print("按 S：保存当前画面和检测结果")
    print("按 Q 或 ESC：退出")

    last_vis = None
    last_slots = None
    last_matches = None
    last_smooth_status = None

    try:
        while True:
            ok, frame = cap.read()

            if not ok:
                print("读取摄像头失败")
                break

            frame_h, frame_w = frame.shape[:2]

            slots = [scale_slot_to_frame(s, frame_w, frame_h) for s in all_raw_slots]
            dets = run_yolo_detection(model, frame)
            matches = match_detections_to_slots(slots, dets)
            smooth_status = update_smooth_status(status_history, slots, matches)

            now = time.time()
            cur_fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now

            if fps == 0:
                fps = cur_fps
            else:
                fps = 0.9 * fps + 0.1 * cur_fps

            occupied_list, empty_list = get_status_lists(slots, smooth_status)

            if PLC_ENABLE and plc_writer is not None:
                if now - last_plc_send_time >= PLC_SEND_INTERVAL:
                    try:
                        plc_writer.write_empty_shelves(empty_list)
                    except Exception as e:
                        print(f"PLC状态写入失败：{e}")
                    last_plc_send_time = now

                if now - last_plc_done_read_time >= 0.3:
                    try:
                        done_flags = plc_writer.read_done_flags()
                        outbound_done = done_flags["outbound_done"]
                        inbound_done = done_flags["inbound_done"]
                        web_state.update_done_latch(outbound_done, inbound_done)
                    except Exception as e:
                        print(f"PLC完成反馈读取失败：{e}")
                    last_plc_done_read_time = now

            web_state.update_web_status(
                occupied_list=occupied_list,
                empty_list=empty_list,
                fps=fps,
                plc_connected=(plc_writer.connected if plc_writer is not None else False),
                outbound_done=outbound_done,
                inbound_done=inbound_done,
            )

            if now - last_print_time >= PRINT_INTERVAL:
                print_shelf_status(slots, smooth_status)
                last_print_time = now

            vis = draw_results(
                frame=frame,
                slots=slots,
                dets=dets,
                matches=matches,
                smooth_status=smooth_status,
                fps=fps,
            )

            web_state.update_latest_frame(vis)

            if SHOW_LOCAL_WINDOW:
                cv2.imshow("Shelf Occupancy 1-20 Realtime Detection", vis)

            last_vis = vis
            last_slots = slots
            last_matches = matches
            last_smooth_status = smooth_status

            if SHOW_LOCAL_WINDOW:
                key = cv2.waitKey(1) & 0xFF

                if key == 27 or key == ord("q") or key == ord("Q"):
                    break

                elif key == ord("s") or key == ord("S"):
                    if last_vis is not None:
                        save_current_result(
                            frame=last_vis,
                            slots=last_slots,
                            matches=last_matches,
                            smooth_status=last_smooth_status,
                        )

    except KeyboardInterrupt:
        print("\n用户中断，正在退出...")

    finally:
        if plc_writer is not None:
            plc_writer.close()
        cap.release()
        cv2.destroyAllWindows()
