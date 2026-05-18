import cv2

from .detector import get_status_lists


def draw_results(frame, slots, dets, matches, smooth_status, fps):
    vis = frame.copy()

    # 画 YOLO 检测框
    for det in dets:
        x1, y1, x2, y2 = map(int, [det.x1, det.y1, det.x2, det.y2])

        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 200, 0), 2)
        cv2.putText(
            vis,
            f"det {det.det_id} {det.conf:.2f}",
            (x1, max(20, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 200, 0),
            2,
            cv2.LINE_AA,
        )

    # 画 1~20 号货位框
    for slot in slots:
        occupied = smooth_status.get(slot.slot_id, False)

        if occupied:
            color = (0, 255, 0)
            status_text = "OCC"
        else:
            color = (0, 0, 255)
            status_text = "EMPTY"

        x1, y1, x2, y2 = map(int, [slot.x1, slot.y1, slot.x2, slot.y2])
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 3)
        cv2.putText(
            vis,
            f"{int(slot.slot_id):02d} {status_text}",
            (x1, max(25, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            color,
            2,
            cv2.LINE_AA,
        )

    cv2.putText(
        vis,
        "Shelf 1-20 Realtime Detection | S save | Q quit",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.78,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        vis,
        f"FPS: {fps:.2f}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.78,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    panel_x = max(20, vis.shape[1] - 330)
    panel_y = 110

    cv2.putText(
        vis,
        "Shelf Status 1-20",
        (panel_x, panel_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    y = panel_y + 35
    for slot in sorted(slots, key=lambda s: int(s.slot_id)):
        occupied = smooth_status.get(slot.slot_id, False)
        text = f"{int(slot.slot_id):02d}: {'Occupied' if occupied else 'Empty'}"
        color = (0, 255, 0) if occupied else (0, 0, 255)

        cv2.putText(
            vis,
            text,
            (panel_x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            color,
            2,
            cv2.LINE_AA,
        )
        y += 25

    occupied_list, empty_list = get_status_lists(slots, smooth_status)

    cv2.putText(
        vis,
        f"Occupied: {occupied_list}",
        (20, vis.shape[0] - 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        vis,
        f"Empty: {empty_list}",
        (20, vis.shape[0] - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )

    return vis
