from collections import deque

from ultralytics import YOLO

from .config import (
    ALLOWED_CLASS_IDS,
    CONF_THRES,
    IMGSZ,
    MATCH_IOU_THRES,
    NMS_IOU_THRES,
    SMOOTH_OCCUPIED_COUNT,
    SMOOTH_WINDOW,
    USE_CENTER_MATCH,
)
from .data_types import DetBox
from .geometry import bbox_iou, point_in_box


def load_yolo_model(model_path):
    return YOLO(str(model_path))


def run_yolo_detection(model, frame):
    results = model.predict(
        source=frame,
        imgsz=IMGSZ,
        conf=CONF_THRES,
        iou=NMS_IOU_THRES,
        verbose=False,
    )

    if len(results) == 0:
        return []

    result = results[0]

    if result.boxes is None or len(result.boxes) == 0:
        return []

    xyxy = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    clss = result.boxes.cls.cpu().numpy().astype(int)

    dets = []

    for i, (box, conf, cls_id) in enumerate(zip(xyxy, confs, clss)):
        if ALLOWED_CLASS_IDS is not None and cls_id not in ALLOWED_CLASS_IDS:
            continue

        x1, y1, x2, y2 = box.tolist()

        dets.append(
            DetBox(
                det_id=i,
                cls_id=int(cls_id),
                conf=float(conf),
                x1=float(x1),
                y1=float(y1),
                x2=float(x2),
                y2=float(y2),
            )
        )

    return dets


def match_detections_to_slots(slots, dets):
    """一对一匹配：一个检测框最多匹配一个货位；一个货位最多匹配一个检测框。"""
    candidates = []

    for det in dets:
        det_box = det.xyxy()

        for slot in slots:
            slot_box = slot.xyxy()
            iou = bbox_iou(det_box, slot_box)
            center_inside = point_in_box(det.cx, det.cy, slot_box)

            valid = False

            if USE_CENTER_MATCH and center_inside:
                valid = True

            if iou >= MATCH_IOU_THRES:
                valid = True

            if not valid:
                continue

            score = 0.0
            if center_inside:
                score += 2.0
            score += iou
            score += det.conf * 0.01

            candidates.append({
                "slot_id": slot.slot_id,
                "det_id": det.det_id,
                "score": score,
                "iou": iou,
                "center_inside": center_inside,
                "slot": slot,
                "det": det,
            })

    candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)

    matched_slots = set()
    matched_dets = set()
    matches = {}

    for c in candidates:
        slot_id = c["slot_id"]
        det_id = c["det_id"]

        if slot_id in matched_slots:
            continue

        if det_id in matched_dets:
            continue

        matched_slots.add(slot_id)
        matched_dets.add(det_id)
        matches[slot_id] = c

    return matches


def update_smooth_status(status_history, slots, matches):
    smooth_status = {}

    for slot in slots:
        occupied_now = slot.slot_id in matches

        if slot.slot_id not in status_history:
            status_history[slot.slot_id] = deque(maxlen=SMOOTH_WINDOW)

        status_history[slot.slot_id].append(occupied_now)

        occupied_count = sum(status_history[slot.slot_id])
        smooth_occupied = occupied_count >= SMOOTH_OCCUPIED_COUNT
        smooth_status[slot.slot_id] = smooth_occupied

    return smooth_status


def get_status_lists(slots, smooth_status):
    occupied_list = []
    empty_list = []

    for slot in sorted(slots, key=lambda s: int(s.slot_id)):
        shelf_no = int(slot.slot_id)

        if smooth_status.get(slot.slot_id, False):
            occupied_list.append(shelf_no)
        else:
            empty_list.append(shelf_no)

    return occupied_list, empty_list


def print_shelf_status(slots, smooth_status):
    occupied_list, empty_list = get_status_lists(slots, smooth_status)

    print("\n==============================")
    print("当前货架状态")
    print("==============================")
    print("有货货架：", occupied_list)
    print("空闲货架：", empty_list)
