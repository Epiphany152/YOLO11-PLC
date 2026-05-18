import json
import time

import pandas as pd

from .config import OUTPUT_DIR
from .detector import get_status_lists
from .image_io import imwrite_unicode


def save_current_result(frame, slots, matches, smooth_status):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    save_dir = OUTPUT_DIR / timestamp
    save_dir.mkdir(parents=True, exist_ok=True)

    image_path = save_dir / "realtime_shelf_1_20.jpg"
    csv_path = save_dir / "status_shelf_1_20.csv"
    json_path = save_dir / "status_shelf_1_20.json"

    imwrite_unicode(image_path, frame)

    rows = []

    for slot in sorted(slots, key=lambda s: int(s.slot_id)):
        occupied = smooth_status.get(slot.slot_id, False)
        match = matches.get(slot.slot_id)

        if match is None:
            det_id = None
            det_conf = None
            match_iou = None
            center_inside = None
        else:
            det = match["det"]
            det_id = det.det_id
            det_conf = det.conf
            match_iou = match["iou"]
            center_inside = match["center_inside"]

        rows.append({
            "shelf_no": int(slot.slot_id),
            "layer_id": slot.layer_id,
            "order_in_layer": slot.order_in_layer,
            "status_cn": "有货" if occupied else "空闲",
            "status_en": "occupied" if occupied else "empty",
            "matched_det_id": det_id,
            "matched_conf": None if det_conf is None else round(det_conf, 4),
            "matched_iou": None if match_iou is None else round(match_iou, 4),
            "center_inside": center_inside,
            "slot_box_xyxy": [
                round(slot.x1, 2),
                round(slot.y1, 2),
                round(slot.x2, 2),
                round(slot.y2, 2),
            ],
        })

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    occupied_list, empty_list = get_status_lists(slots, smooth_status)

    json_data = {
        "timestamp": timestamp,
        "occupied_shelves": occupied_list,
        "empty_shelves": empty_list,
        "details": rows,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)

    print("\n已保存当前检测结果：")
    print(f"图片：{image_path}")
    print(f"CSV ：{csv_path}")
    print(f"JSON：{json_path}")
