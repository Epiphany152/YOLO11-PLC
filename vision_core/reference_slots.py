from pathlib import Path

from .config import MODEL_PATH, REF_CONFIG, SLOTS_PER_LAYER
from .data_types import SlotBox
from .geometry import clamp_box
from .image_io import imread_unicode


def check_files():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"模型文件不存在：{MODEL_PATH}")

    for _, cfg in REF_CONFIG.items():
        if not cfg["ref_image_path"].exists():
            raise FileNotFoundError(f"参考图不存在：{cfg['ref_image_path']}")

        if not cfg["label_path"].exists():
            raise FileNotFoundError(f"标注文件不存在：{cfg['label_path']}")


def read_yolo_label_file(label_path: Path, img_w: int, img_h: int):
    """
    YOLO txt格式：class_id x_center y_center width height，坐标为归一化坐标。
    """
    boxes = []

    with open(label_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line_idx, line in enumerate(lines):
        line = line.strip()

        if not line:
            continue

        parts = line.split()

        if len(parts) < 5:
            raise ValueError(f"标注格式错误：{label_path} 第 {line_idx + 1} 行：{line}")

        cls_id = int(float(parts[0]))
        xc = float(parts[1])
        yc = float(parts[2])
        bw = float(parts[3])
        bh = float(parts[4])

        x1 = (xc - bw / 2) * img_w
        y1 = (yc - bh / 2) * img_h
        x2 = (xc + bw / 2) * img_w
        y2 = (yc + bh / 2) * img_h

        x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2, img_w, img_h)

        boxes.append({
            "cls_id": cls_id,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "cx": (x1 + x2) / 2,
            "cy": (y1 + y2) / 2,
        })

    return boxes


def split_boxes_into_two_layers(raw_boxes, layer_ids):
    """
    每张参考图必须有10个货位框。按 y 坐标分两行，每行按 x 坐标编号。
    """
    if len(layer_ids) != 2:
        raise ValueError("当前程序默认一张参考图对应两层货架。")

    expected_num = SLOTS_PER_LAYER * 2

    if len(raw_boxes) != expected_num:
        raise ValueError(
            f"当前参考图读取到 {len(raw_boxes)} 个标注框，但程序要求每张参考图必须是 {expected_num} 个。\n"
            f"请检查 up.txt / down.txt 是否每个文件都正好标注了10个货位。"
        )

    sorted_by_y = sorted(raw_boxes, key=lambda b: b["cy"])
    upper_boxes = sorted_by_y[:SLOTS_PER_LAYER]
    lower_boxes = sorted_by_y[SLOTS_PER_LAYER:]

    result = []

    for layer_id, layer_boxes in zip(layer_ids, [upper_boxes, lower_boxes]):
        layer_boxes = sorted(layer_boxes, key=lambda b: b["cx"])

        for idx, b in enumerate(layer_boxes, start=1):
            b["layer_id"] = layer_id
            b["order_in_layer"] = idx
            result.append(b)

    return result


def load_reference_slots():
    all_slots = {}

    for key, cfg in REF_CONFIG.items():
        img = imread_unicode(cfg["ref_image_path"])
        img_h, img_w = img.shape[:2]

        raw_boxes = read_yolo_label_file(cfg["label_path"], img_w, img_h)
        layered_boxes = split_boxes_into_two_layers(raw_boxes, cfg["layer_ids"])

        slots = []

        for b in layered_boxes:
            temp_slot_id = f"L{b['layer_id']}-{b['order_in_layer']:02d}"
            slot = SlotBox(
                view_name=cfg["view_name"],
                slot_id=temp_slot_id,
                layer_id=b["layer_id"],
                order_in_layer=b["order_in_layer"],
                cls_id=b["cls_id"],
                x1=b["x1"],
                y1=b["y1"],
                x2=b["x2"],
                y2=b["y2"],
                ref_w=img_w,
                ref_h=img_h,
            )
            slots.append(slot)

        all_slots[key] = slots

    return all_slots


def build_all_slots(reference_slots):
    """合并参考货位，并统一编号为 1~20。"""
    all_slots = []

    for _, slots in reference_slots.items():
        for slot in slots:
            shelf_no = (slot.layer_id - 1) * SLOTS_PER_LAYER + slot.order_in_layer
            slot.slot_id = str(shelf_no)
            all_slots.append(slot)

    all_slots = sorted(all_slots, key=lambda s: int(s.slot_id))
    shelf_ids = [int(s.slot_id) for s in all_slots]

    if len(all_slots) != 20:
        raise ValueError(
            f"当前共读取到 {len(all_slots)} 个货位，不是20个。\n"
            f"当前编号为：{shelf_ids}\n"
            f"请检查 up.txt 和 down.txt 是否各有10个标注框。"
        )

    if shelf_ids != list(range(1, 21)):
        raise ValueError(
            f"货位编号不是1~20连续编号。\n"
            f"当前编号为：{shelf_ids}\n"
            f"请检查分层或标注文件。"
        )

    return all_slots


def scale_slot_to_frame(slot: SlotBox, frame_w: int, frame_h: int):
    """将参考图坐标缩放到当前摄像头帧坐标。"""
    sx = frame_w / slot.ref_w
    sy = frame_h / slot.ref_h

    return SlotBox(
        view_name=slot.view_name,
        slot_id=slot.slot_id,
        layer_id=slot.layer_id,
        order_in_layer=slot.order_in_layer,
        cls_id=slot.cls_id,
        x1=slot.x1 * sx,
        y1=slot.y1 * sy,
        x2=slot.x2 * sx,
        y2=slot.y2 * sy,
        ref_w=frame_w,
        ref_h=frame_h,
    )
