import time
import json
from pathlib import Path
from dataclasses import dataclass
from collections import deque

import cv2
import numpy as np
import pandas as pd
import snap7
from ultralytics import YOLO

from flask import Flask, jsonify, request, send_file, Response
import threading


# ============================================================
# 1. 基本配置
# ============================================================

MODEL_PATH = Path(r"E:\Course\大三下\机器视觉理论与应用\YOLO11\yolo11_local\pt\best.pt")

CAMERA_ID = 1

# 摄像头分辨率
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

# YOLO 推理参数
IMGSZ = 960
CONF_THRES = 0.35
NMS_IOU_THRES = 0.50

# 匹配参数
MATCH_IOU_THRES = 0.20
USE_CENTER_MATCH = True

# 稳定帧数，防止检测结果一闪一闪
SMOOTH_WINDOW = 5
SMOOTH_OCCUPIED_COUNT = 3

# 每隔多少秒在终端输出一次货架状态
PRINT_INTERVAL = 1.0

# 每层货架数量
SLOTS_PER_LAYER = 5

# 如果模型只有一个类别，保持 None 即可
# 如果只检测 class_id=0，可以改成 {0}
ALLOWED_CLASS_IDS = None

OUTPUT_DIR = Path(r"E:\Course\大三下\机器视觉理论与应用\YOLO11\yolo11_local\results_test")


# ============================================================
# PLC 通信配置
# ============================================================
PLC_ENABLE = True
PLC_IP = "192.168.0.3"
PLC_RACK = 0
PLC_SLOT = 1
PLC_DB_NUMBER = 45
PLC_SEND_INTERVAL = 1.0

# DB45 地址分配：
# DBD0   EmptyMask      DWORD，空工位位掩码：1号工位->bit0，2号工位->bit1 ... 20号工位->bit19
# DBW4   EmptyCount     INT，空工位数量
# DBW6   WriteSeq       INT，写入序号
# DBX8.0 DataValid      BOOL，数据有效
# DBX8.1 Heartbeat      BOOL，心跳位
# DBW10  VisionNumber   INT，前端选择的工位号
# DBX12.0 VisionOutbound BOOL，前端出库请求
# DBX12.1 VisionInbound  BOOL，前端入库请求
# DBX12.2 OutboundDone   BOOL，PLC出库完成反馈
# DBX12.3 InboundDone    BOOL，PLC入库完成反馈


# ============================================================
# Web 前端配置
# ============================================================
WEB_ENABLE = True
WEB_HOST = "0.0.0.0"
WEB_PORT = 5000
FRONTEND_HTML = Path(__file__).with_name("frontend_vision_plc_video.html")
SHOW_LOCAL_WINDOW = False  # True=同时弹出OpenCV窗口；False=只在网页中显示

app = Flask(__name__)
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

# 保存最近一次前端命令，用于把 PLC 完成位转换成“完成提示”
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


# ============================================================
# 2. 参考图与标注文件
# ============================================================
# up.jpg + up.txt：第1、2层
# down.jpg + down.txt：第3、4层
#
# 最终统一编号：
# 第1层：1~5
# 第2层：6~10
# 第3层：11~15
# 第4层：16~20

REF_CONFIG = {
    "up": {
        "view_name": "up_two_layers",
        "ref_image_path": Path(r"E:\Course\大三下\机器视觉理论与应用\YOLO11\yolo11_local\库位置标定\上下位置\up.jpg"),
        "label_path": Path(r"E:\Course\大三下\机器视觉理论与应用\YOLO11\yolo11_local\库位置标定\position_label\up.txt"),
        "layer_ids": [1, 2],
    },
    "down": {
        "view_name": "down_two_layers",
        "ref_image_path": Path(r"E:\Course\大三下\机器视觉理论与应用\YOLO11\yolo11_local\库位置标定\上下位置\down.jpg"),
        "label_path": Path(r"E:\Course\大三下\机器视觉理论与应用\YOLO11\yolo11_local\库位置标定\position_label\down.txt"),
        "layer_ids": [3, 4],
    },
}


# ============================================================
# 3. 数据结构
# ============================================================

@dataclass
class SlotBox:
    view_name: str
    slot_id: str
    layer_id: int
    order_in_layer: int
    cls_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    ref_w: int
    ref_h: int

    @property
    def cx(self):
        return (self.x1 + self.x2) / 2

    @property
    def cy(self):
        return (self.y1 + self.y2) / 2

    def xyxy(self):
        return np.array([self.x1, self.y1, self.x2, self.y2], dtype=np.float32)


@dataclass
class DetBox:
    det_id: int
    cls_id: int
    conf: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def cx(self):
        return (self.x1 + self.x2) / 2

    @property
    def cy(self):
        return (self.y1 + self.y2) / 2

    def xyxy(self):
        return np.array([self.x1, self.y1, self.x2, self.y2], dtype=np.float32)


# ============================================================
# 4. 中文路径图像读取与保存
# ============================================================

def imread_unicode(image_path: Path):
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"图片不存在：{image_path}")

    data = np.fromfile(str(image_path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)

    if img is None:
        raise RuntimeError(f"图片读取失败：{image_path}")

    return img


def imwrite_unicode(save_path: Path, img):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    ok, data = cv2.imencode(save_path.suffix, img)
    if not ok:
        raise RuntimeError(f"图片编码失败：{save_path}")

    data.tofile(str(save_path))


# ============================================================
# 5. 工具函数
# ============================================================

def check_files():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"模型文件不存在：{MODEL_PATH}")

    for key, cfg in REF_CONFIG.items():
        if not cfg["ref_image_path"].exists():
            raise FileNotFoundError(f"参考图不存在：{cfg['ref_image_path']}")

        if not cfg["label_path"].exists():
            raise FileNotFoundError(f"标注文件不存在：{cfg['label_path']}")


def clamp_box(x1, y1, x2, y2, w, h):
    x1 = max(0, min(float(x1), w - 1))
    y1 = max(0, min(float(y1), h - 1))
    x2 = max(0, min(float(x2), w - 1))
    y2 = max(0, min(float(y2), h - 1))
    return x1, y1, x2, y2


def bbox_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union_area = area_a + area_b - inter_area

    if union_area <= 0:
        return 0.0

    return inter_area / union_area


def point_in_box(x, y, box):
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2


# ============================================================
# 6. 读取 YOLO 标注，生成参考货位
# ============================================================

def read_yolo_label_file(label_path: Path, img_w: int, img_h: int):
    """
    YOLO txt格式：
    class_id x_center y_center width height
    坐标为归一化坐标。
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
    固定规则：
    每张参考图必须有10个货位框。
    按 y 坐标排序：
        前5个 = 上面一行
        后5个 = 下面一行
    每一行内部再按 x 坐标从左到右编号。

    up 图：
        第一行 -> 1~5
        第二行 -> 6~10

    down 图：
        第三行 -> 11~15
        第四行 -> 16~20
    """
    if len(layer_ids) != 2:
        raise ValueError("当前程序默认一张参考图对应两层货架。")

    expected_num = SLOTS_PER_LAYER * 2

    if len(raw_boxes) != expected_num:
        raise ValueError(
            f"当前参考图读取到 {len(raw_boxes)} 个标注框，但程序要求每张参考图必须是 {expected_num} 个。\n"
            f"请检查 up.txt / down.txt 是否每个文件都正好标注了10个货位。"
        )

    # 先按 y 坐标从上到下排序
    sorted_by_y = sorted(raw_boxes, key=lambda b: b["cy"])

    # 固定前5个为上层，后5个为下层
    upper_boxes = sorted_by_y[:SLOTS_PER_LAYER]
    lower_boxes = sorted_by_y[SLOTS_PER_LAYER:]

    result = []

    for layer_id, layer_boxes in zip(layer_ids, [upper_boxes, lower_boxes]):
        # 同一层按 x 坐标从左到右排序
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
            # 临时编号，后面会统一改成1~20
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
    """
    将所有参考货位合并，并统一编号为 1~20。

    编号规则：
        第1层：1~5
        第2层：6~10
        第3层：11~15
        第4层：16~20
    """
    all_slots = []

    for key, slots in reference_slots.items():
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
    """
    将参考图坐标缩放到当前摄像头帧坐标。
    只能处理分辨率不同，不能处理相机角度变化。
    """
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


# ============================================================
# 7. YOLO 实时检测
# ============================================================

def run_yolo_detection(model, frame):
    results = model.predict(
        source=frame,
        imgsz=IMGSZ,
        conf=CONF_THRES,
        iou=NMS_IOU_THRES,
        verbose=False
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


# ============================================================
# 8. 检测框与货位框匹配
# ============================================================

def match_detections_to_slots(slots, dets):
    """
    一对一匹配：
    一个检测框最多匹配一个货位；
    一个货位最多匹配一个检测框。
    """
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

            # 中心点落入货位框时优先级更高
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


# ============================================================
# 9. 平滑状态，防止闪烁
# ============================================================

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


# ============================================================
# 10. 结果整理与终端输出
# ============================================================

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


# ============================================================
# 11. PLC 通信
# ============================================================

class PLCWriter:
    def __init__(self, ip, rack, slot, db_number):
        self.ip = ip
        self.rack = rack
        self.slot = slot
        self.db_number = db_number
        self.client = snap7.client.Client()
        self.connected = False
        self.write_seq = 0
        self.heartbeat = False
        self.lock = threading.RLock()

    def connect(self):
        with self.lock:
            if self.connected:
                return

            self.client.connect(self.ip, self.rack, self.slot)

            if not self.client.get_connected():
                raise RuntimeError(f"PLC连接失败：{self.ip}")

            self.connected = True
            print(f"PLC已连接：{self.ip}, DB{self.db_number}")

    def close(self):
        with self.lock:
            try:
                if self.connected:
                    self.client.disconnect()
            finally:
                self.connected = False

    @staticmethod
    def build_empty_mask(empty_list):
        mask = 0

        for shelf_no in empty_list:
            shelf_no = int(shelf_no)

            if 1 <= shelf_no <= 20:
                mask |= 1 << (shelf_no - 1)

        return mask

    def write_empty_shelves(self, empty_list):
        """
        周期写视觉检测状态到 DB45：
            DBD0  EmptyMask
            DBW4  EmptyCount
            DBW6  WriteSeq
            DBX8.0 DataValid
            DBX8.1 Heartbeat
        """
        with self.lock:
            self.connect()

            empty_mask = self.build_empty_mask(empty_list)
            empty_count = len(empty_list)

            self.write_seq += 1
            if self.write_seq > 32767:
                self.write_seq = 1

            self.heartbeat = not self.heartbeat

            data = bytearray(10)

            data[0:4] = int(empty_mask).to_bytes(4, byteorder="big", signed=False)
            data[4:6] = int(empty_count).to_bytes(2, byteorder="big", signed=True)
            data[6:8] = int(self.write_seq).to_bytes(2, byteorder="big", signed=True)

            # DBX8.0 = DataValid
            data[8] |= 1 << 0

            # DBX8.1 = Heartbeat
            if self.heartbeat:
                data[8] |= 1 << 1

            self.client.db_write(self.db_number, 0, data)

            print(
                f"已写入PLC状态：IP={self.ip}, "
                f"empty_list={empty_list}, "
                f"empty_mask={empty_mask}, "
                f"empty_count={empty_count}, "
                f"write_seq={self.write_seq}"
            )

    def send_vision_command(self, action, station_no):
        """
        前端指令写入 DB45。

        DB45.DBW10   VisionNumber
        DB45.DBX12.0 VisionOutbound
        DB45.DBX12.1 VisionInbound

        action:
            "outbound" = 出库
            "inbound"  = 入库
        """
        station_no = int(station_no)

        if station_no < 1 or station_no > 20:
            raise ValueError(f"工位号非法：{station_no}")

        if action not in {"outbound", "inbound"}:
            raise ValueError(f"未知动作类型：{action}")

        with self.lock:
            self.connect()

            # 1. 先写工位号：DB45.DBW10
            number_data = int(station_no).to_bytes(2, byteorder="big", signed=True)
            self.client.db_write(self.db_number, 10, number_data)

            # 2. 再置位请求信号：DB45.DBX12.0 或 DB45.DBX12.1
            byte_data = self.client.db_read(self.db_number, 12, 1)

            if action == "outbound":
                byte_data[0] |= 1 << 0       # DBX12.0 = 1
                action_cn = "出库"
            else:
                byte_data[0] |= 1 << 1       # DBX12.1 = 1
                action_cn = "入库"

            self.client.db_write(self.db_number, 12, byte_data)

            print(f"已发送{action_cn}指令：DB{self.db_number}.DBW10={station_no}")

    def read_done_flags(self):
        """
        读取 DB45 完成反馈：
            DB45.DBX12.2 = 出库完成
            DB45.DBX12.3 = 入库完成
        """
        with self.lock:
            self.connect()
            byte_data = self.client.db_read(self.db_number, 12, 1)
            value = int(byte_data[0])

            return {
                "outbound_done": bool(value & (1 << 2)),
                "inbound_done": bool(value & (1 << 3)),
            }


# ============================================================
# 12. Web 接口
# ============================================================

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

        # 等到完成位至少出现过一次 False，再接受下一次 True 作为新完成
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
    """把检测后的OpenCV画面编码为JPEG，供前端 /video_feed 显示。"""
    global latest_jpeg

    if vis_frame is None:
        return

    ok, buffer = cv2.imencode(".jpg", vis_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if not ok:
        return

    with video_lock:
        latest_jpeg = buffer.tobytes()


def mjpeg_generator():
    """MJPEG视频流生成器。"""
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


@app.route("/")
def index_page():
    if not FRONTEND_HTML.exists():
        return "frontend_vision_plc_video.html 不存在", 404
    return send_file(str(FRONTEND_HTML))


@app.route("/api/status", methods=["GET"])
def api_status():
    with status_lock:
        data = dict(current_status)
    return jsonify(data)


@app.route("/video_feed")
def video_feed():
    return Response(
        mjpeg_generator(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/api/command", methods=["POST"])
def api_command():
    global plc_writer_global

    data = request.get_json(silent=True) or {}
    action = data.get("action")
    station_no = data.get("station_no")

    if action not in {"inbound", "outbound"}:
        return jsonify({"ok": False, "msg": "action必须是 inbound 或 outbound"}), 400

    try:
        station_no = int(station_no)
    except Exception:
        return jsonify({"ok": False, "msg": "工位号必须是整数"}), 400

    if station_no < 1 or station_no > 20:
        return jsonify({"ok": False, "msg": "工位号必须在1~20之间"}), 400

    with status_lock:
        occupied_list = list(current_status.get("occupied_shelves", []))
        empty_list = list(current_status.get("empty_shelves", []))

    if action == "inbound" and station_no not in empty_list:
        return jsonify({"ok": False, "msg": f"{station_no}号工位当前不是空位，不能入库"}), 400

    if action == "outbound" and station_no not in occupied_list:
        return jsonify({"ok": False, "msg": f"{station_no}号工位当前不是非空位，不能出库"}), 400

    if plc_writer_global is None:
        return jsonify({"ok": False, "msg": "PLC对象未初始化"}), 500

    try:
        plc_writer_global.send_vision_command(action, station_no)
    except Exception as e:
        return jsonify({"ok": False, "msg": f"PLC写入失败：{e}"}), 500

    # 新命令发出后，清空上一条完成提示，并等待本次命令的完成反馈
    with status_lock:
        current_status["done_message"] = ""
        current_status["done_action"] = ""
        current_status["done_station_no"] = None

        command_state["pending_action"] = action
        command_state["pending_station_no"] = station_no
        command_state["awaiting_done"] = True

        current_done = bool(current_status.get(f"{action}_done", False))
        command_state["done_seen_low"][action] = not current_done

    action_cn = "入库" if action == "inbound" else "出库"
    return jsonify({"ok": True, "msg": f"已发送{action_cn}指令：{station_no}号工位"})


def run_web_server():
    app.run(host=WEB_HOST, port=WEB_PORT, debug=False, use_reloader=False, threaded=True)


# ============================================================
# 13. 可视化
# ============================================================

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
            cv2.LINE_AA
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
            cv2.LINE_AA
        )

    # 左上角信息
    cv2.putText(
        vis,
        "Shelf 1-20 Realtime Detection | S save | Q quit",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.78,
        (0, 255, 255),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        vis,
        f"FPS: {fps:.2f}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.78,
        (0, 255, 255),
        2,
        cv2.LINE_AA
    )

    # 右侧状态栏
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
        cv2.LINE_AA
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
            cv2.LINE_AA
        )

        y += 25

    # 底部汇总
    occupied_list, empty_list = get_status_lists(slots, smooth_status)

    cv2.putText(
        vis,
        f"Occupied: {occupied_list}",
        (20, vis.shape[0] - 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        vis,
        f"Empty: {empty_list}",
        (20, vis.shape[0] - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 0, 255),
        2,
        cv2.LINE_AA
    )

    return vis


# ============================================================
# 12. 保存当前检测结果
# ============================================================

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
                round(slot.y2, 2)
            ],
        })

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    occupied_list, empty_list = get_status_lists(slots, smooth_status)

    json_data = {
        "timestamp": timestamp,
        "occupied_shelves": occupied_list,
        "empty_shelves": empty_list,
        "details": rows
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)

    print("\n已保存当前检测结果：")
    print(f"图片：{image_path}")
    print(f"CSV ：{csv_path}")
    print(f"JSON：{json_path}")


# ============================================================
# 13. 主程序
# ============================================================

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
    model = YOLO(str(MODEL_PATH))

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

    global plc_writer_global

    plc_writer = None
    if PLC_ENABLE:
        plc_writer = PLCWriter(
            ip=PLC_IP,
            rack=PLC_RACK,
            slot=PLC_SLOT,
            db_number=PLC_DB_NUMBER
        )
        plc_writer_global = plc_writer

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

    while True:
        ok, frame = cap.read()

        if not ok:
            print("读取摄像头失败")
            break

        frame_h, frame_w = frame.shape[:2]

        # 将 1~20 号参考货位缩放到当前摄像头画面尺寸
        slots = [scale_slot_to_frame(s, frame_w, frame_h) for s in all_raw_slots]

        # YOLO 检测
        dets = run_yolo_detection(model, frame)

        # 匹配检测框与 1~20 号货位
        matches = match_detections_to_slots(slots, dets)

        # 平滑占用状态
        smooth_status = update_smooth_status(status_history, slots, matches)

        # FPS
        now = time.time()
        cur_fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now

        if fps == 0:
            fps = cur_fps
        else:
            fps = 0.9 * fps + 0.1 * cur_fps

        occupied_list, empty_list = get_status_lists(slots, smooth_status)

        # 每隔 PLC_SEND_INTERVAL 秒把视觉状态写入 PLC，并读取出/入库完成反馈
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
                    update_done_latch(outbound_done, inbound_done)
                except Exception as e:
                    print(f"PLC完成反馈读取失败：{e}")

                last_plc_done_read_time = now

        update_web_status(
            occupied_list=occupied_list,
            empty_list=empty_list,
            fps=fps,
            plc_connected=(plc_writer.connected if plc_writer is not None else False),
            outbound_done=outbound_done,
            inbound_done=inbound_done
        )

        # 每隔 PRINT_INTERVAL 秒在终端输出一次状态
        if now - last_print_time >= PRINT_INTERVAL:
            print_shelf_status(slots, smooth_status)
            last_print_time = now

        # 绘制画面
        vis = draw_results(
            frame=frame,
            slots=slots,
            dets=dets,
            matches=matches,
            smooth_status=smooth_status,
            fps=fps
        )

        # 更新网页中的实时检测画面
        update_latest_frame(vis)

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
                        smooth_status=last_smooth_status
                    )

    if plc_writer is not None:
        plc_writer.close()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()