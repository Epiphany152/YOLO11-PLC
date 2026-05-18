from pathlib import Path

# ============================================================
# 基础路径
# ============================================================
# vision_core/config.py 的上一级目录就是 yolo11_local
BASE_DIR = Path(__file__).resolve().parent.parent

# ============================================================
# 视觉检测配置
# ============================================================
MODEL_PATH = BASE_DIR / "pt" / "best.pt"

CAMERA_ID = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

IMGSZ = 960
CONF_THRES = 0.35
NMS_IOU_THRES = 0.50

MATCH_IOU_THRES = 0.20
USE_CENTER_MATCH = True

SMOOTH_WINDOW = 5
SMOOTH_OCCUPIED_COUNT = 3
PRINT_INTERVAL = 1.0
SLOTS_PER_LAYER = 5
ALLOWED_CLASS_IDS = None

OUTPUT_DIR = BASE_DIR / "results_test"

# ============================================================
# 参考图与标注文件
# ============================================================
REF_CONFIG = {
    "up": {
        "view_name": "up_two_layers",
        "ref_image_path": BASE_DIR / "库位置标定" / "上下位置" / "up.jpg",
        "label_path": BASE_DIR / "库位置标定" / "position_label" / "up.txt",
        "layer_ids": [1, 2],
    },
    "down": {
        "view_name": "down_two_layers",
        "ref_image_path": BASE_DIR / "库位置标定" / "上下位置" / "down.jpg",
        "label_path": BASE_DIR / "库位置标定" / "position_label" / "down.txt",
        "layer_ids": [3, 4],
    },
}

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
# DBD0    EmptyMask       DWORD，空工位位掩码：1号工位->bit0，2号工位->bit1 ... 20号工位->bit19
# DBW4    EmptyCount      INT，空工位数量
# DBW6    WriteSeq        INT，写入序号
# DBX8.0  DataValid       BOOL，数据有效
# DBX8.1  Heartbeat       BOOL，心跳位
# DBW10   VisionNumber    INT，前端选择的工位号
# DBX12.0 VisionOutbound  BOOL，前端出库请求
# DBX12.1 VisionInbound   BOOL，前端入库请求
# DBX12.2 OutboundDone    BOOL，PLC出库完成反馈
# DBX12.3 InboundDone     BOOL，PLC入库完成反馈

# ============================================================
# Web 前端配置
# ============================================================
WEB_ENABLE = True
WEB_HOST = "0.0.0.0"
WEB_PORT = 5000
FRONTEND_HTML = BASE_DIR / "frontend_vision_plc_video.html"
SHOW_LOCAL_WINDOW = False  # True=同时弹出OpenCV窗口；False=只在网页中显示
