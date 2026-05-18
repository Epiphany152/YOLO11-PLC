# YOLO11-PLC 视觉检测与仓储 PLC 联动系统

本项目用于实现基于 YOLO11 的货架工位视觉检测，并与 Siemens PLC、前端网页和仓储工站进行联动。

系统整体流程为：

```text
摄像头采集货架画面
        ↓
YOLO11 检测货物位置
        ↓
判断 1~20 号工位空 / 非空状态
        ↓
前端网页显示检测画面和工位状态
        ↓
用户选择入库 / 出库工位
        ↓
Python 后端通过 Snap7 写入 PLC
        ↓
PLC 控制仓储工站执行入库 / 出库
        ↓
PLC 返回完成信号，前端显示完成提示
```

---

## 1. 项目功能

本项目主要实现以下功能：

1. 摄像头实时采集货架画面。
2. 使用 YOLO11 模型检测当前货架中的工件。
3. 根据标定工位框判断 1~20 号工位的空 / 非空状态。
4. 将检测画面、空工位、非空工位显示在网页前端。
5. 用户可在前端选择：
   - 空工位进行入库；
   - 非空工位进行出库。
6. Python 后端通过 Snap7 与 PLC 通信。
7. PLC 接收视觉检测状态、入库 / 出库请求和工位号。
8. PLC 执行仓储工站控制逻辑。
9. PLC 完成动作后，前端显示入库 / 出库完成提示。

---

## 2. 项目结构

```text
yolo11_local
├── main.py
├── frontend_vision_plc_video.html
├── realtime_test.py
├── vision_plc_backend_video.py
├── README.md
├── README_模块化说明.txt
├── .gitignore
├── .gitattributes
│
├── pt
│   └── best.pt
│
├── vision_core
│   ├── __init__.py
│   ├── app_main.py
│   ├── config.py
│   ├── data_types.py
│   ├── detector.py
│   ├── geometry.py
│   ├── image_io.py
│   ├── plc_client.py
│   ├── reference_slots.py
│   ├── result_saver.py
│   ├── visualizer.py
│   ├── web_server.py
│   └── web_state.py
│
├── 库位置标定
│   ├── position_label
│   │   ├── classes.txt
│   │   ├── up.txt
│   │   └── down.txt
│   ├── 上下位置
│   │   ├── up.jpg
│   │   └── down.jpg
│   ├── WIN_20260511_15_42_47_Pro.jpg
│   ├── WIN_20260511_15_42_49_Pro (2).jpg
│   ├── WIN_20260511_15_42_49_Pro.jpg
│   ├── WIN_20260511_15_43_32_Pro.jpg
│   ├── WIN_20260511_15_43_35_Pro.jpg
│   ├── WIN_20260511_15_44_20_Pro (2).jpg
│   └── WIN_20260511_15_44_20_Pro.jpg
│
└── PLC程序
    └── 智能制造工站PLC-V2.0.6
        └── PLC 项目文件
```

---

## 3. 主要文件说明

### 3.1 程序入口

| 文件 | 作用 |
|---|---|
| `main.py` | 模块化程序入口，推荐运行该文件 |
| `vision_plc_backend_video.py` | 早期单文件版本，作为备份保留 |
| `frontend_vision_plc_video.html` | 前端网页界面 |
| `realtime_test.py` | 摄像头 / YOLO 实时检测测试文件 |

推荐使用：

```bash
python main.py
```

---

### 3.2 `vision_core` 模块说明

| 文件 | 作用 |
|---|---|
| `app_main.py` | 主循环，负责摄像头、YOLO、PLC、Web 线程调度 |
| `config.py` | 路径、摄像头、YOLO、PLC、Web 配置 |
| `data_types.py` | `SlotBox` / `DetBox` 数据结构 |
| `detector.py` | YOLO 检测、框匹配、状态平滑、空 / 非空列表 |
| `geometry.py` | IOU、点是否在框内等几何函数 |
| `image_io.py` | 支持中文路径的图片读取和保存 |
| `plc_client.py` | Snap7 PLC 读写 |
| `reference_slots.py` | 读取参考图和 YOLO 标注，生成 1~20 工位 |
| `result_saver.py` | 保存当前检测结果 |
| `visualizer.py` | 绘制检测框、工位框和状态信息 |
| `web_server.py` | Flask 后端接口 |
| `web_state.py` | 前端状态、视频流缓存、完成提示保持逻辑 |

---

## 4. 环境依赖

建议使用 Conda 环境。

主要依赖：

```bash
pip install ultralytics opencv-python numpy pandas flask python-snap7
```

如果已经安装好 YOLO11 / Ultralytics 环境，只需要补充：

```bash
pip install flask python-snap7
```

---

## 5. 运行方式

### 5.1 进入项目目录

```bash
cd /d E:\Course\大三下\机器视觉理论与应用\YOLO11\yolo11_local
```

### 5.2 启动程序

```bash
python main.py
```

### 5.3 打开前端网页

浏览器访问：

```text
http://127.0.0.1:5000
```

不要直接双击打开 `frontend_vision_plc_video.html`。  
前端页面需要由 Flask 后端提供。

---

## 6. 配置修改

所有主要配置集中在：

```text
vision_core/config.py
```

常用配置如下：

```python
CAMERA_ID = 1

PLC_ENABLE = True
PLC_IP = "192.168.0.3"
PLC_RACK = 0
PLC_SLOT = 1
PLC_DB_NUMBER = 45

WEB_HOST = "0.0.0.0"
WEB_PORT = 5000
```

### 6.1 修改摄像头编号

如果摄像头打不开，可修改：

```python
CAMERA_ID = 0
```

或：

```python
CAMERA_ID = 1
```

### 6.2 不连接 PLC 时运行

如果只是调试视觉检测和前端页面，可以关闭 PLC 通信：

```python
PLC_ENABLE = False
```

正式联调时再改回：

```python
PLC_ENABLE = True
```

---

## 7. YOLO 模型说明

模型文件路径：

```text
pt/best.pt
```

程序默认加载该模型进行实时检测。

模型文件较大，上传 GitHub 时建议使用 Git LFS 管理。

---

## 8. 工位标定说明

标定文件位于：

```text
库位置标定
├── position_label
│   ├── up.txt
│   └── down.txt
└── 上下位置
    ├── up.jpg
    └── down.jpg
```

其中：

| 文件 | 作用 |
|---|---|
| `up.jpg` | 上两层货架参考图 |
| `down.jpg` | 下两层货架参考图 |
| `up.txt` | 上两层货架工位框标注 |
| `down.txt` | 下两层货架工位框标注 |

每张参考图需要标注 10 个工位框。

最终编号规则：

```text
第1层：1~5
第2层：6~10
第3层：11~15
第4层：16~20
```

程序会自动把参考图中的工位框缩放到当前摄像头画面尺寸，并根据 YOLO 检测框判断工位是否有货。

---

## 9. PLC 通信说明

本项目使用 Python `python-snap7` 与 Siemens PLC 通信。

PLC 端软件：

```text
TIA Portal V16
```

默认 PLC IP：

```text
192.168.0.3
```

电脑需要通过网线或交换机与 PLC 处于同一网段。

---

## 10. DB45 通信变量

Python 与 PLC 主要通过 `DB45` 通信。

| 地址 | 变量名 | 类型 | 说明 |
|---|---|---|---|
| `DB45.DBD0` | `EmptyMask` | `DWord` | 空工位位掩码 |
| `DB45.DBW4` | `EmptyCount` | `Int` | 空工位数量 |
| `DB45.DBW6` | `WriteSeq` | `Int` | 写入序号 |
| `DB45.DBX8.0` | `DataValid` | `Bool` | 数据有效标志 |
| `DB45.DBX8.1` | `Heartbeat` | `Bool` | 心跳信号 |
| `DB45.DBW10` | `VisionNumber` | `Int` | 前端选择的工位号 |
| `DB45.DBX12.0` | `VisionOutbound` | `Bool` | 视觉出库请求 |
| `DB45.DBX12.1` | `VisionInbound` | `Bool` | 视觉入库请求 |
| `DB45.DBX12.2` | `OutboundDone` | `Bool` | 出库完成反馈 |
| `DB45.DBX12.3` | `InboundDone` | `Bool` | 入库完成反馈 |

PLC 端需要关闭 `DB45` 的优化块访问。

---

## 11. 空工位掩码说明

`EmptyMask` 使用一个 `DWord` 表示 1~20 号工位是否为空。

对应关系：

```text
bit0  -> 1号工位
bit1  -> 2号工位
bit2  -> 3号工位
...
bit19 -> 20号工位
```

含义：

```text
bit = 1：该工位为空
bit = 0：该工位有货
```

例如：

```text
空工位：[1, 3, 5]
```

则：

```text
bit0 = 1
bit2 = 1
bit4 = 1
```

---

## 12. 前端操作逻辑

### 12.1 入库

前端选择空工位后，Python 写入：

```text
DB45.DBW10   = 工位号
DB45.DBX12.1 = True
```

PLC 检测到 `VisionInbound` 后执行入库逻辑。

---

### 12.2 出库

前端选择非空工位后，Python 写入：

```text
DB45.DBW10   = 工位号
DB45.DBX12.0 = True
```

PLC 检测到 `VisionOutbound` 后执行出库逻辑。

---

### 12.3 完成反馈

PLC 完成动作后写入：

```text
DB45.DBX12.2 = True    # 出库完成
DB45.DBX12.3 = True    # 入库完成
```

前端检测到完成位后显示：

```text
出库完成：X号工位
入库完成：X号工位
```

完成提示会保持到下一次点击入库 / 出库。

---

## 13. PLC 程序说明

PLC 程序位于：

```text
PLC程序/智能制造工站PLC-V2.0.6
```

该文件夹中保存 TIA Portal 工站联调程序。

PLC 端主要逻辑：

1. 接收 `DB45.VisionNumber`。
2. 接收 `DB45.VisionOutbound` / `DB45.VisionInbound`。
3. 将 `VisionNumber` 移动到 PLC 自动控制变量 `Auto.Number`。
4. 置位对应的入库 / 出库控制变量。
5. 控制仓储工站执行动作。
6. 动作完成后写入 `OutboundDone` / `InboundDone` 反馈给 Python 和前端。

当前项目中 `Auto.Number` 的实际地址为：

```text
DB3.DBW14
```

---

## 14. TIA Portal 侧注意事项

1. `DB45` 需要关闭优化块访问。
2. CPU 需要允许 PUT/GET 通信访问。
3. 电脑与 PLC 需要处于同一网段。
4. 推荐电脑 IP 设置为：

```text
192.168.0.xxx
```

例如：

```text
192.168.0.100
```

5. 默认 PLC IP：

```text
192.168.0.3
```

6. Python 端只写视觉检测 DB，PLC 程序内部再处理实际自动控制变量。

---

## 15. 常见问题

### 15.1 未连接 PLC 时 FPS 很低

如果未连接 PLC，但 `PLC_ENABLE = True`，程序会反复等待 PLC 通信超时，导致检测帧率明显降低。

解决方法：

```python
PLC_ENABLE = False
```

---

### 15.2 前端打不开

确认已经运行：

```bash
python main.py
```

然后访问：

```text
http://127.0.0.1:5000
```

不要直接打开 HTML 文件。

---

### 15.3 摄像头打不开

修改：

```python
CAMERA_ID = 0
```

或：

```python
CAMERA_ID = 1
```

---

### 15.4 PLC 写入失败

检查：

1. PLC 是否通电并处于 RUN 状态。
2. 电脑和 PLC 是否在同一网段。
3. 能否 ping 通 PLC：

```bash
ping 192.168.0.3
```

4. `DB45` 是否已经下载到 PLC。
5. `DB45` 是否关闭优化块访问。
6. CPU 是否允许 PUT/GET 通信访问。

---

## 16. GitHub 说明

本项目包含模型文件和 PLC 程序文件。

建议使用 Git LFS 管理大文件：

```bash
git lfs install
git lfs track "*.pt"
git lfs track "*.pth"
git lfs track "*.onnx"
git lfs track "*.engine"
git lfs track "PLC程序/**"
```

不建议上传以下运行缓存：

```text
__pycache__/
results_test/
*.pyc
.env
plc通讯/.env
```

---

## 17. 启动命令汇总

```bash
cd /d E:\Course\大三下\机器视觉理论与应用\YOLO11\yolo11_local
python main.py
```

浏览器打开：

```text
http://127.0.0.1:5000
```
