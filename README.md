# YOLO11-PLC 视觉检测与仓储 PLC 联动系统

本项目用于实现基于 YOLO11 的货架工位视觉检测，并通过前端网页与 Siemens PLC 进行交互控制。

系统功能包括：

- 摄像头实时采集货架画面
- YOLO11 检测货物位置
- 判断 1~20 号工位的空 / 非空状态
- 前端网页显示实时检测画面和工位状态
- 用户在前端选择入库 / 出库工位
- Python 后端通过 Snap7 与仓储 PLC 通信
- PLC 执行对应入库 / 出库逻辑
- 前端显示入库 / 出库完成提示

---

## 一、项目结构

```text
yolo11_local
├── main.py
├── vision_core
│   ├── __init__.py
│   ├── app_main.py          # 主循环：摄像头、YOLO、PLC、Web线程调度
│   ├── config.py            # 路径、摄像头、YOLO、PLC、Web配置
│   ├── data_types.py        # SlotBox / DetBox 数据结构
│   ├── detector.py          # YOLO检测、框匹配、状态平滑、空/非空列表
│   ├── geometry.py          # IOU、点是否在框内等几何函数
│   ├── image_io.py          # 中文路径图片读取/保存
│   ├── plc_client.py        # Snap7 PLC读写
│   ├── reference_slots.py   # 读取参考图和YOLO标注，生成1~20工位
│   ├── result_saver.py      # 保存当前检测结果
│   ├── visualizer.py        # 绘制检测画面
│   ├── web_server.py        # Flask接口
│   └── web_state.py         # 前端状态、视频流缓存、完成提示保持逻辑
├── frontend_vision_plc_video.html
├── pt
│   └── best.pt
└── 库位置标定
    ├── position_label
    │   ├── classes.txt
    │   ├── up.txt
    │   └── down.txt
    └── 上下位置
        ├── up.jpg
        └── down.jpg
```

---

## 二、主要文件说明

### 1. `main.py`

程序入口文件。

运行该文件即可启动：

- YOLO11 实时检测
- Flask 后端服务
- PLC 通信
- 前端网页视频流显示

---

### 2. `vision_core/config.py`

集中管理主要配置，包括：

- 模型路径
- 摄像头编号
- 摄像头分辨率
- YOLO 推理参数
- PLC IP 地址
- PLC DB 块编号
- Web 服务端口
- 标定图片路径
- 标注文件路径

如果需要修改摄像头、PLC IP、DB 号、模型路径等，优先修改该文件。

---

### 3. `frontend_vision_plc_video.html`

前端网页文件。

主要功能：

- 显示实时检测画面
- 显示空工位和非空工位
- 提供入库 / 出库操作按钮
- 显示 PLC 连接状态
- 显示入库 / 出库完成提示

---

### 4. `pt/best.pt`

YOLO11 训练得到的模型权重文件。

程序默认从该位置加载模型。

---

### 5. `库位置标定`

用于保存货架参考图片和工位标注文件。

其中：

```text
库位置标定/上下位置/up.jpg      # 上两层货架参考图
库位置标定/上下位置/down.jpg    # 下两层货架参考图

库位置标定/position_label/up.txt
库位置标定/position_label/down.txt
```

`up.txt` 和 `down.txt` 使用 YOLO 标注格式，每张图对应 10 个工位框。

最终程序统一生成 1~20 号工位：

```text
第1层：1~5
第2层：6~10
第3层：11~15
第4层：16~20
```

---

## 三、运行环境

建议使用 Conda 环境。

主要依赖：

```bash
pip install ultralytics opencv-python numpy pandas flask python-snap7
```

如果已经配置好 YOLO11 环境，只需要补充安装：

```bash
pip install flask python-snap7
```

---

## 四、运行方式

### 1. 进入项目目录

```bash
cd /d E:\Course\大三下\机器视觉理论与应用\YOLO11\yolo11_local
```

### 2. 启动程序

```bash
python main.py
```

### 3. 打开前端网页

浏览器访问：

```text
http://127.0.0.1:5000
```

---

## 五、PLC 通信说明

PLC 使用 Siemens PLC，软件环境为 TIA Portal V16。

Python 端通过 Snap7 与 PLC 通信。

默认 PLC 参数在 `vision_core/config.py` 中配置：

```python
PLC_ENABLE = True
PLC_IP = "192.168.0.3"
PLC_RACK = 0
PLC_SLOT = 1
PLC_DB_NUMBER = 45
```

---

## 六、DB45 变量说明

Python 与 PLC 通信主要使用 `DB45`。

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
| `DB45.DBX12.2` | `OutboundDone` | `Bool` | PLC 出库完成反馈 |
| `DB45.DBX12.3` | `InboundDone` | `Bool` | PLC 入库完成反馈 |

---

## 七、空工位掩码说明

`EmptyMask` 使用一个 `DWord` 表示 1~20 号工位是否为空。

规则：

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

## 八、前端入库 / 出库逻辑

### 1. 入库

前端选择空工位后，Python 写入：

```text
DB45.DBW10   = 工位号
DB45.DBX12.1 = True
```

PLC 检测到 `VisionInbound` 后执行入库逻辑。

---

### 2. 出库

前端选择非空工位后，Python 写入：

```text
DB45.DBW10   = 工位号
DB45.DBX12.0 = True
```

PLC 检测到 `VisionOutbound` 后执行出库逻辑。

---

### 3. 完成反馈

PLC 完成动作后写入：

```text
DB45.DBX12.2 = True    # 出库完成
DB45.DBX12.3 = True    # 入库完成
```

前端检测到完成位后，会显示：

```text
出库完成：X号工位
入库完成：X号工位
```

完成提示会保持到下一次点击入库 / 出库。

---

## 九、TIA Portal 侧注意事项

1. `DB45` 需要关闭优化块访问。
2. CPU 需要允许 PUT/GET 通信访问。
3. PLC IP 地址需要与电脑在同一网段。
4. 电脑与 PLC 可通过网线直连或交换机连接。
5. Python 端只写视觉检测 DB，PLC 程序内部负责把 `VisionNumber` 转移到 `Auto.Number`。
6. 当前项目中 `Auto.Number` 的实际地址为：

```text
DB3.DBW14
```

---

## 十、常见问题

### 1. 未连接 PLC 时 FPS 很低

如果没有连接 PLC，但 `PLC_ENABLE = True`，程序会反复等待 PLC 通信超时，导致检测帧率降低。

调试视觉时可在 `vision_core/config.py` 中关闭 PLC 通信：

```python
PLC_ENABLE = False
```

正式联调时再改回：

```python
PLC_ENABLE = True
```

---

### 2. 前端打不开

确认程序已经运行：

```bash
python main.py
```

然后浏览器访问：

```text
http://127.0.0.1:5000
```

不要直接双击打开 HTML 文件。

---

### 3. 摄像头打不开

在 `vision_core/config.py` 中修改摄像头编号：

```python
CAMERA_ID = 0
```

或：

```python
CAMERA_ID = 1
```

根据实际摄像头编号选择。

---

### 4. 修改 PLC 地址

在 `vision_core/config.py` 中修改：

```python
PLC_IP = "192.168.0.3"
PLC_DB_NUMBER = 45
```

---

## 十一、GitHub 上传说明

建议不要上传以下内容：

```text
__pycache__/
results_test/
*.pyc
.env
```

如果上传 `pt/best.pt`，建议使用 Git LFS 管理模型权重文件。

---

## 十二、启动命令汇总

```bash
cd /d E:\Course\大三下\机器视觉理论与应用\YOLO11\yolo11_local
python main.py
```

浏览器打开：

```text
http://127.0.0.1:5000
```