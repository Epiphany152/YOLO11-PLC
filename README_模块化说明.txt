模块化后的放置方式：

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
    │   ├── up.txt
    │   └── down.txt
    └── 上下位置
        ├── up.jpg
        └── down.jpg

运行方式：
1. 把 main.py 和 vision_core 文件夹复制到 yolo11_local 下。
2. 保证 frontend_vision_plc_video.html 仍在 yolo11_local 下。
3. CMD/Anaconda Prompt 进入 yolo11_local：
   cd /d E:\Course\大三下\机器视觉理论与应用\YOLO11\yolo11_local
4. 运行：
   python main.py
5. 浏览器打开：
   http://127.0.0.1:5000

需要改摄像头、PLC IP、DB号、模型路径等，都只改：
vision_core/config.py
