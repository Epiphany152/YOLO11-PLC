from flask import Flask, Response, jsonify, request, send_file

from .config import FRONTEND_HTML, WEB_HOST, WEB_PORT
from . import web_state as state

app = Flask(__name__)


@app.route("/")
def index_page():
    if not FRONTEND_HTML.exists():
        return "frontend_vision_plc_video.html 不存在", 404
    return send_file(str(FRONTEND_HTML))


@app.route("/api/status", methods=["GET"])
def api_status():
    with state.status_lock:
        data = dict(state.current_status)
    return jsonify(data)


@app.route("/video_feed")
def video_feed():
    return Response(
        state.mjpeg_generator(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/api/command", methods=["POST"])
def api_command():
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

    with state.status_lock:
        occupied_list = list(state.current_status.get("occupied_shelves", []))
        empty_list = list(state.current_status.get("empty_shelves", []))

    if action == "inbound" and station_no not in empty_list:
        return jsonify({"ok": False, "msg": f"{station_no}号工位当前不是空位，不能入库"}), 400

    if action == "outbound" and station_no not in occupied_list:
        return jsonify({"ok": False, "msg": f"{station_no}号工位当前不是非空位，不能出库"}), 400

    if state.plc_writer_global is None:
        return jsonify({"ok": False, "msg": "PLC对象未初始化"}), 500

    try:
        state.plc_writer_global.send_vision_command(action, station_no)
    except Exception as e:
        return jsonify({"ok": False, "msg": f"PLC写入失败：{e}"}), 500

    # 新命令发出后，清空上一条完成提示，并等待本次命令的完成反馈
    with state.status_lock:
        state.current_status["done_message"] = ""
        state.current_status["done_action"] = ""
        state.current_status["done_station_no"] = None

        state.command_state["pending_action"] = action
        state.command_state["pending_station_no"] = station_no
        state.command_state["awaiting_done"] = True

        current_done = bool(state.current_status.get(f"{action}_done", False))
        state.command_state["done_seen_low"][action] = not current_done

    action_cn = "入库" if action == "inbound" else "出库"
    return jsonify({"ok": True, "msg": f"已发送{action_cn}指令：{station_no}号工位"})


def run_web_server():
    app.run(host=WEB_HOST, port=WEB_PORT, debug=False, use_reloader=False, threaded=True)
