import threading

import snap7


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
            DBD0   EmptyMask
            DBW4   EmptyCount
            DBW6   WriteSeq
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

            data[8] |= 1 << 0  # DBX8.0 = DataValid
            if self.heartbeat:
                data[8] |= 1 << 1  # DBX8.1 = Heartbeat

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

            number_data = int(station_no).to_bytes(2, byteorder="big", signed=True)
            self.client.db_write(self.db_number, 10, number_data)

            byte_data = self.client.db_read(self.db_number, 12, 1)

            if action == "outbound":
                byte_data[0] |= 1 << 0  # DBX12.0 = 1
                action_cn = "出库"
            else:
                byte_data[0] |= 1 << 1  # DBX12.1 = 1
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
