from dataclasses import dataclass
import numpy as np


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
