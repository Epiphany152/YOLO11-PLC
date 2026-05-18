from pathlib import Path
import cv2
import numpy as np


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
