"""vision_service.app.visualizer
Functions to render bounding boxes, alpha-blended overlays, and zone boundaries.
"""

from typing import Dict, List, Optional
import cv2
import numpy as np

from vision_service.app.schemas import Detection


def render_annotated_image(
    img_bgr: np.ndarray,
    detections: List[Detection],
    zones: Optional[Dict[str, List[float]]] = None,
    alpha: float = 0.35,
) -> np.ndarray:
    """วาด Bounding Box, ป้ายชื่อคลาส, โซนพื้นที่ และ Alpha Overlay ลงบนภาพ"""
    annotated = img_bgr.copy()
    H, W = annotated.shape[:2]

    # 1. วาดโซนพื้นที่และ Alpha Overlay สำหรับแต่ละโซน (ถ้ามีการกำหนดโซน)
    if zones:
        zone_overlay = annotated.copy()
        zone_colors = [
            (255, 128, 0),  # น้ำเงินอมฟ้า
            (0, 165, 255),  # ส้ม
            (128, 0, 128),  # ม่วง
            (0, 255, 255),  # เหลือง
        ]

        for idx, (zone_name, zbox) in enumerate(zones.items()):
            color = zone_colors[idx % len(zone_colors)]
            zx1, zy1, zx2, zy2 = [int(v) for v in zbox]
            # จำกัดพิกัดไม่ให้ล้นขอบภาพ
            zx1, zx2 = max(0, min(zx1, W)), max(0, min(zx2, W))
            zy1, zy2 = max(0, min(zy1, H)), max(0, min(zy2, H))

            # ระบายสีโปร่งแสงลงบนเลเยอร์ overlay
            cv2.rectangle(zone_overlay, (zx1, zy1), (zx2, zy2), color, -1)
            # วาดเส้นขอบโซน
            cv2.rectangle(annotated, (zx1, zy1), (zx2, zy2), color, 2)
            # วาดป้ายชื่อโซน
            cv2.putText(
                annotated,
                f"ZONE: {zone_name}",
                (zx1 + 10, zy1 + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        # รวมเลเยอร์ภาพด้วย cv2.addWeighted
        cv2.addWeighted(zone_overlay, alpha, annotated, 1.0 - alpha, 0, annotated)

    # 2. วาด Bounding Box และป้ายข้อมูลวัตถุ
    for det in detections:
        x1, y1, x2, y2 = (
            int(det.box.x1),
            int(det.box.y1),
            int(det.box.x2),
            int(det.box.y2),
        )

        # สีขอบกล่อง (เขียวสด)
        box_color = (0, 255, 0)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)

        # ข้อความระบุชื่อคลาสและความมั่นใจ
        label = f"{det.class_name} {det.confidence:.2f}"
        (tw, th), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )

        # วาดแถบพื้นหลังป้ายชื่อ
        badge_y1 = max(0, y1 - th - baseline - 4)
        cv2.rectangle(
            annotated,
            (x1, badge_y1),
            (x1 + tw + 6, y1),
            box_color,
            -1,
        )
        cv2.putText(
            annotated,
            label,
            (x1 + 3, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    return annotated


def encode_image_to_jpeg(img_bgr: np.ndarray, quality: int = 90) -> bytes:
    """แปลง NumPy array เป็น JPEG bytes สำหรับส่งออก HTTP Response"""
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    success, buffer = cv2.imencode(".jpg", img_bgr, encode_params)
    if not success:
        raise ValueError("ไม่สามารถเข้ารหัสรูปภาพเป็น JPEG ได้")
    return buffer.tobytes()
