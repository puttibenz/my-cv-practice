import cv2
import numpy as np
import pytest
from vision_service.app.pipeline import VisionPipeline
from vision_service.app.schemas import BoundingBox, Detection


@pytest.fixture(scope="session")
def pipeline():
    # โหลด pipeline ครั้งเดียวสำหรับ test session
    return VisionPipeline(model_path="yolov8n.pt")


def test_decode_image_valid(pipeline):
    dummy = np.full((100, 100, 3), 128, dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", dummy)
    decoded = pipeline.decode_image(buf.tobytes())
    assert decoded.shape == (100, 100, 3)
    assert decoded.dtype == np.uint8


def test_decode_image_invalid(pipeline):
    with pytest.raises(ValueError):
        pipeline.decode_image(b"not an image file")

    with pytest.raises(ValueError):
        pipeline.decode_image(b"")


def test_compute_analytics_with_zones(pipeline):
    det1 = Detection(
        class_id=0,
        class_name="person",
        confidence=0.9,
        box=BoundingBox(x1=10, y1=10, x2=50, y2=50),  # center = (30, 30)
    )
    det2 = Detection(
        class_id=2,
        class_name="car",
        confidence=0.85,
        box=BoundingBox(x1=150, y1=150, x2=250, y2=250),  # center = (200, 200)
    )

    zones = {
        "zone_top_left": [0.0, 0.0, 100.0, 100.0],
        "zone_bottom_right": [150.0, 150.0, 300.0, 300.0],
    }

    summary = pipeline.compute_analytics([det1, det2], zones=zones)
    assert summary.total_detections == 2
    assert summary.counts_by_class == {"person": 1, "car": 1}
    assert summary.counts_by_zone["zone_top_left"] == 1
    assert summary.counts_by_zone["zone_bottom_right"] == 1


def test_crop_rois_safe_bounds(pipeline):
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    # Box ที่ล้นขอบภาพ (x: -50 ถึง 150, y: -20 ถึง 120)
    det_out = Detection(
        class_id=0,
        class_name="person",
        confidence=0.8,
        box=BoundingBox(x1=-50, y1=-20, x2=150, y2=120),
    )
    crops = pipeline.crop_rois(img, [det_out])
    assert len(crops) == 1
    # ขนาดที่ถูก crop ออกมาต้องถูก clip ให้อยู่ภายในขอบภาพ (100, 100, 3)
    assert crops[0].shape == (100, 100, 3)


def test_run_inference_synthetic_image(pipeline):
    dummy = np.full((320, 320, 3), 200, dtype=np.uint8)
    detections, elapsed_ms = pipeline.run_inference(dummy)
    assert isinstance(detections, list)
    assert elapsed_ms > 0
