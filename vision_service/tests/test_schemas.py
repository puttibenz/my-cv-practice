import pytest
from pydantic import ValidationError
from vision_service.app.schemas import (
    BoundingBox,
    Detection,
    AnalyticsSummary,
    VisionInspectionResponse,
)


def test_bounding_box_valid():
    box = BoundingBox(x1=10.0, y1=20.0, x2=110.0, y2=120.0)
    assert box.width == 100.0
    assert box.height == 100.0
    assert box.area == 10000.0
    assert box.center == (60.0, 70.0)


def test_bounding_box_invalid_coords():
    # x2 < x1 should raise ValidationError
    with pytest.raises(ValidationError):
        BoundingBox(x1=100.0, y1=10.0, x2=50.0, y2=80.0)

    # y2 < y1 should raise ValidationError
    with pytest.raises(ValidationError):
        BoundingBox(x1=10.0, y1=100.0, x2=50.0, y2=80.0)


def test_detection_confidence_range():
    # Confidence must be between 0.0 and 1.0
    box = BoundingBox(x1=0.0, y1=0.0, x2=10.0, y2=10.0)
    with pytest.raises(ValidationError):
        Detection(class_id=0, class_name="person", confidence=1.5, box=box)

    with pytest.raises(ValidationError):
        Detection(class_id=0, class_name="person", confidence=-0.1, box=box)


def test_full_response_serialization():
    box = BoundingBox(x1=10.0, y1=10.0, x2=50.0, y2=50.0)
    det = Detection(class_id=2, class_name="car", confidence=0.92, box=box)
    summary = AnalyticsSummary(
        total_detections=1,
        counts_by_class={"car": 1},
        counts_by_zone={"gate_entry": 1},
    )
    resp = VisionInspectionResponse(
        success=True,
        image_shape=(1080, 1920, 3),
        detections=[det],
        analytics=summary,
        inference_time_ms=12.5,
    )
    payload = resp.model_dump()
    assert payload["success"] is True
    assert payload["detections"][0]["class_name"] == "car"
    assert payload["analytics"]["counts_by_zone"]["gate_entry"] == 1
