import io
import json
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from vision_service.app.main import app


@pytest.fixture(scope="session")
def client():
    # ใช้ TestClient พร้อม context manager เพื่อเรียก lifespan startup/shutdown
    with TestClient(app) as test_client:
        yield test_client


def make_dummy_jpeg(width: int = 200, height: int = 200) -> bytes:
    img = np.full((height, width, 3), 180, dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "device" in data


def test_predict_endpoint_success(client):
    img_bytes = make_dummy_jpeg()
    files = {"file": ("test.jpg", io.BytesIO(img_bytes), "image/jpeg")}
    response = client.post("/predict", files=files)

    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    assert "detections" in payload
    assert "analytics" in payload
    assert payload["image_shape"] == [200, 200, 3]


def test_predict_with_zones(client):
    img_bytes = make_dummy_jpeg(300, 300)
    zones = {"zone_center": [50.0, 50.0, 250.0, 250.0]}
    files = {"file": ("test.jpg", io.BytesIO(img_bytes), "image/jpeg")}
    data = {
        "conf_threshold": "0.3",
        "zones_json": json.dumps(zones),
    }

    response = client.post("/predict", files=files, data=data)
    assert response.status_code == 200
    payload = response.json()
    assert "zone_center" in payload["analytics"]["counts_by_zone"]


def test_predict_visualize_endpoint(client):
    img_bytes = make_dummy_jpeg()
    zones = {"zone_full": [0.0, 0.0, 200.0, 200.0]}
    files = {"file": ("test.jpg", io.BytesIO(img_bytes), "image/jpeg")}
    data = {"zones_json": json.dumps(zones)}

    response = client.post("/predict/visualize", files=files, data=data)
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    # ตรวจสอบว่าเป็นภาพ JPEG ที่ถอดรหัสได้จริง
    decoded = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None
    assert decoded.shape == (200, 200, 3)


def test_predict_corrupted_image(client):
    files = {"file": ("bad.jpg", io.BytesIO(b"invalid corrupt data"), "image/jpeg")}
    response = client.post("/predict", files=files)
    assert response.status_code == 400
