"""vision_service.demo_client
Demo script demonstrating end-to-end inference and visualization on bus_sample.jpg.
"""

import json
from pathlib import Path
from fastapi.testclient import TestClient
from vision_service.app.main import app


def run_demo():
    print("🚀 เริ่มต้นรันการทดสอบ Smart Visual Inspector Client...")
    sample_path = Path("bus_sample.jpg")
    if not sample_path.exists():
        print("❌ ไม่พบไฟล์ bus_sample.jpg")
        return

    with TestClient(app) as client:
        # 1. ทดสอบ Health Check
        health = client.get("/health").json()
        print(f"\n[1] Health Check Response: {health}")

        # 2. ส่งภาพไปวิเคราะห์ที่ /predict
        zones = {
            "zone_left_bus": [0.0, 0.0, 500.0, 1080.0],
            "zone_right_pedestrian": [500.0, 0.0, 1080.0, 1080.0],
        }

        with open(sample_path, "rb") as f:
            files = {"file": ("bus_sample.jpg", f, "image/jpeg")}
            data = {
                "conf_threshold": "0.3",
                "zones_json": json.dumps(zones),
            }
            resp = client.post("/predict", files=files, data=data)

        if resp.status_code == 200:
            payload = resp.json()
            print("\n[2] Predict Structured Response:")
            print(f"  - Inference Time: {payload['inference_time_ms']} ms")
            print(f"  - Total Detections: {payload['analytics']['total_detections']}")
            print(f"  - Counts by Class: {payload['analytics']['counts_by_class']}")
            print(f"  - Counts by Zone: {payload['analytics']['counts_by_zone']}")
        else:
            print(f"❌ /predict Error: {resp.text}")

        # 3. ขอภาพที่วาด Bounding Box + Alpha Overlay ที่ /predict/visualize
        with open(sample_path, "rb") as f:
            files = {"file": ("bus_sample.jpg", f, "image/jpeg")}
            data = {
                "conf_threshold": "0.3",
                "zones_json": json.dumps(zones),
            }
            vis_resp = client.post("/predict/visualize", files=files, data=data)

        if vis_resp.status_code == 200:
            out_path = Path("demo_annotated_output.jpg")
            out_path.write_bytes(vis_resp.content)
            print(f"\n[3] บันทึกภาพผลลัพธ์สำเร็จ: {out_path.resolve()}")
            print("🎉 การทดสอบเสร็จสมบูรณ์ 100%!")


if __name__ == "__main__":
    run_demo()
