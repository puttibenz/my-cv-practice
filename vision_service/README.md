# Smart Visual Inspector & Object Counter Microservice 🚀

Production-ready Computer Vision REST API microservice built with **FastAPI**, **YOLOv8**, and **OpenCV**.

---

## 🏗️ Architecture & Features

```
Client (HTTP / REST)
       │
       ▼
┌────────────────────────────────────────────────────────┐
│ FastAPI Application (`vision_service/app/main.py`)      │
│   ├── /health              (Health check & device info)│
│   ├── /predict             (Structured JSON Analytics) │
│   └── /predict/visualize   (Annotated JPEG Streaming)  │
└───────────────────────┬────────────────────────────────┘
                        │
                        ▼
┌────────────────────────────────────────────────────────┐
│ Vision Pipeline Orchestrator (`pipeline.py`)          │
│   ├── 1. Safe Image Decoder (OpenCV BGR Array)         │
│   ├── 2. YOLOv8 GPU/CUDA Inference Engine              │
│   ├── 3. Spatial Zone Analytics (Center-Point Math)    │
│   └── 4. Safe ROI Cropper & Alpha Blended Visualizer   │
└────────────────────────────────────────────────────────┘
```

- **GPU Acceleration**: รองรับ NVIDIA CUDA GPU โดยอัตโนมัติ (Fallback สู่ CPU หากไม่มี GPU)
- **Data Contracts**: ตรวจสอบความถูกต้องของ Input/Output ด้วย Pydantic v2
- **Spatial Zone Analytics**: วิเคราะห์จำนวนวัตถุแยกตามพิกัดโซนที่กำหนดแบบไดนามิก
- **Visualizer Engine**: วาด Bounding Box ป้ายชื่อคลาส ความมั่นใจ และ Alpha Blended Zone Overlay
- **CI/CD & Container**: พร้อมไฟล์ Dockerfile และ GitHub Actions Automated Testing Pipeline

---

## ⚡ วิธีรัน Microservice

### 1. รันบน Local Environment
```bash
# รัน FastAPI ผ่าน Uvicorn
uvicorn vision_service.app.main:app --reload --port 8000
```
เปิด Swagger UI ได้ที่: [http://localhost:8000/docs](http://localhost:8000/docs)

### 2. รันผ่าน Docker Container
```bash
# Build Docker Image
docker build -t vision-service:latest -f vision_service/Dockerfile .

# รัน Container
docker run -p 8000:8000 vision-service:latest
```

---

## 📡 ตัวอย่างการเรียกใช้งาน API

### 1. Health Check
```bash
curl -X GET "http://localhost:8000/health"
```
**ตัวอย่างผลลัพธ์:**
```json
{
  "status": "healthy",
  "service": "Smart Visual Inspector API",
  "version": "1.0.0",
  "device": "cuda:0"
}
```

### 2. วิเคราะห์ภาพและนับวัตถุแยกตามโซน (`/predict`)
```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@bus_sample.jpg" \
  -F "conf_threshold=0.25" \
  -F 'zones_json={"zone_left": [0, 0, 400, 1080], "zone_right": [400, 0, 1080, 1080]}'
```
**ตัวอย่างผลลัพธ์:**
```json
{
  "success": true,
  "image_shape": [1080, 810, 3],
  "detections": [
    {
      "class_id": 5,
      "class_name": "bus",
      "confidence": 0.88,
      "box": {
        "x1": 21.0,
        "y1": 234.0,
        "x2": 801.0,
        "y2": 765.0
      }
    }
  ],
  "analytics": {
    "total_detections": 5,
    "counts_by_class": {
      "bus": 1,
      "person": 4
    },
    "counts_by_zone": {
      "zone_left": 3,
      "zone_right": 2
    }
  },
  "inference_time_ms": 11.45
}
```

### 3. ขอรับภาพผลลัพธ์พร้อม Bounding Box & โซนสีโปร่งแสง (`/predict/visualize`)
```bash
curl -X POST "http://localhost:8000/predict/visualize" \
  -F "file=@bus_sample.jpg" \
  -F 'zones_json={"zone_left": [0, 0, 400, 1080]}' \
  --output result_annotated.jpg
```

---

## 🧪 การรัน Automated Test Suite

```bash
pytest vision_service/tests/ -v
```
ครอบคลุม Unit Tests สำหรับ:
- `test_schemas.py`: ทดสอบ Pydantic Bounding Box coordinates, Area calculation, Confidence bounds
- `test_pipeline.py`: ทดสอบ Image Byte decoding, Spatial zone counting, Safe boundary clipping
- `test_api.py`: ทดสอบ REST Endpoints (`/health`, `/predict`, `/predict/visualize`) ด้วย TestClient
