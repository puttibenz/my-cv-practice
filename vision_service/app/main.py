"""vision_service.app.main
FastAPI application entrypoint providing REST API endpoints for image inspection,
spatial zone analytics, and visual annotation rendering.
"""

from contextlib import asynccontextmanager
import json
from typing import Optional
from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware

from vision_service.app.pipeline import VisionPipeline
from vision_service.app.schemas import VisionInspectionResponse
from vision_service.app.visualizer import encode_image_to_jpeg, render_annotated_image


@asynccontextmanager
async def lifespan(app: FastAPI):
    """โหลดโมเดลเข้า GPU และเตรียมทรัพยากรเมื่อเริ่มต้น Service"""
    print("[LifeCycle] กำลังเตรียมความพร้อม Vision Pipeline...")
    app.state.pipeline = VisionPipeline(model_path="yolov8n.pt")
    yield
    print("[LifeCycle] ปิดการทำงาน Vision Service เรียบร้อย")


app = FastAPI(
    title="Smart Visual Inspector & Object Counter API",
    description="Production REST Microservice สำหรับวิเคราะห์ภาพและตรวจจับวัตถุเชิงพื้นที่ (FastAPI + YOLOv8 + OpenCV)",
    version="1.0.0",
    lifespan=lifespan,
)

# เปิดใช้งาน CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["System"])
@app.get("/health", tags=["System"])
def health_check():
    """ตรวจสอบสถานะความพร้อมและอุปกรณ์ประมวลผล (CPU / CUDA GPU)"""
    pipeline: VisionPipeline = getattr(app.state, "pipeline", None)
    return {
        "status": "healthy",
        "service": "Smart Visual Inspector API",
        "version": "1.0.0",
        "device": pipeline.device if pipeline else "uninitialized",
    }


def parse_zones(zones_json: Optional[str]) -> Optional[dict]:
    """แปลง JSON string ของการกำหนดพิกัดโซนเป็น Dictionary"""
    if not zones_json:
        return None
    try:
        zones = json.loads(zones_json)
        if not isinstance(zones, dict):
            raise ValueError("zones_json ต้องเป็น JSON Object (Dictionary)")
        return zones
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"รูปแบบ zones_json ไม่ถูกต้อง: {str(e)}",
        )


@app.post(
    "/predict",
    response_model=VisionInspectionResponse,
    tags=["Computer Vision"],
    summary="วิเคราะห์รูปภาพและนับวัตถุแยกตามโซน (Structured JSON)",
)
async def predict_image(
    file: UploadFile = File(..., description="ไฟล์รูปภาพ (JPEG, PNG ฯลฯ)"),
    conf_threshold: float = Form(0.25, ge=0.0, le=1.0, description="เกณฑ์ความมั่นใจขั้นต่ำ"),
    zones_json: Optional[str] = Form(
        None,
        description='พิกัดโซนในรูปแบบ JSON เช่น `{"zone_a": [0, 0, 300, 300]}',
    ),
):
    """
    รับไฟล์ภาพเข้ามาประมวลผลผ่าน Pipeline:
    1. ตรวจสอบและแปลง Byte เป็นภาพ OpenCV
    2. รัน YOLOv8 Inference บน GPU
    3. คำนวณ Spatial Analytics นับจำนวนแยกตามโซน
    4. ส่งผลลัพธ์เป็น Structured JSON ตาม Schema
    """
    pipeline: VisionPipeline = app.state.pipeline
    zones = parse_zones(zones_json)

    try:
        image_bytes = await file.read()
        response, _ = pipeline.process(
            image_bytes=image_bytes,
            zones=zones,
            conf_threshold=conf_threshold,
        )
        return response
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(val_err),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"เกิดข้อผิดพลาดในการประมวลผล: {str(exc)}",
        )


@app.post(
    "/predict/visualize",
    tags=["Computer Vision"],
    summary="วิเคราะห์รูปภาพและวาด Bounding Box + โซนโปร่งแสง (คืนค่าเป็นไฟล์ JPEG)",
    responses={
        200: {
            "content": {"image/jpeg": {}},
            "description": "ภาพผลลัพธ์พร้อม Bounding Box และ Overlay",
        }
    },
)
async def predict_and_visualize(
    file: UploadFile = File(..., description="ไฟล์รูปภาพ (JPEG, PNG ฯลฯ)"),
    conf_threshold: float = Form(0.25, ge=0.0, le=1.0, description="เกณฑ์ความมั่นใจขั้นต่ำ"),
    zones_json: Optional[str] = Form(
        None,
        description='พิกัดโซนในรูปแบบ JSON เช่น `{"zone_a": [0, 0, 300, 300]}',
    ),
):
    """
    ประมวลผลภาพและวาด Annotation:
    - Bounding Box สีเขียวพร้อมชื่อคลาสและ Confidence
    - Alpha Blended Overlay สีโปร่งแสงแสดงขอบเขตโซน
    - ส่งกลับเป็นข้อมูลภาพ JPEG โดยตรง
    """
    pipeline: VisionPipeline = app.state.pipeline
    zones = parse_zones(zones_json)

    try:
        image_bytes = await file.read()
        response, img_bgr = pipeline.process(
            image_bytes=image_bytes,
            zones=zones,
            conf_threshold=conf_threshold,
        )

        annotated_bgr = render_annotated_image(
            img_bgr=img_bgr,
            detections=response.detections,
            zones=zones,
            alpha=0.35,
        )

        jpeg_bytes = encode_image_to_jpeg(annotated_bgr)
        return Response(content=jpeg_bytes, media_type="image/jpeg")

    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(val_err),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"เกิดข้อผิดพลาดในการประมวลผล: {str(exc)}",
        )
