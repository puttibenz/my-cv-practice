# Multi-stage / Production Dockerfile for Smart Visual Inspector Microservice
FROM python:3.12-slim AS runtime

# ป้องกัน Python เขียนไฟล์ .pyc และบังคับให้ Flush Log ทันที
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# ติดตั้ง System Libraries พื้นฐานที่จำเป็นสำหรับ OpenCV และ PyTorch
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# คัดลอกและติดตั้ง Dependencies
COPY vision_service/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# คัดลอกซอร์สโค้ดและโมเดล
COPY vision_service /app/vision_service
COPY yolov8n.pt /app/yolov8n.pt

# สร้าง non-root user เพื่อความปลอดภัยใน Production
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# ตรวจสอบความพร้อมของ Container
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["uvicorn", "vision_service.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
