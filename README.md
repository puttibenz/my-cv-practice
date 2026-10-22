# Computer Vision Engineering Bootcamp (`my-cv-practice`)

โปรเจกต์ฝึกฝนและทบทวน **Computer Vision (CV)** ตั้งแต่คณิตศาสตร์ระดับพิกเซลด้วย `NumPy` ไปจนถึง `OpenCV`, `PyTorch CNN (GPU)`, `YOLOv8/v11 Object Detection` และการสร้างระบบ **Production Vision API ด้วย FastAPI + Docker + CI/CD**

---

## โครงสร้างหลักสูตร (100% Complete)
ดูรายละเอียดและความคืบหน้าทั้งหมดได้ที่ไฟล์ [**`STUDY_PLAN.md`**](./STUDY_PLAN.md)

1. **Module 1:** Pixel Math, Image Tensors & 2D Convolution from Scratch (`NumPy`) ✅
2. **Module 2:** Classical Image Processing & Contour Analysis (`OpenCV`) ✅
3. **Module 3:** Deep Learning for Vision & CNN Architecture on GPU (`PyTorch`) ✅
4. **Module 4:** Transfer Learning & Real-time Object Detection (`YOLOv8/v11`) ✅
5. **Module 5:** Image Segmentation, Cropping & Region Analytics (`OpenCV` + `NumPy`) ✅
6. **Production Capstone:** Smart Visual Inspector & Object Counter Microservice (`FastAPI` + `Docker` + `GitHub Actions`) ✅

---

## 🚀 วิธีเปิดใช้งาน Production Vision Microservice

```bash
# 1. รันการทดสอบทั้งหมด (14 Tests)
pytest vision_service/tests/ -v

# 2. รัน API Service บน Local
uvicorn vision_service.app.main:app --reload --port 8000

# 3. เปิดดู API Docs (Swagger UI)
# ไปที่ http://localhost:8000/docs
```
