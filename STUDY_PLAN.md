# แผนการฝึกปฏิบัติ Computer Vision (From Pixel Math to Production AI)
> ออกแบบเฉพาะสำหรับเรียนรู้ตั้งแต่ระดับโครงสร้างพิกเซลและคณิตศาสตร์เบื้องหลัง (Pure NumPy) สู่การประยุกต์ใช้งานจริงระดับ Production บน GPU (NVIDIA RTX 3060) โดยไม่ต้องหาโหลด Dataset เองจากภายนอก

---

## โครงสร้างหลักสูตร 5 Modules + 1 Production Capstone

### [x] Module 1: Pixel Math, Color Spaces & 2D Convolution from Scratch (Pure NumPy) ✅
* **แนวคิดหลัก:**
  - โครงสร้างภาพดิจิทัล: มิติของภาพ `(H, W, C)`, ชนิดข้อมูล `uint8` (0–255) vs `float32` (0.0–1.0)
  - การแปลง RGB เป็น Grayscale ด้วยสมการความสว่างของดวงตามนุษย์ (Luminance Equation: $0.299R + 0.587G + 0.114B$)
  - การปรับความสว่าง/คอนทราสต์ และการทำ **Min-Max / Clipping** ป้องกันปัญหา Integer Overflow
  - การเขียน **2D Spatial Convolution (`conv2d_numpy`)** จากศูนย์ด้วย NumPy (Sliding Window + Zero Padding)
  - การสกัดขอบภาพด้วย **Sobel Edge Detector** ($G_x, G_y, G = \sqrt{G_x^2 + G_y^2}$)
  - การลดขนาดภาพด้วย **2D Max Pooling (`maxpool2d_numpy`)** ซึ่งเป็นหัวใจของ CNN
* **ไฟล์ฝึกหัด:** [`practice/module_01_pixel_math_numpy.py`](./practice/module_01_pixel_math_numpy.py)
* **สถานะ:** ผ่านการทดสอบครบ 4/4 ข้อสมบูรณ์ 100% (Pass All Asserts)

---

### [x] Module 2: Classical Image Processing & Document/Object Geometry (OpenCV) ✅
* **แนวคิดหลัก:**
  - ข้อควรระวังอันดับ 1 ของ OpenCV: ระบบสี **`BGR` vs `RGB`**
  - Image Filtering & Denoising: Gaussian Blur, Median Blur, Bilateral Filter
  - Thresholding (Global vs Otsu's vs Adaptive) & Morphological Operations (Erosion, Dilation, Opening, Closing)
  - Canny Edge Detection & Contour Analysis: การหาวัตถุ นับจำนวน วัดพื้นที่ (`cv2.contourArea`) และตีกรอบ Bounding Box รอบวัตถุอัตโนมัติ
* **ไฟล์ฝึกหัด:** [`practice/module_02_classical_opencv.py`](./practice/module_02_classical_opencv.py)
* **สถานะ:** ผ่านการทดสอบครบ 4/4 ข้อสมบูรณ์ 100% (Pass All Asserts)

---

### [x] Module 3: Deep Learning for Vision & CNN from Scratch on GPU (PyTorch) ✅
* **แนวคิดหลัก:**
  - ความแตกต่างของแกนข้อมูล: OpenCV/NumPy `(H, W, C)` vs PyTorch Tensor `(B, C, H, W)`
  - การสร้าง Data Augmentation Pipeline (`torchvision.transforms.v2`) และ `DataLoader` (ใช้ชุดข้อมูล `FashionMNIST` / `CIFAR-10` ที่ดาวน์โหลดอัตโนมัติในโค้ด)
  - การประกอบร่าง **Convolutional Neural Network (CNN)** (`nn.Conv2d`, `nn.BatchNorm2d`, `nn.ReLU`, `nn.MaxPool2d`, `nn.Linear`)
  - การคำนวณมิติของ Feature Map และการรัน Training / Validation Loop บน **CUDA GPU (RTX 3060)**
* **ไฟล์ฝึกหัด:** [`practice/module_03_pytorch_cnn_gpu.py`](./practice/module_03_pytorch_cnn_gpu.py)
* **สถานะ:** ผ่านการทดสอบครบ 4/4 ข้อสมบูรณ์ 100% (Loss: 0.3944, Accuracy: 85.89% บน GPU RTX 3060)

---

### [x] Module 4: Transfer Learning & Real-Time Object Detection (Ultralytics YOLO) ✅
* **แนวคิดหลัก:**
  - การทำ Feature Extraction & Fine-tuning โมเดลระดับโลก (`ResNet18` / `MobileNetV3`)
  - คณิตศาสตร์เบื้องหลัง Object Detection: การคำนวณ **IoU (Intersection over Union)** และ **NMS (Non-Maximum Suppression)** ด้วยตัวเอง
  - การใช้งาน **YOLOv8 / YOLOv11** ตรวจจับวัตถุพร้อมดึงพิกัด Bounding Box `(x1, y1, x2, y2)`, Confidence Score และ Class ID บนภาพตัวอย่างอัตโนมัติ
* **ไฟล์ฝึกหัด:** [`practice/module_04_detection_and_yolo.py`](./practice/module_04_detection_and_yolo.py)
* **สถานะ:** ผ่านการทดสอบครบ 4/4 ข้อสมบูรณ์ 100% (Pass IoU, NMS, YOLOv8 GPU Inference & Object Counter)

---

### [x] Module 5: Instance Segmentation, Object Cropping & Visual Analytics Pipeline ✅
* **แนวคิดหลัก:**
  - ความแตกต่างระหว่าง Bounding Box Detection vs Pixel-level Segmentation Mask
  - การทำ Region of Interest (ROI) Cropping: ตัดภาพเฉพาะวัตถุที่ตรวจจับได้ออกมาวิเคราะห์ต่อ พร้อม Safe Boundary Clipping ป้องกัน Index Error
  - การทำ Alpha Blending ไฮไลต์วัตถุโปร่งแสงด้วย `cv2.addWeighted`
  - การวิเคราะห์เชิงพื้นที่ (Spatial Zone Analytics) โดยการตรวจสอบ Center Point เทียบกับพิกัดโซน
  - การประกอบ Structured Analytics JSON Payload สำหรับส่งออกไปยังระบบ REST API ระดับ Production
* **ไฟล์ฝึกหัด:** [`practice/module_05_segmentation_and_analytics.py`](./practice/module_05_segmentation_and_analytics.py)
* **สถานะ:** ผ่านการทดสอบครบ 4/4 ข้อสมบูรณ์ 100% (Pass Safe Cropping, Alpha Overlay, Spatial Zone Analytics & Production Report)

---

### [ ] Production Capstone: Smart Visual Inspector & Object Counter Microservice 🚀
* **แนวคิดหลัก:**
  - สร้าง REST API ด้วย **FastAPI** ที่รับไฟล์รูปภาพ (`UploadFile` / Base64 หรือ URL)
  - ประมวลผลผ่าน Pipeline: `Image Validation -> OpenCV Preprocessing -> YOLO GPU Inference -> ROI Analytics -> JSON + Annotated Image Response`
  - เขียน **Automated Pytest Suite** ตรวจสอบทั้ง Schema และ Vision Pipeline
  - แพ็กระบบลง **Docker Container** พร้อมตั้งค่า **GitHub Actions CI/CD**
* **โฟลเดอร์โปรเจกต์:** `vision_service/`
