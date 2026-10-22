# %% [markdown]
# # Module 4: Intersection over Union (IoU), NMS & Real-Time Object Detection (YOLO)
#
# ใน Module 3 คุณได้สร้างและเทรน **CNN** เพื่อจัดหมวดหมู่ภาพทั้งภาพ (Image Classification) ไปแล้ว
# แต่ในงานจริง วัตถุมักจะไม่ได้อยู่โดดๆ กลางภาพ แต่วางปะปนกันอยู่หลายชิ้น
# งานนี้จึงต้องใช้ **Object Detection** (หาว่า "อะไร" อยู่ที่ "พิกัดไหน" ในภาพ)
#
# ในโมดูลนี้ เราจะ:
# 1. เขียนคณิตศาสตร์เบื้องหลังการประเมินกรอบวัตถุ: **IoU (Intersection over Union)**
# 2. เขียนอัลกอริทึมตัดกรอบที่ซ้ำซ้อน: **NMS (Non-Maximum Suppression)** จากศูนย์
# 3. โหลดและรันโมเดล **YOLOv8** บนการ์ดจอ **RTX 3060** เพื่อตรวจจับคน รถ และสิ่งของแบบ Real-Time!

# %%
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
from ultralytics import YOLO

# ตรวจสอบว่า CUDA พร้อมใช้งานกับ YOLO หรือไม่
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"🚀 YOLO Execution Device: {device}")


# %% [markdown]
# ---
# ## โจทย์ข้อที่ 1: คำนวณ Intersection over Union (IoU) จากศูนย์
#
# **IoU** คือมาตรวัดความแม่นยำในการตีกรอบวัตถุ โดยเปรียบเทียบระหว่างกรอบ 2 กล่อง (เช่น กรอบจริง vs กรอบที่โมเดลทำนาย)
# $$\text{IoU} = \frac{\text{Area of Overlap}}{\text{Area of Union}} = \frac{A \cap B}{A \cup B}$$
#
# **รูปแบบพิกัดกล่อง:** `[x1, y1, x2, y2]` (มุมบนซ้าย และมุมล่างขวา)
#
# **ภารกิจ:**
# เขียนฟังก์ชัน `compute_iou(box1: list[float], box2: list[float]) -> float`:
# 1. หาพิกัดของพื้นที่ทับซ้อน (Intersection Box):
#    - `inter_x1 = max(box1[0], box2[0])`
#    - `inter_y1 = max(box1[1], box2[1])`
#    - `inter_x2 = min(box1[2], box2[2])`
#    - `inter_y2 = min(box1[3], box2[3])`
# 2. คำนวณความกว้างและสูงของพื้นที่ทับซ้อน:
#    - `inter_w = max(0.0, inter_x2 - inter_x1)`
#    - `inter_h = max(0.0, inter_y2 - inter_y1)`
#    - `inter_area = inter_w * inter_h`
# 3. คำนวณพื้นที่ของแต่ละกล่อง:
#    - `area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])`
#    - `area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])`
# 4. คำนวณพื้นที่ยูเนียน:
#    - `union_area = area1 + area2 - inter_area`
# 5. คืนค่า `inter_area / union_area` (ถ้า union_area <= 0 ให้คืนค่า `0.0`)

# %%
def compute_iou(box1: list[float], box2: list[float]) -> float:
    """คำนวณ Intersection over Union (IoU) ระหว่าง 2 Bounding Boxes ในพิกัด [x1, y1, x2, y2]"""
    # TODO: เขียนโค้ดตรงนี้
    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[2], box2[2])
    inter_y2 = min(box1[3], box2[3])

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area=  area1 + area2 - inter_area

    return inter_area / union_area if not (union_area <= 0) else 0.0 

# --- ทดสอบโจทย์ข้อที่ 1 ---
# เคสที่ 1: กล่องซ้อนทับกันพอดีเป๊ะ (IoU ต้องเท่ากับ 1.0)
b1 = [0.0, 0.0, 10.0, 10.0]
b2 = [0.0, 0.0, 10.0, 10.0]
assert np.isclose(compute_iou(b1, b2), 1.0), "กล่องทับกันสนิท IoU ต้องเป็น 1.0"

# เคสที่ 2: กล่องไม่ทับกันเลย (IoU ต้องเท่ากับ 0.0)
b3 = [20.0, 20.0, 30.0, 30.0]
assert compute_iou(b1, b3) == 0.0, "กล่องอยู่คนละที่ IoU ต้องเป็น 0.0"

# เคสที่ 3: ทับกันครึ่งหนึ่ง (พื้นที่ทับซ้อน 50, ยูเนียน 150 -> IoU = 50/150 = 0.3333)
b4 = [0.0, 0.0, 10.0, 10.0]   # area = 100
b5 = [5.0, 0.0, 15.0, 10.0]   # area = 100, inter = 5 * 10 = 50, union = 150
assert np.isclose(compute_iou(b4, b5), 50.0 / 150.0, atol=1e-3), "การคำนวณ IoU ไม่ถูกต้อง"
print("✅ ข้อที่ 1 ผ่าน! (คำนวณคณิตศาสตร์ IoU ถูกต้องแม่นยำ)")


# %% [markdown]
# ---
# ## โจทย์ข้อที่ 2: อัลกอริทึมตัดกรอบซ้ำซ้อน (Non-Maximum Suppression - NMS) จากศูนย์
#
# เวลาโมเดลตรวจจับวัตถุ มันมักจะพ่นกรอบสี่เหลี่ยมออกมาซ้ำซ้อนรอบวัตถุชิ้นเดิมหลายอัน
# **NMS** คืออัลกอริทึมที่ทำหน้าที่เลือกเฉพาะ **"กรอบที่ดีที่สุดเพียงกรอบเดียว"** และลบกรอบที่ซ้อนทับกันทิ้ง
#
# **หลักการทำงาน:**
# 1. เรียงลำดับรายการกล่องตามคะแนนความมั่นใจ (`score`) จากมากไปหาน้อย
# 2. หยิบกล่องที่มีคะแนนสูงสุด (`best_box`) ออกมาเก็บไว้ในรายการผลลัพธ์
# 3. วนตรวจกล่องที่เหลือทั้งหมด: ถ้ากล่องใดมีค่า $\text{IoU}$ กับ `best_box` เกินเกณฑ์ `iou_threshold` ให้ **ลบทิ้งทันที** (เพราะถือว่าเป็นกรอบซ้ำของวัตถุชิ้นเดียวกัน)
# 4. ทำซ้ำจนกว่าจะตรวจครบทุกกล่อง
#
# **ภารกิจ:**
# เขียนฟังก์ชัน `nms_from_scratch(boxes: list[list[float]], scores: list[float], iou_threshold: float = 0.5) -> list[int]`:
# - คืนค่าเป็น `list` ของ Index ของกล่องที่ผ่านการคัดเลือก (เช่น `[0, 2]`)

# %%
def nms_from_scratch(
    boxes: list[list[float]], scores: list[float], iou_threshold: float = 0.5
) -> list[int]:
    """
    คัดเลือกกล่องที่ดีที่สุดด้วยอัลกอริทึม Non-Maximum Suppression (NMS)
    Input:
        boxes: รายการกล่อง [[x1, y1, x2, y2], ...]
        scores: รายการคะแนนความมั่นใจ [0.95, 0.88, ...]
        iou_threshold: เกณฑ์ตัดสินว่าซ้อนทับกันเกินไปหรือไม่ (ค่าเริ่มต้น 0.5)
    Output:
        keep: รายการดัชนี (indices) ของกล่องที่ได้รับเลือก
    """
    # TODO: เขียนโค้ดตรงนี้
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    keep = []

    while order:
        best_idx = order.pop(0)
        keep.append(best_idx)

        remaining = []
        for idx in order:
            if compute_iou(boxes[best_idx], boxes[idx]) <= iou_threshold:
                remaining.append(idx)

        order = remaining

    return keep


# --- ทดสอบโจทย์ข้อที่ 2 ---
# จำลองกล่อง 3 กล่อง:
# กล่องที่ 0 และ 1 ซ้อนทับวัตถุชิ้นเดียวกัน (คะแนน 0.90 กับ 0.75) -> ต้องเก็บกล่อง 0 และตัดกล่อง 1 ทิ้ง
# กล่องที่ 2 อยู่อีกตำแหน่งหนึ่ง (คะแนน 0.85) -> ต้องเก็บไว้
sample_boxes = [
    [10.0, 10.0, 50.0, 50.0],  # Box 0: รถยนต์คันที่ 1 (score 0.90)
    [12.0, 11.0, 52.0, 49.0],  # Box 1: กรอบซ้ำของรถยนต์คันที่ 1 (score 0.75)
    [100.0, 100.0, 160.0, 160.0], # Box 2: คนเดินถนน (score 0.85)
]
sample_scores = [0.90, 0.75, 0.85]

keep_indices = nms_from_scratch(sample_boxes, sample_scores, iou_threshold=0.5)
assert keep_indices is not None, "กรุณาเติมโค้ดในข้อที่ 2"
assert set(keep_indices) == {0, 2}, f"ควรเก็บกล่อง [0, 2] แต่ได้ {keep_indices}"
print("✅ ข้อที่ 2 ผ่าน! (สร้างอัลกอริทึม Non-Maximum Suppression สำเร็จ)")


# %% [markdown]
# ---
# ## โจทย์ข้อที่ 3: โหลดและรันโมเดล YOLOv8 บน GPU RTX 3060
#
# เราจะใช้ไลบรารี **Ultralytics** ซึ่งเป็นเฟรมเวิร์ก Object Detection ยอดนิยมระดับโลก
# ตัวโมเดล `yolov8n.pt` (Nano) จะถูกดาวน์โหลดอัตโนมัติในโค้ด (ขนาดเพียง 6 MB)
#
# **ภารกิจ:**
# เขียนฟังก์ชัน `detect_with_yolo(model: YOLO, img_bgr: np.ndarray, conf: float = 0.25) -> list[dict]`:
# 1. รัน Inference ผ่านโมเดล: `results = model(img_bgr, conf=conf, device=device, verbose=False)`
# 2. ดึงกล่องที่ตรวจพบจากผลลัพธ์แรก: `boxes = results[0].boxes`
# 3. วนลูปผ่านทุกการตรวจจับ:
#    - ดึงพิกัด: `xyxy = box.xyxy[0].cpu().numpy().tolist()` -> `[x1, y1, x2, y2]`
#    - ดึงความมั่นใจ: `confidence = float(box.conf[0].cpu().numpy())`
#    - ดึง Class ID และ Class Name:
#      `cls_id = int(box.cls[0].cpu().numpy())`
#      `class_name = model.names[cls_id]`
#    - เก็บเป็น dict:
#      `{"box": xyxy, "confidence": confidence, "class_id": cls_id, "class_name": class_name}`
# 4. คืนค่าเป็น `list` ของผลลัพธ์ทั้งหมด

# %%
def detect_with_yolo(model: YOLO, img_bgr: np.ndarray, conf: float = 0.25) -> list[dict]:
    """ส่งภาพเข้าตรวจจับวัตถุด้วยโมเดล YOLOv8 บน GPU และสกัดผลลัพธ์โครงสร้างข้อมูล"""
    # TODO: เขียนโค้ดตรงนี้
    results = model(img_bgr, conf=conf, device=device, verbose=False)

    boxes = results[0].boxes

    detections = []
    for box in boxes:
        xyxy = box.xyxy[0].cpu().numpy().tolist()
        confidence = float(box.conf[0].cpu().numpy())
        cls_id = int(box.cls[0].cpu().numpy())
        class_name = model.names[cls_id]

        detections.append({
            "box": xyxy,
            "confidence": confidence,
            "class_id": cls_id,
            "class_name": class_name
        })

    return detections



# --- ทดสอบโจทย์ข้อที่ 3 ---
# สร้างภาพทดสอบจำลองที่มีรถ 2 คันและคน 1 คน หรือดาวน์โหลดภาพตัวอย่างของ Ultralytics อัตโนมัติ
print("⚡ กำลังโหลดโมเดล YOLOv8 และทดสอบการตรวจจับบน GPU...")
yolo_model = YOLO("yolov8n.pt")  # ดาวน์โหลดอัตโนมัติในครั้งแรก
sample_bus_url = "https://ultralytics.com/images/bus.jpg"

# โหลดภาพตัวอย่างจาก URL
import urllib.request
urllib.request.urlretrieve(sample_bus_url, "bus_sample.jpg")
bus_img = cv2.imread("bus_sample.jpg")

detections = detect_with_yolo(yolo_model, bus_img, conf=0.3)

assert detections is not None, "กรุณาเติมโค้ดในข้อที่ 3"
assert len(detections) > 0, "โมเดลควรตรวจพบวัตถุในภาพ bus.jpg อย่างน้อย 1 ชิ้น"
# ในภาพ bus.jpg มีคน (person) และรถบัส (bus) อยู่ชัดเจน
detected_classes = {d["class_name"] for d in detections}
assert "person" in detected_classes or "bus" in detected_classes, f"ควรตรวจพบ person หรือ bus แต่พบ: {detected_classes}"
print(f"✅ ข้อที่ 3 ผ่าน! (ตรวจพบวัตถุ {len(detections)} ชิ้นในภาพ bus.jpg ได้แก่ {detected_classes})")


# %% [markdown]
# ---
# ## โจทย์ข้อที่ 4: สรุปและนับจำนวนวัตถุแยกตามประเภท (Object Counter Analytics)
#
# ในงานระดับ Production เช่น นับรถที่วิ่งผ่านกล้อง หรือนับสินค้าในโกดัง
# เราต้องสรุปสถิติจำนวนชิ้นของแต่ละประเภทออกมาเป็น Structured Summary
#
# **ภารกิจ:**
# เขียนฟังก์ชัน `summarize_detections(detections: list[dict]) -> dict[str, int]`:
# - รับรายการ `detections` จากข้อที่ 3
# - นับความถี่ของแต่ละ `class_name`
# - คืนค่าเป็น Dictionary เช่น `{"person": 4, "bus": 1}`

# %%
def summarize_detections(detections: list[dict]) -> dict[str, int]:
    """นับจำนวนวัตถุแยกตามประเภท (Object Counting)"""
    # TODO: เขียนโค้ดตรงนี้
    
    summary = {}
    for d in detections:
        if summary.get(d["class_name"]):
            summary[d["class_name"]] += 1
        else:
            summary[d["class_name"]] = 1

    return summary



# --- ทดสอบโจทย์ข้อที่ 4 ---
counts = summarize_detections(detections)
assert counts is not None, "กรุณาเติมโค้ดในข้อที่ 4"
assert isinstance(counts, dict)
assert sum(counts.values()) == len(detections), "ยอดรวมจำนวนวัตถุต้องเท่ากับความยาวของ detections"
print(f"✅ ข้อที่ 4 ผ่าน! (สรุปจำนวนวัตถุในภาพสำเร็จ: {counts})")
print("\n🎉 ยินดีด้วย! คุณผ่านการประเมิน Object Detection & YOLOv8 ใน Module 4 ครบทุกข้อแล้ว!")

# %%
