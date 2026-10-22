# %% [markdown]
# # Module 5: ROI Cropping, Segmentation & Visual Analytics Pipeline
#
# ใน Module 4 คุณได้ตรวจจับวัตถุด้วย **YOLOv8** บน GPU RTX 3060 ไปแล้ว
# แต่ในงานจริง การรู้แค่พิกัดยังไม่พอ เราต้องนำภาพวัตถุที่ตรวจจับได้ไป "ต่อยอด" ในเชิงธุรกิจ เช่น:
# - การตัดเฉพาะตัววัตถุออกมา (**ROI Cropping**) เพื่อส่งไปอ่านป้ายทะเบียน, ตรวจรอยตำหนิ, หรือจำแนกสี
# - การทำ **Mask Overlay (Alpha Blending)** ระบายสีโปร่งแสงทับวัตถุแบบสวยงาม
# - การนับวัตถุแยกตามโซนพื้นที่ (**Spatial Zone Analytics / Virtual Tripwire**)
# - การสรุปผลลัพธ์เป็น **Structured Analytics JSON Payload** พร้อมส่งออก API

# %%
import cv2
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
from ultralytics import YOLO

# โหลดภาพตัวอย่างและโมเดล YOLOv8
yolo_model = YOLO("yolov8n.pt")
sample_url = "https://ultralytics.com/images/bus.jpg"
urllib.request.urlretrieve(sample_url, "bus_sample.jpg")
bus_image = cv2.imread("bus_sample.jpg")


# %% [markdown]
# ---
# ## โจทย์ข้อที่ 1: การตัดชิ้นส่วนภาพวัตถุ (Region of Interest - ROI Cropping)
#
# ในงาน Computer Vision เรามักจะต้อง Crop วัตถุที่ตรวจจับได้ออกมาทีละชิ้น (เช่น Crop คนเพื่อส่งเข้าโมเดลจดจำใบหน้า)
#
# **สิ่งที่ต้องระวัง:** พิกัด Bounding Box บางครั้งอาจล้นขอบภาพ (เช่น $x_1 < 0$ หรือ $x_2 > Width$) ต้องทำการ **Clip ขอบเขต** เสมอ!
#
# **ภารกิจ:**
# เขียนฟังก์ชัน `crop_objects(img_bgr: np.ndarray, detections: list[dict]) -> list[np.ndarray]`:
# 1. ดึงขนาดภาพ `H, W = img_bgr.shape[:2]`
# 2. วนลูปผ่านแต่ละวัตถุใน `detections`:
#    - ดึงพิกัด `[x1, y1, x2, y2] = d["box"]`
#    - ป้องกันพิกัดล้นขอบภาพ:
#      `x1_safe = max(0, min(int(x1), W))`
#      `y1_safe = max(0, min(int(y1), H))`
#      `x2_safe = max(0, min(int(x2), W))`
#      `y2_safe = max(0, min(int(y2), H))`
#    - ตัดภาพด้วยการ Slice: `crop = img_bgr[y1_safe:y2_safe, x1_safe:x2_safe]`
#    - ถ้าพื้นที่ `crop` มีขนาดกว้างและสูง $> 0$ ให้เพิ่มเข้าในลิสต์ผลลัพธ์
# 3. คืนค่าเป็น `list` ของภาพ Crop ทั้งหมด

# %%
def crop_objects(img_bgr: np.ndarray, detections: list[dict]) -> list[np.ndarray]:
    """ตัดชิ้นส่วนภาพวัตถุ (ROI) จากพิกัด Bounding Box พร้อมป้องกันพิกัดล้นขอบภาพ"""
    # TODO: เขียนโค้ดตรงนี้
    pass


# --- ทดสอบโจทย์ข้อที่ 1 ---
# จำลองผลลัพธ์การตรวจจับ 2 วัตถุในภาพ bus_sample.jpg
sample_dets = [
    {"box": [0.0, 200.0, 200.0, 700.0], "class_name": "bus"},
    {"box": [200.0, 380.0, 300.0, 700.0], "class_name": "person"},
]
cropped_list = crop_objects(bus_image, sample_dets)

assert cropped_list is not None, "กรุณาเติมโค้ดในข้อที่ 1"
assert len(cropped_list) == 2, f"ควรตัดภาพวัตถุออกมาได้ 2 ชิ้น แต่ได้ {len(cropped_list)}"
assert cropped_list[0].ndim == 3 and cropped_list[0].shape[2] == 3
print("✅ ข้อที่ 1 ผ่าน! (ตัดภาพ ROI พร้อมระบบ Safe Boundary Clipping สำเร็จ)")


# %% [markdown]
# ---
# ## โจทย์ข้อที่ 2: การทำสีไฮไลต์โปร่งแสงทับวัตถุ (Semi-Transparent Overlay ด้วย Alpha Blending)
#
# ในระบบ AI หน้าจอแสดงผลระดับ Professional มักจะระบายสีโปร่งแสงทับวัตถุ เพื่อให้ภาพดูน่าใช้งาน
# สมการ Alpha Blending ของ OpenCV คือ:
# $$\text{Output} = \alpha \cdot \text{Overlay} + (1 - \alpha) \cdot \text{Original} + \gamma$$
#
# **ภารกิจ:**
# เขียนฟังก์ชัน `apply_color_overlay(img_bgr: np.ndarray, box: list[float], color: tuple[int, int, int] = (0, 255, 0), alpha: float = 0.4) -> np.ndarray`:
# 1. สร้างภาพก็อปปี้ `overlay = img_bgr.copy()`
# 2. ระบายสีทึบลงบนพื้นที่กล่องของ `overlay`:
#    `x1, y1, x2, y2 = map(int, box)`
#    `cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)`  # -1 คือถมสีทึบทั้งสี่เหลี่ยม
# 3. ผสมภาพด้วย `cv2.addWeighted(overlay, alpha, img_bgr, 1 - alpha, 0)` แล้วคืนค่า

# %%
def apply_color_overlay(
    img_bgr: np.ndarray,
    box: list[float],
    color: tuple[int, int, int] = (0, 255, 0),
    alpha: float = 0.4,
) -> np.ndarray:
    """ระบายสีโปร่งแสงไฮไลต์ทับตำแหน่งวัตถุด้วย cv2.addWeighted"""
    # TODO: เขียนโค้ดตรงนี้
    pass


# --- ทดสอบโจทย์ข้อที่ 2 ---
test_box = [100.0, 100.0, 300.0, 400.0]
blended_img = apply_color_overlay(bus_image, test_box, color=(0, 255, 0), alpha=0.4)

assert blended_img is not None, "กรุณาเติมโค้ดในข้อที่ 2"
assert blended_img.shape == bus_image.shape
# ในบริเวณกล่อง ค่าสีเขียว (ช่องที่ 1 ใน BGR) ต้องสว่างขึ้นจากการผสมสีโปร่งแสง
assert blended_img[200, 200, 1] >= bus_image[200, 200, 1], "พื้นที่ในกล่องต้องถูกผสมสีเขียวเพิ่มขึ้น"
print("✅ ข้อที่ 2 ผ่าน! (สร้างไฮไลต์โปร่งแสง Alpha Blending สำเร็จ)")


# %% [markdown]
# ---
# ## โจทย์ข้อที่ 3: ระบบนับวัตถุแยกตามโซนพื้นที่ (Spatial Zone Analytics)
#
# ในห้างสรรพสินค้า หรือสี่แยกไฟแดง เรามักจะแบ่งภาพออกเป็นโซน (เช่น "Zone A: เลนซ้าย", "Zone B: เลนขวา")
# และนับว่ามีวัตถุอยู่ในแต่ละโซนกี่ชิ้น โดยดูจาก **จุดศูนย์กลางของวัตถุ (Center Point)**
# $$c_x = \frac{x_1 + x_2}{2}, \quad c_y = \frac{y_1 + y_2}{2}$$
#
# **ภารกิจ:**
# เขียนฟังก์ชัน `count_objects_in_zone(detections: list[dict], zone_box: list[float]) -> int`:
# - `zone_box` คือพิกัดของพื้นที่โซน `[zx1, zy1, zx2, zy2]`
# - สำหรับแต่ละวัตถุใน `detections`:
#   - หาจุดกึ่งกลาง $(c_x, c_y)$ ของวัตถุ
#   - ตรวจสอบว่า $z_{x1} \le c_x \le z_{x2}$ และ $z_{y1} \le c_y \le z_{y2}$ หรือไม่
#   - ถ้านับเข้าโซน ให้บวกคะแนน $+1$
# - คืนค่าจำนวนเต็มของวัตถุที่อยู่ในโซน

# %%
def count_objects_in_zone(detections: list[dict], zone_box: list[float]) -> int:
    """นับจำนวนวัตถุที่จุดศูนย์กลางตกลงในโซนพื้นที่ที่กำหนด"""
    # TODO: เขียนโค้ดตรงนี้
    pass


# --- ทดสอบโจทย์ข้อที่ 3 ---
mock_detections = [
    {"box": [50.0, 50.0, 150.0, 150.0], "class_name": "car"},     # center = (100, 100) -> อยู่ในโซน
    {"box": [120.0, 120.0, 180.0, 180.0], "class_name": "car"}, # center = (150, 150) -> อยู่ในโซน
    {"box": [400.0, 400.0, 500.0, 500.0], "class_name": "car"}, # center = (450, 450) -> อยู่นอกโซน
]
zone_a = [0.0, 0.0, 200.0, 200.0]
zone_count = count_objects_in_zone(mock_detections, zone_a)

assert zone_count is not None, "กรุณาเติมโค้ดในข้อที่ 3"
assert zone_count == 2, f"ควรมีวัตถุอยู่ในโซน 2 ชิ้น แต่ได้ {zone_count}"
print("✅ ข้อที่ 3 ผ่าน! (ระบบวิเคราะห์วัตถุในโซนพื้นที่ Spatial Analytics สำเร็จ)")


# %% [markdown]
# ---
# ## โจทย์ข้อที่ 4: สรุปข้อมูลส่งออกระบบ Production (Structured Analytics JSON)
#
# **ภารกิจ:**
# เขียนฟังก์ชัน `build_analytics_report(detections: list[dict], zones: dict[str, list[float]]) -> dict`:
# 1. นับจำนวนวัตถุทั้งหมด: `total_objects = len(detections)`
# 2. นับจำนวนแยกตามประเภท (`counts_by_class`) เช่น `{"car": 2, "person": 1}`
# 3. นับจำนวนวัตถุในแต่ละโซน (`counts_by_zone`) โดยใช้ฟังก์ชัน `count_objects_in_zone` จากข้อที่ 3
# 4. คืนค่าผลลัพธ์เป็น Dictionary ที่มีโครงสร้างดังนี้:
#    ```python
#    {
#        "total_objects": int,
#        "counts_by_class": dict[str, int],
#        "counts_by_zone": dict[str, int],
#    }
#    ```

# %%
def build_analytics_report(
    detections: list[dict], zones: dict[str, list[float]]
) -> dict:
    """สร้างรายงานสรุปสถิติการตรวจจับวัตถุสำหรับส่งออก REST API ระดับ Production"""
    # TODO: เขียนโค้ดตรงนี้
    pass


# --- ทดสอบโจทย์ข้อที่ 4 ---
zones_config = {
    "zone_left": [0.0, 0.0, 200.0, 500.0],
    "zone_right": [200.0, 0.0, 600.0, 500.0],
}
report = build_analytics_report(mock_detections, zones_config)

assert report is not None, "กรุณาเติมโค้ดในข้อที่ 4"
assert report["total_objects"] == 3
assert "counts_by_class" in report and "counts_by_zone" in report
assert report["counts_by_zone"]["zone_left"] == 2
assert report["counts_by_zone"]["zone_right"] == 0
print("✅ ข้อที่ 4 ผ่าน! (สร้าง Structured Analytics Payload สำหรับ Production สำเร็จ)")
print("\n🎉 ยอดเยี่ยมมาก! คุณผ่าน Module 5 ครบทุกหัวข้อ และพร้อมสร้าง Production Capstone Microservice แล้ว!")
