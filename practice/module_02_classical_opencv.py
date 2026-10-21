# %% [markdown]
# # Module 2: Classical Image Processing, Thresholding & Contours (OpenCV)
#
# ยินดีด้วยที่ผ่าน Module 1! ตอนนี้คุณเข้าใจแล้วว่า:
# - การคูณ Filter บนภาพทำงานอย่างไร (Convolution)
# - การยุบขนาดภาพทำอย่างไร (Max Pooling)
#
# ใน Module 2 นี้ เราจะเริ่มใช้ **`OpenCV` (`cv2`)** ซึ่งเป็นไลบรารีมาตรฐานระดับโลกด้าน Computer Vision
# เพื่อสร้าง **Classical Object Detection Pipeline** สำหรับค้นหา, วัดขนาด, และนับจำนวนวัตถุในภาพ!
#
# ### กับดักที่พบบ่อยที่สุดใน OpenCV:
# - OpenCV อ่านและเขียนภาพด้วยระบบสี **`BGR` (Blue, Green, Red)** ไม่ใช่ `RGB`!
# - ถ้าเอาภาพ BGR ไปพล็อตด้วย `plt.imshow()` ของ Matplotlib หน้าคนจะกลายเป็นสีฟ้าทันที!

# %%
import cv2
import numpy as np
import matplotlib.pyplot as plt


def create_shapes_test_image(size: int = 256) -> np.ndarray:
    """
    สร้างภาพทดสอบที่มีวัตถุรูปทรงต่างๆ พร้อมสัญญาณรบกวน (Noise)
    เพื่อใช้ทดสอบ Pipeline การกรองสัญญาณ และการค้นหาวัตถุ (Contours)
    """
    img = np.zeros((size, size, 3), dtype=np.uint8)
    # วัตถุที่ 1: วงกลมสีขาวขนาดใหญ่ตรงกลาง (รัศมี 35)
    cv2.circle(img, (size // 2, size // 2), 35, (255, 255, 255), -1)
    # วัตถุที่ 2: สี่เหลี่ยมสีขาวมุมซ้ายบน
    cv2.rectangle(img, (30, 30), (80, 80), (255, 255, 255), -1)
    # วัตถุที่ 3: สี่เหลี่ยมผืนผ้าสีขาวมุมขวาล่าง
    cv2.rectangle(img, (160, 170), (230, 210), (255, 255, 255), -1)

    # เพิ่มจุดสัญญาณรบกวนเล็กๆ (Salt-and-Pepper Noise) เพื่อทดสอบการลบ Noise
    rng = np.random.default_rng(42)
    noise_points = rng.integers(0, size, size=(40, 2))
    for pt in noise_points:
        img[pt[0], pt[1]] = 255

    return img


def plot_cv_steps(images: list[np.ndarray], titles: list[str]) -> None:
    """ฟังก์ชันช่วยพล็อตแสดงสเต็ปการประมวลผลของ OpenCV เรียงต่อกัน"""
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, im, title in zip(axes, images, titles):
        if im.ndim == 2:
            ax.imshow(im, cmap="gray")
        else:
            # ถ้าเป็นภาพ 3 มิติ ตรวจสอบว่าเป็น RGB ก่อนพล็อต
            ax.imshow(im)
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    plt.show()


# %% [markdown]
# ---
# ## โจทย์ข้อที่ 1: การจัดการระบบสี (BGR $\leftrightarrow$ RGB Conversion)
#
# **ภารกิจ:**
# เขียนฟังก์ชัน `fix_bgr_to_rgb(img_bgr: np.ndarray) -> np.ndarray`
# เพื่อสลับช่องสีจาก BGR เป็น RGB ให้ถูกต้องสำหรับการนำไปแสดงผลบน Matplotlib หรือส่งเข้าโมเดล AI
#
# **เงื่อนไข:**
# - ใช้ฟังก์ชัน `cv2.cvtColor(...)` ร่วมกับแฟล็ก `cv2.COLOR_BGR2RGB`
# - คืนค่าเป็นอาเรย์ที่มีขนาดเท่าเดิม แต่ช่องสีสลับเป็น R, G, B เรียบร้อย

# %%
def fix_bgr_to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    """แปลงภาพจากระบบสี BGR เป็น RGB ด้วย cv2.cvtColor"""
    # TODO: เขียนโค้ดตรงนี้
    pass


# --- ทดสอบโจทย์ข้อที่ 1 ---
# จำลองภาพ BGR สีแดงสด: B=0, G=0, R=255
bgr_red = np.zeros((10, 10, 3), dtype=np.uint8)
bgr_red[:, :, 2] = 255  # ใน BGR ช่องที่ 2 คือ Red

rgb_converted = fix_bgr_to_rgb(bgr_red)
assert rgb_converted is not None, "กรุณาเติมโค้ดในข้อที่ 1"
assert rgb_converted.shape == bgr_red.shape
# ใน RGB ช่องที่ 0 ต้องกลายเป็นสีแดง 255 และช่องที่ 2 ต้องกลายเป็น 0
assert rgb_converted[0, 0, 0] == 255, f"ช่องที่ 0 ควรเป็นสีแดง 255 แต่ได้ {rgb_converted[0, 0, 0]}"
assert rgb_converted[0, 0, 2] == 0, f"ช่องที่ 2 ควรเป็นสีน้ำเงิน 0 แต่ได้ {rgb_converted[0, 0, 2]}"
print("✅ ข้อที่ 1 ผ่าน! (เข้าใจความต่างของระบบสี BGR vs RGB เรียบร้อย)")


# %% [markdown]
# ---
# ## โจทย์ข้อที่ 2: การลบสัญญาณรบกวนด้วย Gaussian Blur & Morphological Operations
#
# ในงานจริง ภาพมักจะมีจุดรบกวน (Noise) หรือเส้นขอบที่ไม่เรียบ เราจึงต้องทำความสะอาดภาพก่อน
#
# **ภารกิจ:**
# เขียนฟังก์ชัน `denoise_and_clean(img_gray: np.ndarray) -> np.ndarray`:
# 1. ทำภาพให้เรียบเนียนด้วย `cv2.GaussianBlur(img_gray, ksize=(5, 5), sigmaX=0)`
# 2. แปลงภาพเป็นขาว-ดำสนิท (Binary Thresholding) ด้วยวิธี **Otsu's Thresholding**:
#    `_, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)`
# 3. กำจัดจุดรบกวนเล็กๆ ด้วย **Morphological Opening** (Erosion ตามด้วย Dilation):
#    - สร้างเคอร์เนลวงกลม/สี่เหลี่ยมขนาด `3x3`: `kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))`
#    - ใช้ `cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)`
# 4. คืนค่าเป็นภาพ Binary Mask ขนาด `(H, W)` ที่สะอาดเรียบร้อย

# %%
def denoise_and_clean(img_gray: np.ndarray) -> np.ndarray:
    """ทำความสะอาดภาพด้วย Gaussian Blur + Otsu Thresholding + Morphological Opening"""
    # TODO: เขียนโค้ดตรงนี้
    pass


# --- ทดสอบโจทย์ข้อที่ 2 ---
test_canvas = create_shapes_test_image(256)
gray_canvas = cv2.cvtColor(test_canvas, cv2.COLOR_BGR2GRAY)
clean_mask = denoise_and_clean(gray_canvas)

assert clean_mask is not None, "กรุณาเติมโค้ดในข้อที่ 2"
assert clean_mask.shape == gray_canvas.shape
assert clean_mask.dtype == np.uint8
# จุด Noise เล็กๆ ที่ตำแหน่งสุ่มควรถูกลบหายไปหมดแล้ว (พิกเซลส่วนใหญ่ในพื้นหลังต้องเป็น 0)
assert np.sum(clean_mask == 255) < np.sum(gray_canvas > 0), "Noise เล็กๆ ควรถูกลบออกไปหลังทำ Morphological Opening"
print("✅ ข้อที่ 2 ผ่าน! (ทำความสะอาดภาพและสกัด Binary Mask สำเร็จ)")
# plot_cv_steps([gray_canvas, clean_mask], ["Input Grayscale", "Cleaned Mask (Otsu + Morph)"])


# %% [markdown]
# ---
# ## โจทย์ข้อที่ 3: สกัดโครงร่าง (Contours), กรองขนาด และคำนวณ Bounding Box
#
# **Contour (เส้นรอบรูป)** คือการเชื่อมต่อพิกเซลสีขาวที่อยู่ติดกันเป็นขอบเขตของวัตถุ 1 ชิ้น
#
# **ภารกิจ:**
# เขียนฟังก์ชัน `detect_objects(binary_mask: np.ndarray, min_area: float = 500.0) -> list[dict]`:
# 1. หา Contours จาก `binary_mask`:
#    `contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)`
# 2. วนลูปผ่านทุก `cnt` ใน `contours`:
#    - คำนวณพื้นที่ของวัตถุ: `area = cv2.contourArea(cnt)`
#    - กรองเฉพาะวัตถุที่มีพื้นที่ $\ge$ `min_area` (เพื่อตัดเศษขยะเล็กๆ ทิ้ง)
#    - ดึงพิกัดกรอบสี่เหลี่ยม Bounding Box: `x, y, w, h = cv2.boundingRect(cnt)`
#    - เก็บข้อมูลลงใน dictionary:
#      `{"x": int(x), "y": int(y), "w": int(w), "h": int(h), "area": float(area)}`
# 3. คืนค่าเป็น `list` ของ dictionaries ของวัตถุที่ตรวจพบ

# %%
def detect_objects(binary_mask: np.ndarray, min_area: float = 500.0) -> list[dict]:
    """ค้นหาวัตถุในภาพ กรองด้วยขนาดพื้นที่ และคืนค่าพิกัด Bounding Box"""
    # TODO: เขียนโค้ดตรงนี้
    pass


# --- ทดสอบโจทย์ข้อที่ 3 ---
detected_items = detect_objects(clean_mask, min_area=500.0)

assert detected_items is not None, "กรุณาเติมโค้ดในข้อที่ 3"
# ในภาพจำลองของเรามีวัตถุหลักขนาดใหญ่อยู่ 3 ชิ้น (วงกลมกลาง, สี่เหลี่ยมบนซ้าย, สี่เหลี่ยมล่างขวา)
assert len(detected_items) == 3, f"ควรตรวจพบวัตถุหลัก 3 ชิ้น แต่ตรวจพบ {len(detected_items)} ชิ้น"
for obj in detected_items:
    assert all(k in obj for k in ("x", "y", "w", "h", "area")), "คีย์ใน dictionary ไม่ครบถ้วน"
    assert obj["area"] >= 500.0, "มีวัตถุขนาดเล็กกว่า min_area หลุดรอดมา"

print(f"✅ ข้อที่ 3 ผ่าน! (ตรวจพบวัตถุ {len(detected_items)} ชิ้นพร้อมพิกัด Bounding Box ถูกต้อง)")


# %% [markdown]
# ---
# ## โจทย์ข้อที่ 4: วาดกรอบ Bounding Box และป้ายกำกับลงบนภาพ (Annotation Pipeline)
#
# **ภารกิจ:**
# เขียนฟังก์ชัน `draw_detections(img_rgb: np.ndarray, objects: list[dict]) -> np.ndarray`:
# 1. ทำการ Copy ภาพต้นฉบับ: `annotated = img_rgb.copy()`
# 2. วนลูปผ่านแต่ละวัตถุใน `objects`:
#    - ดึงพิกัด `(x, y, w, h)`
#    - วาดกรอบสี่เหลี่ยมสีเขียว (ใน RGB คือ `(0, 255, 0)` หนา 2 พิกเซล) ด้วย `cv2.rectangle`:
#      `cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)`
#    - ใส่ข้อความป้ายกำกับบอกขนาดพื้นที่ เช่น `f"Area: {int(obj['area'])}"` ด้วย `cv2.putText`:
#      `cv2.putText(annotated, f"Area: {int(obj['area'])}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)`
# 3. คืนค่าภาพ `annotated` ที่วาดกรอบเสร็จเรียบร้อย

# %%
def draw_detections(img_rgb: np.ndarray, objects: list[dict]) -> np.ndarray:
    """วาดกรอบสี่เหลี่ยมและป้ายกำกับลงบนภาพสำหรับโชว์ผลลัพธ์ Object Detection"""
    # TODO: เขียนโค้ดตรงนี้
    pass


# --- ทดสอบโจทย์ข้อที่ 4 ---
annotated_result = draw_detections(test_canvas, detected_items)
assert annotated_result is not None, "กรุณาเติมโค้ดในข้อที่ 4"
assert annotated_result.shape == test_canvas.shape
# ภาพผลลัพธ์ต้องมีเส้นสีเขียว (0, 255, 0) ถูกวาดเพิ่มเข้ามา
assert np.any((annotated_result[:, :, 1] == 255) & (annotated_result[:, :, 0] == 0)), "ไม่พบเส้นตีกรอบสีเขียวในภาพ"

print("✅ ข้อที่ 4 ผ่าน! (สร้าง Pipeline การตีกรอบและป้ายกำกับวัตถุสำเร็จ)")
print("\n🎉 ยอดเยี่ยมมาก! คุณสร้าง Classical Object Detection Pipeline ด้วย OpenCV สำเร็จครบทั้ง 4 ขั้นตอนแล้ว!")
# plot_cv_steps([test_canvas, clean_mask, annotated_result], ["Original Image", "Binary Mask", "Detected Objects"])
