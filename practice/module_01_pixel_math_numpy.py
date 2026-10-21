# %% [markdown]
# # Module 1: Pixel Math, Color Spaces & 2D Convolution from Scratch (Pure NumPy)
# 
# ยินดีต้อนรับสู่คอร์ส **Computer Vision Engineering**!
# ในโมดูลแรกนี้ เราจะผ่าตัดดูโครงสร้างของ "รูปภาพดิจิทัล" ในระดับพิกเซลและคณิตศาสตร์เบื้องหลัง
# โดยใช้ **`NumPy` ล้วนๆ** ก่อนที่จะไปใช้ฟังก์ชันสำเร็จรูปใน OpenCV และ PyTorch
#
# ### ความรู้พื้นฐานที่ต้องรู้ก่อนเริ่ม:
# 1. **รูปภาพคือตารางตัวเลข (3D Array):**
#    - ภาพสี RGB มีมิติเป็น `(Height, Width, Channels)` หรือ `(H, W, 3)`
#    - ค่าพิกเซลมาตรฐานเก็บเป็นชนิดข้อมูล `np.uint8` มีค่าตั้งแต่ `0` (มืดสุด) ถึง `255` (สว่างสุด)
# 2. **กับดัก Integer Overflow ใน `uint8`:**
#    - ถ้าเอาพิกเซล `uint8` ที่มีค่า `250 + 20` ใน NumPy ค่าจะทะลุ 255 แล้ววนกลับไปเป็น `14` (มืดตึ๋อ!)
#    - ดังนั้นเวลาคำนวณคณิตศาสตร์ ต้องแปลงเป็น `float32` ก่อนเสมอ แล้วค่อย `np.clip(..., 0, 255).astype(np.uint8)`

# %%
import numpy as np
import matplotlib.pyplot as plt


def create_synthetic_test_image(size: int = 64) -> np.ndarray:
    """
    ฟังก์ชันสร้างภาพทดสอบจำลองขนาด (size, size, 3) แบบอัตโนมัติ (ไม่ต้องโหลดไฟล์จากข้างนอก)
    ประกอบด้วย:
      - พื้นหลังไล่เฉดสี
      - สี่เหลี่ยมสีแดงสด (Red Square)
      - สี่เหลี่ยมสีเขียวสด (Green Square)
      - วงกลมสีน้ำเงินสด (Blue Circle)
    """
    img = np.zeros((size, size, 3), dtype=np.uint8)
    # สี่เหลี่ยมสีแดงที่มุมซ้ายบน
    img[8:28, 8:28, 0] = 240
    # สี่เหลี่ยมสีเขียวที่มุมขวาบน
    img[8:28, 36:56, 1] = 240
    # วงกลมสีน้ำเงินตรงกลางล่าง
    y, x = np.ogrid[:size, :size]
    mask = (x - size // 2) ** 2 + (y - int(size * 0.7)) ** 2 <= 12**2
    img[mask, 2] = 250
    # เส้นขอบสีขาวตรงกลางเพื่อทดสอบ Edge Detection
    img[30:34, 10:54, :] = 255
    return img


def plot_before_after(
    img1: np.ndarray,
    title1: str,
    img2: np.ndarray,
    title2: str,
    cmap2: str | None = "gray",
) -> None:
    """ฟังก์ชันช่วยแสดงผลภาพเปรียบเทียบ Before / After ใน Interactive Cell"""
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(img1, cmap="gray" if img1.ndim == 2 else None)
    axes[0].set_title(title1)
    axes[0].axis("off")

    axes[1].imshow(img2, cmap=cmap2 if img2.ndim == 2 else None)
    axes[1].set_title(title2)
    axes[1].axis("off")
    plt.tight_layout()
    plt.show()


# %% [markdown]
# ---
# ## โจทย์ข้อที่ 1: แปลงภาพ RGB เป็น Grayscale ด้วยสมการความไวแสงของตามนุษย์
# 
# ดวงตาของมนุษย์มีความไวต่อสีเขียวมากที่สุด รองลงมาคือสีแดง และไวต่อสีน้ำเงินน้อยที่สุด
# ดังนั้น มาตรฐานสากล (ITU-R BT.601 ที่ใช้ใน OpenCV) จึงไม่ใช้วิธีหาค่าเฉลี่ย `(R+G+B)/3`
# แต่ใช้สมการถ่วงน้ำหนัก (Weighted Luminance):
# $$Y = 0.299 \cdot R + 0.587 \cdot G + 0.114 \cdot B$$
#
# **ภารกิจ:**
# เขียนฟังก์ชัน `rgb_to_grayscale(img_rgb)` โดยใช้ Vectorization ของ NumPy (ห้ามใช้ลูป `for`)
# - **Input:** `img_rgb` ขนาด `(H, W, 3)` ชนิด `uint8`
# - **Output:** อาเรย์ 2 มิติ ขนาด `(H, W)` ชนิด `np.float32`

# %%
def rgb_to_grayscale(img_rgb: np.ndarray) -> np.ndarray:
    """
    แปลงภาพสี RGB (H, W, 3) เป็นภาพขาวดำ Grayscale (H, W) ชนิด float32
    คำใบ้:
      - ดึงช่องสี R = img_rgb[:, :, 0], G = img_rgb[:, :, 1], B = img_rgb[:, :, 2]
        หรือใช้ dot product: img_rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
      - อย่าลืมแปลงผลลัพธ์เป็น .astype(np.float32)
    """
    # TODO: เขียนโค้ดตรงนี้
    dot_product = img_rgb @ np.array([0.299, 0.587, 0.114])

    return dot_product.astype(np.float32)

# --- ทดสอบโจทย์ข้อที่ 1 ---
test_img = create_synthetic_test_image(64)
gray_img = rgb_to_grayscale(test_img)

assert gray_img is not None, "ฟังก์ชันยังคืนค่า None อยู่ กรุณาเติมโค้ดในข้อที่ 1"
assert gray_img.shape == (64, 64), f"มิติภาพต้องเป็น (64, 64) แต่ได้ {gray_img.shape}"
assert gray_img.dtype == np.float32, f"ชนิดข้อมูลต้องเป็น float32 แต่ได้ {gray_img.dtype}"
# พิกเซลสีเขียว (240) ต้องดูสว่างกว่าพิกเซลสีน้ำเงิน (250) ในภาพขาวดำ!
green_lum = gray_img[15, 45]  # 0.587 * 240 = 140.88
blue_lum = gray_img[45, 32]   # 0.114 * 250 = 28.50
assert np.isclose(green_lum, 0.587 * 240, atol=1e-2), f"ค่าความสว่างสีเขียวไม่ตรง: {green_lum}"
assert np.isclose(blue_lum, 0.114 * 250, atol=1e-2), f"ค่าความสว่างสีน้ำเงินไม่ตรง: {blue_lum}"
print("✅ ข้อที่ 1 ผ่าน! (สีเขียวสว่างกว่าสีน้ำเงินตามหลักสรีรวิทยาของดวงตามนุษย์)")
plot_before_after(test_img, "Original RGB", gray_img, "Grayscale (Luminance)")


# %% [markdown]
# ---
# ## โจทย์ข้อที่ 2: ปรับความสว่างและคอนทราสต์อย่างปลอดภัย (Brightness & Contrast Clipping)
#
# สมการเชิงเส้นสำหรับปรับแต่งภาพคือ:
# $$I_{\text{out}} = \text{clip}(\alpha \cdot I_{\text{in}} + \beta, \; 0, \; 255)$$
# - $\alpha$ (`alpha`): ค่า Contrast ($\alpha > 1$ ภาพจัดจ้านขึ้น, $\alpha < 1$ ภาพหม่นลง)
# - $\beta$ (`beta`): ค่า Brightness ($\beta > 0$ ภาพสว่างขึ้น, $\beta < 0$ ภาพมืดลง)
#
# **ภารกิจ:**
# เขียนฟังก์ชัน `adjust_brightness_contrast(img, alpha, beta)`:
# 1. แปลง `img` เป็น `np.float32` ก่อนคำนวณ เพื่อป้องกัน Integer Overflow!
# 2. คำนวณ `alpha * img + beta`
# 3. ใช้ `np.clip(..., 0, 255)` บังคับค่าให้อยู่ในช่วง `[0, 255]`
# 4. แปลงกลับเป็น `.astype(np.uint8)` แล้วคืนค่า

# %%
def adjust_brightness_contrast(
    img: np.ndarray, alpha: float = 1.0, beta: float = 0.0
) -> np.ndarray:
    """ปรับความสว่างและคอนทราสต์ของภาพพร้อมป้องกันปัญหา uint8 overflow"""
    # TODO: เขียนโค้ดตรงนี้
    img_float = img.astype(np.float32)

    adjusted = alpha * img_float + beta

    out = np.clip(adjusted, 0, 255)

    return out.astype(np.uint8)


# --- ทดสอบโจทย์ข้อที่ 2 ---
bright_img = adjust_brightness_contrast(test_img, alpha=1.2, beta=40.0)
assert bright_img is not None, "กรุณาเติมโค้ดในข้อที่ 2"
assert bright_img.dtype == np.uint8, f"ต้องคืนค่าเป็น uint8 แต่ได้ {bright_img.dtype}"
assert bright_img.shape == test_img.shape
# พิกเซลสีแดงเดิม 240 -> 240 * 1.2 + 40 = 328 -> เมื่อถูก clip ต้องได้ 255 พอดี (ไม่ล้นกลับเป็นเลขน้อย)
assert bright_img[15, 15, 0] == 255, f"เกิด Overflow! ค่าที่ได้คือ {bright_img[15, 15, 0]} แทนที่จะเป็น 255"
# พิกเซลสีดำเดิม 0 -> 0 * 1.2 + 40 = 40
assert bright_img[0, 0, 0] == 40, f"ค่าความสว่างพื้นหลังควรเป็น 40 แต่ได้ {bright_img[0, 0, 0]}"
print("✅ ข้อที่ 2 ผ่าน! (ป้องกัน uint8 overflow ด้วย np.clip สำเร็จ)")
# plot_before_after(test_img, "Original RGB", bright_img, "Alpha=1.2, Beta=+40", cmap2=None)


# %% [markdown]
# ---
# ## โจทย์ข้อที่ 3: สร้าง 2D Spatial Convolution จากศูนย์ (`conv2d_numpy`)
#
# **Convolution** คือหัวใจของทั้งการทำ Image Filtering (ทำให้เบลอ, คมชัด, หาเส้นขอบ) และเป็นเลเยอร์หลักของ **CNN (Convolutional Neural Network)**!
#
# **หลักการทำงาน:**
# 1. เรามีภาพ 2 มิติ `image` ขนาด `(H, W)` และหน้าต่างตัวกรอง `kernel` ขนาด `(kH, kW)` (โดยทั่วไปเป็นเลขคี่ เช่น `3x3`)
# 2. เพื่อให้ภาพผลลัพธ์มีขนาด `(H, W)` เท่าเดิม (**Same Padding**) เราต้องเติมขอบเลขศูนย์ (`Zero Padding`) รอบภาพ:
#    - `pad_h = kH // 2`, `pad_w = kW // 2`
#    - ใช้ `padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode="constant")`
# 3. เลื่อนหน้าต่างไปทุกพิกัด `(i, j)` ของภาพผลลัพธ์:
#    - ดึงพื้นที่ย่อย `patch = padded[i : i + kH, j : j + kW]`
#    - คำนวณผลรวมการคูณแบบจุดต่อจุด: `output[i, j] = np.sum(patch * kernel)`

# %%
def conv2d_numpy(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    คำนวณ 2D Spatial Convolution (Cross-Correlation) แบบ Same Padding
    Input:
        image: 2D array ขนาด (H, W)
        kernel: 2D array ขนาด (kH, kW)
    Output:
        output: 2D array ขนาด (H, W) ชนิด np.float32
    """
    # TODO: เขียนโค้ดตรงนี้
    h, w = image.shape

    kh, kw = kernel.shape
    pad_h = kh // 2
    pad_w = kw // 2

    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode="constant")

    output = np.zeros((h, w), dtype=np.float32)
    for i in range(h):
        for j in range(w):
            patch = padded[i : i + kh, j : j + kw]
            output[i, j] = np.sum(patch * kernel)

    return output 


# --- ทดสอบโจทย์ข้อที่ 3 (Box Blur & Sobel Edge Detection) ---
# 3.1 ทดสอบกับ Sobel Kernels เพื่อสกัดเส้นขอบแนวดิ่ง (Gx) และแนวราบ (Gy)
sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

gx = conv2d_numpy(gray_img, sobel_x)
gy = conv2d_numpy(gray_img, sobel_y)

assert gx is not None and gy is not None, "กรุณาเติมโค้ดในข้อที่ 3"
assert gx.shape == gray_img.shape, f"ขนาดภาพหลังทำ Convolution ต้องเท่าเดิม {gray_img.shape} แต่ได้ {gx.shape}"
assert gx.dtype == np.float32, f"ชนิดข้อมูลควรเป็น float32 แต่ได้ {gx.dtype}"

# คำนวณความแรงของเส้นขอบรวม (Edge Magnitude): G = sqrt(Gx^2 + Gy^2)
edge_magnitude = np.sqrt(gx**2 + gy**2)
# บริเวณเส้นขอบของแถบสีขาว (แถวที่ 30) ต้องมีความแรงของขอบสูงกว่าพื้นที่เรียบๆ (แถวที่ 2)
assert edge_magnitude[30, 32] > 500.0, f"ควรตรวจจับเส้นขอบแนวนอนได้ชัดเจน แต่ได้ค่า {edge_magnitude[30, 32]}"
assert edge_magnitude[2, 2] == 0.0, "พื้นที่สีดำเรียบๆ ต้องมีค่าขอบเป็น 0"
print("✅ ข้อที่ 3 ผ่าน! (สร้าง 2D Convolution และสกัดเส้นขอบด้วย Sobel Filter สำเร็จ)")
# plot_before_after(gray_img, "Grayscale Input", edge_magnitude, "Sobel Edge Magnitude")


# %% [markdown]
# ---
# ## โจทย์ข้อที่ 4: สร้าง 2D Max Pooling จากศูนย์ (`maxpool2d_numpy`)
#
# ในโครงข่ายประสาทเทียม **CNN** หลังจากผ่านชั้น Convolution แล้ว เรามักจะลดขนาดภาพลงครึ่งหนึ่งด้วยชั้น **Max Pooling**
# เพื่อให้โมเดลประมวลผลเร็วขึ้น และสนใจเฉพาะจุดที่เด่นที่สุด (Maximum Activation) ในแต่ละบล็อก
#
# **หลักการทำงาน:**
# - รับ `feature_map` ขนาด `(H, W)` และขนาดหน้าต่าง `pool_size` (เช่น `2`)
# - แบ่งภาพเป็นบล็อกขนาด `pool_size x pool_size` แบบไม่ทับซ้อนกัน (ก้าวทีละ `pool_size`)
# - ภาพผลลัพธ์จะมีขนาด `(H // pool_size, W // pool_size)`
# - พิกัด `(i, j)` ของภาพผลลัพธ์ เก็บค่าสูงสุด (`np.max`) ของบล็อก:
#   `feature_map[i * pool_size : (i + 1) * pool_size, j * pool_size : (j + 1) * pool_size]`

# %%
def maxpool2d_numpy(feature_map: np.ndarray, pool_size: int = 2) -> np.ndarray:
    """
    ลดขนาดภาพ 2D ลงด้วยวิธี Max Pooling (Stride = pool_size)
    Input:
        feature_map: 2D array ขนาด (H, W)
        pool_size: ขนาดหน้าต่าง pooling (ค่าเริ่มต้น 2)
    Output:
        pooled: 2D array ขนาด (H // pool_size, W // pool_size) ชนิด np.float32
    """
    # TODO: เขียนโค้ดตรงนี้
    H, W = feature_map.shape
    h, w = H // pool_size, W // pool_size

    pooled = np.zeros((h, w), dtype=np.float32)

    for i in range(h):
        for j in range(w):
            patch = feature_map[i * pool_size : (i + 1) * pool_size, j * pool_size : (j + 1) * pool_size]
            pooled[i, j] = np.max(patch)

    return pooled


    


# --- ทดสอบโจทย์ข้อที่ 4 ---
pooled_edges = maxpool2d_numpy(edge_magnitude, pool_size=2)
assert pooled_edges is not None, "กรุณาเติมโค้ดในข้อที่ 4"
assert pooled_edges.shape == (32, 32), f"ขนาดหลังทำ 2x2 MaxPool จาก (64, 64) ต้องได้ (32, 32) แต่ได้ {pooled_edges.shape}"

# ทดสอบความถูกต้องด้วยตาราง 4x4 ที่รู้ค่าแน่นอน
sample_4x4 = np.array(
    [
        [1, 5, 2, 0],
        [3, 4, 8, 1],
        [9, 2, 0, 3],
        [1, 0, 4, 7],
    ],
    dtype=np.float32,
)
expected_2x2 = np.array([[5, 8], [9, 7]], dtype=np.float32)
assert np.array_equal(maxpool2d_numpy(sample_4x4, pool_size=2), expected_2x2), "ค่า Max Pooling ยังไม่ถูกต้อง"
print("✅ ข้อที่ 4 ผ่าน! (สร้าง 2D Max Pooling สำเร็จ)")
print("\n🎉 ยินดีด้วย! คุณผ่านการทดสอบคณิตศาสตร์ระดับพิกเซลใน Module 1 ครบทุกข้อแล้ว!")
# plot_before_after(edge_magnitude, "Original Edges (64x64)", pooled_edges, "2x2 Max Pooled Edges (32x32)")

# %%
