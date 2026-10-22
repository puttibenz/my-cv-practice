# %% [markdown]
# # Module 3: Deep Learning for Vision & CNN from Scratch on GPU (PyTorch)
#
# ยินดีต้อนรับสู่โลก **Deep Learning for Computer Vision**!
# ใน Module 1 คุณได้เขียน Convolution และ Max Pooling ด้วย NumPy ด้วยตัวเองแล้ว
# ใน Module 2 คุณได้ใช้ OpenCV สกัด Contours และ Bounding Box
#
# ตอนนี้เราจะกระโดดมาที่ **PyTorch** เพื่อฝึกฝนการสร้าง **Convolutional Neural Network (CNN)**
# และรัน Training บนการ์ดจอ **NVIDIA GeForce RTX 3060 (12GB VRAM)** บนเครื่องของคุณ!
#
# ### ความแตกต่างสำคัญระหว่าง OpenCV vs PyTorch:
# | คุณสมบัติ | OpenCV / NumPy | PyTorch Tensor |
# | :--- | :--- | :--- |
# | **การเรียงแกนภาพ** | `(Height, Width, Channels)` เช่น `(28, 28, 3)` | `(Batch, Channels, Height, Width)` เช่น `(B, 3, 28, 28)` |
# | **ช่วงค่าพิกเซล** | `uint8` (0 ถึง 255) | `float32` ปรับค่าเป็นมาตรฐาน (0.0 ถึง 1.0) |
# | **การประมวลผล** | CPU ล้วน | รันขนานบน **CUDA GPU (RTX 3060)** เร็วกว่า CPU 20-50 เท่า! |

# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

# ตรวจสอบการเชื่อมต่อการ์ดจอ RTX 3060
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 PyTorch Device: {device}")
if device.type == "cuda":
    print(f"   GPU Model: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")


# %% [markdown]
# ---
# ## โจทย์ข้อที่ 1: การแปลงมิติภาพจาก NumPy `(H, W, C)` เป็น PyTorch Tensor `(B, C, H, W)`
#
# นี่คืองานที่ Computer Vision Engineer ต้องทำทุกวัน: รับภาพจากกล้อง/OpenCV มาเป็น NumPy แล้วแปลงเข้าโมเดล PyTorch
#
# **ภารกิจ:**
# เขียนฟังก์ชัน `numpy_to_torch_batch(img_numpy: np.ndarray, target_device: torch.device) -> torch.Tensor`:
# 1. ปรับสเกลค่าพิกเซลจาก `[0, 255]` ให้กลายเป็น `[0.0, 1.0]` ชนิด `float32`:
#    `img_float = img_numpy.astype(np.float32) / 255.0`
# 2. แปลงเป็น PyTorch Tensor ด้วย `torch.from_numpy(...)`
# 3. สลับแกน (Permute) จาก `(H, W, C)` $\rightarrow$ `(C, H, W)`:
#    `tensor_chw = tensor.permute(2, 0, 1)`
# 4. เพิ่มมิติ Batch เข้าไปข้างหน้าให้กลายเป็น `(1, C, H, W)`:
#    `batch_tensor = tensor_chw.unsqueeze(0)`
# 5. ย้าย Tensor ไปไว้บน `target_device` (`.to(target_device)`) แล้วคืนค่า

# %%
def numpy_to_torch_batch(img_numpy: np.ndarray, target_device: torch.device) -> torch.Tensor:
    """แปลงภาพ NumPy (H, W, C) เป็น PyTorch Batch Tensor (1, C, H, W) บน Device ที่กำหนด"""
    # TODO: เขียนโค้ดตรงนี้
    pass


# --- ทดสอบโจทย์ข้อที่ 1 ---
dummy_img = np.random.randint(0, 256, (64, 48, 3), dtype=np.uint8)
t_batch = numpy_to_torch_batch(dummy_img, device)

assert t_batch is not None, "กรุณาเติมโค้ดในข้อที่ 1"
assert t_batch.shape == (1, 3, 64, 48), f"ขนาด Shape ต้องเป็น (1, 3, 64, 48) แต่ได้ {t_batch.shape}"
assert t_batch.dtype == torch.float32, f"ชนิดข้อมูลต้องเป็น float32 แต่ได้ {t_batch.dtype}"
assert t_batch.min() >= 0.0 and t_batch.max() <= 1.0, "ค่าพิกเซลต้องถูก Normalize ให้อยู่ในช่วง 0.0 ถึง 1.0"
assert t_batch.device.type == device.type, f"Tensor ต้องอยู่บน {device} แต่พบว่าอยู่บน {t_batch.device}"
print(f"✅ ข้อที่ 1 ผ่าน! (แปลงมิติภาพเป็น (1, C, H, W) และส่งขึ้น {device} สำเร็จ)")


# %% [markdown]
# ---
# ## โจทย์ข้อที่ 2: ออกแบบสถาปัตยกรรม CNN Architecture จากศูนย์ด้วย PyTorch
#
# เราจะสร้างโมเดลจัดหมวดหมู่ภาพชื่อ **`SimpleCNN`** สำหรับจำแนกภาพขนาด `(1, 28, 28)` (เช่น เสื้อผ้า รองเท้า กระเป๋า 10 หมวดหมู่):
#
# **โครงสร้างโมเดล:**
# 1. **Block 1 (Conv + BatchNorm + ReLU + MaxPool):**
#    - `nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)`  # ขนาดคงที่ 28x28
#    - `nn.BatchNorm2d(16)`
#    - `nn.ReLU()`
#    - `nn.MaxPool2d(kernel_size=2, stride=2)`  # ย่อขนาดลงครึ่งหนึ่งเหลือ 14x14
# 2. **Block 2 (Conv + BatchNorm + ReLU + MaxPool):**
#    - `nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)` # ขนาดคงที่ 14x14
#    - `nn.BatchNorm2d(32)`
#    - `nn.ReLU()`
#    - `nn.MaxPool2d(kernel_size=2, stride=2)`  # ย่อขนาดลงครึ่งหนึ่งเหลือ 7x7
# 3. **Classifier (Fully-Connected Layers):**
#    - ยุบมิติด้วย `nn.Flatten()` (จะได้เวกเตอร์ขนาด $32 \times 7 \times 7 = 1568$)
#    - `nn.Linear(32 * 7 * 7, 64)` + `nn.ReLU()`
#    - `nn.Linear(64, num_classes)` (คืนค่า Logits 10 คลาส)

# %%
class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        # TODO: กำหนดเลเยอร์ต่างๆ ภายใน __init__
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: ส่งผ่านข้อมูล x ตามลำดับบล็อกใน forward
        pass


# --- ทดสอบโจทย์ข้อที่ 2 ---
model = SimpleCNN(num_classes=10).to(device)
dummy_input = torch.randn(4, 1, 28, 28, device=device)  # จำลอง Batch ขนาด 4 รูป
dummy_output = model(dummy_input)

assert dummy_output is not None, "กรุณาเติมโค้ดในข้อที่ 2"
assert dummy_output.shape == (4, 10), f"ผลลัพธ์ Logits ต้องมีขนาด (4, 10) แต่ได้ {dummy_output.shape}"
print("✅ ข้อที่ 2 ผ่าน! (โครงสร้าง CNN สมบูรณ์ มิติ Forward pass ถูกต้อง)")


# %% [markdown]
# ---
# ## โจทย์ข้อที่ 3: โหลดชุดข้อมูล FashionMNIST อัตโนมัติ (Zero-Friction Dataset)
#
# ไม่ต้องเสียเวลาไปดาวน์โหลดไฟล์จากเว็บข้างนอก! PyTorch มีระบบดาวน์โหลดชุดข้อมูลมาตรฐานใน 1 บรรทัด
#
# **ภารกิจ:**
# เขียนฟังก์ชัน `get_fashion_mnist_loaders(batch_size: int = 64)`:
# 1. กำหนด `transform = transforms.Compose([transforms.ToTensor()])` (แปลงเป็น Tensor 0.0 - 1.0 อัตโนมัติ)
# 2. โหลด Train Dataset:
#    `torchvision.datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)`
# 3. โหลด Test Dataset:
#    `torchvision.datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)`
# 4. สร้างและคืนค่า `(train_loader, test_loader)` ด้วย `DataLoader(..., batch_size=batch_size, shuffle=True/False)`

# %%
def get_fashion_mnist_loaders(batch_size: int = 64) -> tuple[DataLoader, DataLoader]:
    """ดาวน์โหลดและสร้าง DataLoader สำหรับชุดข้อมูล FashionMNIST อัตโนมัติ"""
    # TODO: เขียนโค้ดตรงนี้
    pass


# --- ทดสอบโจทย์ข้อที่ 3 ---
train_loader, test_loader = get_fashion_mnist_loaders(batch_size=64)
sample_images, sample_labels = next(iter(train_loader))

assert sample_images.shape == (64, 1, 28, 28), f"Batch shape ต้องเป็น (64, 1, 28, 28) แต่ได้ {sample_images.shape}"
assert len(sample_labels) == 64
print(f"✅ ข้อที่ 3 ผ่าน! (ดาวน์โหลดและเตรียม DataLoader จำนวน {len(train_loader.dataset)} ภาพเรียบร้อย)")


# %% [markdown]
# ---
# ## โจทย์ข้อที่ 4: การเทรน CNN 1 Epoch บน GPU RTX 3060 พร้อมคำนวณ Loss & Accuracy
#
# **ภารกิจ:**
# เขียนฟังก์ชัน `train_one_epoch(model, loader, optimizer, criterion, device) -> tuple[float, float]`:
# 1. ตั้งโมเดลเข้าสู่โหมดเทรน: `model.train()`
# 2. สร้างตัวแปรเก็บ `total_loss = 0.0`, `correct = 0`, `total = 0`
# 3. วนลูป `for images, labels in loader:`:
#    - ย้ายข้อมูลไปที่ GPU: `images, labels = images.to(device), labels.to(device)`
#    - ล้าง Gradient เดิม: `optimizer.zero_grad()`
#    - ยิง Forward Pass: `outputs = model(images)`
#    - คำนวณ Loss: `loss = criterion(outputs, labels)`
#    - ทำ Backward Pass: `loss.backward()`
#    - อัปเดต Weights: `optimizer.step()`
#    - สะสม `total_loss += loss.item()`
#    - คำนวณความแม่นยำ: หาคลาสที่โมเดลทายด้วย `preds = outputs.argmax(dim=1)`
#      และสะสม `correct += (preds == labels).sum().item()`, `total += labels.size(0)`
# 4. คืนค่า `(avg_loss, accuracy)`

# %%
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """รัน Training Loop 1 Epoch และคืนค่า (Average Loss, Accuracy)"""
    # TODO: เขียนโค้ดตรงนี้
    pass


# --- ทดสอบโจทย์ข้อที่ 4 (รันเทรนจริงบน RTX 3060) ---
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("⚡ กำลังรัน Training Loop บน GPU RTX 3060...")
avg_loss, acc = train_one_epoch(model, train_loader, optimizer, criterion, device)

assert avg_loss is not None and acc is not None, "กรุณาเติมโค้ดในข้อที่ 4"
assert avg_loss > 0.0, "Loss ต้องเป็นบวก"
assert acc >= 0.70, f"โมเดล CNN ควรได้ Accuracy ใน Epoch แรกอย่างน้อย 70% แต่ได้ {acc*100:.2f}%"

print(f"✅ ข้อที่ 4 ผ่าน! [Epoch 1] Loss: {avg_loss:.4f} | Accuracy: {acc*100:.2f}%")
print("\n🎉 สุดยอดมาก! คุณฝึกสอนโมเดล CNN บนการ์ดจอ RTX 3060 สำเร็จเรียบร้อยแล้ว!")
