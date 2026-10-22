"""vision_service.app.schemas
Data contracts and Pydantic v2 validation models for the Vision Microservice.
"""

from typing import Dict, List, Tuple
from pydantic import BaseModel, Field, model_validator


class BoundingBox(BaseModel):
    """พิกัด Bounding Box ในรูปแบบ [x1, y1, x2, y2]"""

    x1: float = Field(..., description="พิกัดแกน X ซ้ายสุด")
    y1: float = Field(..., description="พิกัดแกน Y บนสุด")
    x2: float = Field(..., description="พิกัดแกน X ขวาสุด")
    y2: float = Field(..., description="พิกัดแกน Y ล่างสุด")

    @model_validator(mode="after")
    def validate_coordinates(self) -> "BoundingBox":
        if self.x2 < self.x1:
            raise ValueError(f"x2 ({self.x2}) ต้องมากกว่าหรือเท่ากับ x1 ({self.x1})")
        if self.y2 < self.y1:
            raise ValueError(f"y2 ({self.y2}) ต้องมากกว่าหรือเท่ากับ y1 ({self.y1})")
        return self

    @property
    def center(self) -> Tuple[float, float]:
        """คำนวณจุดกึ่งกลาง (cx, cy) ของ Bounding Box"""
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height


class Detection(BaseModel):
    """ผลการตรวจจับวัตถุ 1 ชิ้นจากโมเดล YOLO"""

    class_id: int = Field(..., description="รหัสประเภทวัตถุ")
    class_name: str = Field(..., description="ชื่อประเภทวัตถุ เช่น person, car")
    confidence: float = Field(..., ge=0.0, le=1.0, description="ค่าความมั่นใจ (0.0 - 1.0)")
    box: BoundingBox = Field(..., description="พิกัด Bounding Box")


class AnalyticsSummary(BaseModel):
    """สรุปสถิติเชิงพื้นที่และการนับจำนวนวัตถุ"""

    total_detections: int = Field(0, description="จำนวนวัตถุที่ตรวจจับได้ทั้งหมด")
    counts_by_class: Dict[str, int] = Field(
        default_factory=dict, description="จำนวนวัตถุแยกตามประเภท"
    )
    counts_by_zone: Dict[str, int] = Field(
        default_factory=dict, description="จำนวนวัตถุแยกตามโซนพื้นที่"
    )


class VisionInspectionResponse(BaseModel):
    """Payload ตอบกลับหลักของ REST API /predict"""

    success: bool = True
    image_shape: Tuple[int, int, int] = Field(
        ..., description="ขนาดของภาพประมวลผล (Height, Width, Channels)"
    )
    detections: List[Detection] = Field(
        default_factory=list, description="รายการวัตถุที่ตรวจพบ"
    )
    analytics: AnalyticsSummary = Field(
        ..., description="สถิติเชิงพื้นที่และการนับจำนวน"
    )
    inference_time_ms: float = Field(
        ..., description="เวลาที่ใช้ในการประมวลผลโมเดล (มิลลิวินาที)"
    )
