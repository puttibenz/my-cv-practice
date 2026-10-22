"""vision_service.app.pipeline
Core computer vision pipeline combining OpenCV, YOLO GPU inference,
ROI cropping, and spatial zone analytics.
"""

import time
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
import torch
from ultralytics import YOLO

from vision_service.app.schemas import (
    AnalyticsSummary,
    BoundingBox,
    Detection,
    VisionInspectionResponse,
)


class VisionPipeline:
    """Production Vision Pipeline สำหรับตรวจสอบภาพและวิเคราะห์วัตถุเชิงพื้นที่"""

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        device: Optional[str] = None,
    ):
        if device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"[VisionPipeline] กำลังโหลดโมเดล {model_path} บนอุปกรณ์ {self.device}...")
        self.model = YOLO(model_path)
        # ทำการ Warmup โมเดล 1 ครั้งเพื่อโหลด CUDA Kernels
        self._warmup()

    def _warmup(self) -> None:
        """Warmup โมเดลด้วยภาพเปล่า เพื่อลด Latency ใน Request แรก"""
        dummy = np.zeros((320, 320, 3), dtype=np.uint8)
        self.model(dummy, device=self.device, verbose=False)

    def decode_image(self, image_bytes: bytes) -> np.ndarray:
        """แปลง Raw Bytes เป็นภาพ OpenCV BGR array พร้อมระบบตรวจสอบความถูกต้อง"""
        if not image_bytes or len(image_bytes) == 0:
            raise ValueError("ข้อมูลรูปภาพว่างเปล่า (Empty image payload)")

        np_arr = np.frombuffer(image_bytes, np.uint8)
        img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img_bgr is None or img_bgr.size == 0:
            raise ValueError("ไม่สามารถถอดรหัสรูปภาพได้ ไฟล์อาจเสียหายหรือไม่ใช่รูปแบบภาพ")

        return img_bgr

    def run_inference(
        self,
        img_bgr: np.ndarray,
        conf_threshold: float = 0.25,
    ) -> Tuple[List[Detection], float]:
        """ประมวลผล Object Detection ผ่าน YOLO และจับเวลา Inference Time (ms)"""
        start_time = time.perf_counter()

        results = self.model(
            img_bgr,
            conf=conf_threshold,
            device=self.device,
            verbose=False,
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        detections: List[Detection] = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            xyxy_list = boxes.xyxy.cpu().numpy()
            conf_list = boxes.conf.cpu().numpy()
            cls_list = boxes.cls.cpu().numpy()
            names = results[0].names

            for xyxy, conf, cls_id in zip(xyxy_list, conf_list, cls_list):
                x1, y1, x2, y2 = [float(v) for v in xyxy]
                class_id = int(cls_id)
                class_name = str(names.get(class_id, f"class_{class_id}"))

                bbox = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)
                detections.append(
                    Detection(
                        class_id=class_id,
                        class_name=class_name,
                        confidence=float(conf),
                        box=bbox,
                    )
                )

        return detections, elapsed_ms

    def compute_analytics(
        self,
        detections: List[Detection],
        zones: Optional[Dict[str, List[float]]] = None,
    ) -> AnalyticsSummary:
        """คำนวณสถิติการตรวจจับและการแจกแจงตามโซนพื้นที่ (Spatial Analytics)"""
        total = len(detections)

        # 1. นับแยกตามประเภทวัตถุ
        counts_by_class: Dict[str, int] = {}
        for det in detections:
            counts_by_class[det.class_name] = (
                counts_by_class.get(det.class_name, 0) + 1
            )

        # 2. นับวัตถุที่จุดศูนย์กลางตกลงในแต่ละโซน
        counts_by_zone: Dict[str, int] = {}
        if zones:
            for zone_name, zbox in zones.items():
                zx1, zy1, zx2, zy2 = [float(v) for v in zbox]
                zone_count = 0
                for det in detections:
                    cx, cy = det.box.center
                    if zx1 <= cx <= zx2 and zy1 <= cy <= zy2:
                        zone_count += 1
                counts_by_zone[zone_name] = zone_count

        return AnalyticsSummary(
            total_detections=total,
            counts_by_class=counts_by_class,
            counts_by_zone=counts_by_zone,
        )

    def crop_rois(
        self,
        img_bgr: np.ndarray,
        detections: List[Detection],
    ) -> List[np.ndarray]:
        """ตัดภาพ Region of Interest (ROI) โดยมี Safe Boundary Clipping ป้องกัน Index Error"""
        H, W = img_bgr.shape[:2]
        crops: List[np.ndarray] = []

        for det in detections:
            x1_safe = max(0, min(int(det.box.x1), W))
            y1_safe = max(0, min(int(det.box.y1), H))
            x2_safe = max(0, min(int(det.box.x2), W))
            y2_safe = max(0, min(int(det.box.y2), H))

            crop = img_bgr[y1_safe:y2_safe, x1_safe:x2_safe]
            if crop.size > 0:
                crops.append(crop)

        return crops

    def process(
        self,
        image_bytes: bytes,
        zones: Optional[Dict[str, List[float]]] = None,
        conf_threshold: float = 0.25,
    ) -> Tuple[VisionInspectionResponse, np.ndarray]:
        """Orchestrate ข้อมูลตั้งแต่ Bytes สู่ผลวิเคราะห์และภาพ BGR พร้อมใช้งาน"""
        img_bgr = self.decode_image(image_bytes)
        detections, elapsed_ms = self.run_inference(
            img_bgr, conf_threshold=conf_threshold
        )
        analytics = self.compute_analytics(detections, zones=zones)

        response = VisionInspectionResponse(
            success=True,
            image_shape=img_bgr.shape,
            detections=detections,
            analytics=analytics,
            inference_time_ms=round(elapsed_ms, 2),
        )

        return response, img_bgr
