# src/ocr.py

import easyocr
from dataclasses import dataclass
from ultralytics import YOLO
import pytesseract
import cv2
import numpy as np
from PIL import Image
import re

@dataclass
class ExtractorData:
    product_name: str = ""
    bill_id: str = ""
    phone: str = ""
    address: str = ""
    datetime: str = ""
    sub_tprice: str = ""
    tprice: str = ""
    tdiscount: str = ""
    tax: str = ""
    discount: str = ""
    discount_percentage: str = ""
    fprice: str = ""

LABEL_MAP = {
    "PRODUCT_NAME": "product_name",
    "BILLID": "bill_id",
    "PHONE": "phone",
    "ADDRESSS": "address",
    "ADDRESS": "address",
    "DATETIME": "datetime",
    "SUB_TPRICE": "sub_tprice",
    "TPRICE": "tprice",
    "TDISCOUNT": "tdiscount",
    "TAX": "tax",
    "DISCOUNT": "discount",
    "DISCOUNT_PERCENTAGE": "discount_percentage",
    "FPRICE": "fprice",
}

NUMERIC_FIELDS = {
    "sub_tprice",
    "tprice",
    "tdiscount",
    "tax",
    "discount",
    "discount_percentage",
    "fprice",
}

class YoloExtractor:
    def __init__(self, model_path: str, tess_config: str):
        self.model = YOLO(model_path)
        self.tess_config = tess_config

    def _preprocess(self, image: Image.Image) -> Image.Image:
        # Grayscale + CLAHE + adaptive threshold
        img = image.convert("L")
        arr = np.array(img)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        arr = clahe.apply(arr)
        th = cv2.adaptiveThreshold(
            arr, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11, 2
        )
        return Image.fromarray(th)

    # alias for backward compatibility
    _grey_scale = _preprocess

    def extract(
        self,
        image: Image.Image,
        imgsz: int = 640,
        conf: float = 0.4,
        iou: float = 0.45,
        pad: int = 8
    ) -> ExtractorData:
        # Run YOLO inference with confidence and NMS filtering
        results = self.model(image, imgsz=imgsz, conf=conf, iou=iou)
        w_img, h_img = image.size

        # Keep only the largest box per class
        best_boxes = {}
        for r in results:
            for box in r.boxes:
                idx = int(box.cls[0])
                label = self.model.names[idx]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                # Apply padding and clamp
                x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
                x2, y2 = min(w_img, x2 + pad), min(h_img, y2 + pad)
                area = (x2 - x1) * (y2 - y1)
                if label not in best_boxes or area > best_boxes[label][0]:
                    best_boxes[label] = (area, (x1, y1, x2, y2))

        data = ExtractorData()

        for label, (_, (x1, y1, x2, y2)) in best_boxes.items():
            crop = image.crop((x1, y1, x2, y2))

            # Handle PRODUCTS rows separately
            if label == "PRODUCTS":
                w_c, h_c = crop.size
                split1 = int(0.6 * w_c)
                split2 = int(0.8 * w_c)
                # Product name slice
                slice_name = crop.crop((0, 0, split1, h_c))
                pre_name = self._preprocess(slice_name)
                text_name = pytesseract.image_to_string(
                    pre_name,
                    config=self.tess_config
                ).strip()
                data.product_name = self._clean(text_name)
                # Sub-total price slice
                slice_price = crop.crop((split2, 0, w_c, h_c))
                pre_price = self._preprocess(slice_price)
                text_price = pytesseract.image_to_string(
                    pre_price,
                    config=self.tess_config + " -c tessedit_char_whitelist=0123456789£$.,"
                ).strip()
                data.sub_tprice = self._clean(text_price)
                continue

            field_name = LABEL_MAP.get(label)
            if not field_name:
                continue

            pre = self._preprocess(crop)
            # Choose config based on field type
            if field_name in NUMERIC_FIELDS:
                config = self.tess_config + " -c tessedit_char_whitelist=0123456789£$.,"
            else:
                config = self.tess_config

            raw_text = pytesseract.image_to_string(pre, config=config).strip()
            setattr(data, field_name, self._clean(raw_text))

        return data

    def _clean(self, txt: str) -> str:
        # Strip leading/trailing non-word characters (but keep currency symbols)
        return re.sub(r"^[^\w£$%]+|[^\w£$%]+$", "", txt).strip()


class EasyOCRExtractor:
    def __init__(self, langs=None, gpu=False):
        self.reader = easyocr.Reader(langs or ["en"], gpu=gpu)

    def extract(self, image: Image.Image) -> ExtractorData:
        arr = np.array(image)
        results = self.reader.readtext(arr)
        data = ExtractorData()
        h = arr.shape[0]

        for bbox, text, prob in results:
            y0 = min(pt[1] for pt in bbox)
            txt = text.strip()

            # Header region (top 20%)
            if y0 < h * 0.2:
                key = txt.lower().split()[0]
                if key.startswith("bill"):
                    data.bill_id = txt
                elif "date" in key:
                    data.datetime = txt

            # Footer region (bottom 20%) — look for final price
            elif y0 > h * 0.8:
                if any(sym in txt for sym in ["£", "$"]):
                    data.fprice = txt if not data.fprice else data.fprice

            # Middle region — product names
            else:
                data.product_name = txt if not data.product_name else data.product_name + "; " + txt

        return data, results
