import easyocr
import numpy as np
from typing import List, Dict, Tuple, Union
from pathlib import Path
from PIL import Image
import cv2
import supervision as sv


class OCR:
    def __init__(self, languages: List[str] = None, gpu: bool = True):
        self.reader = easyocr.Reader(languages or ["en"], gpu=gpu)

    @staticmethod
    def _load_bgr(img: Union[str, Path, np.ndarray, Image.Image]) -> np.ndarray:
        if isinstance(img, (str, Path)):
            arr = cv2.imread(str(img))
        elif isinstance(img, Image.Image):
            arr = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
        else:  # np.ndarray
            arr = img.copy()
        return arr

    def extract(
        self, img: Union[str, Path, np.ndarray, Image.Image]
    ) -> List[Dict[str, Union[str, float, List[Tuple[int, int]]]]]:
        bgr = self._load_bgr(img)
        raw = self.reader.readtext(bgr)
        return [
            {"bbox": bbox, "text": txt.strip(), "confidence": float(conf)}
            for bbox, txt, conf in raw
        ]

    def annotate(
        self,
        img: Union[str, Path, np.ndarray, Image.Image],
        results: List[Dict[str, Union[str, float, List[Tuple[int, int]]]]] = None,
        *,
        box_thickness: int = 2,
    ) -> np.ndarray:
        bgr = self._load_bgr(img)
        results = results or self.extract(bgr)
        xyxy = []
        for res in results:
            bbox = res["bbox"]
            xs = [p[0] for p in bbox]
            ys = [p[1] for p in bbox]
            x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
            xyxy.append([x0, y0, x1, y1])

        if xyxy:
            detections = sv.Detections(
                xyxy=np.asarray(xyxy, dtype=int),
                confidence=np.ones(len(xyxy)),
                class_id=np.zeros(len(xyxy), dtype=int),
            )
            annotated = sv.BoxAnnotator(thickness=box_thickness).annotate(
                bgr, detections
            )
        else:
            annotated = bgr
        return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
