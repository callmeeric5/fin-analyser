import easyocr
import numpy as np
from typing import List, Dict, Tuple, Union
from pathlib import Path
from PIL import Image
import cv2
import supervision as sv


class OCR:
    def __init__(self, languages: List[str] | None = None, gpu: bool = True):
        self.reader = easyocr.Reader(languages or ["en"], gpu=gpu)

    @staticmethod
    def _load_bgr(img: Union[str, Path, np.ndarray, Image.Image]) -> np.ndarray:
        if isinstance(img, (str, Path)):
            arr = cv2.imread(str(img))
        elif isinstance(img, Image.Image):
            arr = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
        else:  # assume np.ndarray
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
        results: (
            List[Dict[str, Union[str, float, List[Tuple[int, int]]]]] | None
        ) = None,
        *,
        scale: float = 0.5,  # 50 % size by default; set to 1 for original size
        show_confidence: bool = True,
    ) -> np.ndarray:
        bgr = self._load_bgr(img)
        results = results or self.extract(bgr)

        xyxy, confs, labels = [], [], []
        for res in results:
            xs = [p[0] for p in res["bbox"]]
            ys = [p[1] for p in res["bbox"]]
            x0, y0, x1, y1 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
            xyxy.append([x0, y0, x1, y1])
            confs.append(res["confidence"])
            labels.append(
                f"{res['text']} ({res['confidence']:.2f})"
                if show_confidence
                else res["text"]
            )

        detections = sv.Detections(
            xyxy=np.asarray(xyxy, dtype=int),
            confidence=np.asarray(confs, dtype=float),
            class_id=np.zeros(len(xyxy), dtype=int),
        )
        annotated = sv.BoxAnnotator(thickness=5).annotate(bgr, detections)
        annotated = sv.LabelAnnotator(text_scale=2, smart_position=True).annotate(
            annotated, detections, labels
        )

        if 0 < scale < 1:
            h, w = annotated.shape[:2]
            annotated = cv2.resize(
                annotated,
                (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_AREA,
            )
        rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        return rgb
