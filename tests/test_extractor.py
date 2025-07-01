# tests/test_extractor.py
import pytest
import subprocess
from pathlib import Path
from PIL import Image
import numpy as np

from src.ocr import YoloExtractor, ExtractorData

def has_tesseract():
    try:
        subprocess.run(
            ["tesseract","--version"], check=True,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        return True
    except:
        return False

def has_model_weights():
    return Path("model/best.pt").is_file()

@pytest.fixture
def yolo_extractor():
    path = "model/best.pt"
    if not has_model_weights():
        pytest.skip(f"YOLO weights missing at {path}")
    return YoloExtractor(path, "--oem 1 --psm 7")

def test_yolo_extractor_init(yolo_extractor):
    assert isinstance(yolo_extractor, YoloExtractor)
    assert yolo_extractor.tess_config == "--oem 1 --psm 7"

def test_grey_scale(yolo_extractor):
    w, h = 100, 50
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    img = Image.fromarray(arr, "RGB")

    grey = yolo_extractor._grey_scale(img)
    assert isinstance(grey, Image.Image)
    assert grey.mode == "L"
    assert grey.size == (w, h)
    assert set(np.array(grey).ravel()) <= {0, 255}

def test_extract_end_to_end(yolo_extractor):
    if not has_tesseract():
        pytest.skip("tesseract CLI not available")
    img_path = Path("data/test/images/600.jpg")
    if not img_path.exists():
        pytest.skip(f"Test image missing: {img_path}")

    image = Image.open(img_path)
    data = yolo_extractor.extract(image, imgsz=640)
    print(data)
    assert isinstance(data, ExtractorData)
    # We simply verify that the method returns an ExtractorData, 
    # without asserting on specific field contents.
