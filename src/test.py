from PIL import Image

from ocr import EasyOCRExtractor, ExtractorData
import easyocr
import cv2
import numpy as np
import supervision as sv

img_path = "data/test/images/700.jpg"
img = Image.open(img_path)

# Image path


# Initialize EasyOCR reader (English language, CPU)
reader = easyocr.Reader(['en'], gpu=False)

# Perform text detection on the image
result = reader.readtext(img_path)

# Load image using OpenCV
image = cv2.imread(img_path)

# Prepare lists for bounding boxes, confidences, class IDs, and labels
xyxy, confidences, class_ids, label = [], [], [], []

# Extract data from OCR result
for detection in result:
    bbox, text, confidence = detection[0], detection[1], detection[2]
   
    # Convert bounding box format
    x_min = int(min([point[0] for point in bbox]))
    y_min = int(min([point[1] for point in bbox]))
    x_max = int(max([point[0] for point in bbox]))
    y_max = int(max([point[1] for point in bbox]))
   
    # Append data to lists
    xyxy.append([x_min, y_min, x_max, y_max])
    label.append(text)
    confidences.append(confidence)
    class_ids.append(0)  

# Convert to NumPy arrays
detections = sv.Detections(
    xyxy=np.array(xyxy),
    confidence=np.array(confidences),
    class_id=np.array(class_ids)
)

# Annotate image with bounding boxes and labels
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

annotated_image = box_annotator.annotate(scene=image, detections=detections)
annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=label)

# Display and save the annotated image
sv.plot_image(image=annotated_image)
cv2.imwrite("Output.jpg", annotated_image)