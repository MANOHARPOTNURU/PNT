import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ultralytics import YOLO
from scipy.stats import gaussian_kde

# -------------------------------
# STEP 1: Load Image and Model
# -------------------------------
image_path = r"G:/C-folder/Downloads/car-parking-images/6.jpg"
model = YOLO("runs/detect/train/weights/best.pt")
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError("Image not found.")

height, width = image.shape[:2]
total_image_area = height * width

# -------------------------------
# STEP 2: Detect Cars with YOLO
# -------------------------------
results = model.predict(source=image_path, conf=0.1, save=False)
boxes = results[0].boxes

raw_boxes = []
centers = []
areas = []

for box in boxes:
    x1, y1, x2, y2 = box.xyxy[0].tolist()
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    area = (x2 - x1) * (y2 - y1)
    raw_boxes.append([x1, y1, x2, y2])
    centers.append([cx, cy])
    areas.append(area)

# -------------------------------
# STEP 3: Filter Out Small/Large Boxes
# -------------------------------
avg_area = np.mean(areas)
box_area_threshold_low = 0.3 * avg_area
box_area_threshold_high = 2.0 * avg_area

filtered_boxes = []
filtered_centers = []
for i, area in enumerate(areas):
    if box_area_threshold_low < area < box_area_threshold_high:
        filtered_boxes.append(raw_boxes[i])
        filtered_centers.append(centers[i])

# -------------------------------
# STEP 4: Remove Overlapping Boxes (Greedy NMS)
# -------------------------------
def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union = areaA + areaB - interArea
    return interArea / union if union > 0 else 0

final_boxes = []
skip_indices = set()

for i in range(len(filtered_boxes)):
    if i in skip_indices:
        continue
    boxA = filtered_boxes[i]
    final_boxes.append(boxA)

    for j in range(i + 1, len(filtered_boxes)):
        if j in skip_indices:
            continue
        boxB = filtered_boxes[j]
        if compute_iou(boxA, boxB) > 0.1:
            skip_indices.add(j)  # Discard overlapping future box


# -------------------------------
# STEP 5: Identify Usable Area
# -------------------------------
x_coords = [b[0] for b in final_boxes] + [b[2] for b in final_boxes]
y_coords = [b[1] for b in final_boxes] + [b[3] for b in final_boxes]
x_min, x_max = int(min(x_coords)), int(max(x_coords))
y_min, y_max = int(min(y_coords)), int(max(y_coords))

usable_area_mask = np.zeros((height, width), dtype=np.uint8)
us_area_rect = (x_min, y_min, x_max, y_max)
us_area_pixels = (x_max - x_min) * (y_max - y_min)
cv2.rectangle(usable_area_mask, (x_min, y_min), (x_max, y_max), 255, -1)

usable_area_ratio = us_area_pixels / total_image_area

# -------------------------------
# STEP 6: Binary Preprocessing
# -------------------------------
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
_, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# -------------------------------
# STEP 7: Analyze Outside Area
# -------------------------------
inverse_mask = cv2.bitwise_not(usable_area_mask)
outside_mask = cv2.bitwise_and(binary, binary, mask=inverse_mask)
outside_area = np.count_nonzero(inverse_mask)
outside_smooth = np.count_nonzero(outside_mask)
outside_ignored = outside_area / total_image_area < 0.05

avg_final_area = np.median([(b[2] - b[0]) * (b[3] - b[1]) for b in final_boxes])

if not outside_ignored and avg_final_area > 0:
    contours, _ = cv2.findContours(outside_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    est_empty_outside = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 0.5 * avg_final_area < area < 1.5 * avg_final_area:
            est_empty_outside += 1
else:
    est_empty_outside = 0

# -------------------------------
# STEP 8: Analyze Usable Area
# -------------------------------
inside_mask = cv2.bitwise_and(binary, binary, mask=usable_area_mask)
inside_smooth = np.count_nonzero(inside_mask)
occupied_area = sum([(b[2] - b[0]) * (b[3] - b[1]) for b in final_boxes])
estimated_empty_area = max(us_area_pixels - occupied_area, 0)

if avg_final_area > 0:
    estimated_empty = int(estimated_empty_area / avg_final_area)
else:
    estimated_empty = 0

if inside_smooth >= estimated_empty_area:
    status_msg = "✅ All cars in usable area likely detected."
else:
    status_msg = "⚠️ Some cars may not be detected in usable area."

# -------------------------------
# STEP 9: Annotate and Save
# -------------------------------
img_draw = image.copy()
cv2.rectangle(img_draw, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

slot_data = []
for i, box in enumerate(final_boxes):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(img_draw, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img_draw, f"S{i+1}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    slot_data.append([f"S{i+1}", x1, y1, x2, y2, "Occupied"])

for j in range(estimated_empty):
    slot_data.append([f"E{j+1}", -1, -1, -1, -1, "Estimated Empty"])

for j in range(est_empty_outside):
    slot_data.append([f"OE{j+1}", -1, -1, -1, -1, "Estimated Outside Empty"])

# -------------------------------
# STEP 10: Export CSV Report
# -------------------------------
df = pd.DataFrame(slot_data, columns=["Slot_ID", "x1", "y1", "x2", "y2", "Status"])
df.to_csv("parking_slot_status.csv", index=False)
print("✅ Slot status saved to parking_slot_status.csv")
print(f"Usable area ratio: {usable_area_ratio:.2f}")
print(f"Remaining (outside) area ignored: {outside_ignored}")
print(f"Estimated empty slots in usable area: {estimated_empty}")
print(f"Estimated empty slots in outside area: {est_empty_outside}")
print(status_msg)

# -------------------------------
# STEP 11: Show Final Output
# -------------------------------
plt.figure(figsize=(12, 8))
plt.imshow(cv2.cvtColor(img_draw, cv2.COLOR_BGR2RGB))
plt.title(f"Occupied: {len(final_boxes)} | Empty: {estimated_empty} | Outside: {est_empty_outside}\n{status_msg}")
plt.axis(False)
plt.tight_layout()
plt.show()
