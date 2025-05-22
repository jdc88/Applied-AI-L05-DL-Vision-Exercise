from ultralytics import YOLO
import cv2
import torch

# Load the YOLOv5 pretrained model (YOLOv8 also works the same way)
model = YOLO("yolov8n.pt")  # You can replace with yolov5n.pt if using YOLOv5

# Load and detect objects in the image
image_path = "objects.jpg"
results = model(image_path)[0]  # Get the first prediction result

# Load the image to display dimensions
image = cv2.imread(image_path)

# Actual image size
h, w, _ = image.shape

# Get detected labels and bounding boxes
predicted_labels = []
locations = []

for box in results.boxes:
    cls_id = int(box.cls.item())            # class ID
    label = model.names[cls_id]             # class name
    predicted_labels.append(label)

    x1, y1, x2, y2 = map(int, box.xyxy[0])  # bounding box coordinates
    locations.append((x1, y1, x2, y2))

# Remove duplicates
unique_labels = list(set(predicted_labels))

# Print results
print("✅ Detected Objects:\n")
print("List of predicted object names:")
for label in unique_labels:
    print(f"- {label}")

print("\nList of predicted objects with locations:")
for label, loc in zip(predicted_labels, locations):
    print(f"{label}: {loc}")
