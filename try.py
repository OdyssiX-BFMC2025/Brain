import torch
import cv2
import numpy as np
from picamera2 import Picamera2
from torchvision.transforms import ToTensor

# Load YOLOv5 Nano model
model = torch.hub.load("yolov5", "custom", path="yolov5n.pt", force_reload=True)
model.eval()

# Initialize Raspberry Pi Camera
picam2 = Picamera2()
picam2.preview_configuration.main.size = (640, 480)
picam2.preview_configuration.main.format = "RGB888"
picam2.configure("preview")
picam2.start()

print("Starting object detection... Press 'q' to quit.")

while True:
    frame = picam2.capture_array()  # Capture frame from camera
    img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV
    results = model(img)  # Run YOLOv5 detection

    # Process results
    for det in results.xyxy[0]:  # x1, y1, x2, y2, confidence, class
        x1, y1, x2, y2, conf, cls = det
        if conf > 0.5:  # Confidence threshold
            label = f"{model.names[int(cls)]} ({conf:.2f})"
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(img, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("YOLOv5 Nano Detection", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

picam2.stop()
cv2.destroyAllWindows()
