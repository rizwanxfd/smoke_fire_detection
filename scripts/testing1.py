import cv2
from ultralytics import YOLO

# Load pretrained fire/smoke model
model = YOLO("/home/rizwan.tahir/Documents/smoke_fire_detection/models/best.pt")   

# Open video
video_path = "/home/rizwan.tahir/Documents/smoke_fire_detection/dataset/smoke_testing.mp4"  
cap = cv2.VideoCapture(video_path)

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Output video writer
out = cv2.VideoWriter(
    "output_detected.mp4",
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,
    (width, height)
)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run inference
    results = model(frame, conf=0.3)

    # Draw results
    annotated_frame = results[0].plot()

    # Show
    cv2.imshow("Fire & Smoke Detection", annotated_frame)

    # Save output
    out.write(annotated_frame)

    # Press q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("Processing complete. Saved as output_detected.mp4")