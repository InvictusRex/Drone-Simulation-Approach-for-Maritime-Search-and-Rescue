import cv2
from ultralytics import YOLO

# Path to the YOLOv8 model weights
model_weights_path = r"C:\Users\TheKi\Downloads\best_v8l.pt"

# Load the YOLOv8 model
model = YOLO(model_weights_path)

# UDP stream address (OBS input or any other source)
udp_stream = "udp://192.168.211.4:9999?pkt_size=1316"

# Initialize video capture for the UDP stream
cap = cv2.VideoCapture(udp_stream)

if not cap.isOpened():
    print("Error: Unable to open the UDP stream.")
    exit()

print("Starting UDP stream...")

# Define confidence threshold
confidence_threshold = 0.45

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to retrieve frame from stream.")
            break

        # Run inference on the frame using the model
        results = model.predict(source=frame, conf=confidence_threshold, show=False)

        # Annotate the frame with the predictions
        annotated_frame = results[0].plot()

        # Display the frame in a window
        cv2.imshow("Object Detection", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quitting the stream...")
            break

except KeyboardInterrupt:
    print("\nStream interrupted by user.")

# Release resources
cap.release()
cv2.destroyAllWindows()
print("Stream ended.")
