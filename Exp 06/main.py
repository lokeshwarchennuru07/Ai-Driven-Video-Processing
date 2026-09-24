import cv2
import numpy as np
from pathlib import Path

# =========================================================
# 1. Load Traffic Video
# =========================================================
video_candidates = [
    Path(__file__).with_name("traffic.mp4"),
    Path(__file__).with_name("traffic.mp4.mp4"),
]
video_path = next((path for path in video_candidates if path.is_file()), None)

cap = cv2.VideoCapture(str(video_path) if video_path else "")

if not cap.isOpened():
    print("Error: Could not open a traffic video beside main.py")
    exit()

# =========================================================
# 2. Display Video Properties
# =========================================================
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print("=" * 45)
print("           VIDEO PROPERTIES")
print("=" * 45)
print("Video           :", video_path)
print("Resolution      :", width, "x", height)
print("FPS             :", round(fps, 2))
print("Total Frames    :", total_frames)

if fps > 0:
    duration = total_frames / fps
    print("Duration        :", round(duration, 2), "seconds")

print("=" * 45)

# =========================================================
# 3. Background Subtractor
# =========================================================
background_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=500,
    varThreshold=50,
    detectShadows=True
)

# =========================================================
# 4. Frame Extraction Interval
# =========================================================
interval = 20
frame_number = 0
min_area = 500

# =========================================================
# 5. Process Video
# =========================================================
while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Process every 20th frame
    if frame_number % interval == 0:

        print("\n--------------------------------------")
        print("Frame Number:", frame_number)

        # =================================================
        # 6. Detect Moving Objects
        # =================================================
        foreground_mask = background_subtractor.apply(frame)

        # Remove shadows
        _, threshold = cv2.threshold(
            foreground_mask,
            200,
            255,
            cv2.THRESH_BINARY
        )

        # Remove noise
        kernel = np.ones((5, 5), np.uint8)

        threshold = cv2.morphologyEx(
            threshold,
            cv2.MORPH_OPEN,
            kernel
        )

        threshold = cv2.dilate(
            threshold,
            kernel,
            iterations=2
        )

        # =================================================
        # 7. Find Objects
        # =================================================
        contours, _ = cv2.findContours(
            threshold,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        object_count = 0

        for contour in contours:

            area = cv2.contourArea(contour)

            # Ignore small objects
            if area < min_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            object_count += 1

            # Draw bounding box
            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            # Object label
            cv2.putText(
                frame,
                f"Object {object_count}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        # =================================================
        # 8. Print Object Count
        # =================================================
        print("Objects Detected:", object_count)

        # =================================================
        # 9. Display Information
        # =================================================
        cv2.putText(
            frame,
            f"Objects Detected: {object_count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

        cv2.putText(
            frame,
            f"Frame: {frame_number}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 0),
            2
        )

        # =================================================
        # 10. Display Frame
        # =================================================
        cv2.imshow("Traffic Video - Object Detection", frame)

        # Press Q to Quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    frame_number += 1

# =========================================================
# 11. Release Resources
# =========================================================
cap.release()
cv2.destroyAllWindows()

print("\n" + "=" * 45)
print("Video processing completed successfully.")
print("=" * 45)