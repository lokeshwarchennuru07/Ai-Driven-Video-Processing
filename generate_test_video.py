import cv2
import os
import numpy as np

output_path = r"D:\work\E02\AI driv\exp31.mp4"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Video properties
width, height = 640, 360
fps = 20
duration_seconds = 5
frame_count = fps * duration_seconds

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

for i in range(frame_count):
    # create a simple moving rectangle on a colored background
    frame = np.full((height, width, 3), 200, dtype='uint8')

    # draw moving rectangle
    x = int((width - 100) * (i / frame_count))
    y = height // 3
    cv2.rectangle(frame, (x, y), (x + 100, y + 50), (0, 0, 255), -1)

    out.write(frame)

out.release()

print("Generated test video:", output_path)
