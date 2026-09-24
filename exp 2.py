from pathlib import Path
import csv
import sys
import time
import math
import argparse
from typing import List, Dict
import cv2


def format_timestamp(seconds: float) -> str:
    """
    Convert seconds into HH:MM:SS.mmm format.
    """

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    remaining_seconds = seconds % 60

    return f"{hours:02d}:{minutes:02d}:{remaining_seconds:06.3f}"


def extract_frames(
    video_path: str,
    output_folder: str,
    interval_seconds: float
) -> None:
    """
    Extract frames from a video at regular time intervals.

    Parameters
    ----------
    video_path:
        Path of the input video.

    output_folder:
        Folder in which extracted frames are saved.

    interval_seconds:
        Time interval between extracted frames.
    """

    video_path = video_path.strip().strip('"').strip("'")
    input_path = Path(video_path).expanduser()

    # Check whether the input video exists
    if not input_path.is_file():
        raise FileNotFoundError(
            f"Video file not found: {input_path.resolve()}"
        )

    if interval_seconds <= 0:
        raise ValueError(
            "Extraction interval must be greater than zero."
        )

    # Create output folder if it does not exist
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    # Open the video
    video = cv2.VideoCapture(str(input_path))

    if not video.isOpened():
        raise RuntimeError(
            "OpenCV could not open the video. "
            "Check the video path, format and codec."
        )

    try:
        # Read video properties
        fps = float(video.get(cv2.CAP_PROP_FPS))
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

        if not (math.isfinite(fps) and fps > 0):
            # Some containers do not report FPS; try to infer from duration
            fps = None

        duration_seconds = None
        if fps:
            duration_seconds = total_frames / fps if total_frames > 0 else None
        # Fallback: query duration via milliseconds if available
        if duration_seconds is None:
            duration_ms = video.get(cv2.CAP_PROP_POS_MSEC)
            if duration_ms and duration_ms > 0:
                duration_seconds = duration_ms / 1000.0

        # If still unknown, we will iterate until read() fails
        # interval_frames is not used when seeking by time; keep for info
        interval_frames = None

        print("\n========== VIDEO INFORMATION ==========")
        print(f"Video Name          : {input_path.name}")
        print(f"Resolution          : {width} × {height}")
        print(f"Frame Rate          : {fps:.2f} FPS")
        print(f"Total Frames        : {total_frames}")
        print(f"Duration            : {duration_seconds:.2f} seconds")
        print(f"Extraction Interval : {interval_seconds:.2f} seconds")
        print(f"Frame Interval      : Every {interval_frames} frames")
        print("=======================================")

        frame_index = 0
        extracted_count = 0
        temporal_records: List[Dict] = []

        print("\nPress Q or Esc to stop the program.\n")

        # Prefer seeking by timestamp to avoid decoding every frame
        t = 0.0
        display_available = True
        while True:
            # Seek to desired time (in milliseconds)
            video.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            success, frame = video.read()
            if not success:
                break

            extracted_count += 1

            # Timestamp for this frame (seconds)
            timestamp_seconds = t
            timestamp_text = format_timestamp(timestamp_seconds)

            frame_filename = f"frame_{extracted_count:04d}_time_{timestamp_seconds:.3f}s.jpg"
            frame_file_path = output_path / frame_filename

            # Annotate a display copy
            display_frame = frame.copy()
            cv2.putText(display_frame, f"Frame: {extracted_count}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(display_frame, f"Time: {timestamp_text}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

            saved = cv2.imwrite(str(frame_file_path), frame)
            if not saved:
                print(f"Warning: could not save {frame_filename}")
            else:
                temporal_records.append({
                    "extracted_frame": extracted_count,
                    "timestamp_seconds": round(timestamp_seconds, 3),
                    "timestamp_formatted": timestamp_text,
                    "filename": frame_filename
                })

                print(f"Extracted {extracted_count:03d} | Time: {timestamp_text} | Saved: {frame_filename}")

            # Display if possible
            if display_available:
                try:
                    cv2.imshow("Extracted Frames", display_frame)
                    key = cv2.waitKey(int(interval_seconds * 1000)) & 0xFF
                    if key == ord('q') or key == 27:
                        print("\nExtraction stopped by the user.")
                        break
                except cv2.error:
                    display_available = False
                    time.sleep(interval_seconds)

            # Advance time
            t += interval_seconds
            # If duration known and exceeded, stop
            if duration_seconds and t > duration_seconds:
                break

        # Save timestamp information as a CSV file
        csv_path = output_path / "timestamps.csv"
        with csv_path.open(mode="w", newline="", encoding="utf-8") as csv_file:
            fieldnames = ["extracted_frame", "timestamp_seconds", "timestamp_formatted", "filename"]
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(temporal_records)

        print("\n========== EXTRACTION SUMMARY ==========")
        print(f"Frames extracted : {extracted_count}")
        print(f"Output folder    : {output_path.resolve()}")
        print(f"Timestamp report : {csv_path.resolve()}")
        print("========================================")

    finally:
        video.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":

    try:
        input_video = input(
            "Enter the video path: "
        )

        destination_folder = input(
            "Enter the output folder name "
            "[extracted_frames]: "
        ).strip()

        if not destination_folder:
            destination_folder = "extracted_frames"

        interval = float(
            input(
                "Enter extraction interval in seconds "
                "(example: 2): "
            )
        )

        extract_frames(
            input_video,
            destination_folder,
            interval
        )

    except (
        FileNotFoundError,
        ValueError,
        RuntimeError
    ) as error:

        print(f"\nError: {error}")