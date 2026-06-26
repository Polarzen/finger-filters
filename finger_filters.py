import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import Image, ImageFormat
import numpy as np
import sys
import os

FINGER_TIPS = [4, 8, 12, 16, 20]  # thumb, index, middle, ring, pinky

LINE_COLORS = [
    (255, 80, 80),
    (80, 255, 80),
    (80, 255, 255),
    (255, 80, 255),
    (80, 160, 255),
]

REGION_COLORS = [
    (255, 120, 120),
    (120, 255, 120),
    (120, 255, 255),
    (255, 120, 255),
]


def count_raised_fingers(landmarks, handedness):
    """Count raised fingers using joint angle. Straight finger = angle > 120 deg."""
    tips = [4, 8, 12, 16, 20]
    pips = [3, 6, 10, 14, 18]
    mcps = [2, 5, 9, 13, 17]

    count = 0
    for tip_id, pip_id, mcp_id in zip(tips, pips, mcps):
        tip = np.array([landmarks[tip_id].x, landmarks[tip_id].y])
        pip = np.array([landmarks[pip_id].x, landmarks[pip_id].y])
        mcp = np.array([landmarks[mcp_id].x, landmarks[mcp_id].y])

        v1 = mcp - pip
        v2 = tip - pip

        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 > 0 and n2 > 0:
            cos_angle = np.dot(v1, v2) / (n1 * n2)
            if tip_id == 4:
                wrist = np.array([landmarks[0].x, landmarks[0].y])
                index_mcp = np.array([landmarks[5].x, landmarks[5].y])
                pinky_mcp = np.array([landmarks[17].x, landmarks[17].y])
                palm_center = (wrist + index_mcp + pinky_mcp) / 3
                tip_to_palm = np.linalg.norm(tip - palm_center)
                ip_to_palm = np.linalg.norm(pip - palm_center)
                if tip_to_palm > ip_to_palm * 1.1:
                    count += 1
            else:
                if cos_angle < -0.5:
                    count += 1
    return count


def apply_filter_to_frame(frame, filter_idx):
    if filter_idx == 0:
        return frame
    elif filter_idx == 1:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    elif filter_idx == 2:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 40, 120)
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    elif filter_idx == 3:
        return cv2.GaussianBlur(frame, (65, 65), 0)
    elif filter_idx == 4:
        color = cv2.bilateralFilter(frame, 15, 300, 300)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 7)
        edges = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 3
        )
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        return cv2.bitwise_and(color, edges_colored)
    elif filter_idx == 5:
        return cv2.bitwise_not(frame)
    return frame


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera.")
        return

    model_path = os.path.join(sys._MEIPASS, "hand_landmarker.task") if getattr(sys, 'frozen', False) else "hand_landmarker.task"
    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )
    landmarker = vision.HandLandmarker.create_from_options(options)

    print("Dual Hand Region Filters started. Press 'q' to quit.")
    print("Show both hands. Left hand finger count controls the filter.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = Image(image_format=ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_image)

        display = frame.copy()
        left_tips = None
        right_tips = None
        left_finger_count = 0

        if result.hand_landmarks:
            for i, landmarks in enumerate(result.hand_landmarks):
                handedness = result.handedness[i][0].category_name
                tips = []
                for tip_id in FINGER_TIPS:
                    lm = landmarks[tip_id]
                    tips.append((int(lm.x * w), int(lm.y * h)))

                if handedness == "Left":
                    left_tips = tips
                    left_finger_count = count_raised_fingers(landmarks, handedness)
                else:
                    right_tips = tips

        filter_idx = min(left_finger_count, 5) if left_tips is not None else 0

        if filter_idx > 0 and left_tips is not None and right_tips is not None:
            for i in range(5):
                cv2.line(display, left_tips[i], right_tips[i], LINE_COLORS[i], 3)
                cv2.circle(display, left_tips[i], 8, LINE_COLORS[i], -1)
                cv2.circle(display, right_tips[i], 8, LINE_COLORS[i], -1)

            for i in range(4):
                pts = np.array(
                    [left_tips[i], right_tips[i], right_tips[i + 1], left_tips[i + 1]],
                    np.int32,
                )

                if filter_idx != 0:
                    mask = np.zeros((h, w), dtype=np.uint8)
                    cv2.fillPoly(mask, [pts], 255)
                    filtered = apply_filter_to_frame(frame, filter_idx)
                    mask_3ch = (
                        cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR).astype(np.float32)
                        / 255.0
                    )
                    display = (
                        display.astype(np.float32) * (1 - mask_3ch)
                        + filtered.astype(np.float32) * mask_3ch
                    ).astype(np.uint8)

                cv2.polylines(display, [pts], True, REGION_COLORS[i], 2)

            cv2.putText(
                display,
                f"Filter: {filter_idx} | Fingers: {left_finger_count}",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
        else:
            cv2.putText(
                display,
                "Show both hands with fingers raised",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

        cv2.imshow("Finger Region Filters", display)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    main()