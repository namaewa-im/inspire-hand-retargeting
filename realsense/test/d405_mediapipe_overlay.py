from __future__ import annotations

import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pyrealsense2 as rs


MODEL_PATH = Path("models/hand_landmarker.task")

# 특정 카메라를 지정하려면 시리얼 번호를 문자열로 입력합니다.
# 예: CAMERA_SERIAL = "123456789012"
CAMERA_SERIAL: str | None = None

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 30

# MediaPipe의 21개 랜드마크를 연결하는 손 골격 구조
HAND_CONNECTIONS = (
    # 엄지
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),

    # 검지
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),

    # 중지
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),

    # 약지
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),

    # 소지
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),

    # 손바닥 아래쪽
    (0, 17),
)


def draw_hand_landmarks(
    image: np.ndarray,
    landmarks,
    handedness=None,
) -> None:
    """MediaPipe normalized landmarks를 OpenCV 영상 위에 그립니다."""

    height, width = image.shape[:2]

    points: list[tuple[int, int]] = []

    for landmark in landmarks:
        x = int(np.clip(landmark.x, 0.0, 1.0) * (width - 1))
        y = int(np.clip(landmark.y, 0.0, 1.0) * (height - 1))
        points.append((x, y))

    # 관절 사이 연결선
    for start_idx, end_idx in HAND_CONNECTIONS:
        cv2.line(
            image,
            points[start_idx],
            points[end_idx],
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    # 각 관절점
    for landmark_idx, point in enumerate(points):
        cv2.circle(
            image,
            point,
            5,
            (0, 0, 255),
            -1,
            cv2.LINE_AA,
        )

        # 랜드마크 번호 표시
        cv2.putText(
            image,
            str(landmark_idx),
            (point[0] + 4, point[1] - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    # Left/Right 및 confidence 표시
    if handedness:
        category = handedness[0]

        label = category.category_name or "Hand"
        score = category.score

        wrist_x, wrist_y = points[0]

        cv2.putText(
            image,
            f"{label}: {score:.2f}",
            (wrist_x, max(25, wrist_y - 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"MediaPipe model not found: {MODEL_PATH.resolve()}\n"
            "먼저 hand_landmarker.task 파일을 다운로드하세요."
        )

    # ------------------------------------------------------------------
    # RealSense 설정
    # ------------------------------------------------------------------
    pipeline = rs.pipeline()
    config = rs.config()

    if CAMERA_SERIAL is not None:
        config.enable_device(CAMERA_SERIAL)

    config.enable_stream(
        rs.stream.color,
        FRAME_WIDTH,
        FRAME_HEIGHT,
        rs.format.bgr8,
        FPS,
    )

    # ------------------------------------------------------------------
    # MediaPipe Hand Landmarker 설정
    # ------------------------------------------------------------------
    base_options = mp.tasks.BaseOptions(
        model_asset_path=str(MODEL_PATH)
    )

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)

    profile = pipeline.start(config)

    device = profile.get_device()
    camera_name = device.get_info(rs.camera_info.name)
    camera_serial = device.get_info(rs.camera_info.serial_number)

    print(f"Camera: {camera_name}")
    print(f"Serial: {camera_serial}")
    print("종료: q 또는 ESC")

    cv2.namedWindow("D405 Raw", cv2.WINDOW_NORMAL)
    cv2.namedWindow("MediaPipe Overlay", cv2.WINDOW_NORMAL)

    cv2.resizeWindow("D405 Raw", FRAME_WIDTH, FRAME_HEIGHT)
    cv2.resizeWindow("MediaPipe Overlay", FRAME_WIDTH, FRAME_HEIGHT)

    # 두 창을 좌우로 배치
    cv2.moveWindow("D405 Raw", 50, 50)
    cv2.moveWindow("MediaPipe Overlay", 750, 50)

    previous_time = time.perf_counter()
    display_fps = 0.0

    # detect_for_video에는 계속 증가하는 timestamp가 필요합니다.
    start_time_ns = time.monotonic_ns()

    try:
        while True:
            frames = pipeline.wait_for_frames()

            color_frame = frames.get_color_frame()

            if not color_frame:
                continue

            # RealSense BGR frame → NumPy
            color_image = np.asanyarray(color_frame.get_data())

            # 원본 창에 사용할 복사본
            raw_image = color_image.copy()

            # overlay 창에 사용할 복사본
            overlay_image = color_image.copy()

            # OpenCV BGR → MediaPipe RGB
            rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            rgb_image = np.ascontiguousarray(rgb_image)

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_image,
            )

            timestamp_ms = int(
                (time.monotonic_ns() - start_time_ns) / 1_000_000
            )

            result = landmarker.detect_for_video(
                mp_image,
                timestamp_ms,
            )

            # 검출된 모든 손을 overlay에 그림
            for hand_idx, hand_landmarks in enumerate(result.hand_landmarks):
                handedness = None

                if hand_idx < len(result.handedness):
                    handedness = result.handedness[hand_idx]

                draw_hand_landmarks(
                    overlay_image,
                    hand_landmarks,
                    handedness,
                )

            # FPS 계산
            current_time = time.perf_counter()
            frame_dt = current_time - previous_time
            previous_time = current_time

            if frame_dt > 0:
                instantaneous_fps = 1.0 / frame_dt
                display_fps = 0.9 * display_fps + 0.1 * instantaneous_fps

            cv2.putText(
                raw_image,
                f"FPS: {display_fps:.1f}",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                overlay_image,
                f"FPS: {display_fps:.1f} | Hands: {len(result.hand_landmarks)}",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("D405 Raw", raw_image)
            cv2.imshow("MediaPipe Overlay", overlay_image)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == 27:
                break

            # 창의 X 버튼으로 닫은 경우
            raw_visible = cv2.getWindowProperty(
                "D405 Raw",
                cv2.WND_PROP_VISIBLE,
            )
            overlay_visible = cv2.getWindowProperty(
                "MediaPipe Overlay",
                cv2.WND_PROP_VISIBLE,
            )

            if raw_visible < 1 or overlay_visible < 1:
                break

    finally:
        pipeline.stop()
        landmarker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
