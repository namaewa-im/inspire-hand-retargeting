from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import cv2
import mediapipe as mp
import numpy as np
import pyrealsense2 as rs
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision


# This file is expected to be placed in ~/maniskill/realsense/.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "hand_landmarker.task"

JOINT_NAMES: tuple[str, ...] = (
    "thumb_cmc_yaw",
    "thumb_cmc_pitch",
    "index_mcp",
    "middle_mcp",
    "ring_mcp",
    "pinky_mcp",
)

JOINT_LABELS: tuple[str, ...] = (
    "Thumb CMC Yaw",
    "Thumb CMC Pitch",
    "Index MCP",
    "Middle MCP",
    "Ring MCP",
    "Pinky MCP",
)

# Keep the legacy constant name for UI import compatibility.
# Thumb outputs are normalized to [0, 1].
# The four-finger MCP outputs are signed values in [-1, 1]:
# flexion is positive and hyperextension is negative.
JOINT_LIMITS_RAD: tuple[tuple[float, float], ...] = (
    (0.00, 1.00),   # thumb CMC yaw
    (0.00, 1.00),   # thumb CMC pitch
    (-1.00, 1.00),  # index MCP
    (-1.00, 1.00),  # middle MCP
    (-1.00, 1.00),  # ring MCP
    (-1.00, 1.00),  # pinky MCP
)

JOINT_UNITS: tuple[str, ...] = (
    "normalized",
    "normalized",
    "normalized",
    "normalized",
    "normalized",
    "normalized",
)

# Thumb normalization calibration.
#
# Yaw is represented by how far the Thumb CMC->MCP segment leaves the
# palm plane. The perpendicular distance is divided by the segment length,
# so the result is scale-independent and already lies in [0, 1].
#
#   yaw = |dot(CMC->MCP, palm_normal)| / ||CMC->MCP||
#       = |dot(normalize(CMC->MCP), palm_normal)|

# Pitch is the bend angle between the thumb CMC->MCP and MCP->IP
# segments. A straight thumb is 0. The observed usable maximum is
# approximately 70 degrees, which is mapped to 1.
THUMB_PITCH_MAX_BEND_RAD = float(np.deg2rad(70.0))

# The four-finger MCP score still uses this sign calibration.
# Keep the value from the previous estimator until the palm-normal
# convention is recalibrated from recorded poses.
FINGER_MCP_FLEXION_SIGN = -1.0

# MediaPipe hand landmark graph
HAND_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)


@dataclass(slots=True)
class HandEstimate:
    """Estimator output for a single frame."""

    raw_bgr: np.ndarray
    overlay_bgr: np.ndarray
    joint_values: Optional[np.ndarray]
    handedness: Optional[str]
    handedness_score: float
    fps: float
    camera_name: str
    camera_serial: str
    timestamp_s: float

    @property
    def has_hand(self) -> bool:
        return self.joint_values is not None


class HandJointEstimator:
    """
    Estimate MediaPipe hand landmarks from a RealSense D405 stream and
    return the following six normalized values.

    [0] thumb CMC yaw   [0, 1]
        Measure the perpendicular distance of Thumb CMC->MCP from the
        palm plane, normalized by the CMC->MCP segment length.

        segment lies in palm plane -> 0
        segment points along normal -> 1

    [1] thumb CMC pitch [0, 1]
        Compute the unsigned bend angle between thumb CMC->MCP and
        thumb MCP->IP.

        straight (0 deg) -> 0
        70 deg bend      -> 1

    [2] index MCP       [-1, 1]
    [3] middle MCP      [-1, 1]
    [4] ring MCP        [-1, 1]
    [5] pinky MCP       [-1, 1]
        signed dot product between the palmar normal and MCP -> PIP

    Each MCP score is 0 when the proximal phalanx lies in the palm plane.
    Flexion toward the palm is positive, while dorsal hyperextension
    remains negative instead of being clamped to 0.
    """

    def __init__(
        self,
        model_path: Path | str = DEFAULT_MODEL_PATH,
        camera_serial: Optional[str] = None,
        preferred_hand: Optional[str] = None,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        min_detection_confidence: float = 0.5,
        min_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        self.model_path = Path(model_path)
        self.camera_serial = camera_serial
        self.preferred_hand = self._normalize_hand_name(preferred_hand)
        self.width = width
        self.height = height
        self.requested_fps = fps

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"MediaPipe model not found: {self.model_path}\n"
                "Expected location: ~/maniskill/models/hand_landmarker.task"
            )

        base_options = mp_python.BaseOptions(
            model_asset_path=str(self.model_path),
        )
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)

        self._pipeline = rs.pipeline()
        self._config = rs.config()
        if self.camera_serial:
            self._config.enable_device(self.camera_serial)
        self._config.enable_stream(
            rs.stream.color,
            width,
            height,
            rs.format.bgr8,
            fps,
        )

        self._started = False
        self._camera_name = "RealSense"
        self._camera_serial = camera_serial or "unknown"
        self._start_ns = 0
        self._last_frame_time = 0.0
        self._smoothed_fps = 0.0
        self._last_timestamp_ms = -1

    @property
    def camera_name(self) -> str:
        return self._camera_name

    @property
    def serial_number(self) -> str:
        return self._camera_serial

    def start(self) -> None:
        if self._started:
            return

        profile = self._pipeline.start(self._config)
        device = profile.get_device()

        self._camera_name = self._safe_device_info(
            device,
            rs.camera_info.name,
            default="RealSense",
        )
        self._camera_serial = self._safe_device_info(
            device,
            rs.camera_info.serial_number,
            default=self.camera_serial or "unknown",
        )

        self._start_ns = time.monotonic_ns()
        self._last_frame_time = time.perf_counter()
        self._smoothed_fps = 0.0
        self._last_timestamp_ms = -1
        self._started = True

    def close(self) -> None:
        if self._started:
            self._pipeline.stop()
            self._started = False
        self._landmarker.close()

    def __enter__(self) -> "HandJointEstimator":
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def read(self, timeout_ms: int = 1000) -> HandEstimate:
        """Read one camera frame and return its estimate."""
        if not self._started:
            self.start()

        frames = self._pipeline.wait_for_frames(timeout_ms)
        color_frame = frames.get_color_frame()
        if not color_frame:
            raise RuntimeError("RealSense color frame was not received.")

        raw_bgr = np.asanyarray(color_frame.get_data()).copy()
        overlay_bgr = raw_bgr.copy()

        rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(rgb),
        )

        timestamp_ms = int((time.monotonic_ns() - self._start_ns) / 1_000_000)
        # MediaPipe VIDEO mode requires strictly increasing timestamps.
        timestamp_ms = max(timestamp_ms, self._last_timestamp_ms + 1)
        self._last_timestamp_ms = timestamp_ms

        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)
        selected_index = self._select_hand_index(result)

        joint_values: Optional[np.ndarray] = None
        handedness: Optional[str] = None
        handedness_score = 0.0

        if selected_index is not None:
            image_landmarks = result.hand_landmarks[selected_index]
            handedness, handedness_score = self._get_handedness(result, selected_index)

            if selected_index < len(result.hand_world_landmarks):
                geometry_landmarks = result.hand_world_landmarks[selected_index]
            else:
                geometry_landmarks = image_landmarks

            geometry_hand = handedness or self.preferred_hand
            joint_values = self._calculate_joint_values(
                geometry_landmarks,
                geometry_hand,
            )
            joint_values = self._clip_joint_values(joint_values)

            self._draw_hand_landmarks(
                overlay_bgr,
                image_landmarks,
                handedness,
                handedness_score,
            )
            self._draw_joint_values(overlay_bgr, joint_values)
        else:
            cv2.putText(
                overlay_bgr,
                "No hand detected",
                (20, 42),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                (0, 180, 255),
                2,
                cv2.LINE_AA,
            )

        fps = self._update_fps()
        cv2.putText(
            overlay_bgr,
            f"FPS {fps:.1f}",
            (20, overlay_bgr.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        return HandEstimate(
            raw_bgr=raw_bgr,
            overlay_bgr=overlay_bgr,
            joint_values=joint_values,
            handedness=handedness,
            handedness_score=handedness_score,
            fps=fps,
            camera_name=self._camera_name,
            camera_serial=self._camera_serial,
            timestamp_s=timestamp_ms / 1000.0,
        )

    @staticmethod
    def _normalize_hand_name(hand_name: Optional[str]) -> Optional[str]:
        if hand_name is None:
            return None
        normalized = hand_name.strip().lower()
        if normalized == "left":
            return "Left"
        if normalized == "right":
            return "Right"
        raise ValueError("preferred_hand must be 'Left', 'Right', or None.")

    @staticmethod
    def _safe_device_info(device, info, default: str) -> str:
        try:
            if device.supports(info):
                return device.get_info(info)
        except RuntimeError:
            pass
        return default

    def _select_hand_index(self, result) -> Optional[int]:
        if not result.hand_landmarks:
            return None

        candidates: list[tuple[int, str, float]] = []
        for index in range(len(result.hand_landmarks)):
            label, score = self._get_handedness(result, index)
            candidates.append((index, label or "Unknown", score))

        if self.preferred_hand is not None:
            preferred = [item for item in candidates if item[1] == self.preferred_hand]
            if preferred:
                return max(preferred, key=lambda item: item[2])[0]

        return max(candidates, key=lambda item: item[2])[0]

    @staticmethod
    def _get_handedness(result, index: int) -> tuple[Optional[str], float]:
        if index >= len(result.handedness) or not result.handedness[index]:
            return None, 0.0

        category = result.handedness[index][0]
        return category.category_name, float(category.score)

    @classmethod
    def _calculate_joint_values(
        cls,
        landmarks: Sequence,
        handedness: Optional[str] = None,
    ) -> np.ndarray:
        points = np.asarray(
            [[lm.x, lm.y, lm.z] for lm in landmarks],
            dtype=np.float64,
        )
        if points.shape != (21, 3):
            raise ValueError(f"Expected 21x3 landmarks, got {points.shape}.")

        # MediaPipe landmark indices
        wrist = points[0]
        thumb_cmc = points[1]
        thumb_mcp = points[2]
        thumb_ip = points[3]
        index_mcp = points[5]
        index_pip = points[6]
        middle_mcp = points[9]
        middle_pip = points[10]
        ring_mcp = points[13]
        ring_pip = points[14]
        pinky_mcp = points[17]
        pinky_pip = points[18]

        # Palm-fixed coordinate frame.
        # lateral: pinky MCP -> index MCP
        # forward: wrist -> palm center
        # palmar_normal: oriented toward the flexion side of the palm
        palm_center = np.mean(
            [index_mcp, middle_mcp, ring_mcp, pinky_mcp],
            axis=0,
        )
        palm_lateral_seed = cls._normalize(index_mcp - pinky_mcp)
        palm_forward_seed = cls._normalize(palm_center - wrist)
        raw_palm_normal = cls._normalize(
            np.cross(palm_lateral_seed, palm_forward_seed)
        )
        palm_normal = cls._orient_palmar_normal(
            raw_palm_normal,
            handedness,
        )

        # Thumb CMC yaw:
        # Use the normalized perpendicular distance of the CMC->MCP
        # segment from the palm plane. This is scale-independent.
        thumb_cmc_yaw = cls._thumb_yaw_score(
            palm_normal=palm_normal,
            cmc_to_mcp=thumb_mcp - thumb_cmc,
        )

        # Thumb pitch:
        # Use only the geometric bend between CMC->MCP and MCP->IP.
        # aligned segments -> 0
        # increasing bend  -> increasing score toward 1
        thumb_cmc_pitch = cls._thumb_pitch_score(
            cmc_to_mcp=thumb_mcp - thumb_cmc,
            mcp_to_ip=thumb_ip - thumb_mcp,
        )

        # Four-finger MCP scores:
        # finger in palm plane        -> n dot v = 0
        # flexion toward the palm     -> positive score
        # dorsal hyperextension       -> negative score
        index_flexion = cls._mcp_normal_score(
            palm_normal,
            index_pip - index_mcp,
        )
        middle_flexion = cls._mcp_normal_score(
            palm_normal,
            middle_pip - middle_mcp,
        )
        ring_flexion = cls._mcp_normal_score(
            palm_normal,
            ring_pip - ring_mcp,
        )
        pinky_flexion = cls._mcp_normal_score(
            palm_normal,
            pinky_pip - pinky_mcp,
        )

        return np.asarray(
            [
                thumb_cmc_yaw,
                thumb_cmc_pitch,
                index_flexion,
                middle_flexion,
                ring_flexion,
                pinky_flexion,
            ],
            dtype=np.float32,
        )


    @classmethod
    def _thumb_yaw_score(
        cls,
        palm_normal: np.ndarray,
        cmc_to_mcp: np.ndarray,
    ) -> float:
        """
        Return the normalized distance of CMC->MCP from the palm plane.

        The palm plane is defined by its unit normal. If CMC is treated as
        lying on that plane, the perpendicular MCP distance is

            abs(dot(cmc_to_mcp, palm_normal)).

        Dividing by the CMC->MCP length removes hand-size dependence. The
        resulting score is the absolute normal component of the unit thumb
        segment: 0 in the palm plane and 1 along the palm normal.
        """
        segment_length = float(np.linalg.norm(cmc_to_mcp))
        if segment_length < 1e-8:
            raise ValueError(
                "Degenerate thumb yaw geometry: zero-length CMC->MCP segment."
            )

        perpendicular_distance = abs(
            float(np.dot(cmc_to_mcp, palm_normal))
        )
        normalized_distance = perpendicular_distance / segment_length

        return float(np.clip(normalized_distance, 0.0, 1.0))

    @classmethod
    def _thumb_pitch_score(
        cls,
        cmc_to_mcp: np.ndarray,
        mcp_to_ip: np.ndarray,
    ) -> float:
        """
        Compute the unsigned 3-D angle between CMC->MCP and MCP->IP.

        Collinear segments produce zero. Bending in either direction
        increases the angle, so no palmar-side sign test or clamping of
        hyperextension is applied before normalization.
        """
        proximal_direction = cls._normalize(cmc_to_mcp)
        distal_direction = cls._normalize(mcp_to_ip)
        bend_angle = cls._angle_between_unit_vectors(
            proximal_direction,
            distal_direction,
        )

        if THUMB_PITCH_MAX_BEND_RAD <= 0.0:
            raise ValueError(
                "THUMB_PITCH_MAX_BEND_RAD must be greater than zero."
            )

        return float(
            np.clip(
                bend_angle / THUMB_PITCH_MAX_BEND_RAD,
                0.0,
                1.0,
            )
        )

    @staticmethod
    def _orient_palmar_normal(
        raw_normal: np.ndarray,
        handedness: Optional[str],
    ) -> np.ndarray:
        """
        Orient the palm normal toward the anatomical palmar flexion side.

        With the lateral axis defined from pinky MCP to index MCP, the raw
        cross product points toward the palmar side for a left hand and the
        dorsal side for a right hand. The sign correction below makes MCP
        flexion positive for both hands.
        """
        if handedness == "Right":
            return -raw_normal
        return raw_normal

    @staticmethod
    def _project_onto_plane(
        vector: np.ndarray,
        plane_normal: np.ndarray,
    ) -> np.ndarray:
        """Orthogonally project a vector onto a plane with a unit normal."""
        return vector - float(np.dot(vector, plane_normal)) * plane_normal

    @staticmethod
    def _angle_between_unit_vectors(
        unit_a: np.ndarray,
        unit_b: np.ndarray,
    ) -> float:
        """Return the unsigned angle between two unit vectors in radians."""
        cosine = float(np.clip(np.dot(unit_a, unit_b), -1.0, 1.0))
        return float(np.arccos(cosine))

    @classmethod
    def _normalized_angle_between(
        cls,
        vector_a: np.ndarray,
        vector_b: np.ndarray,
    ) -> float:
        """Return the unsigned 3-D angle divided by pi, in [0, 1]."""
        unit_a = cls._normalize(vector_a)
        unit_b = cls._normalize(vector_b)
        cosine = float(np.clip(np.dot(unit_a, unit_b), -1.0, 1.0))
        angle_rad = float(np.arccos(cosine))
        return float(np.clip(angle_rad / np.pi, 0.0, 1.0))

    @classmethod
    def _mcp_normal_score(
        cls,
        palmar_normal: np.ndarray,
        mcp_to_pip: np.ndarray,
    ) -> float:
        """
        Return signed MCP flexion toward the palmar side.

        A proximal phalanx in the palm plane produces 0. The raw normal
        score is converted with FINGER_MCP_FLEXION_SIGN so anatomical
        flexion is positive. Hyperextension remains negative and is
        preserved in the output.
        """
        finger_direction = cls._normalize(mcp_to_pip)
        raw_signed_score = float(
            np.dot(palmar_normal, finger_direction)
        )

        # Convert the observed camera-coordinate sign into the anatomical
        # convention: flexion is positive, hyperextension is negative.
        signed_flexion_score = (
            FINGER_MCP_FLEXION_SIGN * raw_signed_score
        )
        return float(np.clip(signed_flexion_score, -1.0, 1.0))

    @staticmethod
    def _normalize(vector: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        norm = float(np.linalg.norm(vector))
        if norm < eps:
            raise ValueError("Degenerate hand geometry: zero-length vector.")
        return vector / norm

    @staticmethod
    def _clip_joint_values(values: np.ndarray) -> np.ndarray:
        clipped = values.astype(np.float32, copy=True)
        for index, (lower, upper) in enumerate(JOINT_LIMITS_RAD):
            clipped[index] = np.clip(clipped[index], lower, upper)
        return clipped

    @staticmethod
    def _draw_hand_landmarks(
        image: np.ndarray,
        landmarks: Sequence,
        handedness: Optional[str],
        score: float,
    ) -> None:
        height, width = image.shape[:2]
        points: list[tuple[int, int]] = []

        for landmark in landmarks:
            x = int(np.clip(landmark.x, 0.0, 1.0) * (width - 1))
            y = int(np.clip(landmark.y, 0.0, 1.0) * (height - 1))
            points.append((x, y))

        for start, end in HAND_CONNECTIONS:
            cv2.line(
                image,
                points[start],
                points[end],
                (80, 220, 130),
                2,
                cv2.LINE_AA,
            )

        for index, point in enumerate(points):
            radius = 6 if index in (0, 1, 2, 5, 9, 13, 17) else 4
            cv2.circle(image, point, radius, (40, 90, 255), -1, cv2.LINE_AA)

        label = handedness or "Hand"
        cv2.putText(
            image,
            f"{label} {score:.2f}",
            (20, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 230, 80),
            2,
            cv2.LINE_AA,
        )

    @staticmethod
    def _draw_joint_values(image: np.ndarray, values: np.ndarray) -> None:
        x = 20
        y = 72
        for label, value, _unit in zip(JOINT_LABELS, values, JOINT_UNITS):
            text = f"{label}: {value:.3f}"

            cv2.putText(
                image,
                text,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (245, 245, 245),
                1,
                cv2.LINE_AA,
            )
            y += 22

    def _update_fps(self) -> float:
        now = time.perf_counter()
        dt = now - self._last_frame_time
        self._last_frame_time = now
        if dt <= 0:
            return self._smoothed_fps

        instantaneous = 1.0 / dt
        if self._smoothed_fps <= 0:
            self._smoothed_fps = instantaneous
        else:
            self._smoothed_fps = 0.9 * self._smoothed_fps + 0.1 * instantaneous
        return self._smoothed_fps
