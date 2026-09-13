from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, deque
from dataclasses import dataclass, field
from math import acos, degrees, hypot
from pathlib import Path
from statistics import median
from typing import Optional

import cv2
import numpy as np
import torch
from ultralytics import YOLO, __version__ as ultralytics_version


# =============================================================================
# V12 최종 통합 버전
#
# YOLO Head Detector
#   -> ByteTrack local ID (시간 정규화 track_buffer)
#   -> Stable ID manager (30fps 기준 시간을 입력 FPS에 맞게 정규화)
#       - tracklet geometry recovery
#       - recovery geometry score gate <= 0.45
#       - Pool ReID cosine gate >= 0.86
#       - alias/history merge
#   -> tentative trajectory backfill
#   -> 5초 익수 특징/판별
#
# 현재 권장:
#   Duplicate suppression = OFF
#   Tracklet recovery      = ON
#   Alias/history merge    = ON
#   Tentative backfill     = ON
#
# 목적:
#   추적 안정성을 유지하면서 tentative 확인 때문에 익수 history가 늦게 쌓이는
#   문제를 줄인다.
# =============================================================================


# =============================================================================
# Drowning rule settings — 기존 기준 유지
# =============================================================================

WINDOW_SECONDS = 5.0
MIN_HISTORY_SECONDS = 5.0
MIN_VISIBLE_HISTORY_SECONDS = 2.0
MOTION_SAMPLE_SECONDS = 0.10
MIN_DIRECTION_STEP_HEADS = 0.10
DIRECTION_CHANGE_DEGREES = 70.0
SUBMERGE_MIN_SECONDS = 0.35

MOTION_TOTAL_DISTANCE_REFERENCE = 6.0
MOTION_VERTICAL_REFERENCE = 4.0
MOTION_DIRECTION_CHANGES_REFERENCE = 5.0
MOTION_TOTAL_DISTANCE_WEIGHT = 0.45
MOTION_VERTICAL_WEIGHT = 0.35
MOTION_DIRECTION_WEIGHT = 0.20
HIGH_MOTION_SCORE_THRESHOLD = 0.45

ACTIVE_MAX_PROGRESS_RATIO = 0.35
ACTIVE_MIN_TOTAL_DISTANCE = 5.0
ACTIVE_MIN_VERTICAL_MOTION = 3.0
ACTIVE_MIN_DIRECTION_CHANGES = 4

PASSIVE_MAX_NET_DISTANCE = 1.5
PASSIVE_MIN_DISAPPEAR_RATIO = 0.45
PASSIVE_MIN_CONTINUOUS_MISSING_SECONDS = 2.0
PASSIVE_MIN_SUBMERGE_CYCLES = 2
PASSIVE_CYCLES_MIN_DISAPPEAR_RATIO = 0.20


# =============================================================================
# Stable ID settings — V33 계열 기반
# =============================================================================

ID_RECOVERY_MAX_GAP_SECONDS = 2.0
ID_RECOVERY_DISTANCE_FACTOR = 3.2

MIN_NEW_ID_CONFIRM_FRAMES = 6
MAX_PREDICTION_FRAMES = 5

VERY_SHORT_VISIBLE_FRAMES = 8
VERY_SHORT_TRACK_PENALTY = 0.50
SHORT_VISIBLE_FRAMES = 15
SHORT_TRACK_PENALTY = 0.25

MIN_RECOVERY_TRACKLET_FRAMES = 3
TRACKLET_HISTORY_SIZE = 8
TENTATIVE_MAX_MISSING_FRAMES = 2

ALIAS_MAX_SHORT_VISIBLE_FRAMES = 15
ALIAS_MIN_ESTABLISHED_FRAMES = 20
ALIAS_BIRTH_MAX_GAP_FRAMES = 30
ALIAS_MAX_SCORE = 0.45

MIN_SCALE_RATIO = 0.50
MAX_SCALE_RATIO = 2.00

DISTANCE_WEIGHT = 0.70
GAP_WEIGHT = 0.20
SCALE_WEIGHT = 0.10

# =============================================================================
# V12 validated recovery settings (Task 90 GT)
# =============================================================================

RECOVERY_GEOMETRY_MAX_SCORE = 0.45
REID_MIN_SIMILARITY = 0.86
REID_STRONG_SIMILARITY = 0.90
REID_IMGSZ = 224
REID_CROP_MARGIN = 0.20
REID_EMA_ALPHA = 0.20

V12_DISTANCE_WEIGHT = 0.55
V12_GAP_WEIGHT = 0.15
V12_SCALE_WEIGHT = 0.10
V12_REID_WEIGHT = 0.20

# 30fps에서 사용하던 frame 상수의 실제 시간 의미를 보존한다.
REF_FPS = 30.0
NEW_ID_CONFIRM_SECONDS = 6.0 / REF_FPS
PREDICTION_SECONDS = 5.0 / REF_FPS
VERY_SHORT_SECONDS = 8.0 / REF_FPS
SHORT_SECONDS = 15.0 / REF_FPS
RECOVERY_TRACKLET_SECONDS = 3.0 / REF_FPS
TRACKLET_HISTORY_SECONDS = 8.0 / REF_FPS
TENTATIVE_MAX_MISSING_SECONDS = 2.0 / REF_FPS
ALIAS_MAX_SHORT_SECONDS = 15.0 / REF_FPS
ALIAS_MIN_ESTABLISHED_SECONDS = 20.0 / REF_FPS
ALIAS_BIRTH_MAX_GAP_SECONDS = 30.0 / REF_FPS

# Task 90 평가에 사용한 ByteTrack 시간 기준.
BT_TRACK_HIGH_THRESH = 0.35
BT_TRACK_LOW_THRESH = 0.10
BT_NEW_TRACK_THRESH = 0.40
BT_TRACK_BUFFER_SECONDS = 4.0
BT_MATCH_THRESH = 0.80


# =============================================================================
# Duplicate suppression
#
# 현재 OFF.
# V40에서 129개까지 suppression되어 PASSIVE 오탐 증가 가능성이 있었고,
# V50에서도 추적 개선폭 대비 판정 지연/누락 우려가 있어 우선 비활성화한다.
# =============================================================================

ENABLE_DUPLICATE_SUPPRESSION = False

DUPLICATE_HIGH_CONF_MIN = 0.70
DUPLICATE_LOW_CONF_MAX = 0.35
DUPLICATE_MAX_CENTER_HEADS = 0.85
DUPLICATE_MIN_X_OVERLAP = 0.35


# =============================================================================
# Paths
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_SOURCE = Path(
    r"C:\Users\이승희\Desktop\Video Project 6.mp4"
)

DEFAULT_MODEL = Path(
    r"D:\임베디드 경진대회\머리 추적\models\pool_head_best.pt"
)

DEFAULT_REID_MODEL = Path(
    r"D:\임베디드 경진대회\머리 추적\ReID_runs\pool_head_reid_yolo11n_cls\weights\best.pt"
)

DEFAULT_TRACKER = Path(
    r"C:\Users\이승희\Desktop\bytetrack_pool.yaml"
)

DEFAULT_OUTPUT_DIR = Path(
    r"D:\임베디드 경진대회\머리 추적\익수자 판별 학습"
)


MOTION_LOW = "LOW_MOTION"
MOTION_HIGH = "HIGH_MOTION"

CLASS_NORMAL = "NORMAL"
CLASS_ACTIVE = "ACTIVE_DROWNING"
CLASS_PASSIVE = "PASSIVE_DROWNING"


# =============================================================================
# Basic data classes
# =============================================================================

@dataclass(frozen=True)
class HeadObservation:
    local_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    embedding: Optional[np.ndarray] = None


@dataclass
class TrackletPoint:
    frame: int
    center_x: float
    center_y: float
    scale: float
    embedding: Optional[np.ndarray] = None


@dataclass
class StableTrack:
    stable_id: int
    current_local_id: int
    first_frame: int
    last_seen_frame: int

    x1: float
    y1: float
    x2: float
    y2: float

    center_x: float
    center_y: float
    scale: float

    velocity_x: float = 0.0
    velocity_y: float = 0.0

    visible_frames: int = 1
    fragment_count: int = 1
    reid_embedding: Optional[np.ndarray] = None

    tail: deque[TrackletPoint] = field(
        default_factory=lambda: deque(maxlen=TRACKLET_HISTORY_SIZE)
    )


@dataclass
class TentativeTrack:
    local_id: int
    first_frame: int
    last_frame: int
    consecutive_frames: int

    points: deque[TrackletPoint] = field(
        default_factory=lambda: deque(maxlen=TRACKLET_HISTORY_SIZE)
    )

    latest_observation: Optional[HeadObservation] = None
    embeddings: deque[np.ndarray] = field(
        default_factory=lambda: deque(maxlen=TRACKLET_HISTORY_SIZE)
    )


# =============================================================================
# Pool ReID helpers
# =============================================================================

def l2_normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return vector
    return vector / norm


def cosine_similarity(
    first: Optional[np.ndarray],
    second: Optional[np.ndarray],
) -> Optional[float]:
    if first is None or second is None:
        return None
    first = l2_normalize(first)
    second = l2_normalize(second)
    return float(np.clip(np.dot(first, second), -1.0, 1.0))


def average_embeddings(values) -> Optional[np.ndarray]:
    vectors = [l2_normalize(value) for value in values if value is not None]
    if not vectors:
        return None
    return l2_normalize(np.mean(np.stack(vectors), axis=0))


def configure_stable_id_timing(fps: float) -> dict[str, int]:
    """30fps에서 검증한 Stable-ID 시간 의미를 입력 FPS에 맞게 환산한다."""
    global MIN_NEW_ID_CONFIRM_FRAMES
    global MAX_PREDICTION_FRAMES
    global VERY_SHORT_VISIBLE_FRAMES
    global SHORT_VISIBLE_FRAMES
    global MIN_RECOVERY_TRACKLET_FRAMES
    global TRACKLET_HISTORY_SIZE
    global TENTATIVE_MAX_MISSING_FRAMES
    global ALIAS_MAX_SHORT_VISIBLE_FRAMES
    global ALIAS_MIN_ESTABLISHED_FRAMES
    global ALIAS_BIRTH_MAX_GAP_FRAMES

    def frames(seconds: float) -> int:
        return max(1, int(round(seconds * fps)))

    MIN_NEW_ID_CONFIRM_FRAMES = frames(NEW_ID_CONFIRM_SECONDS)
    MAX_PREDICTION_FRAMES = frames(PREDICTION_SECONDS)
    VERY_SHORT_VISIBLE_FRAMES = frames(VERY_SHORT_SECONDS)
    SHORT_VISIBLE_FRAMES = frames(SHORT_SECONDS)
    MIN_RECOVERY_TRACKLET_FRAMES = frames(RECOVERY_TRACKLET_SECONDS)
    TRACKLET_HISTORY_SIZE = frames(TRACKLET_HISTORY_SECONDS)
    TENTATIVE_MAX_MISSING_FRAMES = frames(TENTATIVE_MAX_MISSING_SECONDS)
    ALIAS_MAX_SHORT_VISIBLE_FRAMES = frames(ALIAS_MAX_SHORT_SECONDS)
    ALIAS_MIN_ESTABLISHED_FRAMES = frames(ALIAS_MIN_ESTABLISHED_SECONDS)
    ALIAS_BIRTH_MAX_GAP_FRAMES = frames(ALIAS_BIRTH_MAX_GAP_SECONDS)

    return {
        "confirm_frames": MIN_NEW_ID_CONFIRM_FRAMES,
        "prediction_frames": MAX_PREDICTION_FRAMES,
        "very_short_frames": VERY_SHORT_VISIBLE_FRAMES,
        "short_frames": SHORT_VISIBLE_FRAMES,
        "recovery_tracklet_frames": MIN_RECOVERY_TRACKLET_FRAMES,
        "tracklet_history_frames": TRACKLET_HISTORY_SIZE,
        "tentative_max_missing_frames": TENTATIVE_MAX_MISSING_FRAMES,
        "alias_max_short_frames": ALIAS_MAX_SHORT_VISIBLE_FRAMES,
        "alias_min_established_frames": ALIAS_MIN_ESTABLISHED_FRAMES,
        "alias_birth_max_gap_frames": ALIAS_BIRTH_MAX_GAP_FRAMES,
    }


class PoolReIDEncoder:
    """YOLO classification fine-tune의 embedding을 cosine ReID에 사용한다."""

    def __init__(self, model_path: Path, device) -> None:
        self.model_path = Path(model_path)
        self.device = device
        self.model = YOLO(str(self.model_path))
        self.calls = 0
        self.images = 0

    @staticmethod
    def _crop(frame, observation: HeadObservation):
        height, width = frame.shape[:2]
        box_w = observation.x2 - observation.x1
        box_h = observation.y2 - observation.y1
        margin_x = box_w * REID_CROP_MARGIN
        margin_y = box_h * REID_CROP_MARGIN
        x1 = max(0, int(round(observation.x1 - margin_x)))
        y1 = max(0, int(round(observation.y1 - margin_y)))
        x2 = min(width, int(round(observation.x2 + margin_x)))
        y2 = min(height, int(round(observation.y2 + margin_y)))
        if x2 <= x1 or y2 <= y1:
            return None
        crop = frame[y1:y2, x1:x2]
        return crop if crop.size else None

    @staticmethod
    def _to_vector(value) -> np.ndarray:
        if isinstance(value, torch.Tensor):
            value = value.detach().float().cpu().numpy()
        return l2_normalize(np.asarray(value, dtype=np.float32).reshape(-1))

    def encode_observations(
        self,
        frame,
        observations: dict[int, HeadObservation],
    ) -> None:
        if not observations:
            return

        local_ids = []
        crops = []
        for local_id, observation in observations.items():
            crop = self._crop(frame, observation)
            if crop is None:
                continue
            local_ids.append(local_id)
            crops.append(crop)

        if not crops:
            return

        try:
            outputs = list(
                self.model.embed(
                    source=crops,
                    imgsz=REID_IMGSZ,
                    device=self.device,
                    verbose=False,
                )
            )
            if len(outputs) != len(crops):
                raise RuntimeError(
                    f"ReID embedding count mismatch: {len(outputs)} != {len(crops)}"
                )
            for local_id, output in zip(local_ids, outputs):
                object.__setattr__(
                    observations[local_id],
                    "embedding",
                    self._to_vector(output),
                )
            self.calls += 1
            self.images += len(crops)
        except Exception:
            # 일부 Ultralytics 버전에서 batch embed 반환 형식이 다를 수 있어
            # 개별 crop으로 안전하게 fallback한다.
            for local_id, crop in zip(local_ids, crops):
                outputs = list(
                    self.model.embed(
                        source=crop,
                        imgsz=REID_IMGSZ,
                        device=self.device,
                        verbose=False,
                    )
                )
                if not outputs:
                    continue
                object.__setattr__(
                    observations[local_id],
                    "embedding",
                    self._to_vector(outputs[0]),
                )
                self.calls += 1
                self.images += 1


# =============================================================================
# Persistent Stable ID Manager
# =============================================================================

class PersistentHeadIDManager:

    def __init__(
        self,
        max_gap_frames: int,
        distance_factor: float,
        reid_threshold: float = REID_MIN_SIMILARITY,
        geometry_max_score: float = RECOVERY_GEOMETRY_MAX_SCORE,
    ) -> None:

        self.max_gap_frames = max(1, int(max_gap_frames))
        self.distance_factor = float(distance_factor)
        self.reid_threshold = float(reid_threshold)
        self.geometry_max_score = float(geometry_max_score)

        self.next_stable_id = 1

        self.tracks: dict[int, StableTrack] = {}
        self.local_to_stable: dict[int, int] = {}
        self.tentative_tracks: dict[int, TentativeTrack] = {}
        self.alias_map: dict[int, int] = {}

        self.created_count = 0
        self.recovered_count = 0
        self.alias_merge_count = 0
        self.tentative_created_count = 0
        self.tentative_discarded_count = 0
        self.geometry_gate_rejected_count = 0
        self.reid_candidate_checks = 0
        self.reid_gate_rejected_count = 0
        self.reid_recovery_count = 0
        self.reid_alias_count = 0

    @staticmethod
    def _center(
        observation: HeadObservation,
    ) -> tuple[float, float]:

        return (
            (observation.x1 + observation.x2) / 2.0,
            (observation.y1 + observation.y2) / 2.0,
        )

    @staticmethod
    def _scale(
        observation: HeadObservation,
    ) -> float:

        width = max(
            1.0,
            float(observation.x2 - observation.x1),
        )

        height = max(
            1.0,
            float(observation.y2 - observation.y1),
        )

        return max(
            1.0,
            (width * height) ** 0.5,
        )

    def canonical_id(
        self,
        stable_id: int,
    ) -> int:

        seen: set[int] = set()

        while stable_id in self.alias_map:

            if stable_id in seen:
                break

            seen.add(stable_id)
            stable_id = self.alias_map[stable_id]

        return stable_id

    def _predicted_center_at(
        self,
        track: StableTrack,
        frame_index: int,
    ) -> tuple[float, float]:

        gap_frames = max(
            0,
            frame_index - track.last_seen_frame,
        )

        prediction_frames = min(
            gap_frames,
            MAX_PREDICTION_FRAMES,
        )

        return (
            track.center_x + track.velocity_x * prediction_frames,
            track.center_y + track.velocity_y * prediction_frames,
        )

    @staticmethod
    def _short_track_penalty(
        visible_frames: int,
    ) -> float:

        if visible_frames < VERY_SHORT_VISIBLE_FRAMES:
            return VERY_SHORT_TRACK_PENALTY

        if visible_frames < SHORT_VISIBLE_FRAMES:
            return SHORT_TRACK_PENALTY

        return 0.0

    @staticmethod
    def _tentative_embedding(
        tentative: TentativeTrack,
    ) -> Optional[np.ndarray]:
        return average_embeddings(tentative.embeddings)

    @staticmethod
    def _update_reid_embedding(
        track: StableTrack,
        embedding: Optional[np.ndarray],
    ) -> None:
        if embedding is None:
            return
        embedding = l2_normalize(embedding)
        if track.reid_embedding is None:
            track.reid_embedding = embedding
            return
        track.reid_embedding = l2_normalize(
            (1.0 - REID_EMA_ALPHA) * track.reid_embedding
            + REID_EMA_ALPHA * embedding
        )

    def _create_stable_track(
        self,
        frame_index: int,
        observation: HeadObservation,
        tentative: Optional[TentativeTrack] = None,
    ) -> tuple[int, dict[str, object]]:

        stable_id = self.next_stable_id
        self.next_stable_id += 1

        center_x, center_y = self._center(observation)
        scale = self._scale(observation)

        track = StableTrack(
            stable_id=stable_id,
            current_local_id=observation.local_id,
            first_frame=(
                tentative.first_frame
                if tentative is not None
                else frame_index
            ),
            last_seen_frame=frame_index,
            x1=float(observation.x1),
            y1=float(observation.y1),
            x2=float(observation.x2),
            y2=float(observation.y2),
            center_x=center_x,
            center_y=center_y,
            scale=scale,
            visible_frames=(
                tentative.consecutive_frames
                if tentative is not None
                else 1
            ),
        )

        backfill_points: list[TrackletPoint] = []

        if tentative is not None and tentative.points:

            for point in tentative.points:
                track.tail.append(point)
                backfill_points.append(point)

        else:

            point = TrackletPoint(
                frame=frame_index,
                center_x=center_x,
                center_y=center_y,
                scale=scale,
                embedding=observation.embedding,
            )

            track.tail.append(point)
            backfill_points.append(point)

        if len(track.tail) >= 2:

            previous = track.tail[-2]
            current = track.tail[-1]

            dt = max(
                1,
                current.frame - previous.frame,
            )

            track.velocity_x = (
                current.center_x - previous.center_x
            ) / dt

            track.velocity_y = (
                current.center_y - previous.center_y
            ) / dt

        if tentative is not None:
            track.reid_embedding = self._tentative_embedding(tentative)
        if track.reid_embedding is None:
            track.reid_embedding = observation.embedding

        self.tracks[stable_id] = track
        self.local_to_stable[observation.local_id] = stable_id

        self.created_count += 1

        return stable_id, {
            "event": "CREATE",
            "display_id": stable_id,
            "local_id": observation.local_id,
            "previous_local_id": "",
            "gap_frames": 0,
            "distance_ratio": "",
            "scale_ratio": "",
            "score": "",
            "backfill_points": backfill_points,
        }

    def _update_track(
        self,
        track: StableTrack,
        observation: HeadObservation,
        frame_index: int,
    ) -> None:

        new_x, new_y = self._center(observation)

        gap_frames = max(
            1,
            frame_index - track.last_seen_frame,
        )

        track.velocity_x = (
            new_x - track.center_x
        ) / gap_frames

        track.velocity_y = (
            new_y - track.center_y
        ) / gap_frames

        track.center_x = new_x
        track.center_y = new_y

        track.x1 = float(observation.x1)
        track.y1 = float(observation.y1)
        track.x2 = float(observation.x2)
        track.y2 = float(observation.y2)

        track.scale = self._scale(observation)
        track.current_local_id = observation.local_id
        track.last_seen_frame = frame_index
        track.visible_frames += 1

        track.tail.append(
            TrackletPoint(
                frame=frame_index,
                center_x=new_x,
                center_y=new_y,
                scale=track.scale,
                embedding=observation.embedding,
            )
        )

        self._update_reid_embedding(
            track,
            observation.embedding,
        )

    def _score_tracklet_candidate(
        self,
        track: StableTrack,
        tentative: TentativeTrack,
    ) -> Optional[dict[str, float]]:

        if not tentative.points:
            return None

        first_point = tentative.points[0]

        gap_frames = (
            first_point.frame - track.last_seen_frame
        )

        if gap_frames <= 0:
            return None

        if gap_frames > self.max_gap_frames:
            return None

        distance_ratios: list[float] = []
        scale_ratios: list[float] = []

        for point in tentative.points:

            predicted_x, predicted_y = self._predicted_center_at(
                track,
                point.frame,
            )

            normalization_scale = max(
                track.scale,
                point.scale,
                1.0,
            )

            distance_pixels = hypot(
                point.center_x - predicted_x,
                point.center_y - predicted_y,
            )

            distance_ratios.append(
                distance_pixels / normalization_scale
            )

            scale_ratios.append(
                point.scale / max(track.scale, 1.0)
            )

        mean_distance_ratio = (
            sum(distance_ratios) / len(distance_ratios)
        )

        mean_scale_ratio = (
            sum(scale_ratios) / len(scale_ratios)
        )

        if mean_distance_ratio > self.distance_factor:
            return None

        if not (
            MIN_SCALE_RATIO
            <= mean_scale_ratio
            <= MAX_SCALE_RATIO
        ):
            return None

        gap_ratio = (
            gap_frames / self.max_gap_frames
        )

        scale_difference = abs(
            1.0 - mean_scale_ratio
        )

        normalized_distance = (
            mean_distance_ratio
            / max(self.distance_factor, 1e-6)
        )

        base_score = (
            DISTANCE_WEIGHT * normalized_distance
            + GAP_WEIGHT * gap_ratio
            + SCALE_WEIGHT * scale_difference
        )

        penalty = self._short_track_penalty(
            track.visible_frames
        )

        geometry_score = base_score + penalty

        # V12: geometry가 애매하면 ReID가 높더라도 강제 recovery하지 않는다.
        if geometry_score > self.geometry_max_score:
            self.geometry_gate_rejected_count += 1
            return None

        reid_similarity = None
        candidate_embedding = self._tentative_embedding(tentative)

        if (
            track.reid_embedding is not None
            and candidate_embedding is not None
        ):
            self.reid_candidate_checks += 1
            reid_similarity = cosine_similarity(
                track.reid_embedding,
                candidate_embedding,
            )

            if (
                reid_similarity is not None
                and reid_similarity < self.reid_threshold
            ):
                self.reid_gate_rejected_count += 1
                return None

        if reid_similarity is None:
            final_score = geometry_score
            reid_used = False
        else:
            # cosine [-1, 1] -> distance [0, 1]
            appearance_distance = (1.0 - reid_similarity) / 2.0
            final_score = (
                V12_DISTANCE_WEIGHT * normalized_distance
                + V12_GAP_WEIGHT * gap_ratio
                + V12_SCALE_WEIGHT * scale_difference
                + V12_REID_WEIGHT * appearance_distance
                + penalty
            )
            reid_used = True

        return {
            "gap_frames": float(gap_frames),
            "distance_ratio": float(mean_distance_ratio),
            "scale_ratio": float(mean_scale_ratio),
            "base_score": float(base_score),
            "short_track_penalty": float(penalty),
            "geometry_score": float(geometry_score),
            "reid_similarity": (
                "" if reid_similarity is None else float(reid_similarity)
            ),
            "reid_used": bool(reid_used),
            "score": float(final_score),
        }

    def _find_recovery_candidate(
        self,
        tentative: TentativeTrack,
        used_stable_ids: set[int],
    ) -> tuple[
        Optional[int],
        Optional[dict[str, float]],
    ]:

        best_id = None
        best_details = None
        best_score = float("inf")

        for stable_id, track in self.tracks.items():

            canonical = self.canonical_id(stable_id)

            if canonical != stable_id:
                continue

            if stable_id in used_stable_ids:
                continue

            details = self._score_tracklet_candidate(
                track,
                tentative,
            )

            if details is None:
                continue

            if details["score"] < best_score:

                best_score = details["score"]
                best_id = stable_id
                best_details = details

        return best_id, best_details

    def _recover_track(
        self,
        stable_id: int,
        frame_index: int,
        observation: HeadObservation,
        tentative: TentativeTrack,
        details: dict[str, float],
    ) -> dict[str, object]:

        stable_id = self.canonical_id(stable_id)

        track = self.tracks[stable_id]
        previous_local_id = track.current_local_id

        if (
            previous_local_id in self.local_to_stable
            and self.canonical_id(
                self.local_to_stable[previous_local_id]
            )
            == stable_id
        ):
            self.local_to_stable.pop(
                previous_local_id,
                None,
            )

        # 현재 frame은 뒤의 정상 update에서 다시 들어갈 수 있으므로,
        # backfill에서는 current frame을 제외한다.
        backfill_points = [
            point
            for point in tentative.points
            if point.frame < frame_index
        ]

        # tentative의 과거 point를 StableTrack tail에도 보존
        existing_frames = {
            point.frame
            for point in track.tail
        }

        merged_tail = list(track.tail)

        for point in backfill_points:

            if point.frame not in existing_frames:
                merged_tail.append(point)

        merged_tail.sort(
            key=lambda point: point.frame
        )

        track.tail.clear()

        for point in merged_tail[-TRACKLET_HISTORY_SIZE:]:
            track.tail.append(point)

        track.fragment_count += 1

        self._update_track(
            track,
            observation,
            frame_index,
        )

        self.local_to_stable[
            observation.local_id
        ] = stable_id

        self.recovered_count += 1
        if details.get("reid_used", False):
            self.reid_recovery_count += 1

        return {
            "event": "RECOVER",
            "display_id": stable_id,
            "local_id": observation.local_id,
            "previous_local_id": previous_local_id,
            "gap_frames": details.get("gap_frames", ""),
            "distance_ratio": details.get("distance_ratio", ""),
            "scale_ratio": details.get("scale_ratio", ""),
            "score": details.get("score", ""),
            "geometry_score": details.get("geometry_score", ""),
            "reid_similarity": details.get("reid_similarity", ""),
            "reid_used": details.get("reid_used", False),
            "backfill_points": backfill_points,
        }

    def _update_tentative(
        self,
        frame_index: int,
        observation: HeadObservation,
    ) -> TentativeTrack:

        local_id = observation.local_id
        center_x, center_y = self._center(observation)

        point = TrackletPoint(
            frame=frame_index,
            center_x=center_x,
            center_y=center_y,
            scale=self._scale(observation),
            embedding=observation.embedding,
        )

        tentative = self.tentative_tracks.get(
            local_id
        )

        if tentative is None:

            tentative = TentativeTrack(
                local_id=local_id,
                first_frame=frame_index,
                last_frame=frame_index,
                consecutive_frames=1,
                latest_observation=observation,
            )

            tentative.points.append(point)
            if observation.embedding is not None:
                tentative.embeddings.append(observation.embedding)

            self.tentative_tracks[
                local_id
            ] = tentative

            self.tentative_created_count += 1

            return tentative

        if (
            frame_index - tentative.last_frame
            == 1
        ):
            tentative.consecutive_frames += 1

        else:

            tentative.first_frame = frame_index
            tentative.consecutive_frames = 1
            tentative.points.clear()

        tentative.last_frame = frame_index
        tentative.latest_observation = observation
        tentative.points.append(point)
        if observation.embedding is not None:
            tentative.embeddings.append(observation.embedding)

        return tentative

    def _cleanup_tentatives(
        self,
        frame_index: int,
        visible_local_ids: set[int],
    ) -> None:

        remove_ids: list[int] = []

        for local_id, tentative in self.tentative_tracks.items():

            if local_id in visible_local_ids:
                continue

            if (
                frame_index - tentative.last_frame
                > TENTATIVE_MAX_MISSING_FRAMES
            ):
                remove_ids.append(local_id)

        for local_id in remove_ids:

            self.tentative_tracks.pop(
                local_id,
                None,
            )

            self.tentative_discarded_count += 1

    def _find_alias_parent(
        self,
        short_track: StableTrack,
        current_observation: HeadObservation,
        frame_index: int,
        used_stable_ids: set[int],
    ) -> tuple[
        Optional[int],
        Optional[dict[str, float]],
    ]:

        if (
            short_track.visible_frames
            > ALIAS_MAX_SHORT_VISIBLE_FRAMES
        ):
            return None, None

        temp = TentativeTrack(
            local_id=current_observation.local_id,
            first_frame=short_track.first_frame,
            last_frame=frame_index,
            consecutive_frames=max(
                1,
                len(short_track.tail),
            ),
            latest_observation=current_observation,
        )

        for point in short_track.tail:
            temp.points.append(point)

        best_parent = None
        best_details = None
        best_score = float("inf")

        for candidate_id, candidate_track in self.tracks.items():

            if candidate_id == short_track.stable_id:
                continue

            if self.canonical_id(candidate_id) != candidate_id:
                continue

            if candidate_id in used_stable_ids:
                continue

            if (
                candidate_track.visible_frames
                < ALIAS_MIN_ESTABLISHED_FRAMES
            ):
                continue

            birth_gap = (
                short_track.first_frame
                - candidate_track.last_seen_frame
            )

            if birth_gap <= 0:
                continue

            if birth_gap > min(
                ALIAS_BIRTH_MAX_GAP_FRAMES,
                self.max_gap_frames,
            ):
                continue

            details = self._score_tracklet_candidate(
                candidate_track,
                temp,
            )

            if details is None:
                continue

            if details["score"] > ALIAS_MAX_SCORE:
                continue

            if details["score"] < best_score:

                best_score = details["score"]
                best_parent = candidate_id
                best_details = details

        return best_parent, best_details

    def _merge_alias(
        self,
        alias_id: int,
        canonical_id: int,
        current_local_id: int,
        details: dict[str, float],
    ) -> dict[str, object]:

        alias_id = self.canonical_id(alias_id)
        canonical_id = self.canonical_id(canonical_id)

        if alias_id == canonical_id:

            return {
                "event": "ALIAS_MERGE",
                "display_id": canonical_id,
                "local_id": current_local_id,
                "alias_from": alias_id,
                "alias_to": canonical_id,
            }

        alias_track = self.tracks.get(alias_id)
        canonical_track = self.tracks.get(canonical_id)

        if (
            alias_track is None
            or canonical_track is None
        ):
            return {
                "event": "ALIAS_MERGE_SKIPPED",
                "display_id": canonical_id,
                "local_id": current_local_id,
                "alias_from": alias_id,
                "alias_to": canonical_id,
            }

        canonical_track.fragment_count += (
            alias_track.fragment_count
        )

        merged_points = (
            list(canonical_track.tail)
            + list(alias_track.tail)
        )

        merged_points.sort(
            key=lambda point: point.frame
        )

        by_frame: dict[int, TrackletPoint] = {}

        for point in merged_points:
            by_frame[point.frame] = point

        canonical_track.tail.clear()

        for frame in sorted(by_frame)[
            -TRACKLET_HISTORY_SIZE:
        ]:
            canonical_track.tail.append(
                by_frame[frame]
            )

        if (
            alias_track.last_seen_frame
            > canonical_track.last_seen_frame
        ):
            canonical_track.last_seen_frame = (
                alias_track.last_seen_frame
            )
            canonical_track.current_local_id = (
                alias_track.current_local_id
            )
            canonical_track.x1 = alias_track.x1
            canonical_track.y1 = alias_track.y1
            canonical_track.x2 = alias_track.x2
            canonical_track.y2 = alias_track.y2
            canonical_track.center_x = alias_track.center_x
            canonical_track.center_y = alias_track.center_y
            canonical_track.scale = alias_track.scale
            canonical_track.velocity_x = alias_track.velocity_x
            canonical_track.velocity_y = alias_track.velocity_y

        canonical_track.visible_frames = max(
            canonical_track.visible_frames,
            alias_track.visible_frames,
        )

        if canonical_track.reid_embedding is None:
            canonical_track.reid_embedding = alias_track.reid_embedding
        elif alias_track.reid_embedding is not None:
            canonical_track.reid_embedding = l2_normalize(
                canonical_track.reid_embedding + alias_track.reid_embedding
            )

        for local_id, mapped_id in list(
            self.local_to_stable.items()
        ):
            if self.canonical_id(mapped_id) == alias_id:
                self.local_to_stable[local_id] = canonical_id

        self.local_to_stable[
            current_local_id
        ] = canonical_id

        self.alias_map[
            alias_id
        ] = canonical_id

        self.tracks.pop(
            alias_id,
            None,
        )

        self.alias_merge_count += 1
        if details.get("reid_used", False):
            self.reid_alias_count += 1

        return {
            "event": "ALIAS_MERGE",
            "display_id": canonical_id,
            "local_id": current_local_id,
            "alias_from": alias_id,
            "alias_to": canonical_id,
            "gap_frames": details.get("gap_frames", ""),
            "distance_ratio": details.get("distance_ratio", ""),
            "scale_ratio": details.get("scale_ratio", ""),
            "score": details.get("score", ""),
            "geometry_score": details.get("geometry_score", ""),
            "reid_similarity": details.get("reid_similarity", ""),
            "reid_used": details.get("reid_used", False),
        }

    def update(
        self,
        frame_index: int,
        observations: dict[int, HeadObservation],
    ) -> tuple[
        dict[int, int],
        list[dict[str, object]],
    ]:

        assignments: dict[int, int] = {}
        events: list[dict[str, object]] = []

        used_stable_ids: set[int] = set()
        visible_local_ids = set(observations.keys())

        unknown_observations: list[
            HeadObservation
        ] = []

        # PASS 1: 기존 local ID
        for local_id, observation in observations.items():

            mapped_id = self.local_to_stable.get(
                local_id
            )

            if mapped_id is None:

                unknown_observations.append(
                    observation
                )

                continue

            stable_id = self.canonical_id(
                mapped_id
            )

            track = self.tracks.get(
                stable_id
            )

            if track is None:

                self.local_to_stable.pop(
                    local_id,
                    None,
                )

                unknown_observations.append(
                    observation
                )

                continue

            if (
                track.visible_frames
                <= ALIAS_MAX_SHORT_VISIBLE_FRAMES
            ):

                parent_id, details = self._find_alias_parent(
                    short_track=track,
                    current_observation=observation,
                    frame_index=frame_index,
                    used_stable_ids=used_stable_ids,
                )

                if (
                    parent_id is not None
                    and details is not None
                ):

                    alias_event = self._merge_alias(
                        alias_id=stable_id,
                        canonical_id=parent_id,
                        current_local_id=local_id,
                        details=details,
                    )

                    canonical_id = self.canonical_id(
                        parent_id
                    )

                    canonical_track = self.tracks.get(
                        canonical_id
                    )

                    if canonical_track is not None:

                        self._update_track(
                            canonical_track,
                            observation,
                            frame_index,
                        )

                        assignments[
                            local_id
                        ] = canonical_id

                        used_stable_ids.add(
                            canonical_id
                        )

                        events.append(
                            alias_event
                        )

                        continue

            self._update_track(
                track,
                observation,
                frame_index,
            )

            assignments[
                local_id
            ] = stable_id

            used_stable_ids.add(
                stable_id
            )

        # PASS 2: 새로운 local ID
        for observation in unknown_observations:

            local_id = observation.local_id

            tentative = self._update_tentative(
                frame_index,
                observation,
            )

            # Tracklet recovery
            if (
                tentative.consecutive_frames
                >= MIN_RECOVERY_TRACKLET_FRAMES
            ):

                candidate_id, details = self._find_recovery_candidate(
                    tentative,
                    used_stable_ids,
                )

                if (
                    candidate_id is not None
                    and details is not None
                ):

                    canonical_candidate = self.canonical_id(
                        candidate_id
                    )

                    event = self._recover_track(
                        stable_id=canonical_candidate,
                        frame_index=frame_index,
                        observation=observation,
                        tentative=tentative,
                        details=details,
                    )

                    assignments[
                        local_id
                    ] = canonical_candidate

                    used_stable_ids.add(
                        canonical_candidate
                    )

                    events.append(
                        event
                    )

                    self.tentative_tracks.pop(
                        local_id,
                        None,
                    )

                    continue

            # 신규 Stable ID 확정
            if (
                tentative.consecutive_frames
                >= MIN_NEW_ID_CONFIRM_FRAMES
            ):

                stable_id, event = self._create_stable_track(
                    frame_index=frame_index,
                    observation=observation,
                    tentative=tentative,
                )

                assignments[
                    local_id
                ] = stable_id

                used_stable_ids.add(
                    stable_id
                )

                events.append(
                    event
                )

                self.tentative_tracks.pop(
                    local_id,
                    None,
                )

        self._cleanup_tentatives(
            frame_index,
            visible_local_ids,
        )

        return assignments, events

    def fragment_count(
        self,
        stable_id: int,
    ) -> int:

        stable_id = self.canonical_id(
            stable_id
        )

        track = self.tracks.get(
            stable_id
        )

        if track is None:
            return 0

        return track.fragment_count


# =============================================================================
# Detection / drowning state
# =============================================================================

@dataclass(frozen=True)
class Detection:
    local_id: int
    stable_id: int
    confidence: float

    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def center(
        self,
    ) -> tuple[float, float]:

        return (
            (self.x1 + self.x2) / 2.0,
            (self.y1 + self.y2) / 2.0,
        )

    @property
    def scale(
        self,
    ) -> float:

        width = max(
            1.0,
            float(self.x2 - self.x1),
        )

        height = max(
            1.0,
            float(self.y2 - self.y1),
        )

        return max(
            1.0,
            (width * height) ** 0.5,
        )


@dataclass(frozen=True)
class TrajectoryPoint:
    frame: int
    center: tuple[float, float] | None
    scale: float | None


@dataclass(frozen=True)
class DrowningFeatures:
    total_distance: float = 0.0
    net_distance: float = 0.0
    progress_ratio: float = 0.0
    vertical_motion: float = 0.0
    direction_changes: int = 0
    disappear_ratio: float = 0.0
    submerge_cycles: int = 0
    continuous_missing_time: float = 0.0
    motion_score: float = 0.0
    direction_change_rate: float = 0.0
    history_seconds: float = 0.0
    visible_ratio: float = 0.0


@dataclass
class StableTrackState:
    stable_id: int
    first_frame: int
    last_seen_frame: int

    points: deque[
        TrajectoryPoint
    ]

    latest_features: DrowningFeatures = field(
        default_factory=DrowningFeatures
    )

    motion_class: str = MOTION_LOW
    drowning_class: str = CLASS_NORMAL
    window_ready: bool = False

    visible_frames: int = 0
    missing_frames: int = 0

    max_motion_score: float = 0.0
    max_disappear_ratio: float = 0.0
    max_continuous_missing_time: float = 0.0

    class_frames: Counter[str] = field(
        default_factory=Counter
    )


# =============================================================================
# Optional duplicate suppression
# =============================================================================

def bbox_scale(
    box: list[float],
) -> float:

    x1, y1, x2, y2 = box

    width = max(
        1.0,
        x2 - x1,
    )

    height = max(
        1.0,
        y2 - y1,
    )

    return max(
        1.0,
        (width * height) ** 0.5,
    )


def bbox_center(
    box: list[float],
) -> tuple[float, float]:

    x1, y1, x2, y2 = box

    return (
        (x1 + x2) / 2.0,
        (y1 + y2) / 2.0,
    )


def x_overlap_ratio(
    first: list[float],
    second: list[float],
) -> float:

    overlap = max(
        0.0,
        min(first[2], second[2])
        -
        max(first[0], second[0]),
    )

    width_first = max(
        1.0,
        first[2] - first[0],
    )

    width_second = max(
        1.0,
        second[2] - second[0],
    )

    return (
        overlap
        /
        min(
            width_first,
            width_second,
        )
    )


def suppress_duplicate_heads(
    raw_boxes: dict[
        int,
        tuple[
            list[float],
            float,
        ],
    ],
) -> tuple[
    dict[
        int,
        tuple[
            list[float],
            float,
        ],
    ],
    list[int],
]:

    if (
        not ENABLE_DUPLICATE_SUPPRESSION
        or len(raw_boxes) <= 1
    ):
        return raw_boxes, []

    ordered = sorted(
        raw_boxes.items(),
        key=lambda item:
            item[1][1],
        reverse=True,
    )

    kept: dict[
        int,
        tuple[
            list[float],
            float,
        ],
    ] = {}

    suppressed: list[int] = []

    for local_id, (
        box,
        confidence,
    ) in ordered:

        remove = False

        for (
            kept_box,
            kept_conf,
        ) in kept.values():

            if (
                kept_conf
                <
                DUPLICATE_HIGH_CONF_MIN
            ):
                continue

            if (
                confidence
                >
                DUPLICATE_LOW_CONF_MAX
            ):
                continue

            cx, cy = bbox_center(
                box
            )

            kx, ky = bbox_center(
                kept_box
            )

            distance_heads = (
                hypot(
                    cx - kx,
                    cy - ky,
                )
                /
                max(
                    bbox_scale(box),
                    bbox_scale(kept_box),
                    1.0,
                )
            )

            overlap = x_overlap_ratio(
                box,
                kept_box,
            )

            if (
                distance_heads
                <= DUPLICATE_MAX_CENTER_HEADS
                and
                overlap
                >= DUPLICATE_MIN_X_OVERLAP
            ):
                remove = True
                suppressed.append(local_id)
                break

        if not remove:

            kept[
                local_id
            ] = (
                box,
                confidence,
            )

    return kept, suppressed


# =============================================================================
# Drowning Rule Engine
# =============================================================================

class DrowningRuleEngine:

    def __init__(
        self,
        fps: float,
        retention_frames: int,
    ) -> None:

        self.fps = max(
            float(fps),
            1e-6,
        )

        self.window_frames = max(
            2,
            int(
                round(
                    WINDOW_SECONDS
                    *
                    self.fps
                )
            ),
        )

        self.minimum_history_frames = max(
            2,
            int(
                round(
                    MIN_HISTORY_SECONDS
                    *
                    self.fps
                )
            ),
        )

        self.minimum_visible_frames = max(
            1,
            int(
                round(
                    MIN_VISIBLE_HISTORY_SECONDS
                    *
                    self.fps
                )
            ),
        )

        self.sample_frames = max(
            1,
            int(
                round(
                    MOTION_SAMPLE_SECONDS
                    *
                    self.fps
                )
            ),
        )

        self.submerge_min_frames = max(
            1,
            int(
                round(
                    SUBMERGE_MIN_SECONDS
                    *
                    self.fps
                )
            ),
        )

        self.retention_frames = max(
            int(retention_frames),
            self.window_frames,
        )

        self.states: dict[
            int,
            StableTrackState
        ] = {}

    def backfill_tracklet(
        self,
        stable_id: int,
        points: list[TrackletPoint],
    ) -> None:
        """
        tentative 기간의 trajectory를 Stable ID 확정/복구 직후
        익수 history에 되돌려 넣는다.

        같은 frame이 이미 존재하면 visible point를 우선하고 중복하지 않는다.
        """

        if not points:
            return

        state = self.states.get(
            stable_id
        )

        first_frame = min(
            point.frame
            for point in points
        )

        last_frame = max(
            point.frame
            for point in points
        )

        if state is None:

            state = StableTrackState(
                stable_id=stable_id,
                first_frame=first_frame,
                last_seen_frame=last_frame,
                points=deque(),
            )

            self.states[
                stable_id
            ] = state

        else:

            state.first_frame = min(
                state.first_frame,
                first_frame,
            )

            state.last_seen_frame = max(
                state.last_seen_frame,
                last_frame,
            )

        by_frame: dict[
            int,
            TrajectoryPoint
        ] = {
            point.frame: point
            for point in state.points
        }

        for point in points:

            by_frame[
                point.frame
            ] = TrajectoryPoint(
                frame=point.frame,
                center=(
                    point.center_x,
                    point.center_y,
                ),
                scale=point.scale,
            )

        merged = sorted(
            by_frame.values(),
            key=lambda item:
                item.frame,
        )

        state.points.clear()

        for point in merged[
            -self.window_frames:
        ]:
            state.points.append(
                point
            )

        # backfill된 visible frame 수를 보정
        state.visible_frames = sum(
            point.center is not None
            for point in state.points
        )

    def merge_states(
        self,
        alias_id: int,
        canonical_id: int,
    ) -> None:

        if alias_id == canonical_id:
            return

        alias_state = self.states.get(
            alias_id
        )

        if alias_state is None:
            return

        canonical_state = self.states.get(
            canonical_id
        )

        if canonical_state is None:

            alias_state.stable_id = (
                canonical_id
            )

            self.states[
                canonical_id
            ] = alias_state

            self.states.pop(
                alias_id,
                None,
            )

            return

        by_frame: dict[
            int,
            TrajectoryPoint
        ] = {}

        for point in canonical_state.points:
            by_frame[point.frame] = point

        for point in alias_state.points:

            current = by_frame.get(
                point.frame
            )

            if (
                current is None
                or
                (
                    current.center is None
                    and point.center is not None
                )
            ):
                by_frame[point.frame] = point

        merged_points = sorted(
            by_frame.values(),
            key=lambda point:
                point.frame,
        )

        canonical_state.points.clear()

        for point in merged_points[
            -self.window_frames:
        ]:
            canonical_state.points.append(
                point
            )

        canonical_state.first_frame = min(
            canonical_state.first_frame,
            alias_state.first_frame,
        )

        canonical_state.last_seen_frame = max(
            canonical_state.last_seen_frame,
            alias_state.last_seen_frame,
        )

        canonical_state.visible_frames = sum(
            point.center is not None
            for point in canonical_state.points
        )

        canonical_state.max_motion_score = max(
            canonical_state.max_motion_score,
            alias_state.max_motion_score,
        )

        canonical_state.max_disappear_ratio = max(
            canonical_state.max_disappear_ratio,
            alias_state.max_disappear_ratio,
        )

        canonical_state.max_continuous_missing_time = max(
            canonical_state.max_continuous_missing_time,
            alias_state.max_continuous_missing_time,
        )

        canonical_state.class_frames.update(
            alias_state.class_frames
        )

        self.states.pop(
            alias_id,
            None,
        )

    def update(
        self,
        frame: int,
        detections: dict[
            int,
            Detection
        ],
    ) -> dict[
        int,
        StableTrackState
    ]:

        for stable_id, detection in detections.items():

            state = self.states.get(
                stable_id
            )

            if state is None:

                state = StableTrackState(
                    stable_id=stable_id,
                    first_frame=frame,
                    last_seen_frame=frame,
                    points=deque(),
                )

                self.states[
                    stable_id
                ] = state

            state.last_seen_frame = frame

            # 같은 frame이 backfill로 이미 들어온 경우 교체
            existing = {
                point.frame: point
                for point in state.points
            }

            existing[
                frame
            ] = TrajectoryPoint(
                frame=frame,
                center=detection.center,
                scale=detection.scale,
            )

            state.points.clear()

            for point in sorted(
                existing.values(),
                key=lambda item:
                    item.frame,
            )[
                -self.window_frames:
            ]:
                state.points.append(
                    point
                )

        for stable_id, state in list(
            self.states.items()
        ):

            if (
                stable_id not in detections
                and
                frame - state.last_seen_frame
                <= self.retention_frames
            ):

                # 이 frame이 아직 없을 때만 missing을 추가
                if not any(
                    point.frame == frame
                    for point in state.points
                ):
                    state.points.append(
                        TrajectoryPoint(
                            frame=frame,
                            center=None,
                            scale=None,
                        )
                    )

            oldest_allowed = (
                frame
                -
                self.window_frames
                +
                1
            )

            while (
                state.points
                and
                state.points[0].frame
                <
                oldest_allowed
            ):
                state.points.popleft()

            if not state.points:
                continue

            if (
                frame - state.last_seen_frame
                > self.retention_frames
            ):
                continue

            state.visible_frames = sum(
                point.center is not None
                for point in state.points
            )

            state.missing_frames = sum(
                point.center is None
                for point in state.points
            )

            state.latest_features = self._calculate_features(
                state.points
            )

            state.window_ready = (
                frame
                -
                state.first_frame
                +
                1
                >=
                self.minimum_history_frames

                and

                len(state.points)
                >=
                self.minimum_history_frames

                and

                state.visible_frames
                >=
                self.minimum_visible_frames
            )

            (
                state.motion_class,
                state.drowning_class,
            ) = self._classify(
                state.latest_features,
                state.window_ready,
            )

            state.max_motion_score = max(
                state.max_motion_score,
                state.latest_features.motion_score,
            )

            state.max_disappear_ratio = max(
                state.max_disappear_ratio,
                state.latest_features.disappear_ratio,
            )

            state.max_continuous_missing_time = max(
                state.max_continuous_missing_time,
                state.latest_features.continuous_missing_time,
            )

            state.class_frames[
                state.drowning_class
            ] += 1

        return {
            stable_id: state
            for stable_id, state in self.states.items()
            if (
                state.points
                and
                frame - state.last_seen_frame
                <=
                self.retention_frames
            )
        }

    def _motion_samples(
        self,
        points: list[
            TrajectoryPoint
        ],
    ):

        if not points:
            return []

        buckets: dict[
            int,
            list[
                TrajectoryPoint
            ],
        ] = {}

        for point in points:

            bucket_index = (
                point.frame
                //
                self.sample_frames
            )

            buckets.setdefault(
                bucket_index,
                [],
            ).append(
                point
            )

        samples = []

        for bucket_index in range(
            min(buckets),
            max(buckets) + 1,
        ):

            bucket_points = buckets.get(
                bucket_index,
                [],
            )

            visible = [
                point
                for point in bucket_points
                if point.center is not None
            ]

            if not visible:

                samples.append(
                    None
                )

                continue

            x_values = [
                point.center[0]
                for point in visible
            ]

            y_values = [
                point.center[1]
                for point in visible
            ]

            samples.append(
                (
                    visible[-1].frame,
                    (
                        median(x_values),
                        median(y_values),
                    ),
                )
            )

        return samples

    def _calculate_features(
        self,
        point_deque,
    ) -> DrowningFeatures:

        points = list(
            point_deque
        )

        if not points:
            return DrowningFeatures()

        visible_points = [
            point
            for point in points
            if point.center is not None
        ]

        scales = [
            point.scale
            for point in visible_points
            if point.scale is not None
        ]

        head_scale = (
            max(
                1.0,
                median(scales),
            )
            if scales
            else
            1.0
        )

        history_seconds = (
            len(points)
            /
            self.fps
        )

        missing_flags = [
            point.center is None
            for point in points
        ]

        disappear_ratio = (
            sum(missing_flags)
            /
            len(points)
        )

        samples = self._motion_samples(
            points
        )

        segment_runs = []
        current_run = []
        visible_centers = []
        previous_center = None

        for sample in samples:

            if sample is None:

                if current_run:

                    segment_runs.append(
                        current_run
                    )

                    current_run = []

                previous_center = None
                continue

            _, center = sample

            visible_centers.append(
                center
            )

            if previous_center is not None:

                current_run.append(
                    (
                        (
                            center[0]
                            -
                            previous_center[0]
                        )
                        /
                        head_scale,
                        (
                            center[1]
                            -
                            previous_center[1]
                        )
                        /
                        head_scale,
                    )
                )

            previous_center = center

        if current_run:
            segment_runs.append(
                current_run
            )

        segments = [
            segment
            for run in segment_runs
            for segment in run
        ]

        total_distance = sum(
            hypot(dx, dy)
            for dx, dy in segments
        )

        vertical_motion = sum(
            abs(dy)
            for _, dy in segments
        )

        if len(visible_centers) >= 2:

            first_center = visible_centers[0]
            last_center = visible_centers[-1]

            net_distance = (
                hypot(
                    last_center[0]
                    -
                    first_center[0],
                    last_center[1]
                    -
                    first_center[1],
                )
                /
                head_scale
            )

        else:
            net_distance = 0.0

        progress_ratio = (
            min(
                1.0,
                net_distance / total_distance,
            )
            if total_distance > 1e-6
            else
            0.0
        )

        direction_changes = sum(
            self._count_direction_changes(
                run
            )
            for run in segment_runs
        )

        direction_change_rate = (
            direction_changes
            /
            max(
                history_seconds,
                1e-6,
            )
        )

        (
            submerge_cycles,
            longest_missing_frames,
        ) = self._missing_features(
            missing_flags
        )

        continuous_missing_time = (
            longest_missing_frames
            /
            self.fps
        )

        distance_component = min(
            1.0,
            total_distance
            /
            MOTION_TOTAL_DISTANCE_REFERENCE,
        )

        vertical_component = min(
            1.0,
            vertical_motion
            /
            MOTION_VERTICAL_REFERENCE,
        )

        direction_component = min(
            1.0,
            direction_changes
            /
            MOTION_DIRECTION_CHANGES_REFERENCE,
        )

        motion_score = (
            MOTION_TOTAL_DISTANCE_WEIGHT
            *
            distance_component
            +
            MOTION_VERTICAL_WEIGHT
            *
            vertical_component
            +
            MOTION_DIRECTION_WEIGHT
            *
            direction_component
        )

        return DrowningFeatures(
            total_distance=total_distance,
            net_distance=net_distance,
            progress_ratio=progress_ratio,
            vertical_motion=vertical_motion,
            direction_changes=direction_changes,
            disappear_ratio=disappear_ratio,
            submerge_cycles=submerge_cycles,
            continuous_missing_time=continuous_missing_time,
            motion_score=motion_score,
            direction_change_rate=direction_change_rate,
            history_seconds=history_seconds,
            visible_ratio=(
                1.0
                -
                disappear_ratio
            ),
        )

    @staticmethod
    def _count_direction_changes(
        segments,
    ) -> int:

        meaningful = [
            (dx, dy)
            for dx, dy in segments
            if hypot(dx, dy)
            >=
            MIN_DIRECTION_STEP_HEADS
        ]

        changes = 0

        for first, second in zip(
            meaningful,
            meaningful[1:],
        ):

            first_length = hypot(
                *first
            )

            second_length = hypot(
                *second
            )

            if (
                first_length <= 1e-9
                or
                second_length <= 1e-9
            ):
                continue

            cosine = (
                first[0]
                *
                second[0]
                +
                first[1]
                *
                second[1]
            ) / (
                first_length
                *
                second_length
            )

            angle = degrees(
                acos(
                    max(
                        -1.0,
                        min(
                            1.0,
                            cosine,
                        ),
                    )
                )
            )

            if (
                angle
                >=
                DIRECTION_CHANGE_DEGREES
            ):
                changes += 1

        return changes

    def _missing_features(
        self,
        missing_flags,
    ):

        cycles = 0
        longest = 0
        index = 0

        while index < len(
            missing_flags
        ):

            if not missing_flags[index]:

                index += 1
                continue

            start = index

            while (
                index
                <
                len(missing_flags)
                and
                missing_flags[index]
            ):
                index += 1

            run_length = (
                index
                -
                start
            )

            longest = max(
                longest,
                run_length,
            )

            is_bracketed = (
                start > 0
                and
                index
                <
                len(missing_flags)
            )

            if (
                is_bracketed
                and
                run_length
                >=
                self.submerge_min_frames
            ):
                cycles += 1

        return cycles, longest

    @staticmethod
    def _classify(
        features,
        window_ready,
    ):

        motion_class = (
            MOTION_HIGH
            if (
                features.motion_score
                >=
                HIGH_MOTION_SCORE_THRESHOLD
            )
            else
            MOTION_LOW
        )

        if not window_ready:

            return (
                motion_class,
                CLASS_NORMAL,
            )

        if motion_class == MOTION_HIGH:

            active = (
                features.progress_ratio
                <=
                ACTIVE_MAX_PROGRESS_RATIO
                and
                features.total_distance
                >=
                ACTIVE_MIN_TOTAL_DISTANCE
                and
                (
                    features.vertical_motion
                    >=
                    ACTIVE_MIN_VERTICAL_MOTION
                    or
                    features.direction_changes
                    >=
                    ACTIVE_MIN_DIRECTION_CHANGES
                )
            )

            return (
                motion_class,
                (
                    CLASS_ACTIVE
                    if active
                    else
                    CLASS_NORMAL
                ),
            )

        long_missing = (
            features.continuous_missing_time
            >=
            PASSIVE_MIN_CONTINUOUS_MISSING_SECONDS
        )

        disappeared_after_submerge = (
            features.disappear_ratio
            >=
            PASSIVE_MIN_DISAPPEAR_RATIO
            and
            features.submerge_cycles
            >=
            1
            and
            features.net_distance
            <=
            PASSIVE_MAX_NET_DISTANCE
        )

        repeated_submerge = (
            features.submerge_cycles
            >=
            PASSIVE_MIN_SUBMERGE_CYCLES
            and
            features.disappear_ratio
            >=
            PASSIVE_CYCLES_MIN_DISAPPEAR_RATIO
            and
            features.net_distance
            <=
            PASSIVE_MAX_NET_DISTANCE
        )

        passive = (
            long_missing
            or
            disappeared_after_submerge
            or
            repeated_submerge
        )

        return (
            motion_class,
            (
                CLASS_PASSIVE
                if passive
                else
                CLASS_NORMAL
            ),
        )


# =============================================================================
# CLI / video / output
# =============================================================================

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
    )

    parser.add_argument(
        "--tracker",
        type=Path,
        default=DEFAULT_TRACKER,
    )

    parser.add_argument(
        "--reid-model",
        type=Path,
        default=DEFAULT_REID_MODEL,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=960,
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--iou",
        type=float,
        default=0.70,
    )

    parser.add_argument(
        "--device",
        default="auto",
    )

    parser.add_argument(
        "--id-recovery-max-gap-s",
        type=float,
        default=ID_RECOVERY_MAX_GAP_SECONDS,
    )

    parser.add_argument(
        "--id-recovery-distance-factor",
        type=float,
        default=ID_RECOVERY_DISTANCE_FACTOR,
    )

    parser.add_argument(
        "--reid-threshold",
        type=float,
        default=REID_MIN_SIMILARITY,
    )

    parser.add_argument(
        "--recovery-geometry-max-score",
        type=float,
        default=RECOVERY_GEOMETRY_MAX_SCORE,
    )

    parser.add_argument(
        "--no-normalize-tracker-time",
        action="store_true",
        help="기존 tracker yaml을 그대로 사용하고 4초 track_buffer 정규화를 끕니다.",
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
    )

    return parser.parse_args()


def resolve_device(
    value: str,
):

    if value.lower() == "auto":

        return (
            0
            if torch.cuda.is_available()
            else
            "cpu"
        )

    return (
        int(value)
        if value.isdigit()
        else
        value
    )


def video_info(
    path: Path,
):

    capture = cv2.VideoCapture(
        str(path)
    )

    if not capture.isOpened():

        raise RuntimeError(
            f"영상을 열 수 없습니다: {path}"
        )

    fps = float(
        capture.get(
            cv2.CAP_PROP_FPS
        )
    )

    width = int(
        capture.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        capture.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    frame_count = int(
        capture.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    capture.release()

    if fps <= 0:
        fps = 30.0

    return (
        fps,
        width,
        height,
        frame_count,
    )


def create_next_version_dir(
    output_root: Path,
):

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    pattern = re.compile(
        r"^V(\d+)$",
        re.IGNORECASE,
    )

    versions = []

    for path in output_root.iterdir():

        if not path.is_dir():
            continue

        match = pattern.fullmatch(
            path.name
        )

        if match:

            versions.append(
                int(
                    match.group(1)
                )
            )

    number = (
        max(
            versions,
            default=0,
        )
        +
        1
    )

    while True:

        run_dir = (
            output_root
            /
            f"V{number}"
        )

        try:

            run_dir.mkdir()

            return number, run_dir

        except FileExistsError:
            number += 1


def create_runtime_tracker_yaml(
    run_dir: Path,
    fps: float,
) -> Path:
    track_buffer = max(1, int(round(BT_TRACK_BUFFER_SECONDS * fps)))
    path = run_dir / "bytetrack_pool_v12_runtime.yaml"
    path.write_text(
        "\n".join(
            [
                "tracker_type: bytetrack",
                "",
                f"track_high_thresh: {BT_TRACK_HIGH_THRESH}",
                f"track_low_thresh: {BT_TRACK_LOW_THRESH}",
                f"new_track_thresh: {BT_NEW_TRACK_THRESH}",
                "",
                f"track_buffer: {track_buffer}",
                "",
                f"match_thresh: {BT_MATCH_THRESH}",
                "",
                "fuse_score: True",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


def class_color(
    drowning_class: str,
):

    if drowning_class == CLASS_ACTIVE:
        return (0, 80, 255)

    if drowning_class == CLASS_PASSIVE:
        return (0, 0, 255)

    return (70, 210, 70)


def draw_detection(
    frame,
    detection: Detection,
    state: StableTrackState,
):

    color = class_color(
        state.drowning_class
    )

    cv2.rectangle(
        frame,
        (
            detection.x1,
            detection.y1,
        ),
        (
            detection.x2,
            detection.y2,
        ),
        color,
        1,
    )

    motion_text = (
        "HIGH"
        if state.motion_class == MOTION_HIGH
        else
        "LOW"
    )

    label = (
        f"ID {detection.stable_id} "
        f"{motion_text} "
        f"{state.drowning_class} "
        f"S={state.latest_features.motion_score:.2f}"
    )

    cv2.putText(
        frame,
        label,
        (
            detection.x1,
            max(
                20,
                detection.y1 - 5,
            ),
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        color,
        1,
        cv2.LINE_AA,
    )


CSV_COLUMNS = [
    "record_type",
    "frame",
    "time_s",
    "stable_id",
    "local_id",
    "visible",
    "window_ready",
    "motion_class",
    "drowning_class",
    "motion_score",
    "total_distance",
    "net_distance",
    "progress_ratio",
    "vertical_motion",
    "direction_changes",
    "direction_change_rate",
    "disappear_ratio",
    "submerge_cycles",
    "continuous_missing_time",
    "history_seconds",
    "visible_ratio",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2",
    "center_x",
    "center_y",
    "head_scale",
    "event",
    "previous_local_id",
    "gap_frames",
    "distance_ratio",
    "scale_ratio",
    "id_recovery_score",
    "geometry_score",
    "reid_similarity",
    "reid_used",
    "alias_from",
    "alias_to",
    "backfill_frames",
    "first_frame",
    "last_seen_frame",
    "visible_frames",
    "fragment_count",
    "normal_frames",
    "active_drowning_frames",
    "passive_drowning_frames",
    "final_drowning_class",
]


def write_feature_row(
    writer,
    frame_index,
    fps,
    state,
    detection,
):

    feature = state.latest_features

    row = {
        "record_type": "FRAME_FEATURE",
        "frame": frame_index,
        "time_s": f"{frame_index / fps:.3f}",
        "stable_id": state.stable_id,
        "local_id": (
            detection.local_id
            if detection is not None
            else
            ""
        ),
        "visible": int(
            detection is not None
        ),
        "window_ready": int(
            state.window_ready
        ),
        "motion_class": state.motion_class,
        "drowning_class": state.drowning_class,
        "motion_score": f"{feature.motion_score:.6f}",
        "total_distance": f"{feature.total_distance:.6f}",
        "net_distance": f"{feature.net_distance:.6f}",
        "progress_ratio": f"{feature.progress_ratio:.6f}",
        "vertical_motion": f"{feature.vertical_motion:.6f}",
        "direction_changes": feature.direction_changes,
        "direction_change_rate": f"{feature.direction_change_rate:.6f}",
        "disappear_ratio": f"{feature.disappear_ratio:.6f}",
        "submerge_cycles": feature.submerge_cycles,
        "continuous_missing_time": f"{feature.continuous_missing_time:.6f}",
        "history_seconds": f"{feature.history_seconds:.6f}",
        "visible_ratio": f"{feature.visible_ratio:.6f}",
    }

    if detection is not None:

        row.update(
            {
                "confidence": f"{detection.confidence:.6f}",
                "x1": detection.x1,
                "y1": detection.y1,
                "x2": detection.x2,
                "y2": detection.y2,
                "center_x": f"{detection.center[0]:.3f}",
                "center_y": f"{detection.center[1]:.3f}",
                "head_scale": f"{detection.scale:.3f}",
            }
        )

    writer.writerow(
        row
    )


def main() -> int:

    args = parse_args()

    source = args.source.expanduser().resolve()
    model_path = args.model.expanduser().resolve()
    tracker_path = args.tracker.expanduser().resolve()
    reid_model_path = args.reid_model.expanduser().resolve()
    output_root = args.output_dir.expanduser().resolve()

    for label, path in (
        ("영상", source),
        ("모델", model_path),
        ("트래커", tracker_path),
        ("ReID 모델", reid_model_path),
    ):

        if not path.is_file():

            print(
                f"{label} 파일 없음: {path}"
            )

            return 2

    (
        fps,
        width,
        height,
        source_frames,
    ) = video_info(
        source
    )

    device = resolve_device(
        args.device
    )

    max_gap_frames = max(
        1,
        int(
            round(
                args.id_recovery_max_gap_s
                *
                fps
            )
        ),
    )

    stable_timing = configure_stable_id_timing(fps)

    print(
        "[최신 추적 + 익수 판별]"
    )

    print(
        f"영상: {source}"
    )

    print(
        f"모델: {model_path}"
    )

    print(
        f"트래커: {tracker_path}"
    )

    print(
        f"Pool ReID: {reid_model_path}"
    )

    print(
        f"V12 gate: geometry<={args.recovery_geometry_max_score:.2f}, "
        f"ReID>={args.reid_threshold:.2f}"
    )

    print(
        f"FPS: {fps:.3f}"
    )

    print(
        f"해상도: {width}x{height}"
    )

    print(
        f"frames: {source_frames}"
    )

    print(
        f"device: {device}"
    )

    print(
        f"Duplicate suppression: "
        f"{'ON' if ENABLE_DUPLICATE_SUPPRESSION else 'OFF'}"
    )

    print(
        "Tracklet recovery: ON"
    )

    print(
        "Alias/history merge: ON"
    )

    print(
        "Tentative history backfill: ON"
    )

    print(
        "Stable-ID timing: "
        + ", ".join(f"{key}={value}" for key, value in stable_timing.items())
    )

    print(
        f"ByteTrack effective buffer: "
        f"{int(round(BT_TRACK_BUFFER_SECONDS * fps))} frames "
        f"({BT_TRACK_BUFFER_SECONDS:.1f}s)"
    )

    if args.dry_run:

        print(
            "dry-run OK"
        )

        return 0

    model = YOLO(
        str(
            model_path
        )
    )

    reid_encoder = PoolReIDEncoder(
        reid_model_path,
        device,
    )

    (
        version,
        run_dir,
    ) = create_next_version_dir(
        output_root
    )

    version_name = (
        f"V{version}"
    )

    effective_tracker_path = tracker_path
    if not args.no_normalize_tracker_time:
        effective_tracker_path = create_runtime_tracker_yaml(
            run_dir,
            fps,
        )

    output_video = (
        run_dir
        /
        f"{version_name}_drowning_result.mp4"
    )

    output_csv = (
        run_dir
        /
        f"{version_name}_drowning_all.csv"
    )

    output_json = (
        run_dir
        /
        f"{version_name}_drowning_summary.json"
    )

    video_writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (
            width,
            height,
        ),
    )

    if not video_writer.isOpened():

        raise RuntimeError(
            f"출력 비디오를 만들 수 없습니다: {output_video}"
        )

    stable_manager = PersistentHeadIDManager(
        max_gap_frames=max_gap_frames,
        distance_factor=args.id_recovery_distance_factor,
        reid_threshold=args.reid_threshold,
        geometry_max_score=args.recovery_geometry_max_score,
    )

    rule_engine = DrowningRuleEngine(
        fps=fps,
        retention_frames=max_gap_frames,
    )

    duplicate_suppressed_total = 0
    backfill_total = 0
    processed_frames = 0
    raw_local_ids = set()

    with output_csv.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as csv_handle:

        csv_writer = csv.DictWriter(
            csv_handle,
            fieldnames=CSV_COLUMNS,
            extrasaction="ignore",
        )

        csv_writer.writeheader()

        results = model.track(
            source=str(source),
            stream=True,
            persist=False,
            tracker=str(effective_tracker_path),
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=device,
            classes=[0],
            verbose=False,
        )

        try:

            for frame_index, result in enumerate(
                results
            ):

                if (
                    args.max_frames > 0
                    and
                    frame_index >= args.max_frames
                ):
                    break

                frame = result.orig_img.copy()
                boxes = result.boxes

                raw_boxes = {}

                if (
                    boxes is not None
                    and
                    boxes.id is not None
                    and
                    len(boxes) > 0
                ):

                    xyxy_values = (
                        boxes.xyxy
                        .detach()
                        .cpu()
                        .numpy()
                    )

                    local_ids = (
                        boxes.id
                        .int()
                        .detach()
                        .cpu()
                        .tolist()
                    )

                    confidences = (
                        boxes.conf
                        .detach()
                        .cpu()
                        .tolist()
                    )

                    for (
                        xyxy,
                        local_id,
                        confidence,
                    ) in zip(
                        xyxy_values,
                        local_ids,
                        confidences,
                    ):

                        (
                            x1,
                            y1,
                            x2,
                            y2,
                        ) = [
                            int(
                                round(
                                    float(value)
                                )
                            )
                            for value in xyxy
                        ]

                        raw_boxes[
                            int(local_id)
                        ] = (
                            [
                                x1,
                                y1,
                                x2,
                                y2,
                            ],
                            float(
                                confidence
                            ),
                        )

                raw_local_ids.update(
                    raw_boxes.keys()
                )

                (
                    filtered_boxes,
                    suppressed_ids,
                ) = suppress_duplicate_heads(
                    raw_boxes
                )

                duplicate_suppressed_total += len(
                    suppressed_ids
                )

                observations = {}

                for (
                    local_id,
                    (
                        xyxy,
                        _confidence,
                    ),
                ) in filtered_boxes.items():

                    observations[
                        local_id
                    ] = HeadObservation(
                        local_id=local_id,
                        x1=xyxy[0],
                        y1=xyxy[1],
                        x2=xyxy[2],
                        y2=xyxy[3],
                    )

                # V12 Pool ReID embedding.
                # 현재는 GT에서 검증한 동작과 동일하게 모든 visible head에 계산한다.
                # 추후 Raspberry Pi/Hailo 배포 단계에서 ambiguous recovery에만 호출하도록
                # 최적화할 수 있다.
                reid_encoder.encode_observations(
                    frame,
                    observations,
                )

                (
                    assignments,
                    events,
                ) = stable_manager.update(
                    frame_index,
                    observations,
                )

                # -------------------------------------------------------------
                # Alias merge + tentative backfill
                # -------------------------------------------------------------

                for event in events:

                    event_type = event.get(
                        "event"
                    )

                    stable_id = event.get(
                        "display_id"
                    )

                    if (
                        event_type
                        ==
                        "ALIAS_MERGE"
                    ):

                        alias_from = event.get(
                            "alias_from"
                        )

                        alias_to = event.get(
                            "alias_to"
                        )

                        if (
                            alias_from not in (
                                None,
                                "",
                            )
                            and
                            alias_to not in (
                                None,
                                "",
                            )
                        ):

                            rule_engine.merge_states(
                                int(alias_from),
                                int(alias_to),
                            )

                    backfill_points = event.get(
                        "backfill_points",
                        [],
                    )

                    if (
                        stable_id not in (
                            None,
                            "",
                        )
                        and
                        backfill_points
                    ):

                        canonical_id = stable_manager.canonical_id(
                            int(stable_id)
                        )

                        rule_engine.backfill_tracklet(
                            canonical_id,
                            list(
                                backfill_points
                            ),
                        )

                        backfill_total += len(
                            backfill_points
                        )

                    csv_writer.writerow(
                        {
                            "record_type": "ID_EVENT",
                            "frame": frame_index,
                            "time_s": f"{frame_index / fps:.3f}",
                            "stable_id": event.get(
                                "display_id",
                                "",
                            ),
                            "local_id": event.get(
                                "local_id",
                                "",
                            ),
                            "event": event_type,
                            "previous_local_id": event.get(
                                "previous_local_id",
                                "",
                            ),
                            "gap_frames": event.get(
                                "gap_frames",
                                "",
                            ),
                            "distance_ratio": event.get(
                                "distance_ratio",
                                "",
                            ),
                            "scale_ratio": event.get(
                                "scale_ratio",
                                "",
                            ),
                            "id_recovery_score": event.get(
                                "score",
                                "",
                            ),
                            "geometry_score": event.get(
                                "geometry_score",
                                "",
                            ),
                            "reid_similarity": event.get(
                                "reid_similarity",
                                "",
                            ),
                            "reid_used": int(
                                bool(event.get("reid_used", False))
                            ),
                            "alias_from": event.get(
                                "alias_from",
                                "",
                            ),
                            "alias_to": event.get(
                                "alias_to",
                                "",
                            ),
                            "backfill_frames": len(
                                backfill_points
                            ),
                        }
                    )

                detections = {}

                for (
                    local_id,
                    stable_id,
                ) in assignments.items():

                    box_record = filtered_boxes.get(
                        local_id
                    )

                    if box_record is None:
                        continue

                    (
                        xyxy,
                        confidence,
                    ) = box_record

                    canonical_id = stable_manager.canonical_id(
                        stable_id
                    )

                    candidate = Detection(
                        local_id=local_id,
                        stable_id=canonical_id,
                        confidence=confidence,
                        x1=xyxy[0],
                        y1=xyxy[1],
                        x2=xyxy[2],
                        y2=xyxy[3],
                    )

                    existing = detections.get(
                        canonical_id
                    )

                    if (
                        existing is None
                        or
                        candidate.confidence
                        >
                        existing.confidence
                    ):
                        detections[
                            canonical_id
                        ] = candidate

                active_states = rule_engine.update(
                    frame_index,
                    detections,
                )

                for stable_id, state in sorted(
                    active_states.items()
                ):

                    detection = detections.get(
                        stable_id
                    )

                    write_feature_row(
                        csv_writer,
                        frame_index,
                        fps,
                        state,
                        detection,
                    )

                    if detection is not None:

                        draw_detection(
                            frame,
                            detection,
                            state,
                        )

                video_writer.write(
                    frame
                )

                processed_frames += 1

                if (
                    processed_frames
                    %
                    max(
                        1,
                        int(
                            round(
                                fps
                                *
                                5.0
                            )
                        ),
                    )
                    ==
                    0
                ):

                    print(
                        f"{processed_frames}/{source_frames}"
                        f" | Stable={len(stable_manager.tracks)}"
                        f" | Recovery={stable_manager.recovered_count}"
                        f" | Alias={stable_manager.alias_merge_count}"
                        f" | Backfill={backfill_total}"
                        f" | Duplicate={duplicate_suppressed_total}"
                    )

        finally:

            video_writer.release()

    # TRACK_SUMMARY
    with output_csv.open(
        "a",
        newline="",
        encoding="utf-8-sig",
    ) as csv_handle:

        csv_writer = csv.DictWriter(
            csv_handle,
            fieldnames=CSV_COLUMNS,
            extrasaction="ignore",
        )

        for stable_id, state in sorted(
            rule_engine.states.items()
        ):

            if (
                state.class_frames[
                    CLASS_PASSIVE
                ]
                >
                0
            ):
                final_class = CLASS_PASSIVE

            elif (
                state.class_frames[
                    CLASS_ACTIVE
                ]
                >
                0
            ):
                final_class = CLASS_ACTIVE

            else:
                final_class = CLASS_NORMAL

            csv_writer.writerow(
                {
                    "record_type": "TRACK_SUMMARY",
                    "stable_id": stable_id,
                    "first_frame": state.first_frame,
                    "last_seen_frame": state.last_seen_frame,
                    "visible_frames": state.visible_frames,
                    "fragment_count": stable_manager.fragment_count(
                        stable_id
                    ),
                    "normal_frames": state.class_frames[
                        CLASS_NORMAL
                    ],
                    "active_drowning_frames": state.class_frames[
                        CLASS_ACTIVE
                    ],
                    "passive_drowning_frames": state.class_frames[
                        CLASS_PASSIVE
                    ],
                    "final_drowning_class": final_class,
                }
            )

    summary = {
        "version": version_name,
        "source": str(source),
        "model": str(model_path),
        "tracker": str(effective_tracker_path),
        "tracker_source": str(tracker_path),
        "reid_model": str(reid_model_path),
        "fps": fps,
        "resolution": [
            width,
            height,
        ],
        "tracking_enhancements": {
            "duplicate_suppression": ENABLE_DUPLICATE_SUPPRESSION,
            "tracklet_recovery": True,
            "alias_history_merge": True,
            "tentative_history_backfill": True,
            "pool_reid_gate": True,
            "geometry_recovery_gate": True,
            "tracker_time_normalized": not args.no_normalize_tracker_time,
        },
        "tracking_parameters": {
            "confirm_frames": MIN_NEW_ID_CONFIRM_FRAMES,
            "prediction_frames": MAX_PREDICTION_FRAMES,
            "recovery_tracklet_frames": MIN_RECOVERY_TRACKLET_FRAMES,
            "very_short_penalty": VERY_SHORT_TRACK_PENALTY,
            "short_penalty": SHORT_TRACK_PENALTY,
            "alias_max_score": ALIAS_MAX_SCORE,
            "geometry_max_score": args.recovery_geometry_max_score,
            "reid_threshold": args.reid_threshold,
            "reid_imgsz": REID_IMGSZ,
            "stable_timing": stable_timing,
            "bytetrack_buffer_seconds": BT_TRACK_BUFFER_SECONDS,
        },
        "results": {
            "processed_frames": processed_frames,
            "raw_bytetrack_ids": len(
                raw_local_ids
            ),
            "active_rule_states": len(
                rule_engine.states
            ),
            "stable_ids_created": stable_manager.created_count,
            "stable_ids_current": len(
                stable_manager.tracks
            ),
            "stable_ids_recovered": stable_manager.recovered_count,
            "alias_merges": stable_manager.alias_merge_count,
            "tentative_created": stable_manager.tentative_created_count,
            "tentative_discarded": stable_manager.tentative_discarded_count,
            "geometry_gate_rejected": stable_manager.geometry_gate_rejected_count,
            "reid_candidate_checks": stable_manager.reid_candidate_checks,
            "reid_gate_rejected": stable_manager.reid_gate_rejected_count,
            "reid_recoveries": stable_manager.reid_recovery_count,
            "reid_alias_merges": stable_manager.reid_alias_count,
            "reid_embed_calls": reid_encoder.calls,
            "reid_images": reid_encoder.images,
            "backfill_points": backfill_total,
            "duplicate_suppressed": duplicate_suppressed_total,
        },
        "environment": {
            "torch": torch.__version__,
            "ultralytics": ultralytics_version,
            "opencv": cv2.__version__,
            "cuda_available": torch.cuda.is_available(),
        },
    }

    output_json.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("완료")
    print(f"버전: {version_name}")
    print(f"처리 프레임: {processed_frames}")
    print(
        f"Recovery: "
        f"{stable_manager.recovered_count}"
    )
    print(
        f"Alias merge: "
        f"{stable_manager.alias_merge_count}"
    )
    print(
        f"Geometry gate rejected: "
        f"{stable_manager.geometry_gate_rejected_count}"
    )
    print(
        f"ReID checks/rejected/recovered: "
        f"{stable_manager.reid_candidate_checks}/"
        f"{stable_manager.reid_gate_rejected_count}/"
        f"{stable_manager.reid_recovery_count}"
    )
    print(
        f"Tentative backfill points: "
        f"{backfill_total}"
    )
    print(
        f"Duplicate suppressed: "
        f"{duplicate_suppressed_total}"
    )
    print(
        f"결과 폴더: {run_dir}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
