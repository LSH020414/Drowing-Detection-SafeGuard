from __future__ import annotations

import csv
import json
from collections import defaultdict, deque
from dataclasses import dataclass, field
from math import hypot
from pathlib import Path
from statistics import mean, median
from typing import Optional

import cv2
import numpy as np
import torch
from ultralytics import YOLO


# =============================================================================
# PATHS
# =============================================================================

SOURCE = Path(
    r"D:\임베디드 경진대회\머리 추적\CVAT용 영상\converted_10fps\30fps\Wavepool Lifeguard Rescue 12 - Spot the Drowning! (720p)_10fps.mp4"
)

HEAD_MODEL = Path(
    r"D:\임베디드 경진대회\머리 추적\models"
    r"\pool_head_best.pt"
)

POOL_REID_MODEL = Path(
    r"D:\임베디드 경진대회\머리 추적\ReID_runs"
    r"\pool_head_reid_yolo11n_cls\weights\best.pt"
)

OUTPUT_ROOT = Path(
    r"D:\임베디드 경진대회\머리 추적\트레킹 테스트\Wavepool_Rescue_12_V10_V11_compare"
)

TRACKER_10FPS = OUTPUT_ROOT / "bytetrack_pool_10fps.yaml"


# =============================================================================
# YOLO / ByteTrack
# =============================================================================

IMG_SIZE = 960
CONF = 0.05
IOU = 0.70

DEVICE = 0 if torch.cuda.is_available() else "cpu"


# 원래 30fps V10 계열 ByteTrack yaml:
#
# track_high_thresh: 0.35
# track_low_thresh : 0.10
# new_track_thresh : 0.40
# track_buffer     : 120
# match_thresh     : 0.80
#
# 120 / 30fps = 4초
# 10fps에서는 40 frame으로 맞춘다.

TRACK_HIGH_THRESH = 0.35
TRACK_LOW_THRESH = 0.10
NEW_TRACK_THRESH = 0.40
TRACK_BUFFER_SECONDS = 4.0
MATCH_THRESH = 0.80


# =============================================================================
# V10 시간 기준
#
# 기존 30fps frame 값의 실제 시간을 보존한다.
# =============================================================================

NEW_ID_CONFIRM_SECONDS = 6 / 30
PREDICTION_SECONDS = 5 / 30

VERY_SHORT_SECONDS = 8 / 30
SHORT_SECONDS = 15 / 30

RECOVERY_TRACKLET_SECONDS = 3 / 30
TRACKLET_HISTORY_SECONDS = 8 / 30

TENTATIVE_MAX_MISSING_SECONDS = 2 / 30

ALIAS_MAX_SHORT_SECONDS = 15 / 30
ALIAS_MIN_ESTABLISHED_SECONDS = 20 / 30
ALIAS_BIRTH_MAX_GAP_SECONDS = 30 / 30

RECOVERY_MAX_GAP_SECONDS = 2.0


# =============================================================================
# Geometry settings
# =============================================================================

ID_RECOVERY_DISTANCE_FACTOR = 3.2

MIN_SCALE_RATIO = 0.50
MAX_SCALE_RATIO = 2.00

VERY_SHORT_TRACK_PENALTY = 0.50
SHORT_TRACK_PENALTY = 0.25

ALIAS_MAX_SCORE = 0.45


# =============================================================================
# V10 geometry weights
# =============================================================================

V10_DISTANCE_WEIGHT = 0.70
V10_GAP_WEIGHT = 0.20
V10_SCALE_WEIGHT = 0.10


# =============================================================================
# V11 Pool-ReID weights
# =============================================================================

V11_DISTANCE_WEIGHT = 0.55
V11_GAP_WEIGHT = 0.15
V11_SCALE_WEIGHT = 0.10
V11_REID_WEIGHT = 0.20

# 이 값 아래의 appearance는 recovery 거부
REID_MIN_SIMILARITY = 0.30

# 매우 높은 similarity는 강한 근거
REID_STRONG_SIMILARITY = 0.70

REID_IMGSZ = 224
REID_CROP_MARGIN = 0.20

# stable embedding EMA
REID_EMA_ALPHA = 0.20


# =============================================================================
# Runtime config
# =============================================================================

@dataclass
class RuntimeConfig:
    fps: float

    min_new_id_confirm_frames: int
    max_prediction_frames: int

    very_short_frames: int
    short_frames: int

    min_recovery_tracklet_frames: int
    tracklet_history_size: int

    tentative_max_missing_frames: int

    alias_max_short_frames: int
    alias_min_established_frames: int
    alias_birth_max_gap_frames: int

    max_gap_frames: int


def sec_to_frames(
    seconds: float,
    fps: float,
    minimum: int = 1,
) -> int:
    return max(
        minimum,
        int(round(seconds * fps)),
    )


def make_runtime_config(
    fps: float,
) -> RuntimeConfig:

    return RuntimeConfig(
        fps=fps,

        min_new_id_confirm_frames=sec_to_frames(
            NEW_ID_CONFIRM_SECONDS,
            fps,
        ),

        max_prediction_frames=sec_to_frames(
            PREDICTION_SECONDS,
            fps,
        ),

        very_short_frames=sec_to_frames(
            VERY_SHORT_SECONDS,
            fps,
        ),

        short_frames=sec_to_frames(
            SHORT_SECONDS,
            fps,
        ),

        min_recovery_tracklet_frames=sec_to_frames(
            RECOVERY_TRACKLET_SECONDS,
            fps,
        ),

        tracklet_history_size=sec_to_frames(
            TRACKLET_HISTORY_SECONDS,
            fps,
        ),

        tentative_max_missing_frames=sec_to_frames(
            TENTATIVE_MAX_MISSING_SECONDS,
            fps,
        ),

        alias_max_short_frames=sec_to_frames(
            ALIAS_MAX_SHORT_SECONDS,
            fps,
        ),

        alias_min_established_frames=sec_to_frames(
            ALIAS_MIN_ESTABLISHED_SECONDS,
            fps,
        ),

        alias_birth_max_gap_frames=sec_to_frames(
            ALIAS_BIRTH_MAX_GAP_SECONDS,
            fps,
        ),

        max_gap_frames=sec_to_frames(
            RECOVERY_MAX_GAP_SECONDS,
            fps,
        ),
    )


# =============================================================================
# Data
# =============================================================================

@dataclass
class HeadObservation:
    local_id: int

    x1: float
    y1: float
    x2: float
    y2: float

    confidence: float

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

    center_x: float
    center_y: float
    scale: float

    velocity_x: float = 0.0
    velocity_y: float = 0.0

    visible_frames: int = 1
    fragment_count: int = 1

    embedding: Optional[np.ndarray] = None

    tail: deque = field(
        default_factory=deque
    )


@dataclass
class TentativeTrack:
    local_id: int

    first_frame: int
    last_frame: int
    consecutive_frames: int

    points: deque = field(
        default_factory=deque
    )


# =============================================================================
# Embedding helpers
# =============================================================================

def l2_normalize(
    vector: np.ndarray,
) -> np.ndarray:

    vector = vector.astype(
        np.float32,
        copy=False,
    ).reshape(-1)

    norm = float(
        np.linalg.norm(vector)
    )

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

    return float(
        np.clip(
            np.dot(first, second),
            -1.0,
            1.0,
        )
    )


def average_embeddings(
    embeddings: list[np.ndarray],
) -> Optional[np.ndarray]:

    if not embeddings:
        return None

    vectors = [
        l2_normalize(item)
        for item in embeddings
        if item is not None
    ]

    if not vectors:
        return None

    value = np.mean(
        np.stack(vectors),
        axis=0,
    )

    return l2_normalize(value)


def crop_head(
    frame: np.ndarray,
    observation: HeadObservation,
) -> Optional[np.ndarray]:

    height, width = frame.shape[:2]

    box_w = observation.x2 - observation.x1
    box_h = observation.y2 - observation.y1

    margin_x = box_w * REID_CROP_MARGIN
    margin_y = box_h * REID_CROP_MARGIN

    x1 = max(
        0,
        int(round(observation.x1 - margin_x)),
    )

    y1 = max(
        0,
        int(round(observation.y1 - margin_y)),
    )

    x2 = min(
        width,
        int(round(observation.x2 + margin_x)),
    )

    y2 = min(
        height,
        int(round(observation.y2 + margin_y)),
    )

    if x2 <= x1 or y2 <= y1:
        return None

    crop = frame[
        y1:y2,
        x1:x2
    ]

    if crop.size == 0:
        return None

    return crop


class PoolReIDEncoder:

    def __init__(
        self,
        model_path: Path,
        device,
    ) -> None:

        print(
            f"[ReID] Loading: {model_path}"
        )

        self.model = YOLO(
            str(model_path)
        )

        self.device = device

        self.calls = 0
        self.images = 0

    def encode_observations(
        self,
        frame: np.ndarray,
        observations: dict[int, HeadObservation],
    ) -> None:

        if not observations:
            return

        local_ids = []
        crops = []

        for local_id, observation in observations.items():

            crop = crop_head(
                frame,
                observation,
            )

            if crop is None:
                continue

            local_ids.append(
                local_id
            )

            crops.append(
                crop
            )

        if not crops:
            return

        try:

            outputs = self.model.embed(
                source=crops,
                imgsz=REID_IMGSZ,
                device=self.device,
                verbose=False,
            )

            outputs = list(outputs)

            if len(outputs) != len(crops):
                raise RuntimeError(
                    "embedding output count mismatch"
                )

            for local_id, tensor in zip(
                local_ids,
                outputs,
            ):

                if isinstance(
                    tensor,
                    torch.Tensor,
                ):

                    vector = (
                        tensor
                        .detach()
                        .float()
                        .cpu()
                        .numpy()
                        .reshape(-1)
                    )

                else:

                    vector = np.asarray(
                        tensor,
                        dtype=np.float32,
                    ).reshape(-1)

                observations[
                    local_id
                ].embedding = l2_normalize(
                    vector
                )

            self.calls += 1
            self.images += len(crops)

        except Exception as batch_error:

            print(
                "[ReID] Batch embed fallback:",
                batch_error,
            )

            for local_id, crop in zip(
                local_ids,
                crops,
            ):

                outputs = self.model.embed(
                    source=crop,
                    imgsz=REID_IMGSZ,
                    device=self.device,
                    verbose=False,
                )

                outputs = list(outputs)

                if not outputs:
                    continue

                tensor = outputs[0]

                if isinstance(
                    tensor,
                    torch.Tensor,
                ):

                    vector = (
                        tensor
                        .detach()
                        .float()
                        .cpu()
                        .numpy()
                        .reshape(-1)
                    )

                else:

                    vector = np.asarray(
                        tensor,
                        dtype=np.float32,
                    ).reshape(-1)

                observations[
                    local_id
                ].embedding = l2_normalize(
                    vector
                )

                self.calls += 1
                self.images += 1


# =============================================================================
# Stable ID Manager
# =============================================================================

class StableIDManager:

    def __init__(
        self,
        config: RuntimeConfig,
        use_reid: bool,
    ) -> None:

        self.cfg = config
        self.use_reid = use_reid

        self.next_stable_id = 1

        self.tracks: dict[
            int,
            StableTrack
        ] = {}

        self.local_to_stable: dict[
            int,
            int
        ] = {}

        self.tentatives: dict[
            int,
            TentativeTrack
        ] = {}

        self.alias_map: dict[
            int,
            int
        ] = {}

        self.created_count = 0
        self.recovered_count = 0
        self.alias_count = 0

        self.reid_candidate_checks = 0
        self.reid_gate_rejections = 0
        self.reid_recoveries = 0
        self.reid_alias_merges = 0

    # -------------------------------------------------------------------------
    # Geometry
    # -------------------------------------------------------------------------

    @staticmethod
    def center(
        observation: HeadObservation,
    ) -> tuple[float, float]:

        return (
            (
                observation.x1
                +
                observation.x2
            )
            / 2.0,

            (
                observation.y1
                +
                observation.y2
            )
            / 2.0,
        )

    @staticmethod
    def scale(
        observation: HeadObservation,
    ) -> float:

        width = max(
            1.0,
            observation.x2 - observation.x1,
        )

        height = max(
            1.0,
            observation.y2 - observation.y1,
        )

        return max(
            1.0,
            float(
                np.sqrt(
                    width * height
                )
            ),
        )

    def canonical_id(
        self,
        stable_id: int,
    ) -> int:

        seen = set()

        while stable_id in self.alias_map:

            if stable_id in seen:
                break

            seen.add(
                stable_id
            )

            stable_id = self.alias_map[
                stable_id
            ]

        return stable_id

    # -------------------------------------------------------------------------
    # Embedding
    # -------------------------------------------------------------------------

    def update_embedding(
        self,
        track: StableTrack,
        embedding: Optional[np.ndarray],
    ) -> None:

        if (
            not self.use_reid
            or embedding is None
        ):
            return

        embedding = l2_normalize(
            embedding
        )

        if track.embedding is None:

            track.embedding = embedding
            return

        combined = (
            (1.0 - REID_EMA_ALPHA)
            *
            track.embedding

            +

            REID_EMA_ALPHA
            *
            embedding
        )

        track.embedding = l2_normalize(
            combined
        )

    @staticmethod
    def tentative_embedding(
        tentative: TentativeTrack,
    ) -> Optional[np.ndarray]:

        values = [
            point.embedding
            for point in tentative.points
            if point.embedding is not None
        ]

        return average_embeddings(
            values
        )

    # -------------------------------------------------------------------------
    # Prediction
    # -------------------------------------------------------------------------

    def predicted_center(
        self,
        track: StableTrack,
        frame: int,
    ) -> tuple[float, float]:

        gap = max(
            0,
            frame - track.last_seen_frame,
        )

        prediction = min(
            gap,
            self.cfg.max_prediction_frames,
        )

        return (
            track.center_x
            +
            track.velocity_x
            *
            prediction,

            track.center_y
            +
            track.velocity_y
            *
            prediction,
        )

    # -------------------------------------------------------------------------
    # Penalty
    # -------------------------------------------------------------------------

    def short_penalty(
        self,
        visible_frames: int,
    ) -> float:

        if (
            visible_frames
            <
            self.cfg.very_short_frames
        ):

            return (
                VERY_SHORT_TRACK_PENALTY
            )

        if (
            visible_frames
            <
            self.cfg.short_frames
        ):

            return (
                SHORT_TRACK_PENALTY
            )

        return 0.0

    # -------------------------------------------------------------------------
    # Track creation
    # -------------------------------------------------------------------------

    def create_track(
        self,
        frame: int,
        observation: HeadObservation,
        tentative: TentativeTrack,
    ):

        stable_id = self.next_stable_id
        self.next_stable_id += 1

        cx, cy = self.center(
            observation
        )

        track = StableTrack(
            stable_id=stable_id,
            current_local_id=observation.local_id,
            first_frame=tentative.first_frame,
            last_seen_frame=frame,
            center_x=cx,
            center_y=cy,
            scale=self.scale(
                observation
            ),
            visible_frames=max(
                1,
                tentative.consecutive_frames,
            ),
            tail=deque(
                maxlen=self.cfg.tracklet_history_size
            ),
        )

        for point in tentative.points:
            track.tail.append(
                point
            )

        if len(track.tail) >= 2:

            first = track.tail[-2]
            second = track.tail[-1]

            dt = max(
                1,
                second.frame - first.frame,
            )

            track.velocity_x = (
                second.center_x
                -
                first.center_x
            ) / dt

            track.velocity_y = (
                second.center_y
                -
                first.center_y
            ) / dt

        track.embedding = (
            self.tentative_embedding(
                tentative
            )
            if self.use_reid
            else None
        )

        self.tracks[
            stable_id
        ] = track

        self.local_to_stable[
            observation.local_id
        ] = stable_id

        self.created_count += 1

        return stable_id, {
            "event": "CREATE",
            "stable_id": stable_id,
            "local_id": observation.local_id,
        }

    # -------------------------------------------------------------------------
    # Update track
    # -------------------------------------------------------------------------

    def update_track(
        self,
        track: StableTrack,
        observation: HeadObservation,
        frame: int,
    ) -> None:

        cx, cy = self.center(
            observation
        )

        gap = max(
            1,
            frame - track.last_seen_frame,
        )

        track.velocity_x = (
            cx - track.center_x
        ) / gap

        track.velocity_y = (
            cy - track.center_y
        ) / gap

        track.center_x = cx
        track.center_y = cy

        track.scale = self.scale(
            observation
        )

        track.current_local_id = (
            observation.local_id
        )

        track.last_seen_frame = frame
        track.visible_frames += 1

        track.tail.append(
            TrackletPoint(
                frame=frame,
                center_x=cx,
                center_y=cy,
                scale=track.scale,
                embedding=observation.embedding,
            )
        )

        self.update_embedding(
            track,
            observation.embedding,
        )

    # -------------------------------------------------------------------------
    # Tentative
    # -------------------------------------------------------------------------

    def update_tentative(
        self,
        frame: int,
        observation: HeadObservation,
    ) -> TentativeTrack:

        local_id = observation.local_id

        cx, cy = self.center(
            observation
        )

        point = TrackletPoint(
            frame=frame,
            center_x=cx,
            center_y=cy,
            scale=self.scale(
                observation
            ),
            embedding=observation.embedding,
        )

        tentative = self.tentatives.get(
            local_id
        )

        if tentative is None:

            tentative = TentativeTrack(
                local_id=local_id,
                first_frame=frame,
                last_frame=frame,
                consecutive_frames=1,
                points=deque(
                    maxlen=self.cfg.tracklet_history_size
                ),
            )

            tentative.points.append(
                point
            )

            self.tentatives[
                local_id
            ] = tentative

            return tentative

        if (
            frame
            -
            tentative.last_frame
            ==
            1
        ):

            tentative.consecutive_frames += 1

        else:

            tentative.first_frame = frame
            tentative.consecutive_frames = 1
            tentative.points.clear()

        tentative.last_frame = frame

        tentative.points.append(
            point
        )

        return tentative

    def cleanup_tentatives(
        self,
        frame: int,
        visible_local_ids: set[int],
    ) -> None:

        remove = []

        for local_id, tentative in self.tentatives.items():

            if local_id in visible_local_ids:
                continue

            gap = (
                frame
                -
                tentative.last_frame
            )

            if (
                gap
                >
                self.cfg.tentative_max_missing_frames
            ):

                remove.append(
                    local_id
                )

        for local_id in remove:

            self.tentatives.pop(
                local_id,
                None,
            )

    # -------------------------------------------------------------------------
    # Candidate score
    # -------------------------------------------------------------------------

    def score_candidate(
        self,
        track: StableTrack,
        tentative: TentativeTrack,
    ) -> Optional[dict]:

        if not tentative.points:
            return None

        first_point = (
            tentative.points[0]
        )

        gap_frames = (
            first_point.frame
            -
            track.last_seen_frame
        )

        if gap_frames <= 0:
            return None

        if (
            gap_frames
            >
            self.cfg.max_gap_frames
        ):
            return None

        distances = []
        scales = []

        for point in tentative.points:

            px, py = self.predicted_center(
                track,
                point.frame,
            )

            normalizer = max(
                track.scale,
                point.scale,
                1.0,
            )

            distance = hypot(
                point.center_x - px,
                point.center_y - py,
            )

            distances.append(
                distance / normalizer
            )

            scales.append(
                point.scale
                /
                max(
                    track.scale,
                    1.0,
                )
            )

        mean_distance = mean(
            distances
        )

        mean_scale = mean(
            scales
        )

        if (
            mean_distance
            >
            ID_RECOVERY_DISTANCE_FACTOR
        ):
            return None

        if not (
            MIN_SCALE_RATIO
            <=
            mean_scale
            <=
            MAX_SCALE_RATIO
        ):
            return None

        normalized_distance = (
            mean_distance
            /
            ID_RECOVERY_DISTANCE_FACTOR
        )

        gap_ratio = (
            gap_frames
            /
            self.cfg.max_gap_frames
        )

        scale_difference = abs(
            1.0 - mean_scale
        )

        penalty = self.short_penalty(
            track.visible_frames
        )

        reid_similarity = None

        if self.use_reid:

            candidate_embedding = (
                self.tentative_embedding(
                    tentative
                )
            )

            if (
                track.embedding is not None
                and
                candidate_embedding is not None
            ):

                self.reid_candidate_checks += 1

                reid_similarity = cosine_similarity(
                    track.embedding,
                    candidate_embedding,
                )

                if (
                    reid_similarity
                    is not None
                    and
                    reid_similarity
                    <
                    REID_MIN_SIMILARITY
                ):

                    self.reid_gate_rejections += 1

                    return None

        if not self.use_reid:

            score = (
                V10_DISTANCE_WEIGHT
                *
                normalized_distance

                +
                V10_GAP_WEIGHT
                *
                gap_ratio

                +
                V10_SCALE_WEIGHT
                *
                scale_difference

                +
                penalty
            )

        else:

            if reid_similarity is None:

                # embedding이 없는 경우 geometry만 사용하되
                # appearance 가중치를 distance 쪽으로 돌린다.

                score = (
                    0.70
                    *
                    normalized_distance

                    +
                    0.20
                    *
                    gap_ratio

                    +
                    0.10
                    *
                    scale_difference

                    +
                    penalty
                )

            else:

                appearance_distance = (
                    1.0
                    -
                    reid_similarity
                ) / 2.0

                score = (
                    V11_DISTANCE_WEIGHT
                    *
                    normalized_distance

                    +
                    V11_GAP_WEIGHT
                    *
                    gap_ratio

                    +
                    V11_SCALE_WEIGHT
                    *
                    scale_difference

                    +
                    V11_REID_WEIGHT
                    *
                    appearance_distance

                    +
                    penalty
                )

        return {
            "score": float(
                score
            ),

            "gap_frames": int(
                gap_frames
            ),

            "distance_ratio": float(
                mean_distance
            ),

            "scale_ratio": float(
                mean_scale
            ),

            "reid_similarity": (
                ""
                if reid_similarity is None
                else float(
                    reid_similarity
                )
            ),
        }

    # -------------------------------------------------------------------------
    # Recovery
    # -------------------------------------------------------------------------

    def find_recovery(
        self,
        tentative: TentativeTrack,
        used: set[int],
    ):

        candidates = []

        for stable_id, track in self.tracks.items():

            canonical = self.canonical_id(
                stable_id
            )

            if canonical != stable_id:
                continue

            if stable_id in used:
                continue

            details = self.score_candidate(
                track,
                tentative,
            )

            if details is None:
                continue

            candidates.append(
                (
                    details["score"],
                    stable_id,
                    details,
                )
            )

        if not candidates:
            return None, None

        candidates.sort(
            key=lambda item:
                item[0]
        )

        _, stable_id, details = (
            candidates[0]
        )

        return stable_id, details

    def recover(
        self,
        stable_id: int,
        observation: HeadObservation,
        frame: int,
        details: dict,
    ):

        stable_id = self.canonical_id(
            stable_id
        )

        track = self.tracks[
            stable_id
        ]

        old_local = (
            track.current_local_id
        )

        self.local_to_stable.pop(
            old_local,
            None,
        )

        track.fragment_count += 1

        self.update_track(
            track,
            observation,
            frame,
        )

        self.local_to_stable[
            observation.local_id
        ] = stable_id

        self.recovered_count += 1

        if (
            self.use_reid
            and
            details.get(
                "reid_similarity",
                "",
            )
            !=
            ""
        ):

            self.reid_recoveries += 1

        return {
            "event": "RECOVER",
            "stable_id": stable_id,
            "local_id": observation.local_id,
            "previous_local_id": old_local,
            **details,
        }

    # -------------------------------------------------------------------------
    # Alias
    # -------------------------------------------------------------------------

    def find_alias_parent(
        self,
        short_track: StableTrack,
        observation: HeadObservation,
        frame: int,
        used: set[int],
    ):

        if (
            short_track.visible_frames
            >
            self.cfg.alias_max_short_frames
        ):
            return None, None

        temp = TentativeTrack(
            local_id=observation.local_id,
            first_frame=short_track.first_frame,
            last_frame=frame,
            consecutive_frames=max(
                1,
                len(short_track.tail),
            ),
            points=deque(
                short_track.tail,
                maxlen=self.cfg.tracklet_history_size,
            ),
        )

        best = None

        for candidate_id, candidate in self.tracks.items():

            if (
                candidate_id
                ==
                short_track.stable_id
            ):
                continue

            if (
                self.canonical_id(
                    candidate_id
                )
                !=
                candidate_id
            ):
                continue

            if candidate_id in used:
                continue

            if (
                candidate.visible_frames
                <
                self.cfg.alias_min_established_frames
            ):
                continue

            birth_gap = (
                short_track.first_frame
                -
                candidate.last_seen_frame
            )

            if birth_gap <= 0:
                continue

            if (
                birth_gap
                >
                min(
                    self.cfg.alias_birth_max_gap_frames,
                    self.cfg.max_gap_frames,
                )
            ):
                continue

            details = self.score_candidate(
                candidate,
                temp,
            )

            if details is None:
                continue

            if (
                details["score"]
                >
                ALIAS_MAX_SCORE
            ):
                continue

            if (
                best is None
                or
                details["score"]
                <
                best[0]
            ):

                best = (
                    details["score"],
                    candidate_id,
                    details,
                )

        if best is None:
            return None, None

        return (
            best[1],
            best[2],
        )

    def merge_alias(
        self,
        alias_id: int,
        parent_id: int,
        local_id: int,
        details: dict,
    ):

        alias_id = self.canonical_id(
            alias_id
        )

        parent_id = self.canonical_id(
            parent_id
        )

        if alias_id == parent_id:
            return None

        alias = self.tracks.get(
            alias_id
        )

        parent = self.tracks.get(
            parent_id
        )

        if alias is None or parent is None:
            return None

        parent.fragment_count += (
            alias.fragment_count
        )

        # newer state
        if (
            alias.last_seen_frame
            >
            parent.last_seen_frame
        ):

            parent.last_seen_frame = (
                alias.last_seen_frame
            )

            parent.current_local_id = (
                alias.current_local_id
            )

            parent.center_x = (
                alias.center_x
            )

            parent.center_y = (
                alias.center_y
            )

            parent.scale = (
                alias.scale
            )

            parent.velocity_x = (
                alias.velocity_x
            )

            parent.velocity_y = (
                alias.velocity_y
            )

        parent.visible_frames = max(
            parent.visible_frames,
            alias.visible_frames,
        )

        if self.use_reid:

            if (
                parent.embedding is None
                and
                alias.embedding is not None
            ):

                parent.embedding = (
                    alias.embedding.copy()
                )

            elif (
                parent.embedding is not None
                and
                alias.embedding is not None
            ):

                parent.embedding = (
                    l2_normalize(
                        parent.embedding
                        +
                        alias.embedding
                    )
                )

        merged_points = (
            list(parent.tail)
            +
            list(alias.tail)
        )

        merged_points.sort(
            key=lambda point:
                point.frame
        )

        parent.tail.clear()

        for point in merged_points[
            -self.cfg.tracklet_history_size:
        ]:

            parent.tail.append(
                point
            )

        for lid, mapped in list(
            self.local_to_stable.items()
        ):

            if (
                self.canonical_id(
                    mapped
                )
                ==
                alias_id
            ):

                self.local_to_stable[
                    lid
                ] = parent_id

        self.local_to_stable[
            local_id
        ] = parent_id

        self.alias_map[
            alias_id
        ] = parent_id

        self.tracks.pop(
            alias_id,
            None,
        )

        self.alias_count += 1

        if (
            self.use_reid
            and
            details.get(
                "reid_similarity",
                "",
            )
            !=
            ""
        ):

            self.reid_alias_merges += 1

        return {
            "event": "ALIAS",
            "stable_id": parent_id,
            "alias_from": alias_id,
            "alias_to": parent_id,
            **details,
        }

    # -------------------------------------------------------------------------
    # Main update
    # -------------------------------------------------------------------------

    def update(
        self,
        frame: int,
        observations: dict[int, HeadObservation],
    ):

        assignments = {}
        events = []

        used = set()

        unknown = []

        visible_local_ids = set(
            observations.keys()
        )

        # PASS 1
        for local_id, observation in observations.items():

            mapped = self.local_to_stable.get(
                local_id
            )

            if mapped is None:

                unknown.append(
                    observation
                )

                continue

            stable_id = self.canonical_id(
                mapped
            )

            track = self.tracks.get(
                stable_id
            )

            if track is None:

                self.local_to_stable.pop(
                    local_id,
                    None,
                )

                unknown.append(
                    observation
                )

                continue

            # Short fragment re-check
            if (
                track.visible_frames
                <=
                self.cfg.alias_max_short_frames
            ):

                parent_id, details = (
                    self.find_alias_parent(
                        track,
                        observation,
                        frame,
                        used,
                    )
                )

                if (
                    parent_id is not None
                    and
                    details is not None
                ):

                    event = self.merge_alias(
                        stable_id,
                        parent_id,
                        local_id,
                        details,
                    )

                    canonical = (
                        self.canonical_id(
                            parent_id
                        )
                    )

                    parent = self.tracks.get(
                        canonical
                    )

                    if parent is not None:

                        self.update_track(
                            parent,
                            observation,
                            frame,
                        )

                        assignments[
                            local_id
                        ] = canonical

                        used.add(
                            canonical
                        )

                        if event is not None:
                            events.append(
                                event
                            )

                        continue

            self.update_track(
                track,
                observation,
                frame,
            )

            assignments[
                local_id
            ] = stable_id

            used.add(
                stable_id
            )

        # PASS 2
        for observation in unknown:

            local_id = observation.local_id

            tentative = self.update_tentative(
                frame,
                observation,
            )

            # Recovery
            if (
                tentative.consecutive_frames
                >=
                self.cfg.min_recovery_tracklet_frames
            ):

                candidate, details = (
                    self.find_recovery(
                        tentative,
                        used,
                    )
                )

                if (
                    candidate is not None
                    and
                    details is not None
                ):

                    event = self.recover(
                        candidate,
                        observation,
                        frame,
                        details,
                    )

                    stable_id = (
                        self.canonical_id(
                            candidate
                        )
                    )

                    assignments[
                        local_id
                    ] = stable_id

                    used.add(
                        stable_id
                    )

                    events.append(
                        event
                    )

                    self.tentatives.pop(
                        local_id,
                        None,
                    )

                    continue

            # New ID
            if (
                tentative.consecutive_frames
                >=
                self.cfg.min_new_id_confirm_frames
            ):

                stable_id, event = (
                    self.create_track(
                        frame,
                        observation,
                        tentative,
                    )
                )

                assignments[
                    local_id
                ] = stable_id

                used.add(
                    stable_id
                )

                events.append(
                    event
                )

                self.tentatives.pop(
                    local_id,
                    None,
                )

        self.cleanup_tentatives(
            frame,
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
# Video helpers
# =============================================================================

def get_video_info(
    path: Path,
):

    cap = cv2.VideoCapture(
        str(path)
    )

    if not cap.isOpened():

        raise RuntimeError(
            f"영상 열기 실패: {path}"
        )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    cap.release()

    return (
        float(fps),
        width,
        height,
        frames,
    )


def create_tracker_yaml(
    fps: float,
):

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    buffer_frames = max(
        1,
        round(
            TRACK_BUFFER_SECONDS
            *
            fps
        ),
    )

    text = f"""tracker_type: bytetrack

track_high_thresh: {TRACK_HIGH_THRESH}
track_low_thresh: {TRACK_LOW_THRESH}
new_track_thresh: {NEW_TRACK_THRESH}

track_buffer: {buffer_frames}

match_thresh: {MATCH_THRESH}

fuse_score: True
"""

    TRACKER_10FPS.write_text(
        text,
        encoding="utf-8",
    )

    print(
        f"[ByteTrack] track_buffer = "
        f"{buffer_frames} frames "
        f"({TRACK_BUFFER_SECONDS:.1f}s)"
    )


def draw_box(
    frame,
    observation,
    stable_id,
    mode,
):

    if mode == "V11_POOL_REID":
        color = (
            0,
            220,
            255,
        )
    else:
        color = (
            90,
            220,
            90,
        )

    x1 = int(
        observation.x1
    )

    y1 = int(
        observation.y1
    )

    x2 = int(
        observation.x2
    )

    y2 = int(
        observation.y2
    )

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        color,
        2,
    )

    text = (
        f"ID {stable_id}"
    )

    cv2.putText(
        frame,
        text,
        (
            x1,
            max(
                20,
                y1 - 5,
            ),
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )


# =============================================================================
# One evaluation pass
# =============================================================================

def run_mode(
    mode: str,
    fps: float,
    width: int,
    height: int,
    total_frames: int,
):

    use_reid = (
        mode
        ==
        "V11_POOL_REID"
    )

    print()
    print("=" * 80)
    print(mode)
    print("=" * 80)

    detector = YOLO(
        str(HEAD_MODEL)
    )

    reid_encoder = None

    if use_reid:

        reid_encoder = (
            PoolReIDEncoder(
                POOL_REID_MODEL,
                DEVICE,
            )
        )

    cfg = make_runtime_config(
        fps
    )

    manager = StableIDManager(
        config=cfg,
        use_reid=use_reid,
    )

    video_path = (
        OUTPUT_ROOT
        /
        f"{mode}_tracking.mp4"
    )

    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(
            *"mp4v"
        ),
        fps,
        (
            width,
            height,
        ),
    )

    raw_local_ids = set()

    assignment_records = []

    event_records = []

    raw_box_count = 0

    results = detector.track(
        source=str(SOURCE),
        stream=True,
        persist=False,
        tracker=str(
            TRACKER_10FPS
        ),
        conf=CONF,
        iou=IOU,
        imgsz=IMG_SIZE,
        device=DEVICE,
        classes=[0],
        verbose=False,
    )

    for frame_index, result in enumerate(
        results
    ):

        frame = (
            result.orig_img.copy()
        )

        observations = {}

        boxes = result.boxes

        if (
            boxes is not None
            and
            boxes.id is not None
            and
            len(boxes) > 0
        ):

            coords = (
                boxes.xyxy
                .detach()
                .cpu()
                .numpy()
            )

            ids = (
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

            for box, local_id, confidence in zip(
                coords,
                ids,
                confidences,
            ):

                local_id = int(
                    local_id
                )

                observation = (
                    HeadObservation(
                        local_id=local_id,

                        x1=float(box[0]),
                        y1=float(box[1]),
                        x2=float(box[2]),
                        y2=float(box[3]),

                        confidence=float(
                            confidence
                        ),
                    )
                )

                observations[
                    local_id
                ] = observation

                raw_local_ids.add(
                    local_id
                )

                raw_box_count += 1

        if (
            use_reid
            and
            reid_encoder is not None
        ):

            reid_encoder.encode_observations(
                frame,
                observations,
            )

        assignments, events = (
            manager.update(
                frame_index,
                observations,
            )
        )

        for event in events:

            event_records.append(
                {
                    "mode": mode,
                    "frame": frame_index,
                    "time_s": (
                        frame_index
                        /
                        fps
                    ),
                    **event,
                }
            )

        for local_id, stable_id in assignments.items():

            observation = (
                observations.get(
                    local_id
                )
            )

            if observation is None:
                continue

            canonical = manager.canonical_id(
                stable_id
            )

            assignment_records.append(
                (
                    frame_index,
                    canonical,
                    local_id,
                )
            )

            draw_box(
                frame,
                observation,
                canonical,
                mode,
            )

        cv2.putText(
            frame,
            mode,
            (
                20,
                35,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (
                255,
                255,
                255,
            ),
            2,
            cv2.LINE_AA,
        )

        writer.write(
            frame
        )

        if (
            frame_index + 1
        ) % max(
            1,
            int(
                fps * 5
            ),
        ) == 0:

            print(
                f"{frame_index + 1}"
                f"/{total_frames}"
                f" | Raw IDs="
                f"{len(raw_local_ids)}"
                f" | Stable="
                f"{len(manager.tracks)}"
                f" | Recovery="
                f"{manager.recovered_count}"
                f" | Alias="
                f"{manager.alias_count}"
            )

    writer.release()

    # -------------------------------------------------------------------------
    # 최종 canonical ID 기준으로 다시 집계
    # alias merge 이전 기록도 최종 canonical에 합친다.
    # -------------------------------------------------------------------------

    frame_sets = defaultdict(
        set
    )

    for (
        frame_index,
        stable_id,
        _local_id,
    ) in assignment_records:

        canonical = (
            manager.canonical_id(
                stable_id
            )
        )

        frame_sets[
            canonical
        ].add(
            frame_index
        )

    tracks = []

    for stable_id, frames in sorted(
        frame_sets.items()
    ):

        ordered = sorted(
            frames
        )

        observed_frames = len(
            ordered
        )

        span_frames = (
            ordered[-1]
            -
            ordered[0]
            +
            1
        )

        tracks.append(
            {
                "mode": mode,
                "stable_id": stable_id,

                "first_frame": ordered[0],
                "last_frame": ordered[-1],

                "observed_frames": observed_frames,
                "span_frames": span_frames,

                "observed_seconds": (
                    observed_frames
                    /
                    fps
                ),

                "span_seconds": (
                    span_frames
                    /
                    fps
                ),

                "fragment_count": (
                    manager.fragment_count(
                        stable_id
                    )
                ),
            }
        )

    observed_counts = [
        item[
            "observed_frames"
        ]
        for item in tracks
    ]

    span_counts = [
        item[
            "span_frames"
        ]
        for item in tracks
    ]

    summary = {
        "mode": mode,

        "video_frames": total_frames,
        "fps": fps,

        "raw_bytetrack_ids": len(
            raw_local_ids
        ),

        "raw_boxes": raw_box_count,

        "stable_ids_final": len(
            tracks
        ),

        "stable_ids_created": (
            manager.created_count
        ),

        "recoveries": (
            manager.recovered_count
        ),

        "alias_merges": (
            manager.alias_count
        ),

        "mean_observed_frames": (
            mean(
                observed_counts
            )
            if observed_counts
            else 0
        ),

        "median_observed_frames": (
            median(
                observed_counts
            )
            if observed_counts
            else 0
        ),

        "mean_span_frames": (
            mean(
                span_counts
            )
            if span_counts
            else 0
        ),

        "median_span_frames": (
            median(
                span_counts
            )
            if span_counts
            else 0
        ),

        "tracks_under_1s": sum(
            item[
                "observed_seconds"
            ]
            <
            1.0
            for item in tracks
        ),

        "tracks_under_3s": sum(
            item[
                "observed_seconds"
            ]
            <
            3.0
            for item in tracks
        ),

        "tracks_over_10s": sum(
            item[
                "span_seconds"
            ]
            >=
            10.0
            for item in tracks
        ),

        "total_fragment_count": sum(
            item[
                "fragment_count"
            ]
            for item in tracks
        ),

        "reid_candidate_checks": (
            manager.reid_candidate_checks
        ),

        "reid_gate_rejections": (
            manager.reid_gate_rejections
        ),

        "reid_recoveries": (
            manager.reid_recoveries
        ),

        "reid_alias_merges": (
            manager.reid_alias_merges
        ),
    }

    if reid_encoder is not None:

        summary[
            "reid_embed_calls"
        ] = (
            reid_encoder.calls
        )

        summary[
            "reid_images"
        ] = (
            reid_encoder.images
        )

    return (
        summary,
        tracks,
        event_records,
    )


# =============================================================================
# CSV
# =============================================================================

def write_dict_csv(
    path: Path,
    rows: list[dict],
):

    if not rows:
        return

    keys = []

    seen = set()

    for row in rows:

        for key in row.keys():

            if key not in seen:

                seen.add(
                    key
                )

                keys.append(
                    key
                )

    with path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=keys,
            extrasaction="ignore",
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


# =============================================================================
# Main
# =============================================================================

def main():

    print("=" * 80)
    print("V10 vs V11 POOL ReID TRACKING TEST")
    print("=" * 80)

    for label, path in [
        (
            "SOURCE",
            SOURCE,
        ),
        (
            "HEAD MODEL",
            HEAD_MODEL,
        ),
        (
            "POOL REID",
            POOL_REID_MODEL,
        ),
    ]:

        if not path.is_file():

            raise FileNotFoundError(
                f"{label} 없음:\n{path}"
            )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        fps,
        width,
        height,
        total_frames,
    ) = get_video_info(
        SOURCE
    )

    print(
        f"FPS = {fps}"
    )

    print(
        f"Frames = {total_frames}"
    )

    print(
        f"Resolution = "
        f"{width}x{height}"
    )

    print(
        f"Device = {DEVICE}"
    )

    create_tracker_yaml(
        fps
    )

    cfg = make_runtime_config(
        fps
    )

    print()
    print("[10fps time-normalized Stable ID]")
    print(
        "new ID confirm:",
        cfg.min_new_id_confirm_frames,
    )

    print(
        "prediction:",
        cfg.max_prediction_frames,
    )

    print(
        "recovery tracklet:",
        cfg.min_recovery_tracklet_frames,
    )

    print(
        "tracklet history:",
        cfg.tracklet_history_size,
    )

    print(
        "alias max short:",
        cfg.alias_max_short_frames,
    )

    print(
        "alias established:",
        cfg.alias_min_established_frames,
    )

    print(
        "alias birth gap:",
        cfg.alias_birth_max_gap_frames,
    )

    print(
        "recovery max gap:",
        cfg.max_gap_frames,
    )

    summaries = []
    all_tracks = []
    all_events = []

    for mode in [
        "V10_GEOMETRY",
        "V11_POOL_REID",
    ]:

        (
            summary,
            tracks,
            events,
        ) = run_mode(
            mode,
            fps,
            width,
            height,
            total_frames,
        )

        summaries.append(
            summary
        )

        all_tracks.extend(
            tracks
        )

        all_events.extend(
            events
        )

    write_dict_csv(
        OUTPUT_ROOT
        /
        "comparison_summary.csv",

        summaries,
    )

    write_dict_csv(
        OUTPUT_ROOT
        /
        "track_details.csv",

        all_tracks,
    )

    write_dict_csv(
        OUTPUT_ROOT
        /
        "recovery_events.csv",

        all_events,
    )

    (
        OUTPUT_ROOT
        /
        "comparison_summary.json"
    ).write_text(
        json.dumps(
            summaries,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 80)
    print("FINAL COMPARISON")
    print("=" * 80)

    for result in summaries:

        print()
        print(
            result[
                "mode"
            ]
        )

        print(
            " raw ByteTrack IDs :",
            result[
                "raw_bytetrack_ids"
            ],
        )

        print(
            " final Stable IDs  :",
            result[
                "stable_ids_final"
            ],
        )

        print(
            " Stable created    :",
            result[
                "stable_ids_created"
            ],
        )

        print(
            " Recovery          :",
            result[
                "recoveries"
            ],
        )

        print(
            " Alias merge       :",
            result[
                "alias_merges"
            ],
        )

        print(
            " median observed   :",
            result[
                "median_observed_frames"
            ],
            "frames",
        )

        print(
            " median span       :",
            result[
                "median_span_frames"
            ],
            "frames",
        )

        print(
            " tracks < 1 sec    :",
            result[
                "tracks_under_1s"
            ],
        )

        print(
            " tracks < 3 sec    :",
            result[
                "tracks_under_3s"
            ],
        )

        print(
            " tracks >= 10 sec  :",
            result[
                "tracks_over_10s"
            ],
        )

        print(
            " fragment total    :",
            result[
                "total_fragment_count"
            ],
        )

        if (
            result[
                "mode"
            ]
            ==
            "V11_POOL_REID"
        ):

            print(
                " ReID checks       :",
                result[
                    "reid_candidate_checks"
                ],
            )

            print(
                " ReID rejected     :",
                result[
                    "reid_gate_rejections"
                ],
            )

            print(
                " ReID recoveries   :",
                result[
                    "reid_recoveries"
                ],
            )

    print()
    print(
        "결과 폴더:"
    )

    print(
        OUTPUT_ROOT
    )


if __name__ == "__main__":
    main()