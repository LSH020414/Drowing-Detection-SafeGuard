from __future__ import annotations

import math
from pathlib import Path
from dataclasses import dataclass

import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

import drowning_main_v12 as v12


# =============================================================================
# PATH
# =============================================================================

VIDEO_ROOT = Path(
    r"C:\Users\이승희\Desktop\활동적 익수자"
)

MODEL_PATH = Path(
    r"D:\임베디드 경진대회\머리 추적\models\pool_head_best.pt"
)

OUTPUT_ROOT = Path(
    r"D:\임베디드 경진대회\머리 추적\익수자 판별 학습\ACTIVE_DROWNING_FINAL"
)

VIDEO_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".webm",
    ".m4v",
}


# =============================================================================
# TEMPORAL SETTINGS
# =============================================================================

WINDOW_SECONDS = 5.0
WINDOW_STRIDE_SECONDS = 1.0

# 분석용 temporal sampling
TARGET_SAMPLE_HZ = 10.0

# Detector
DETECT_CONF = 0.05
DETECT_IOU = 0.70
IMAGE_SIZE = 960


# =============================================================================
# IMPORTANT
#
# 이 폴더의 영상은 이미 익수자만 crop된 영상.
#
# 따라서:
#
#   영상 하나 = 동일한 사람 한 명
#
# ByteTrack/Stable ID가 중간에 바뀌었다는 이유로
# 학습 시계열을 잘라내지 않는다.
#
# Detector가 머리를 못 찾으면:
#
#   detected = 0
#   center = None
#
# 으로 저장한다.
#
# LOST 자체가 학습 feature다.
# =============================================================================


@dataclass
class Observation:

    frame: int
    time_s: float

    detected: int
    confidence: float

    center_x: float | None
    center_y: float | None

    scale: float | None

    x1: float | None
    y1: float | None
    x2: float | None
    y2: float | None


# =============================================================================
# VIDEO SEARCH
# =============================================================================

def find_videos(root: Path) -> list[Path]:

    videos = []

    for path in root.rglob("*"):

        if (
            path.is_file()
            and
            path.suffix.lower()
            in VIDEO_EXTENSIONS
        ):
            videos.append(path)

    return sorted(videos)


# =============================================================================
# SINGLE PERSON HEAD SELECTION
# =============================================================================

def choose_single_person_head(
    boxes,
    previous: Observation | None,
):
    """
    이미 익수자 한 명만 crop된 영상이므로
    여러 detection이 나오면 동일 인물의 머리 후보를 선택한다.

    우선순위:
    1. 이전 머리 위치와의 연속성
    2. confidence
    """

    if boxes is None or len(boxes) == 0:
        return None

    candidates = []

    for box in boxes:

        confidence = float(
            box.conf[0]
        )

        x1, y1, x2, y2 = map(
            float,
            box.xyxy[0].tolist()
        )

        width = max(
            1.0,
            x2 - x1
        )

        height = max(
            1.0,
            y2 - y1
        )

        center_x = (
            x1 + x2
        ) / 2.0

        center_y = (
            y1 + y2
        ) / 2.0

        scale = math.sqrt(
            width * height
        )

        if (
            previous is None
            or
            previous.detected == 0
            or
            previous.center_x is None
            or
            previous.center_y is None
        ):

            score = confidence

        else:

            dx = (
                center_x
                - previous.center_x
            )

            dy = (
                center_y
                - previous.center_y
            )

            distance = math.hypot(
                dx,
                dy
            )

            normalization_scale = max(
                scale,
                previous.scale
                if previous.scale is not None
                else scale,
                1.0,
            )

            distance_heads = (
                distance
                /
                normalization_scale
            )

            continuity = max(
                0.0,
                1.0
                -
                distance_heads / 5.0
            )

            score = (
                confidence
                +
                0.50 * continuity
            )

        candidates.append(
            {
                "score":
                    score,

                "confidence":
                    confidence,

                "x1":
                    x1,

                "y1":
                    y1,

                "x2":
                    x2,

                "y2":
                    y2,

                "center_x":
                    center_x,

                "center_y":
                    center_y,

                "scale":
                    scale,
            }
        )

    if not candidates:
        return None

    candidates.sort(
        key=lambda item:
            item["score"],
        reverse=True
    )

    return candidates[0]


# =============================================================================
# VIDEO -> DETECTED / LOST SEQUENCE
# =============================================================================

def process_video(
    model: YOLO,
    video_path: Path,
):

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():

        raise RuntimeError(
            f"영상 열기 실패: {video_path}"
        )

    fps = float(
        cap.get(
            cv2.CAP_PROP_FPS
        )
    )

    if fps <= 0:
        fps = 30.0

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

    frame_count = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    frame_step = max(
        1,
        round(
            fps
            /
            TARGET_SAMPLE_HZ
        )
    )

    actual_sample_hz = (
        fps
        /
        frame_step
    )

    observations = []

    previous_visible = None

    frame_index = 0

    while True:

        ok, frame = cap.read()

        if not ok:
            break

        if (
            frame_index
            %
            frame_step
            != 0
        ):

            frame_index += 1
            continue

        result = model.predict(
            frame,
            imgsz=IMAGE_SIZE,
            conf=DETECT_CONF,
            iou=DETECT_IOU,
            classes=[0],
            verbose=False,
        )[0]

        selected = choose_single_person_head(
            result.boxes,
            previous_visible,
        )

        time_s = (
            frame_index
            /
            fps
        )

        if selected is None:

            observation = Observation(
                frame=frame_index,
                time_s=time_s,

                detected=0,
                confidence=0.0,

                center_x=None,
                center_y=None,

                scale=None,

                x1=None,
                y1=None,
                x2=None,
                y2=None,
            )

        else:

            observation = Observation(
                frame=frame_index,
                time_s=time_s,

                detected=1,

                confidence=
                    selected[
                        "confidence"
                    ],

                center_x=
                    selected[
                        "center_x"
                    ],

                center_y=
                    selected[
                        "center_y"
                    ],

                scale=
                    selected[
                        "scale"
                    ],

                x1=
                    selected["x1"],

                y1=
                    selected["y1"],

                x2=
                    selected["x2"],

                y2=
                    selected["y2"],
            )

            previous_visible = (
                observation
            )

        observations.append(
            observation
        )

        frame_index += 1

    cap.release()

    return observations, {
        "fps":
            fps,

        "sample_hz":
            actual_sample_hz,

        "width":
            width,

        "height":
            height,

        "frame_count":
            frame_count,

        "duration_s":
            (
                frame_count
                /
                fps
            ),
    }


# =============================================================================
# RUN LENGTH
# =============================================================================

def get_runs(
    flags: list[bool],
    target: bool,
):

    runs = []

    current = 0

    for value in flags:

        if value == target:

            current += 1

        else:

            if current > 0:
                runs.append(
                    current
                )

            current = 0

    if current > 0:

        runs.append(
            current
        )

    return runs


def get_completed_lost_runs(
    missing_flags: list[bool],
):
    """
    DETECTED -> LOST -> DETECTED

    즉 양쪽이 visible로 둘러싸인 LOST만 계산.
    """

    runs = []

    index = 0
    count = len(
        missing_flags
    )

    while index < count:

        if not missing_flags[index]:

            index += 1
            continue

        start = index

        while (
            index < count
            and
            missing_flags[index]
        ):

            index += 1

        end = index

        bracketed = (
            start > 0
            and
            end < count
            and
            not missing_flags[
                start - 1
            ]
            and
            not missing_flags[
                end
            ]
        )

        if bracketed:

            runs.append(
                end - start
            )

    return runs


def run_statistics(
    runs,
    sample_hz: float,
):

    if not runs:

        return {
            "count": 0,
            "mean": 0.0,
            "std": 0.0,
            "cv": 0.0,
            "max": 0.0,
            "min": 0.0,
        }

    seconds = (
        np.asarray(
            runs,
            dtype=np.float32
        )
        /
        sample_hz
    )

    mean = float(
        np.mean(
            seconds
        )
    )

    std = float(
        np.std(
            seconds
        )
    )

    cv = (
        std / mean
        if mean > 1e-8
        else 0.0
    )

    return {
        "count":
            len(runs),

        "mean":
            mean,

        "std":
            std,

        "cv":
            cv,

        "max":
            float(
                np.max(
                    seconds
                )
            ),

        "min":
            float(
                np.min(
                    seconds
                )
            ),
    }


# =============================================================================
# MAIN FEATURE EXTRACTION
# =============================================================================

def calculate_features(
    window: list[Observation],
    sample_hz: float,
):

    if not window:
        return {}

    # =========================================================================
    # Visible / missing
    # =========================================================================

    visible = [
        observation
        for observation in window
        if observation.detected
    ]

    missing_flags = [
        not bool(
            observation.detected
        )
        for observation in window
    ]

    visible_ratio = (
        len(visible)
        /
        len(window)
    )

    disappear_ratio = (
        1.0
        -
        visible_ratio
    )

    # =========================================================================
    # Head scale normalization
    # =========================================================================

    scales = [
        observation.scale
        for observation in visible
        if observation.scale
        is not None
    ]

    if scales:

        head_scale = max(
            1.0,
            float(
                np.median(
                    scales
                )
            )
        )

        head_scale_mean = float(
            np.mean(
                scales
            )
        )

        head_scale_std = float(
            np.std(
                scales
            )
        )

        head_scale_change = float(
            (
                max(scales)
                -
                min(scales)
            )
            /
            max(
                head_scale_mean,
                1e-6
            )
        )

    else:

        head_scale = 1.0

        head_scale_mean = 0.0
        head_scale_std = 0.0
        head_scale_change = 0.0

    # =========================================================================
    # Motion segments
    #
    # LOST가 발생하면 motion run을 끊는다.
    # 이는 V12와 같은 방식이다.
    # =========================================================================

    motion_runs = []

    current_run = []

    previous = None

    for observation in window:

        if not observation.detected:

            if current_run:

                motion_runs.append(
                    current_run
                )

                current_run = []

            previous = None

            continue

        if previous is not None:

            dx = (
                observation.center_x
                -
                previous.center_x
            ) / head_scale

            dy = (
                observation.center_y
                -
                previous.center_y
            ) / head_scale

            current_run.append(
                (
                    dx,
                    dy
                )
            )

        previous = observation

    if current_run:

        motion_runs.append(
            current_run
        )

    motion_segments = [
        segment
        for run in motion_runs
        for segment in run
    ]

    # =========================================================================
    # Distance / vertical motion
    # =========================================================================

    total_distance = sum(
        math.hypot(
            dx,
            dy
        )
        for dx, dy
        in motion_segments
    )

    vertical_motion = sum(
        abs(
            dy
        )
        for _, dy
        in motion_segments
    )

    if len(visible) >= 2:

        first = visible[0]
        last = visible[-1]

        net_distance = (
            math.hypot(
                last.center_x
                -
                first.center_x,

                last.center_y
                -
                first.center_y,
            )
            /
            head_scale
        )

    else:

        net_distance = 0.0

    progress_ratio = (
        min(
            1.0,
            net_distance
            /
            total_distance
        )
        if total_distance > 1e-6
        else 0.0
    )

    # =========================================================================
    # Direction changes
    # =========================================================================

    direction_changes = 0

    for run in motion_runs:

        meaningful = [
            (
                dx,
                dy
            )
            for dx, dy
            in run
            if (
                math.hypot(
                    dx,
                    dy
                )
                >=
                v12.MIN_DIRECTION_STEP_HEADS
            )
        ]

        for (
            first_segment,
            second_segment
        ) in zip(
            meaningful,
            meaningful[1:]
        ):

            first_length = (
                math.hypot(
                    *first_segment
                )
            )

            second_length = (
                math.hypot(
                    *second_segment
                )
            )

            if (
                first_length <= 1e-9
                or
                second_length <= 1e-9
            ):

                continue

            cosine = (
                (
                    first_segment[0]
                    *
                    second_segment[0]
                )
                +
                (
                    first_segment[1]
                    *
                    second_segment[1]
                )
            ) / (
                first_length
                *
                second_length
            )

            cosine = max(
                -1.0,
                min(
                    1.0,
                    cosine
                )
            )

            angle = math.degrees(
                math.acos(
                    cosine
                )
            )

            if (
                angle
                >=
                v12.DIRECTION_CHANGE_DEGREES
            ):

                direction_changes += 1

    history_seconds = (
        len(window)
        /
        sample_hz
    )

    direction_change_rate = (
        direction_changes
        /
        max(
            history_seconds,
            1e-6
        )
    )

    # =========================================================================
    # SPEED
    #
    # 단위:
    # head-size / second
    #
    # 각 visible run 내부에서만 계산.
    # LOST를 건너뛰어 속도를 계산하지 않는다.
    # =========================================================================

    dt = (
        1.0
        /
        sample_hz
    )

    all_speeds = []

    speed_runs = []

    for run in motion_runs:

        speeds_this_run = []

        for dx, dy in run:

            distance_heads = (
                math.hypot(
                    dx,
                    dy
                )
            )

            speed = (
                distance_heads
                /
                dt
            )

            speeds_this_run.append(
                speed
            )

            all_speeds.append(
                speed
            )

        if speeds_this_run:

            speed_runs.append(
                speeds_this_run
            )

    if all_speeds:

        mean_speed = float(
            np.mean(
                all_speeds
            )
        )

        max_speed = float(
            np.max(
                all_speeds
            )
        )

        speed_std = float(
            np.std(
                all_speeds
            )
        )

    else:

        mean_speed = 0.0
        max_speed = 0.0
        speed_std = 0.0

    # =========================================================================
    # ACCELERATION
    #
    # 단위:
    # head-size / second^2
    #
    # IMPORTANT:
    # LOST 전후의 speed는 서로 연결하지 않는다.
    # 같은 visible run 안에서만 acceleration 계산.
    # =========================================================================

    accelerations = []

    for speeds_this_run in speed_runs:

        for (
            previous_speed,
            current_speed
        ) in zip(
            speeds_this_run,
            speeds_this_run[1:]
        ):

            acceleration = (
                current_speed
                -
                previous_speed
            ) / dt

            accelerations.append(
                abs(
                    acceleration
                )
            )

    if accelerations:

        mean_acceleration = float(
            np.mean(
                accelerations
            )
        )

        max_acceleration = float(
            np.max(
                accelerations
            )
        )

        acceleration_std = float(
            np.std(
                accelerations
            )
        )

    else:

        mean_acceleration = 0.0
        max_acceleration = 0.0
        acceleration_std = 0.0

    # =========================================================================
    # LOST features
    # =========================================================================

    lost_runs = get_runs(
        missing_flags,
        True
    )

    visible_runs = get_runs(
        missing_flags,
        False
    )

    completed_lost_runs = (
        get_completed_lost_runs(
            missing_flags
        )
    )

    lost_stats = run_statistics(
        lost_runs,
        sample_hz
    )

    completed_lost_stats = (
        run_statistics(
            completed_lost_runs,
            sample_hz
        )
    )

    visible_stats = run_statistics(
        visible_runs,
        sample_hz
    )

    # =========================================================================
    # Transitions
    # =========================================================================

    redetection_count = 0
    lost_entry_count = 0

    for (
        previous_flag,
        current_flag
    ) in zip(
        missing_flags,
        missing_flags[1:]
    ):

        # LOST -> DETECTED
        if (
            previous_flag
            and
            not current_flag
        ):

            redetection_count += 1

        # DETECTED -> LOST
        if (
            not previous_flag
            and
            current_flag
        ):

            lost_entry_count += 1

    # =========================================================================
    # Submerge cycle
    # =========================================================================

    min_submerge_samples = max(
        1,
        round(
            v12.SUBMERGE_MIN_SECONDS
            *
            sample_hz
        )
    )

    submerge_cycles = sum(
        1
        for run
        in completed_lost_runs
        if (
            run
            >=
            min_submerge_samples
        )
    )

    continuous_missing_time = (
        lost_stats[
            "max"
        ]
    )

    # =========================================================================
    # V12 motion score
    # =========================================================================

    distance_component = min(
        1.0,
        total_distance
        /
        v12.MOTION_TOTAL_DISTANCE_REFERENCE
    )

    vertical_component = min(
        1.0,
        vertical_motion
        /
        v12.MOTION_VERTICAL_REFERENCE
    )

    direction_component = min(
        1.0,
        direction_changes
        /
        v12.MOTION_DIRECTION_CHANGES_REFERENCE
    )

    motion_score = (
        (
            v12.MOTION_TOTAL_DISTANCE_WEIGHT
            *
            distance_component
        )
        +
        (
            v12.MOTION_VERTICAL_WEIGHT
            *
            vertical_component
        )
        +
        (
            v12.MOTION_DIRECTION_WEIGHT
            *
            direction_component
        )
    )

    # =========================================================================
    # RETURN
    # =========================================================================

    return {
        # ---------------------------------------------------------------------
        # V12 motion features
        # ---------------------------------------------------------------------

        "total_distance":
            total_distance,

        "net_distance":
            net_distance,

        "progress_ratio":
            progress_ratio,

        "vertical_motion":
            vertical_motion,

        "direction_changes":
            direction_changes,

        "direction_change_rate":
            direction_change_rate,

        "motion_score":
            motion_score,

        # ---------------------------------------------------------------------
        # Speed
        # ---------------------------------------------------------------------

        "mean_speed":
            mean_speed,

        "max_speed":
            max_speed,

        "speed_std":
            speed_std,

        # ---------------------------------------------------------------------
        # Acceleration
        # ---------------------------------------------------------------------

        "mean_acceleration":
            mean_acceleration,

        "max_acceleration":
            max_acceleration,

        "acceleration_std":
            acceleration_std,

        # ---------------------------------------------------------------------
        # Detection / LOST
        # ---------------------------------------------------------------------

        "visible_ratio":
            visible_ratio,

        "disappear_ratio":
            disappear_ratio,

        "continuous_missing_time":
            continuous_missing_time,

        "submerge_cycles":
            submerge_cycles,

        "lost_count":
            lost_stats[
                "count"
            ],

        "lost_duration_mean":
            lost_stats[
                "mean"
            ],

        "lost_duration_std":
            lost_stats[
                "std"
            ],

        "lost_duration_cv":
            lost_stats[
                "cv"
            ],

        "lost_duration_max":
            lost_stats[
                "max"
            ],

        "lost_duration_min":
            lost_stats[
                "min"
            ],

        # ---------------------------------------------------------------------
        # Completed LOST
        # ---------------------------------------------------------------------

        "completed_lost_count":
            completed_lost_stats[
                "count"
            ],

        "completed_lost_duration_mean":
            completed_lost_stats[
                "mean"
            ],

        "completed_lost_duration_std":
            completed_lost_stats[
                "std"
            ],

        "completed_lost_duration_cv":
            completed_lost_stats[
                "cv"
            ],

        # LOST irregularity
        "lost_irregularity":
            completed_lost_stats[
                "cv"
            ],

        # ---------------------------------------------------------------------
        # Visible run
        # ---------------------------------------------------------------------

        "visible_run_count":
            visible_stats[
                "count"
            ],

        "visible_duration_mean":
            visible_stats[
                "mean"
            ],

        "visible_duration_std":
            visible_stats[
                "std"
            ],

        "visible_duration_cv":
            visible_stats[
                "cv"
            ],

        # ---------------------------------------------------------------------
        # Transitions
        # ---------------------------------------------------------------------

        "redetection_count":
            redetection_count,

        "lost_entry_count":
            lost_entry_count,

        # ---------------------------------------------------------------------
        # Head scale
        # ---------------------------------------------------------------------

        "head_scale_mean":
            head_scale_mean,

        "head_scale_std":
            head_scale_std,

        "head_scale_change":
            head_scale_change,

        # ---------------------------------------------------------------------
        # History
        # ---------------------------------------------------------------------

        "history_seconds":
            history_seconds,

        "visible_frames":
            len(
                visible
            ),

        "missing_frames":
            (
                len(window)
                -
                len(visible)
            ),
    }


# =============================================================================
# WINDOW GENERATOR
# =============================================================================

def generate_windows(
    observations,
    sample_hz: float,
):

    window_size = max(
        1,
        round(
            WINDOW_SECONDS
            *
            sample_hz
        )
    )

    stride = max(
        1,
        round(
            WINDOW_STRIDE_SECONDS
            *
            sample_hz
        )
    )

    if (
        len(observations)
        <
        window_size
    ):

        return

    for start in range(
        0,
        (
            len(observations)
            -
            window_size
            +
            1
        ),
        stride
    ):

        end = (
            start
            +
            window_size
        )

        yield (
            start,
            end,
            observations[
                start:end
            ]
        )


# =============================================================================
# MAIN
# =============================================================================

def main():

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 80)

    print(
        "ACTIVE DROWNING FINAL FEATURE EXTRACTION"
    )

    print("=" * 80)

    print(
        f"[VIDEO ROOT] {VIDEO_ROOT}"
    )

    print(
        f"[MODEL] {MODEL_PATH}"
    )

    print()

    if not MODEL_PATH.is_file():

        raise FileNotFoundError(
            f"Head detector 없음: {MODEL_PATH}"
        )

    videos = find_videos(
        VIDEO_ROOT
    )

    print(
        f"[FOUND VIDEOS] {len(videos)}"
    )

    if not videos:

        print(
            "영상이 없습니다."
        )

        return

    device = (
        0
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"[DEVICE] {device}"
    )

    model = YOLO(
        str(
            MODEL_PATH
        )
    )

    window_rows = []
    sequence_rows = []
    video_rows = []

    videos_with_windows = 0

    # =========================================================================
    # VIDEO LOOP
    # =========================================================================

    for video_index, video_path in enumerate(
        videos,
        start=1
    ):

        print()
        print("=" * 80)

        print(
            f"[{video_index}/{len(videos)}]"
        )

        print(
            video_path
        )

        try:

            (
                observations,
                info,
            ) = process_video(
                model,
                video_path
            )

        except Exception as error:

            print(
                f"[ERROR] {error}"
            )

            continue

        visible_total = sum(
            observation.detected
            for observation
            in observations
        )

        overall_visible_ratio = (
            visible_total
            /
            max(
                1,
                len(
                    observations
                )
            )
        )

        video_window_count = 0

        # =====================================================================
        # WINDOW LOOP
        # =====================================================================

        for (
            start,
            end,
            window
        ) in generate_windows(
            observations,
            info[
                "sample_hz"
            ]
        ):

            features = (
                calculate_features(
                    window,
                    info[
                        "sample_hz"
                    ]
                )
            )

            start_s = (
                window[0]
                .time_s
            )

            end_s = (
                window[-1]
                .time_s
            )

            window_id = (
                f"V{video_index:03d}"
                f"_{start_s:.2f}"
                f"_{end_s:.2f}"
            )

            # LOST가 많다고 reject하지 않음.
            # 머리 검출이 5초 동안 0회면 별도 표시만 한다.

            if (
                features[
                    "visible_frames"
                ]
                == 0
            ):

                quality = (
                    "NO_DETECTION"
                )

            else:

                quality = (
                    "USABLE"
                )

            window_rows.append(
                {
                    "video_id":
                        video_path.name,

                    "window_id":
                        window_id,

                    "label":
                        "ACTIVE_DROWNING",

                    "quality":
                        quality,

                    "start_s":
                        start_s,

                    "end_s":
                        end_s,

                    **features,
                }
            )

            # =================================================================
            # RAW TEMPORAL SEQUENCE
            # =================================================================

            for observation in window:

                if (
                    observation.center_x
                    is not None
                ):

                    center_x_norm = (
                        observation.center_x
                        /
                        info["width"]
                    )

                else:

                    center_x_norm = (
                        np.nan
                    )

                if (
                    observation.center_y
                    is not None
                ):

                    center_y_norm = (
                        observation.center_y
                        /
                        info["height"]
                    )

                else:

                    center_y_norm = (
                        np.nan
                    )

                if (
                    observation.scale
                    is not None
                ):

                    head_scale_norm = (
                        observation.scale
                        /
                        math.sqrt(
                            max(
                                1.0,
                                (
                                    info["width"]
                                    *
                                    info["height"]
                                )
                            )
                        )
                    )

                else:

                    head_scale_norm = (
                        np.nan
                    )

                sequence_rows.append(
                    {
                        "video_id":
                            video_path.name,

                        "window_id":
                            window_id,

                        "label":
                            "ACTIVE_DROWNING",

                        "quality":
                            quality,

                        "frame":
                            observation.frame,

                        "time_s":
                            observation.time_s,

                        "detected":
                            observation.detected,

                        "confidence":
                            observation.confidence,

                        "center_x":
                            observation.center_x,

                        "center_y":
                            observation.center_y,

                        "center_x_norm":
                            center_x_norm,

                        "center_y_norm":
                            center_y_norm,

                        "head_scale":
                            observation.scale,

                        "head_scale_norm":
                            head_scale_norm,
                    }
                )

            video_window_count += 1

        if video_window_count > 0:

            videos_with_windows += 1

        video_rows.append(
            {
                "video_id":
                    video_path.name,

                "duration_s":
                    info[
                        "duration_s"
                    ],

                "fps":
                    info[
                        "fps"
                    ],

                "sample_hz":
                    info[
                        "sample_hz"
                    ],

                "width":
                    info[
                        "width"
                    ],

                "height":
                    info[
                        "height"
                    ],

                "sampled_frames":
                    len(
                        observations
                    ),

                "visible_frames":
                    visible_total,

                "overall_visible_ratio":
                    overall_visible_ratio,

                "windows":
                    video_window_count,
            }
        )

        print(
            f"WINDOWS={video_window_count}"
            f" | VisibleRatio="
            f"{overall_visible_ratio:.3f}"
        )

    # =========================================================================
    # DATAFRAME
    # =========================================================================

    windows_df = pd.DataFrame(
        window_rows
    )

    sequences_df = pd.DataFrame(
        sequence_rows
    )

    videos_df = pd.DataFrame(
        video_rows
    )

    # =========================================================================
    # SAVE
    # =========================================================================

    all_windows_path = (
        OUTPUT_ROOT
        /
        "ACTIVE_FINAL_ALL_windows.csv"
    )

    usable_windows_path = (
        OUTPUT_ROOT
        /
        "ACTIVE_FINAL_USABLE_windows.csv"
    )

    sequences_path = (
        OUTPUT_ROOT
        /
        "ACTIVE_FINAL_sequences.csv"
    )

    video_summary_path = (
        OUTPUT_ROOT
        /
        "ACTIVE_FINAL_video_summary.csv"
    )

    windows_df.to_csv(
        all_windows_path,
        index=False,
        encoding="utf-8-sig"
    )

    if not windows_df.empty:

        windows_df[
            windows_df[
                "quality"
            ]
            ==
            "USABLE"
        ].to_csv(
            usable_windows_path,
            index=False,
            encoding="utf-8-sig"
        )

    sequences_df.to_csv(
        sequences_path,
        index=False,
        encoding="utf-8-sig"
    )

    videos_df.to_csv(
        video_summary_path,
        index=False,
        encoding="utf-8-sig"
    )

    # =========================================================================
    # FINAL REPORT
    # =========================================================================

    print()
    print("=" * 80)

    print(
        "FINISHED"
    )

    print("=" * 80)

    print(
        f"VIDEOS              : "
        f"{len(videos)}"
    )

    print(
        f"VIDEOS WITH WINDOWS : "
        f"{videos_with_windows}"
    )

    print(
        f"TOTAL WINDOWS       : "
        f"{len(windows_df)}"
    )

    if not windows_df.empty:

        usable = int(
            (
                windows_df[
                    "quality"
                ]
                ==
                "USABLE"
            ).sum()
        )

        no_detection = int(
            (
                windows_df[
                    "quality"
                ]
                ==
                "NO_DETECTION"
            ).sum()
        )

        print(
            f"USABLE WINDOWS      : "
            f"{usable}"
        )

        print(
            f"NO DETECTION        : "
            f"{no_detection}"
        )

        print()

        print(
            f"Mean disappear ratio : "
            f"{windows_df['disappear_ratio'].mean():.3f}"
        )

        print(
            f"Mean lost count       : "
            f"{windows_df['lost_count'].mean():.3f}"
        )

        print(
            f"Mean lost CV          : "
            f"{windows_df['lost_duration_cv'].mean():.3f}"
        )

        print(
            f"Mean redetection      : "
            f"{windows_df['redetection_count'].mean():.3f}"
        )

        print()

        print(
            f"Mean speed            : "
            f"{windows_df['mean_speed'].mean():.3f}"
        )

        print(
            f"Mean speed std        : "
            f"{windows_df['speed_std'].mean():.3f}"
        )

        print(
            f"Mean acceleration     : "
            f"{windows_df['mean_acceleration'].mean():.3f}"
        )

        print(
            f"Mean acceleration std : "
            f"{windows_df['acceleration_std'].mean():.3f}"
        )

    print()

    print(
        "[OUTPUT]"
    )

    print(
        OUTPUT_ROOT
    )


if __name__ == "__main__":
    main()