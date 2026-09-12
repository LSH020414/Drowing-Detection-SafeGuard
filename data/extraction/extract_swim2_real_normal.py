from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO


# =============================================================================
# PATH
# =============================================================================

# 스크린샷의 현재 폴더 기준
VIDEO_ROOT = Path(
    r"D:\임베디드 경진대회\머리 추적\Swim2_ONLY\Swim2 ONLY\Swim2 ONLY"
)

MODEL_PATH = Path(
    r"D:\임베디드 경진대회\머리 추적\models\pool_head_best.pt"
)

TRACKER_PATH = Path(
    r"C:\Users\이승희\Desktop\bytetrack_pool.yaml"
)

OUTPUT_ROOT = Path(
    r"D:\임베디드 경진대회\머리 추적\익수자 판별 학습\SWIM2_REAL_NORMAL"
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
# DETECTOR
# =============================================================================

IMG_SIZE = 960
CONF = 0.05
IOU = 0.70


# =============================================================================
# TEMPORAL SETTINGS
# =============================================================================

TARGET_HZ = 10.0

WINDOW_SECONDS = 5.0
WINDOW_SAMPLES = int(WINDOW_SECONDS * TARGET_HZ)  # 50

STRIDE_SECONDS = 1.0
STRIDE_SAMPLES = int(STRIDE_SECONDS * TARGET_HZ)  # 10

MIN_VISIBLE_RATIO = 0.50

# 너무 작은 머리 제거
MIN_HEAD_SCALE_NORM = 0.020

# 한 영상에서 지나치게 많은 중첩 window 방지
MAX_WINDOWS_PER_TRACK = 12


# =============================================================================
# VIDEO
# =============================================================================

def find_videos(root: Path):

    videos = []

    for p in root.rglob("*"):

        if (
            p.is_file()
            and p.suffix.lower() in VIDEO_EXTENSIONS
        ):
            videos.append(p)

    return sorted(videos)


def detect_source_view(path: Path):

    parts = [
        p.lower()
        for p in path.parts
    ]

    for p in parts:

        if "above" in p:
            return "ABOVE"

        if "under" in p:
            return "UNDER"

    # 폴더명이 조금 다를 경우
    joined = "\\".join(parts)

    if "above" in joined:
        return "ABOVE"

    if "under" in joined:
        return "UNDER"

    return "UNKNOWN"


def detect_stroke(path: Path):

    joined = "\\".join(
        p.lower()
        for p in path.parts
    )

    if "backstroke" in joined:
        return "BACKSTROKE"

    if "breaststroke" in joined:
        return "BREASTSTROKE"

    if "butterfly" in joined:
        return "BUTTERFLY"

    if "freestyle" in joined:
        return "FREESTYLE"

    return "UNKNOWN"


# =============================================================================
# GEOMETRY
# =============================================================================

def get_center(box):

    x1, y1, x2, y2 = box

    return (
        (x1 + x2) / 2.0,
        (y1 + y2) / 2.0,
    )


def get_scale(box):

    x1, y1, x2, y2 = box

    w = max(
        1.0,
        x2 - x1
    )

    h = max(
        1.0,
        y2 - y1
    )

    return math.sqrt(
        w * h
    )


# =============================================================================
# SPEED / ACCELERATION
# =============================================================================

def calculate_motion_features(
    window_df: pd.DataFrame
):

    visible = window_df[
        window_df["detected"] == 1
    ]

    speeds = []
    accelerations = []

    previous_speed = None
    previous_speed_time = None

    for i in range(
        1,
        len(window_df)
    ):

        prev = window_df.iloc[
            i - 1
        ]

        cur = window_df.iloc[
            i
        ]

        # LOST를 건너서 연결하지 않음
        if (
            int(prev["detected"]) != 1
            or
            int(cur["detected"]) != 1
        ):

            previous_speed = None
            previous_speed_time = None

            continue

        dt = (
            float(cur["time_sec"])
            -
            float(prev["time_sec"])
        )

        if dt <= 0:
            continue

        dx = (
            float(cur["cx"])
            -
            float(prev["cx"])
        )

        dy = (
            float(cur["cy"])
            -
            float(prev["cy"])
        )

        local_scale = max(
            1.0,
            (
                float(cur["head_scale"])
                +
                float(prev["head_scale"])
            )
            /
            2.0
        )

        distance_heads = (
            math.hypot(
                dx,
                dy
            )
            /
            local_scale
        )

        speed = (
            distance_heads
            /
            dt
        )

        speeds.append(
            speed
        )

        current_time = float(
            cur["time_sec"]
        )

        if (
            previous_speed is not None
            and
            previous_speed_time is not None
        ):

            acceleration_dt = (
                current_time
                -
                previous_speed_time
            )

            if acceleration_dt > 0:

                acceleration = abs(
                    speed
                    -
                    previous_speed
                ) / acceleration_dt

                accelerations.append(
                    acceleration
                )

        previous_speed = speed
        previous_speed_time = current_time

    if speeds:

        mean_speed = float(
            np.mean(
                speeds
            )
        )

        max_speed = float(
            np.max(
                speeds
            )
        )

        speed_std = float(
            np.std(
                speeds
            )
        )

    else:

        mean_speed = 0.0
        max_speed = 0.0
        speed_std = 0.0

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

    # -------------------------------------------------------------------------
    # Basic trajectory features
    # -------------------------------------------------------------------------

    total_distance = 0.0
    vertical_motion = 0.0

    first_visible = None
    last_visible = None

    direction_changes = 0

    previous_vector = None

    for i in range(
        1,
        len(window_df)
    ):

        prev = window_df.iloc[
            i - 1
        ]

        cur = window_df.iloc[
            i
        ]

        if (
            int(prev["detected"]) != 1
            or
            int(cur["detected"]) != 1
        ):
            continue

        dx = (
            float(cur["cx"])
            -
            float(prev["cx"])
        )

        dy = (
            float(cur["cy"])
            -
            float(prev["cy"])
        )

        scale = max(
            1.0,
            (
                float(cur["head_scale"])
                +
                float(prev["head_scale"])
            ) / 2.0
        )

        ndx = dx / scale
        ndy = dy / scale

        distance = math.hypot(
            ndx,
            ndy
        )

        total_distance += distance

        vertical_motion += abs(
            ndy
        )

        if distance >= 0.10:

            vector = np.asarray(
                [
                    ndx,
                    ndy
                ],
                dtype=np.float32
            )

            if previous_vector is not None:

                norm_a = np.linalg.norm(
                    previous_vector
                )

                norm_b = np.linalg.norm(
                    vector
                )

                if (
                    norm_a > 0
                    and
                    norm_b > 0
                ):

                    cosine = float(
                        np.dot(
                            previous_vector,
                            vector
                        )
                        /
                        (
                            norm_a
                            *
                            norm_b
                        )
                    )

                    cosine = np.clip(
                        cosine,
                        -1.0,
                        1.0
                    )

                    angle = math.degrees(
                        math.acos(
                            cosine
                        )
                    )

                    if angle >= 70.0:
                        direction_changes += 1

            previous_vector = vector

    if len(visible) > 0:

        first_visible = visible.iloc[0]
        last_visible = visible.iloc[-1]

    if (
        first_visible is not None
        and
        last_visible is not None
    ):

        median_scale = max(
            1.0,
            float(
                visible[
                    "head_scale"
                ].median()
            )
        )

        net_distance = (
            math.hypot(
                float(
                    last_visible["cx"]
                    -
                    first_visible["cx"]
                ),
                float(
                    last_visible["cy"]
                    -
                    first_visible["cy"]
                ),
            )
            /
            median_scale
        )

    else:

        net_distance = 0.0

    progress_ratio = (
        net_distance
        /
        total_distance
        if total_distance > 1e-8
        else 0.0
    )

    return {
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

        "mean_speed":
            mean_speed,

        "max_speed":
            max_speed,

        "speed_std":
            speed_std,

        "mean_acceleration":
            mean_acceleration,

        "max_acceleration":
            max_acceleration,

        "acceleration_std":
            acceleration_std,
    }


# =============================================================================
# MISSING FEATURES
# =============================================================================

def calculate_missing_features(
    detected_values,
    dt
):

    detected_values = [
        int(v)
        for v in detected_values
    ]

    lost_runs = []
    visible_runs = []

    current_state = detected_values[0]
    current_length = 1

    for value in detected_values[
        1:
    ]:

        if value == current_state:

            current_length += 1

        else:

            duration = (
                current_length
                *
                dt
            )

            if current_state == 0:

                lost_runs.append(
                    duration
                )

            else:

                visible_runs.append(
                    duration
                )

            current_state = value
            current_length = 1

    duration = (
        current_length
        *
        dt
    )

    if current_state == 0:

        lost_runs.append(
            duration
        )

    else:

        visible_runs.append(
            duration
        )

    lost_count = len(
        lost_runs
    )

    lost_mean = (
        float(
            np.mean(
                lost_runs
            )
        )
        if lost_runs
        else 0.0
    )

    lost_std = (
        float(
            np.std(
                lost_runs
            )
        )
        if lost_runs
        else 0.0
    )

    lost_cv = (
        lost_std
        /
        lost_mean
        if lost_mean > 1e-8
        else 0.0
    )

    continuous_missing_time = (
        max(
            lost_runs
        )
        if lost_runs
        else 0.0
    )

    redetection_count = 0

    for i in range(
        1,
        len(detected_values)
    ):

        if (
            detected_values[
                i - 1
            ] == 0
            and
            detected_values[
                i
            ] == 1
        ):

            redetection_count += 1

    return {
        "lost_count":
            lost_count,

        "lost_duration_mean":
            lost_mean,

        "lost_duration_std":
            lost_std,

        "lost_duration_cv":
            lost_cv,

        "continuous_missing_time":
            continuous_missing_time,

        "redetection_count":
            redetection_count,
    }


# =============================================================================
# TRACK PROCESS
# =============================================================================

def process_video(
    model,
    video_path,
    device
):

    cap = cv2.VideoCapture(
        str(
            video_path
        )
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

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    cap.release()

    if fps <= 0:
        fps = 30.0

    sample_interval = max(
        1,
        int(
            round(
                fps
                /
                TARGET_HZ
            )
        )
    )

    actual_sample_hz = (
        fps
        /
        sample_interval
    )

    dt = (
        1.0
        /
        actual_sample_hz
    )

    frame_scale = math.sqrt(
        max(
            1,
            width * height
        )
    )

    # -------------------------------------------------------------------------
    # tracker 결과 수집
    # -------------------------------------------------------------------------

    track_data = defaultdict(
        dict
    )

    results = model.track(
        source=str(
            video_path
        ),

        stream=True,

        persist=False,

        tracker=str(
            TRACKER_PATH
        ),

        imgsz=IMG_SIZE,

        conf=CONF,
        iou=IOU,

        classes=[0],

        device=device,

        verbose=False,
    )

    for frame_index, result in enumerate(
        results
    ):

        if (
            frame_index
            %
            sample_interval
            != 0
        ):
            continue

        sample_index = int(
            round(
                frame_index
                /
                sample_interval
            )
        )

        boxes = result.boxes

        if (
            boxes is None
            or
            boxes.id is None
            or
            len(boxes) == 0
        ):
            continue

        ids = (
            boxes.id
            .int()
            .detach()
            .cpu()
            .tolist()
        )

        xyxy = (
            boxes.xyxy
            .detach()
            .cpu()
            .numpy()
        )

        confs = (
            boxes.conf
            .detach()
            .cpu()
            .tolist()
        )

        for (
            track_id,
            box,
            confidence
        ) in zip(
            ids,
            xyxy,
            confs
        ):

            box = [
                float(v)
                for v in box
            ]

            cx, cy = get_center(
                box
            )

            scale = get_scale(
                box
            )

            track_data[
                int(
                    track_id
                )
            ][
                sample_index
            ] = {
                "frame_idx":
                    frame_index,

                "time_sec":
                    frame_index
                    /
                    fps,

                "detected":
                    1,

                "confidence":
                    float(
                        confidence
                    ),

                "cx":
                    cx,

                "cy":
                    cy,

                "head_scale":
                    scale,

                "head_scale_norm":
                    scale
                    /
                    frame_scale,
            }

    window_rows = []
    sequence_rows = []

    source_view = detect_source_view(
        video_path
    )

    stroke = detect_stroke(
        video_path
    )

    relative_path = str(
        video_path.relative_to(
            VIDEO_ROOT
        )
    )

    # -------------------------------------------------------------------------
    # 각 track
    # -------------------------------------------------------------------------

    for (
        track_id,
        samples
    ) in track_data.items():

        if not samples:
            continue

        min_sample = min(
            samples.keys()
        )

        max_sample = max(
            samples.keys()
        )

        track_length = (
            max_sample
            -
            min_sample
            +
            1
        )

        if (
            track_length
            <
            WINDOW_SAMPLES
        ):
            continue

        starts = list(
            range(
                min_sample,
                max_sample
                -
                WINDOW_SAMPLES
                +
                2,
                STRIDE_SAMPLES,
            )
        )

        if len(starts) > MAX_WINDOWS_PER_TRACK:

            # 전체 구간에서 균등하게 뽑음
            indices = np.linspace(
                0,
                len(starts) - 1,
                MAX_WINDOWS_PER_TRACK
            ).round().astype(int)

            starts = [
                starts[i]
                for i in indices
            ]

        for start_sample in starts:

            records = []

            end_sample = (
                start_sample
                +
                WINDOW_SAMPLES
            )

            for sample_index in range(
                start_sample,
                end_sample
            ):

                if sample_index in samples:

                    row = samples[
                        sample_index
                    ].copy()

                else:

                    row = {
                        "frame_idx":
                            int(
                                round(
                                    sample_index
                                    *
                                    sample_interval
                                )
                            ),

                        "time_sec":
                            sample_index
                            /
                            actual_sample_hz,

                        "detected":
                            0,

                        "confidence":
                            0.0,

                        "cx":
                            np.nan,

                        "cy":
                            np.nan,

                        "head_scale":
                            np.nan,

                        "head_scale_norm":
                            np.nan,
                    }

                records.append(
                    row
                )

            window_df = pd.DataFrame(
                records
            )

            visible_ratio = float(
                window_df[
                    "detected"
                ].mean()
            )

            disappear_ratio = (
                1.0
                -
                visible_ratio
            )

            visible = window_df[
                window_df[
                    "detected"
                ] == 1
            ]

            if len(visible) == 0:

                continue

            median_head_scale_norm = float(
                visible[
                    "head_scale_norm"
                ].median()
            )

            # 실제 정상 학습 데이터 품질 필터
            quality = "USABLE"
            reject_reason = []

            if (
                visible_ratio
                <
                MIN_VISIBLE_RATIO
            ):

                quality = "REJECT"

                reject_reason.append(
                    "low_visibility"
                )

            if (
                median_head_scale_norm
                <
                MIN_HEAD_SCALE_NORM
            ):

                quality = "REJECT"

                reject_reason.append(
                    "small_head"
                )

            movement = (
                calculate_motion_features(
                    window_df
                )
            )

            missing = (
                calculate_missing_features(
                    window_df[
                        "detected"
                    ].tolist(),
                    dt
                )
            )

            median_head_scale = float(
                visible[
                    "head_scale"
                ].median()
            )

            mean_head_scale = float(
                visible[
                    "head_scale"
                ].mean()
            )

            std_head_scale = float(
                visible[
                    "head_scale"
                ].std(
                    ddof=0
                )
            )

            # -----------------------------------------------------------------
            # unique window ID
            # -----------------------------------------------------------------

            safe_video = (
                relative_path
                .replace("\\", "__")
                .replace("/", "__")
            )

            window_id = (
                f"{safe_video}"
                f"__T{track_id}"
                f"__S{start_sample}"
            )

            window_rows.append(
                {
                    "window_id":
                        window_id,

                    "video_id":
                        relative_path,

                    "source_view":
                        source_view,

                    "stroke":
                        stroke,

                    "track_id":
                        track_id,

                    "label":
                        "NORMAL_SWIMMING",

                    "quality":
                        quality,

                    "reject_reason":
                        "+".join(
                            reject_reason
                        ),

                    "visible_ratio":
                        visible_ratio,

                    "disappear_ratio":
                        disappear_ratio,

                    "median_head_scale":
                        median_head_scale,

                    "median_head_scale_norm":
                        median_head_scale_norm,

                    "mean_head_scale":
                        mean_head_scale,

                    "std_head_scale":
                        std_head_scale,

                    **movement,

                    **missing,
                }
            )

            # sequence는 REJECT 포함해서 일단 저장
            for (
                sequence_index,
                row
            ) in window_df.iterrows():

                sequence_rows.append(
                    {
                        "window_id":
                            window_id,

                        "video_id":
                            relative_path,

                        "source_view":
                            source_view,

                        "stroke":
                            stroke,

                        "track_id":
                            track_id,

                        "label":
                            "NORMAL_SWIMMING",

                        "quality":
                            quality,

                        "step":
                            sequence_index,

                        "time_sec":
                            float(
                                sequence_index
                                *
                                dt
                            ),

                        "frame_idx":
                            row[
                                "frame_idx"
                            ],

                        "detected":
                            int(
                                row[
                                    "detected"
                                ]
                            ),

                        "confidence":
                            row[
                                "confidence"
                            ],

                        "cx":
                            row[
                                "cx"
                            ],

                        "cy":
                            row[
                                "cy"
                            ],

                        "head_scale":
                            row[
                                "head_scale"
                            ],

                        "head_scale_norm":
                            row[
                                "head_scale_norm"
                            ],
                    }
                )

    return (
        window_rows,
        sequence_rows,
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    if not VIDEO_ROOT.exists():

        raise FileNotFoundError(
            f"VIDEO_ROOT 없음:\n{VIDEO_ROOT}"
        )

    videos = find_videos(
        VIDEO_ROOT
    )

    print(
        "=" * 80
    )

    print(
        "SWIM2 REAL NORMAL EXTRACTION"
    )

    print(
        "=" * 80
    )

    print(
        f"[ROOT] {VIDEO_ROOT}"
    )

    print(
        f"[VIDEOS] {len(videos)}"
    )

    above_count = sum(
        detect_source_view(v)
        ==
        "ABOVE"
        for v in videos
    )

    under_count = sum(
        detect_source_view(v)
        ==
        "UNDER"
        for v in videos
    )

    print(
        f"[ABOVE] {above_count}"
    )

    print(
        f"[UNDER] {under_count}"
    )

    if len(videos) == 0:

        print(
            "영상이 없습니다."
        )

        return

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            MODEL_PATH
        )

    if not TRACKER_PATH.exists():

        raise FileNotFoundError(
            TRACKER_PATH
        )

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

    all_windows = []
    all_sequences = []

    summary_rows = []

    # =========================================================================
    # VIDEO LOOP
    # =========================================================================

    for (
        index,
        video
    ) in enumerate(
        videos,
        start=1
    ):

        print()
        print(
            "=" * 80
        )

        print(
            f"[{index}/{len(videos)}]"
        )

        print(
            video
        )

        try:

            (
                windows,
                sequences
            ) = process_video(
                model=model,
                video_path=video,
                device=device
            )

        except Exception as error:

            print(
                f"[ERROR] {error}"
            )

            continue

        all_windows.extend(
            windows
        )

        all_sequences.extend(
            sequences
        )

        usable_count = sum(
            row[
                "quality"
            ] == "USABLE"
            for row in windows
        )

        reject_count = (
            len(windows)
            -
            usable_count
        )

        source_view = (
            detect_source_view(
                video
            )
        )

        stroke = (
            detect_stroke(
                video
            )
        )

        summary_rows.append(
            {
                "video_id":
                    str(
                        video.relative_to(
                            VIDEO_ROOT
                        )
                    ),

                "source_view":
                    source_view,

                "stroke":
                    stroke,

                "windows":
                    len(
                        windows
                    ),

                "usable":
                    usable_count,

                "reject":
                    reject_count,
            }
        )

        print(
            f"WINDOWS={len(windows)}"
            f" | USABLE={usable_count}"
            f" | REJECT={reject_count}"
        )

        # 중간 저장
        if index % 10 == 0:

            pd.DataFrame(
                all_windows
            ).to_csv(
                OUTPUT_ROOT
                /
                "SWIM2_REAL_ALL_windows.csv",
                index=False,
                encoding="utf-8-sig",
            )

            pd.DataFrame(
                all_sequences
            ).to_csv(
                OUTPUT_ROOT
                /
                "SWIM2_REAL_ALL_sequences.csv",
                index=False,
                encoding="utf-8-sig",
            )

    # =========================================================================
    # SAVE
    # =========================================================================

    windows_df = pd.DataFrame(
        all_windows
    )

    sequences_df = pd.DataFrame(
        all_sequences
    )

    summary_df = pd.DataFrame(
        summary_rows
    )

    all_windows_path = (
        OUTPUT_ROOT
        /
        "SWIM2_REAL_ALL_windows.csv"
    )

    usable_windows_path = (
        OUTPUT_ROOT
        /
        "SWIM2_REAL_USABLE_windows.csv"
    )

    above_windows_path = (
        OUTPUT_ROOT
        /
        "SWIM2_ABOVE_USABLE_windows.csv"
    )

    under_windows_path = (
        OUTPUT_ROOT
        /
        "SWIM2_UNDER_USABLE_windows.csv"
    )

    all_sequences_path = (
        OUTPUT_ROOT
        /
        "SWIM2_REAL_ALL_sequences.csv"
    )

    usable_sequences_path = (
        OUTPUT_ROOT
        /
        "SWIM2_REAL_USABLE_sequences.csv"
    )

    above_sequences_path = (
        OUTPUT_ROOT
        /
        "SWIM2_ABOVE_USABLE_sequences.csv"
    )

    under_sequences_path = (
        OUTPUT_ROOT
        /
        "SWIM2_UNDER_USABLE_sequences.csv"
    )

    summary_path = (
        OUTPUT_ROOT
        /
        "SWIM2_REAL_video_summary.csv"
    )

    windows_df.to_csv(
        all_windows_path,
        index=False,
        encoding="utf-8-sig"
    )

    sequences_df.to_csv(
        all_sequences_path,
        index=False,
        encoding="utf-8-sig"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig"
    )

    # =========================================================================
    # USABLE
    # =========================================================================

    if not windows_df.empty:

        usable_windows = (
            windows_df[
                windows_df[
                    "quality"
                ] == "USABLE"
            ]
            .copy()
        )

        usable_ids = set(
            usable_windows[
                "window_id"
            ].astype(str)
        )

        usable_sequences = (
            sequences_df[
                sequences_df[
                    "window_id"
                ]
                .astype(str)
                .isin(
                    usable_ids
                )
            ]
            .copy()
        )

        above_windows = (
            usable_windows[
                usable_windows[
                    "source_view"
                ] == "ABOVE"
            ]
            .copy()
        )

        under_windows = (
            usable_windows[
                usable_windows[
                    "source_view"
                ] == "UNDER"
            ]
            .copy()
        )

        above_ids = set(
            above_windows[
                "window_id"
            ].astype(str)
        )

        under_ids = set(
            under_windows[
                "window_id"
            ].astype(str)
        )

        above_sequences = (
            usable_sequences[
                usable_sequences[
                    "window_id"
                ]
                .astype(str)
                .isin(
                    above_ids
                )
            ]
            .copy()
        )

        under_sequences = (
            usable_sequences[
                usable_sequences[
                    "window_id"
                ]
                .astype(str)
                .isin(
                    under_ids
                )
            ]
            .copy()
        )

    else:

        usable_windows = pd.DataFrame()
        usable_sequences = pd.DataFrame()

        above_windows = pd.DataFrame()
        under_windows = pd.DataFrame()

        above_sequences = pd.DataFrame()
        under_sequences = pd.DataFrame()

    usable_windows.to_csv(
        usable_windows_path,
        index=False,
        encoding="utf-8-sig"
    )

    usable_sequences.to_csv(
        usable_sequences_path,
        index=False,
        encoding="utf-8-sig"
    )

    above_windows.to_csv(
        above_windows_path,
        index=False,
        encoding="utf-8-sig"
    )

    above_sequences.to_csv(
        above_sequences_path,
        index=False,
        encoding="utf-8-sig"
    )

    under_windows.to_csv(
        under_windows_path,
        index=False,
        encoding="utf-8-sig"
    )

    under_sequences.to_csv(
        under_sequences_path,
        index=False,
        encoding="utf-8-sig"
    )

    # =========================================================================
    # REPORT
    # =========================================================================

    print()
    print(
        "=" * 80
    )

    print(
        "FINISHED"
    )

    print(
        "=" * 80
    )

    print(
        f"VIDEOS TOTAL       : {len(videos)}"
    )

    print(
        f"WINDOWS TOTAL      : {len(windows_df)}"
    )

    print(
        f"USABLE TOTAL       : {len(usable_windows)}"
    )

    print(
        f"ABOVE USABLE       : {len(above_windows)}"
    )

    print(
        f"UNDER USABLE       : {len(under_windows)}"
    )

    print(
        f"SEQUENCE ROWS      : {len(usable_sequences)}"
    )

    if not usable_windows.empty:

        print()
        print(
            "[STROKE DISTRIBUTION]"
        )

        print(
            usable_windows[
                [
                    "source_view",
                    "stroke"
                ]
            ]
            .value_counts()
            .to_string()
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
