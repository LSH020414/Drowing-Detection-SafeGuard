from __future__ import annotations

import math
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO


# =============================================================================
# PATHS
# =============================================================================

VIDEO_ROOT = Path(
    r"D:\임베디드 경진대회\머리 추적\CVAT용 영상\converted_10fps"
)

MANIFEST_PATH = Path(
    r"D:\임베디드 경진대회\머리 추적\익수자 판별 학습\FLOATING_REVIEW\FLOATING_REVIEW_manifest.csv"
)

MODEL_PATH = Path(
    r"D:\임베디드 경진대회\머리 추적\models\pool_head_best.pt"
)

TRACKER_PATH = Path(
    r"C:\Users\이승희\Desktop\bytetrack_pool.yaml"
)

OUTPUT_ROOT = Path(
    r"D:\임베디드 경진대회\머리 추적\익수자 판별 학습\FLOATING_FINAL"
)

SEQUENCE_CSV = (
    OUTPUT_ROOT
    / "FLOATING_FINAL_sequences.csv"
)

WINDOW_CSV = (
    OUTPUT_ROOT
    / "FLOATING_FINAL_USABLE_windows.csv"
)

FAILED_CSV = (
    OUTPUT_ROOT
    / "FLOATING_FINAL_failed.csv"
)


# =============================================================================
# SETTINGS
# =============================================================================

TARGET_HZ = 10.0

WINDOW_SAMPLES = 50

DT = 0.1

IMG_SIZE = 960

CONF = 0.05

IOU = 0.70


# manifest에서 이미 정상적으로 review clip 생성된 것만
REQUIRE_STATUS_OK = True

# 너무 심하게 track이 재현되지 않은 window 제외
# 0.50 = 50개 중 25개 이상 해당 track 검출
MIN_VISIBLE_RATIO = 0.50

# true이면 manual_label 사용
# 아직 manual_label을 안 넣었으므로 False
USE_MANUAL_LABEL = False


# =============================================================================
# HELPERS
# =============================================================================

def normalize_path_text(value):
    """
    CSV에서 _가 escape되어 보이는 문제 등과 무관하게
    실제 문자열 그대로 Path 처리.
    """
    return str(value)


def resolve_video_path(relative_path):

    relative_path = normalize_path_text(
        relative_path
    )

    path = (
        VIDEO_ROOT
        /
        Path(relative_path)
    )

    if path.exists():
        return path

    # 혹시 slash 혼용
    path2 = (
        VIDEO_ROOT
        /
        Path(
            relative_path.replace(
                "\\",
                "/"
            )
        )
    )

    if path2.exists():
        return path2

    return None


def box_to_features(
    xyxy,
    width,
    height,
):

    x1, y1, x2, y2 = [
        float(v)
        for v in xyxy
    ]

    bw = max(
        1.0,
        x2 - x1
    )

    bh = max(
        1.0,
        y2 - y1
    )

    cx = (
        x1 + x2
    ) / 2.0

    cy = (
        y1 + y2
    ) / 2.0

    scale = math.sqrt(
        bw * bh
    )

    frame_diag = math.sqrt(
        width * width
        +
        height * height
    )

    frame_area_scale = math.sqrt(
        width * height
    )

    return {
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,

        "cx": cx,
        "cy": cy,

        "cx_norm":
            cx / width,

        "cy_norm":
            cy / height,

        "width_norm":
            bw / width,

        "height_norm":
            bh / height,

        "head_scale":
            scale,

        "head_scale_norm":
            scale
            /
            frame_area_scale,

        "frame_diag":
            frame_diag,
    }


# =============================================================================
# FEATURE CALCULATION
# =============================================================================

def calculate_sequence_motion(
    rows,
):

    """
    detected가 연속된 두 timestep 사이에서만
    speed / acceleration 계산.

    LOST 구간을 가로질러 연결하지 않는다.
    """

    prev = None
    prev_speed = None

    for row in rows:

        row["speed"] = np.nan
        row["vertical_speed"] = np.nan
        row["acceleration"] = np.nan

        if row["detected"] != 1:

            prev = None
            prev_speed = None
            continue

        if prev is not None:

            mean_scale = (
                row["head_scale"]
                +
                prev["head_scale"]
            ) / 2.0

            mean_scale = max(
                1.0,
                mean_scale
            )

            dx = (
                row["cx"]
                -
                prev["cx"]
            )

            dy = (
                row["cy"]
                -
                prev["cy"]
            )

            distance = math.hypot(
                dx,
                dy
            )

            speed = (
                distance
                /
                mean_scale
                /
                DT
            )

            vertical_speed = (
                abs(dy)
                /
                mean_scale
                /
                DT
            )

            row["speed"] = speed
            row["vertical_speed"] = (
                vertical_speed
            )

            if prev_speed is not None:

                row["acceleration"] = (
                    abs(
                        speed
                        -
                        prev_speed
                    )
                    /
                    DT
                )

            prev_speed = speed

        prev = row

    return rows


def missing_features(
    detected_list,
):

    lost_runs = []

    current = 0

    for value in detected_list:

        if value == 0:

            current += 1

        else:

            if current > 0:
                lost_runs.append(
                    current
                )

            current = 0

    if current > 0:

        lost_runs.append(
            current
        )

    lost_durations = [
        run * DT
        for run in lost_runs
    ]

    total_missing = (
        detected_list.count(0)
    )

    visibility = (
        detected_list.count(1)
        /
        len(detected_list)
    )

    if lost_durations:

        mean_lost = float(
            np.mean(
                lost_durations
            )
        )

        max_lost = float(
            np.max(
                lost_durations
            )
        )

        std_lost = float(
            np.std(
                lost_durations
            )
        )

    else:

        mean_lost = 0.0
        max_lost = 0.0
        std_lost = 0.0

    return {
        "visible_ratio":
            visibility,

        "missing_ratio":
            total_missing
            /
            len(detected_list),

        "lost_count":
            len(lost_runs),

        "lost_duration_mean":
            mean_lost,

        "lost_duration_max":
            max_lost,

        "lost_duration_std":
            std_lost,

        "redetection_count":
            max(
                0,
                len(lost_runs)
                -
                (
                    1
                    if detected_list[-1] == 0
                    else 0
                )
            ),
    }


def aggregate_window(
    rows,
):

    detected = [
        int(r["detected"])
        for r in rows
    ]

    missing = missing_features(
        detected
    )

    valid = [
        r
        for r in rows
        if r["detected"] == 1
    ]

    speeds = [
        r["speed"]
        for r in valid
        if not pd.isna(
            r["speed"]
        )
    ]

    vertical_speeds = [
        r["vertical_speed"]
        for r in valid
        if not pd.isna(
            r["vertical_speed"]
        )
    ]

    accelerations = [
        r["acceleration"]
        for r in valid
        if not pd.isna(
            r["acceleration"]
        )
    ]

    scales = [
        r["head_scale_norm"]
        for r in valid
    ]

    xs = [
        r["cx_norm"]
        for r in valid
    ]

    ys = [
        r["cy_norm"]
        for r in valid
    ]

    result = dict(
        missing
    )

    if speeds:

        result["mean_speed"] = float(
            np.mean(speeds)
        )

        result["max_speed"] = float(
            np.max(speeds)
        )

        result["speed_std"] = float(
            np.std(speeds)
        )

    else:

        result["mean_speed"] = 0.0
        result["max_speed"] = 0.0
        result["speed_std"] = 0.0

    if vertical_speeds:

        result[
            "mean_vertical_speed"
        ] = float(
            np.mean(
                vertical_speeds
            )
        )

        result[
            "max_vertical_speed"
        ] = float(
            np.max(
                vertical_speeds
            )
        )

    else:

        result[
            "mean_vertical_speed"
        ] = 0.0

        result[
            "max_vertical_speed"
        ] = 0.0

    if accelerations:

        result[
            "mean_acceleration"
        ] = float(
            np.mean(
                accelerations
            )
        )

        result[
            "max_acceleration"
        ] = float(
            np.max(
                accelerations
            )
        )

        result[
            "acceleration_std"
        ] = float(
            np.std(
                accelerations
            )
        )

    else:

        result[
            "mean_acceleration"
        ] = 0.0

        result[
            "max_acceleration"
        ] = 0.0

        result[
            "acceleration_std"
        ] = 0.0

    if scales:

        result[
            "mean_head_scale_norm"
        ] = float(
            np.mean(scales)
        )

        result[
            "head_scale_std"
        ] = float(
            np.std(scales)
        )

    else:

        result[
            "mean_head_scale_norm"
        ] = 0.0

        result[
            "head_scale_std"
        ] = 0.0

    if len(xs) >= 2:

        result[
            "net_displacement_norm"
        ] = float(
            math.hypot(
                xs[-1] - xs[0],
                ys[-1] - ys[0],
            )
        )

        result[
            "vertical_range_norm"
        ] = float(
            max(ys)
            -
            min(ys)
        )

    else:

        result[
            "net_displacement_norm"
        ] = 0.0

        result[
            "vertical_range_norm"
        ] = 0.0

    return result


# =============================================================================
# TRACK VIDEO
# =============================================================================

def track_video(
    model,
    video_path,
    device,
):

    """
    한 원본 영상 전체를 딱 한 번 tracking.

    return:
        track_data[track_id][frame_index]
    """

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

    track_data = defaultdict(
        dict
    )

    frame_index = 0

    model.predictor = None

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        result = model.track(
            source=frame,

            persist=True,

            tracker=str(
                TRACKER_PATH
            ),

            imgsz=IMG_SIZE,

            conf=CONF,

            iou=IOU,

            classes=[0],

            device=device,

            verbose=False,
        )[0]

        boxes = result.boxes

        if (
            boxes is not None
            and
            len(boxes) > 0
            and
            boxes.id is not None
        ):

            ids = (
                boxes.id
                .detach()
                .cpu()
                .numpy()
                .astype(int)
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
                .numpy()
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

                features = box_to_features(
                    box,
                    width,
                    height,
                )

                features[
                    "confidence"
                ] = float(
                    confidence
                )

                track_data[
                    int(track_id)
                ][
                    frame_index
                ] = features

        frame_index += 1

    cap.release()

    return (
        track_data,
        fps,
        width,
        height,
        frame_index,
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    print(
        "=" * 80
    )

    print(
        "FLOATING FINAL SEQUENCE EXTRACTION"
    )

    print(
        "=" * 80
    )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    if not MANIFEST_PATH.exists():

        raise FileNotFoundError(
            MANIFEST_PATH
        )

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            MODEL_PATH
        )

    manifest = pd.read_csv(
        MANIFEST_PATH
    )

    print(
        f"MANIFEST ROWS : {len(manifest)}"
    )

    # -------------------------------------------------------------------------
    # 후보 선택
    # -------------------------------------------------------------------------

    selected = manifest.copy()

    if REQUIRE_STATUS_OK:

        selected = selected[
            selected["status"]
            .astype(str)
            .str.upper()
            ==
            "OK"
        ]

    if USE_MANUAL_LABEL:

        selected = selected[
            selected[
                "manual_label"
            ]
            .astype(str)
            .str.upper()
            ==
            "KEEP"
        ]

    selected = (
        selected
        .sort_values(
            [
                "relative_path",
                "clip_start_frame",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    print(
        f"SELECTED      : {len(selected)}"
    )

    device = (
        0
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"DEVICE        : {device}"
    )

    model = YOLO(
        str(
            MODEL_PATH
        )
    )

    all_sequences = []

    all_windows = []

    failed = []

    # -------------------------------------------------------------------------
    # 같은 원본 영상의 후보를 그룹화
    # -------------------------------------------------------------------------

    grouped = selected.groupby(
        "relative_path",
        sort=False,
    )

    total_videos = len(
        grouped
    )

    window_counter = 0

    # =========================================================================
    # VIDEO
    # =========================================================================

    for video_number, (
        relative_path,
        rows
    ) in enumerate(
        grouped,
        start=1
    ):

        print()
        print(
            "=" * 80
        )

        print(
            f"[VIDEO {video_number}/{total_videos}]"
        )

        print(
            relative_path
        )

        video_path = resolve_video_path(
            relative_path
        )

        if video_path is None:

            print(
                "[FAIL] 원본 영상 없음"
            )

            for _, row in rows.iterrows():

                failed.append(
                    {
                        "review_index":
                            row[
                                "review_index"
                            ],

                        "relative_path":
                            relative_path,

                        "reason":
                            "VIDEO_NOT_FOUND",
                    }
                )

            continue

        try:

            (
                track_data,
                fps,
                width,
                height,
                video_frames,
            ) = track_video(
                model,
                video_path,
                device,
            )

        except Exception as e:

            print(
                "[FAIL]",
                e
            )

            continue

        print(
            f"FPS={fps:.3f}"
            f" | frames={video_frames}"
            f" | tracks={len(track_data)}"
        )

        # =====================================================================
        # WINDOW
        # =====================================================================

        for _, candidate in rows.iterrows():

            review_index = int(
                candidate[
                    "review_index"
                ]
            )

            track_id = int(
                candidate[
                    "track_id"
                ]
            )

            start_frame = int(
                candidate[
                    "clip_start_frame"
                ]
            )

            # manifest end 값에 의존하지 않고
            # 정확하게 50 samples 생성
            target_frames = [
                start_frame + step
                for step in range(
                    WINDOW_SAMPLES
                )
            ]

            if (
                target_frames[-1]
                >=
                video_frames
            ):

                failed.append(
                    {
                        "review_index":
                            review_index,

                        "relative_path":
                            relative_path,

                        "track_id":
                            track_id,

                        "reason":
                            "OUT_OF_VIDEO",
                    }
                )

                continue

            sequence_rows = []

            this_track = (
                track_data.get(
                    track_id,
                    {}
                )
            )

            for step, frame_idx in enumerate(
                target_frames
            ):

                observation = (
                    this_track.get(
                        frame_idx
                    )
                )

                base = {
                    "window_id":
                        window_counter,

                    "class":
                        "FLOATING",

                    "review_index":
                        review_index,

                    "video_id":
                        candidate[
                            "video_id"
                        ],

                    "relative_path":
                        relative_path,

                    "track_id":
                        track_id,

                    "step":
                        step,

                    "time_sec":
                        step * DT,

                    "source_frame":
                        frame_idx,
                }

                if observation is None:

                    base.update(
                        {
                            "detected":
                                0,

                            "confidence":
                                0.0,

                            "x1":
                                np.nan,

                            "y1":
                                np.nan,

                            "x2":
                                np.nan,

                            "y2":
                                np.nan,

                            "cx":
                                np.nan,

                            "cy":
                                np.nan,

                            "cx_norm":
                                np.nan,

                            "cy_norm":
                                np.nan,

                            "width_norm":
                                np.nan,

                            "height_norm":
                                np.nan,

                            "head_scale":
                                np.nan,

                            "head_scale_norm":
                                np.nan,
                        }
                    )

                else:

                    base.update(
                        {
                            "detected":
                                1,

                            "confidence":
                                observation[
                                    "confidence"
                                ],

                            "x1":
                                observation[
                                    "x1"
                                ],

                            "y1":
                                observation[
                                    "y1"
                                ],

                            "x2":
                                observation[
                                    "x2"
                                ],

                            "y2":
                                observation[
                                    "y2"
                                ],

                            "cx":
                                observation[
                                    "cx"
                                ],

                            "cy":
                                observation[
                                    "cy"
                                ],

                            "cx_norm":
                                observation[
                                    "cx_norm"
                                ],

                            "cy_norm":
                                observation[
                                    "cy_norm"
                                ],

                            "width_norm":
                                observation[
                                    "width_norm"
                                ],

                            "height_norm":
                                observation[
                                    "height_norm"
                                ],

                            "head_scale":
                                observation[
                                    "head_scale"
                                ],

                            "head_scale_norm":
                                observation[
                                    "head_scale_norm"
                                ],
                        }
                    )

                sequence_rows.append(
                    base
                )

            # -----------------------------------------------------------------
            # 움직임 계산
            # -----------------------------------------------------------------

            sequence_rows = (
                calculate_sequence_motion(
                    sequence_rows
                )
            )

            visible_count = sum(
                r["detected"]
                for r in sequence_rows
            )

            visible_ratio = (
                visible_count
                /
                WINDOW_SAMPLES
            )

            if (
                visible_ratio
                <
                MIN_VISIBLE_RATIO
            ):

                failed.append(
                    {
                        "review_index":
                            review_index,

                        "relative_path":
                            relative_path,

                        "track_id":
                            track_id,

                        "visible_ratio":
                            visible_ratio,

                        "reason":
                            "LOW_REPRODUCED_VISIBILITY",
                    }
                )

                continue

            summary = (
                aggregate_window(
                    sequence_rows
                )
            )

            window_row = {
                "window_id":
                    window_counter,

                "class":
                    "FLOATING",

                "review_index":
                    review_index,

                "video_id":
                    candidate[
                        "video_id"
                    ],

                "relative_path":
                    relative_path,

                "track_id":
                    track_id,

                "start_frame":
                    start_frame,

                "end_frame":
                    start_frame
                    +
                    WINDOW_SAMPLES
                    -
                    1,

                "samples":
                    WINDOW_SAMPLES,

                "sampling_hz":
                    TARGET_HZ,
            }

            window_row.update(
                summary
            )

            all_sequences.extend(
                sequence_rows
            )

            all_windows.append(
                window_row
            )

            window_counter += 1

        print(
            f"CURRENT WINDOWS : {window_counter}"
        )

    # =========================================================================
    # SAVE
    # =========================================================================

    sequence_df = pd.DataFrame(
        all_sequences
    )

    window_df = pd.DataFrame(
        all_windows
    )

    failed_df = pd.DataFrame(
        failed
    )

    sequence_df.to_csv(
        SEQUENCE_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    window_df.to_csv(
        WINDOW_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    failed_df.to_csv(
        FAILED_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    # =========================================================================
    # VALIDATION
    # =========================================================================

    if len(sequence_df) > 0:

        counts = (
            sequence_df
            .groupby(
                "window_id"
            )
            .size()
        )

        bad_counts = counts[
            counts
            !=
            WINDOW_SAMPLES
        ]

        time_bad = 0

        for (
            window_id,
            group
        ) in sequence_df.groupby(
            "window_id"
        ):

            times = (
                group
                .sort_values(
                    "step"
                )[
                    "time_sec"
                ]
                .to_numpy()
            )

            expected = np.arange(
                WINDOW_SAMPLES
            ) * DT

            if not np.allclose(
                times,
                expected,
                atol=1e-8,
            ):

                time_bad += 1

    else:

        bad_counts = []
        time_bad = 0

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
        f"FINAL WINDOWS     : {len(window_df)}"
    )

    print(
        f"SEQUENCE ROWS     : {len(sequence_df)}"
    )

    print(
        f"FAILED WINDOWS    : {len(failed_df)}"
    )

    print(
        f"BAD 50-STEP WIN   : {len(bad_counts)}"
    )

    print(
        f"BAD TIME WINDOWS  : {time_bad}"
    )

    print()
    print(
        "[EXPECTED]"
    )

    print(
        "SEQUENCE ROWS = FINAL WINDOWS x 50"
    )

    print()
    print(
        "[OUTPUT]"
    )

    print(
        SEQUENCE_CSV
    )

    print(
        WINDOW_CSV
    )

    print(
        FAILED_CSV
    )


if __name__ == "__main__":
    main()