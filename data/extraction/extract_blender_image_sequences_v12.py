from __future__ import annotations

import math
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO


# =============================================================================
# PATH
# =============================================================================

DATASET_ROOT = Path(
    r"D:\임베디드 경진대회\머리 추적\omniverse 생산 공장\DATASET_RAW"
)

ACTIVE_ROOT = (
    DATASET_ROOT
    / "C01_M07_DROWNING"
)

FLOATING_ROOT = (
    DATASET_ROOT
    / "C01_M08_FLOAT_IDLE"
)

MODEL_PATH = Path(
    r"D:\임베디드 경진대회\머리 추적\models\pool_head_best.pt"
)

TRACKER_PATH = Path(
    r"C:\Users\이승희\Desktop\bytetrack_pool.yaml"
)

OUTPUT_ROOT = Path(
    r"D:\임베디드 경진대회\머리 추적\익수자 판별 학습\BLENDER_V12"
)


# =============================================================================
# 중요: Omniverse 이미지 생성 FPS
# =============================================================================

# 실제 Omniverse에서 렌더링한 FPS로 변경
#
# 30fps 생성 → 30.0
# 60fps 생성 → 60.0
#
SOURCE_FPS = 30.0

TARGET_FPS = 10.0

WINDOW_SECONDS = 5.0

WINDOW_SAMPLES = 50

# 1초 간격 sliding window
STRIDE_SAMPLES = 10


# =============================================================================
# DETECTOR
# =============================================================================

IMG_SIZE = 960

CONF = 0.05

IOU = 0.70

MIN_HEAD_SCALE_NORM = 0.010


# =============================================================================
# QUALITY
# =============================================================================

# ACTIVE는 머리 LOST 자체가 중요한 feature라 너무 높게 잡으면 안 됨.
MIN_VISIBLE_RATIO = 0.40


# =============================================================================
# IMAGE
# =============================================================================

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".webp",
}


# =============================================================================
# OUTPUT
# =============================================================================

ACTIVE_SEQUENCE_PATH = (
    OUTPUT_ROOT
    / "BLENDER_ACTIVE_sequences.csv"
)

ACTIVE_WINDOW_PATH = (
    OUTPUT_ROOT
    / "BLENDER_ACTIVE_windows.csv"
)

FLOAT_SEQUENCE_PATH = (
    OUTPUT_ROOT
    / "BLENDER_FLOATING_sequences.csv"
)

FLOAT_WINDOW_PATH = (
    OUTPUT_ROOT
    / "BLENDER_FLOATING_windows.csv"
)

COMBINED_SEQUENCE_PATH = (
    OUTPUT_ROOT
    / "BLENDER_COMBINED_sequences.csv"
)

COMBINED_WINDOW_PATH = (
    OUTPUT_ROOT
    / "BLENDER_COMBINED_windows.csv"
)

VIDEO_SUMMARY_PATH = (
    OUTPUT_ROOT
    / "BLENDER_sequence_summary.csv"
)


# =============================================================================
# NATURAL SORT
# =============================================================================

def natural_key(path: Path):

    text = path.name

    return [
        int(x)
        if x.isdigit()
        else x.lower()

        for x in re.split(
            r"(\d+)",
            text
        )
    ]


# =============================================================================
# IMAGE SEQUENCE FINDER
# =============================================================================

def find_image_sequences(root: Path):

    """
    root 아래의 'images' 폴더를 전부 찾고,
    그 안에 이미지가 있으면 하나의 sequence로 취급.
    """

    sequences = []

    if not root.exists():

        print(
            f"[WARNING] ROOT 없음: {root}"
        )

        return sequences

    for folder in root.rglob("images"):

        if not folder.is_dir():
            continue

        images = sorted(
            [
                p
                for p in folder.iterdir()

                if (
                    p.is_file()
                    and
                    p.suffix.lower()
                    in
                    IMAGE_EXTENSIONS
                )
            ],
            key=natural_key,
        )

        if images:

            sequences.append(
                {
                    "folder":
                        folder,

                    "images":
                        images,
                }
            )

    return sequences


# =============================================================================
# BOX
# =============================================================================

def box_features(
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

    head_scale = math.sqrt(
        bw * bh
    )

    frame_scale = math.sqrt(
        width * height
    )

    return {
        "x1":
            x1,

        "y1":
            y1,

        "x2":
            x2,

        "y2":
            y2,

        "cx":
            cx,

        "cy":
            cy,

        "cx_norm":
            cx / width,

        "cy_norm":
            cy / height,

        "width_norm":
            bw / width,

        "height_norm":
            bh / height,

        "head_scale":
            head_scale,

        "head_scale_norm":
            head_scale / frame_scale,
    }


# =============================================================================
# SINGLE SWIMMER STABLE TRACK
# =============================================================================

class SingleSwimmerTracker:

    """
    합성 장면에서 주요 수영자 1명 기준.

    ByteTrack ID가 끊겨도
    이전 위치 / 머리 크기를 보고 동일 인물로 복구.
    """

    def __init__(self):

        self.initialized = False

        self.last_step = None

        self.last_cx = None
        self.last_cy = None

        self.last_scale = None

        self.prev_step = None

        self.prev_cx = None
        self.prev_cy = None

        self.last_raw_id = None

    def predict(
        self,
        step,
    ):

        if not self.initialized:

            return None

        px = self.last_cx
        py = self.last_cy

        if (
            self.prev_step is not None
            and
            self.last_step is not None
        ):

            dt = (
                self.last_step
                -
                self.prev_step
            )

            if dt > 0:

                vx = (
                    self.last_cx
                    -
                    self.prev_cx
                ) / dt

                vy = (
                    self.last_cy
                    -
                    self.prev_cy
                ) / dt

                future = (
                    step
                    -
                    self.last_step
                )

                # 지나친 extrapolation 방지
                future = min(
                    future,
                    10
                )

                px += (
                    vx
                    *
                    future
                )

                py += (
                    vy
                    *
                    future
                )

        return (
            px,
            py
        )

    def select(
        self,
        detections,
        step,
    ):

        if not detections:

            return None

        # 최초
        if not self.initialized:

            selected = max(
                detections,
                key=lambda x:
                    x["confidence"]
            )

            self.update(
                selected,
                step
            )

            return selected

        # 같은 ByteTrack ID가 있으면 우선
        if (
            self.last_raw_id
            is not None
        ):

            same = [
                d
                for d in detections

                if d[
                    "raw_track_id"
                ]
                ==
                self.last_raw_id
            ]

            if same:

                selected = max(
                    same,
                    key=lambda x:
                        x["confidence"]
                )

                self.update(
                    selected,
                    step
                )

                return selected

        pred_x, pred_y = (
            self.predict(
                step
            )
        )

        gap = max(
            1,
            step
            -
            self.last_step
        )

        allowed_distance = min(
            8.0,
            2.5
            +
            0.15
            *
            gap,
        )

        candidates = []

        for detection in detections:

            scale_ratio = (
                detection[
                    "head_scale"
                ]
                /
                max(
                    1.0,
                    self.last_scale
                )
            )

            if not (
                0.30
                <=
                scale_ratio
                <=
                3.50
            ):

                continue

            mean_scale = max(
                1.0,
                (
                    detection[
                        "head_scale"
                    ]
                    +
                    self.last_scale
                )
                /
                2.0
            )

            distance = (
                math.hypot(
                    detection["cx"]
                    -
                    pred_x,

                    detection["cy"]
                    -
                    pred_y,
                )
                /
                mean_scale
            )

            if (
                distance
                >
                allowed_distance
            ):

                continue

            score = (
                distance
                +
                0.35
                *
                abs(
                    math.log(
                        max(
                            1e-6,
                            scale_ratio
                        )
                    )
                )
                -
                0.2
                *
                detection[
                    "confidence"
                ]
            )

            candidates.append(
                (
                    score,
                    detection
                )
            )

        if candidates:

            candidates.sort(
                key=lambda x:
                    x[0]
            )

            selected = (
                candidates[0][1]
            )

            self.update(
                selected,
                step
            )

            return selected

        # 너무 오래 LOST 되었으면 단일 수영자 가정으로 복구
        if (
            step
            -
            self.last_step
            >=
            20
        ):

            selected = max(
                detections,
                key=lambda x:
                    x["confidence"]
            )

            self.update(
                selected,
                step
            )

            return selected

        return None

    def update(
        self,
        detection,
        step,
    ):

        if self.initialized:

            self.prev_step = (
                self.last_step
            )

            self.prev_cx = (
                self.last_cx
            )

            self.prev_cy = (
                self.last_cy
            )

        self.last_step = step

        self.last_cx = (
            detection[
                "cx"
            ]
        )

        self.last_cy = (
            detection[
                "cy"
            ]
        )

        self.last_scale = (
            detection[
                "head_scale"
            ]
        )

        self.last_raw_id = (
            detection[
                "raw_track_id"
            ]
        )

        self.initialized = True


# =============================================================================
# MOTION
# =============================================================================

def add_motion_features(
    rows
):

    prev = None
    prev_speed = None

    for row in rows:

        row[
            "speed"
        ] = np.nan

        row[
            "vertical_speed"
        ] = np.nan

        row[
            "acceleration"
        ] = np.nan

        if (
            row[
                "detected"
            ]
            !=
            1
        ):

            prev = None
            prev_speed = None

            continue

        if prev is not None:

            mean_scale = max(
                1.0,
                (
                    row[
                        "head_scale"
                    ]
                    +
                    prev[
                        "head_scale"
                    ]
                )
                /
                2.0
            )

            dx = (
                row[
                    "cx"
                ]
                -
                prev[
                    "cx"
                ]
            )

            dy = (
                row[
                    "cy"
                ]
                -
                prev[
                    "cy"
                ]
            )

            speed = (
                math.hypot(
                    dx,
                    dy
                )
                /
                mean_scale
                /
                0.1
            )

            vertical_speed = (
                abs(
                    dy
                )
                /
                mean_scale
                /
                0.1
            )

            row[
                "speed"
            ] = speed

            row[
                "vertical_speed"
            ] = vertical_speed

            if (
                prev_speed
                is not None
            ):

                row[
                    "acceleration"
                ] = (
                    abs(
                        speed
                        -
                        prev_speed
                    )
                    /
                    0.1
                )

            prev_speed = speed

        prev = row

    return rows


# =============================================================================
# WINDOW SUMMARY
# =============================================================================

def calculate_window_summary(
    rows
):

    detected = [
        int(
            r[
                "detected"
            ]
        )
        for r in rows
    ]

    visible_ratio = (
        sum(
            detected
        )
        /
        len(
            detected
        )
    )

    # LOST runs
    lost_runs = []

    run = 0

    for d in detected:

        if d == 0:

            run += 1

        else:

            if run > 0:

                lost_runs.append(
                    run
                )

            run = 0

    if run > 0:

        lost_runs.append(
            run
        )

    lost_sec = [
        x * 0.1
        for x in lost_runs
    ]

    redetection_count = 0

    for i in range(
        1,
        len(
            detected
        )
    ):

        if (
            detected[
                i - 1
            ] == 0
            and
            detected[
                i
            ] == 1
        ):

            redetection_count += 1

    visible = [
        r
        for r in rows

        if r[
            "detected"
        ]
        ==
        1
    ]

    speeds = [
        r["speed"]

        for r in visible

        if not pd.isna(
            r[
                "speed"
            ]
        )
    ]

    vertical_speeds = [
        r[
            "vertical_speed"
        ]

        for r in visible

        if not pd.isna(
            r[
                "vertical_speed"
            ]
        )
    ]

    accelerations = [
        r[
            "acceleration"
        ]

        for r in visible

        if not pd.isna(
            r[
                "acceleration"
            ]
        )
    ]

    scales = [
        r[
            "head_scale_norm"
        ]
        for r in visible
    ]

    xs = [
        r[
            "cx_norm"
        ]
        for r in visible
    ]

    ys = [
        r[
            "cy_norm"
        ]
        for r in visible
    ]

    return {
        "visible_ratio":
            visible_ratio,

        "missing_ratio":
            1.0
            -
            visible_ratio,

        "lost_count":
            len(
                lost_runs
            ),

        "lost_duration_mean":
            float(
                np.mean(
                    lost_sec
                )
            )
            if lost_sec
            else 0.0,

        "lost_duration_max":
            float(
                np.max(
                    lost_sec
                )
            )
            if lost_sec
            else 0.0,

        "lost_duration_std":
            float(
                np.std(
                    lost_sec
                )
            )
            if lost_sec
            else 0.0,

        "redetection_count":
            redetection_count,

        "mean_speed":
            float(
                np.mean(
                    speeds
                )
            )
            if speeds
            else 0.0,

        "max_speed":
            float(
                np.max(
                    speeds
                )
            )
            if speeds
            else 0.0,

        "speed_std":
            float(
                np.std(
                    speeds
                )
            )
            if speeds
            else 0.0,

        "mean_vertical_speed":
            float(
                np.mean(
                    vertical_speeds
                )
            )
            if vertical_speeds
            else 0.0,

        "max_vertical_speed":
            float(
                np.max(
                    vertical_speeds
                )
            )
            if vertical_speeds
            else 0.0,

        "mean_acceleration":
            float(
                np.mean(
                    accelerations
                )
            )
            if accelerations
            else 0.0,

        "max_acceleration":
            float(
                np.max(
                    accelerations
                )
            )
            if accelerations
            else 0.0,

        "acceleration_std":
            float(
                np.std(
                    accelerations
                )
            )
            if accelerations
            else 0.0,

        "mean_head_scale_norm":
            float(
                np.mean(
                    scales
                )
            )
            if scales
            else 0.0,

        "head_scale_std":
            float(
                np.std(
                    scales
                )
            )
            if scales
            else 0.0,

        "net_displacement_norm":
            float(
                math.hypot(
                    xs[-1]
                    -
                    xs[0],

                    ys[-1]
                    -
                    ys[0],
                )
            )
            if len(
                xs
            ) >= 2
            else 0.0,

        "vertical_range_norm":
            float(
                max(
                    ys
                )
                -
                min(
                    ys
                )
            )
            if len(
                ys
            ) >= 2
            else 0.0,
    }


# =============================================================================
# PROCESS IMAGE SEQUENCE
# =============================================================================

def process_sequence(
    model,
    sequence_folder,
    images,
    class_name,
    source_name,
    device,
):

    print(
        f"IMAGES RAW : {len(images)}"
    )

    if not images:

        return [], []

    # -------------------------------------------------------------------------
    # sampling interval
    # -------------------------------------------------------------------------

    sample_interval = max(
        1,
        int(
            round(
                SOURCE_FPS
                /
                TARGET_FPS
            )
        )
    )

    actual_fps = (
        SOURCE_FPS
        /
        sample_interval
    )

    print(
        f"SOURCE FPS={SOURCE_FPS:.3f}"
        f" | every={sample_interval}"
        f" | OUTPUT={actual_fps:.3f}fps"
    )

    sampled_images = (
        images[
            ::sample_interval
        ]
    )

    print(
        f"10FPS SAMPLES : {len(sampled_images)}"
    )

    # -------------------------------------------------------------------------
    # tracker reset
    # -------------------------------------------------------------------------

    model.predictor = None

    stable_tracker = (
        SingleSwimmerTracker()
    )

    samples = []

    for sample_step, image_path in enumerate(
        sampled_images
    ):

        frame = cv2.imread(
            str(
                image_path
            )
        )

        if frame is None:

            print(
                f"[WARNING] image read fail: {image_path}"
            )

            continue

        h, w = (
            frame.shape[
                :2
            ]
        )

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

        detections = []

        boxes = result.boxes

        if (
            boxes is not None
            and
            len(
                boxes
            ) > 0
        ):

            xyxy_values = (
                boxes.xyxy
                .detach()
                .cpu()
                .numpy()
            )

            conf_values = (
                boxes.conf
                .detach()
                .cpu()
                .tolist()
            )

            if (
                boxes.id
                is not None
            ):

                raw_ids = (
                    boxes.id
                    .int()
                    .detach()
                    .cpu()
                    .tolist()
                )

            else:

                raw_ids = [
                    -1
                    for _ in range(
                        len(
                            xyxy_values
                        )
                    )
                ]

            for (
                raw_id,
                xyxy,
                conf,
            ) in zip(
                raw_ids,
                xyxy_values,
                conf_values,
            ):

                f = (
                    box_features(
                        xyxy,
                        w,
                        h,
                    )
                )

                if (
                    f[
                        "head_scale_norm"
                    ]
                    <
                    MIN_HEAD_SCALE_NORM
                ):

                    continue

                f[
                    "raw_track_id"
                ] = int(
                    raw_id
                )

                f[
                    "confidence"
                ] = float(
                    conf
                )

                detections.append(
                    f
                )

        selected = (
            stable_tracker.select(
                detections,
                sample_step,
            )
        )

        if selected is None:

            samples.append(
                {
                    "step_global":
                        sample_step,

                    "image_name":
                        image_path.name,

                    "detected":
                        0,

                    "confidence":
                        0.0,

                    "raw_track_id":
                        -1,

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

            samples.append(
                {
                    "step_global":
                        sample_step,

                    "image_name":
                        image_path.name,

                    "detected":
                        1,

                    "confidence":
                        selected[
                            "confidence"
                        ],

                    "raw_track_id":
                        selected[
                            "raw_track_id"
                        ],

                    "cx":
                        selected[
                            "cx"
                        ],

                    "cy":
                        selected[
                            "cy"
                        ],

                    "cx_norm":
                        selected[
                            "cx_norm"
                        ],

                    "cy_norm":
                        selected[
                            "cy_norm"
                        ],

                    "width_norm":
                        selected[
                            "width_norm"
                        ],

                    "height_norm":
                        selected[
                            "height_norm"
                        ],

                    "head_scale":
                        selected[
                            "head_scale"
                        ],

                    "head_scale_norm":
                        selected[
                            "head_scale_norm"
                        ],
                }
            )

    # =========================================================================
    # WINDOWS
    # =========================================================================

    sequence_rows = []
    window_rows = []

    if (
        len(
            samples
        )
        <
        WINDOW_SAMPLES
    ):

        return (
            sequence_rows,
            window_rows
        )

    relative_sequence = str(
        sequence_folder.relative_to(
            DATASET_ROOT
        )
    )

    starts = range(
        0,

        len(
            samples
        )
        -
        WINDOW_SAMPLES
        +
        1,

        STRIDE_SAMPLES,
    )

    for start in starts:

        window = [
            dict(
                x
            )

            for x in samples[
                start:
                start
                +
                WINDOW_SAMPLES
            ]
        ]

        for local_step, row in enumerate(
            window
        ):

            row[
                "step"
            ] = local_step

            row[
                "time_sec"
            ] = (
                local_step
                *
                0.1
            )

        window = (
            add_motion_features(
                window
            )
        )

        visible_ratio = (
            sum(
                r[
                    "detected"
                ]

                for r in window
            )
            /
            50.0
        )

        if (
            visible_ratio
            <
            MIN_VISIBLE_RATIO
        ):

            continue

        window_id = (
            f"{source_name}"
            f"__{relative_sequence}"
            f"__S{start}"
        )

        for row in window:

            sequence_rows.append(
                {
                    "window_id":
                        window_id,

                    "class":
                        class_name,

                    "source_type":
                        "BLENDER",

                    "source_name":
                        source_name,

                    "video_id":
                        relative_sequence,

                    "step":
                        row[
                            "step"
                        ],

                    "time_sec":
                        row[
                            "time_sec"
                        ],

                    "image_name":
                        row[
                            "image_name"
                        ],

                    "detected":
                        row[
                            "detected"
                        ],

                    "confidence":
                        row[
                            "confidence"
                        ],

                    "raw_track_id":
                        row[
                            "raw_track_id"
                        ],

                    "cx":
                        row[
                            "cx"
                        ],

                    "cy":
                        row[
                            "cy"
                        ],

                    "cx_norm":
                        row[
                            "cx_norm"
                        ],

                    "cy_norm":
                        row[
                            "cy_norm"
                        ],

                    "width_norm":
                        row[
                            "width_norm"
                        ],

                    "height_norm":
                        row[
                            "height_norm"
                        ],

                    "head_scale":
                        row[
                            "head_scale"
                        ],

                    "head_scale_norm":
                        row[
                            "head_scale_norm"
                        ],

                    "speed":
                        row[
                            "speed"
                        ],

                    "vertical_speed":
                        row[
                            "vertical_speed"
                        ],

                    "acceleration":
                        row[
                            "acceleration"
                        ],
                }
            )

        summary = (
            calculate_window_summary(
                window
            )
        )

        window_row = {
            "window_id":
                window_id,

            "class":
                class_name,

            "source_type":
                "BLENDER",

            "source_name":
                source_name,

            "video_id":
                relative_sequence,

            "start_step":
                start,

            "end_step":
                start
                +
                49,

            "samples":
                50,

            "sampling_hz":
                10.0,
        }

        window_row.update(
            summary
        )

        window_rows.append(
            window_row
        )

    return (
        sequence_rows,
        window_rows
    )


# =============================================================================
# PROCESS CLASS
# =============================================================================

def process_class(
    model,
    root,
    class_name,
    source_name,
    device,
):

    sequences = (
        find_image_sequences(
            root
        )
    )

    print()
    print(
        "=" * 80
    )

    print(
        source_name
    )

    print(
        "=" * 80
    )

    print(
        f"IMAGE SEQUENCES : {len(sequences)}"
    )

    all_seq = []
    all_win = []
    summary_rows = []

    for i, item in enumerate(
        sequences,
        start=1
    ):

        folder = (
            item[
                "folder"
            ]
        )

        images = (
            item[
                "images"
            ]
        )

        print()
        print(
            "-" * 80
        )

        print(
            f"[{i}/{len(sequences)}]"
        )

        print(
            folder
        )

        try:

            seq_rows, win_rows = (
                process_sequence(
                    model=model,

                    sequence_folder=folder,

                    images=images,

                    class_name=class_name,

                    source_name=source_name,

                    device=device,
                )
            )

        except Exception as e:

            print(
                "[ERROR]",
                e
            )

            summary_rows.append(
                {
                    "source_name":
                        source_name,

                    "class":
                        class_name,

                    "sequence_folder":
                        str(
                            folder
                        ),

                    "raw_images":
                        len(
                            images
                        ),

                    "windows":
                        0,

                    "sequence_rows":
                        0,

                    "status":
                        "ERROR",

                    "message":
                        str(
                            e
                        ),
                }
            )

            continue

        all_seq.extend(
            seq_rows
        )

        all_win.extend(
            win_rows
        )

        summary_rows.append(
            {
                "source_name":
                    source_name,

                "class":
                    class_name,

                "sequence_folder":
                    str(
                        folder.relative_to(
                            DATASET_ROOT
                        )
                    ),

                "raw_images":
                    len(
                        images
                    ),

                "windows":
                    len(
                        win_rows
                    ),

                "sequence_rows":
                    len(
                        seq_rows
                    ),

                "status":
                    "OK",

                "message":
                    "",
            }
        )

        print(
            f"WINDOWS={len(win_rows)}"
            f" | ROWS={len(seq_rows)}"
        )

    return (
        pd.DataFrame(
            all_seq
        ),

        pd.DataFrame(
            all_win
        ),

        pd.DataFrame(
            summary_rows
        ),
    )


# =============================================================================
# VALIDATE
# =============================================================================

def validate(
    df,
    name,
):

    print()
    print(
        f"[VALIDATION] {name}"
    )

    if df.empty:

        print(
            "EMPTY"
        )

        return

    counts = (
        df.groupby(
            "window_id"
        )
        .size()
    )

    bad = (
        counts[
            counts
            !=
            50
        ]
    )

    print(
        f"WINDOWS : {len(counts)}"
    )

    print(
        f"ROWS    : {len(df)}"
    )

    print(
        f"BAD     : {len(bad)}"
    )

    if (
        len(df)
        ==
        len(counts)
        *
        50
    ):

        print(
            "50-STEP CHECK : PASS"
        )

    else:

        print(
            "50-STEP CHECK : FAIL"
        )


# =============================================================================
# MAIN
# =============================================================================

def main():

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    print(
        "=" * 80
    )

    print(
        "BLENDER IMAGE SEQUENCE V12"
    )

    print(
        "=" * 80
    )

    print(
        f"SOURCE FPS : {SOURCE_FPS}"
    )

    print(
        f"TARGET FPS : {TARGET_FPS}"
    )

    print(
        f"MODEL      : {MODEL_PATH}"
    )

    device = (
        0
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"DEVICE     : {device}"
    )

    model = YOLO(
        str(
            MODEL_PATH
        )
    )

    # =========================================================================
    # ACTIVE
    # =========================================================================

    (
        active_seq,
        active_win,
        active_summary,
    ) = process_class(
        model=model,

        root=ACTIVE_ROOT,

        class_name="ACTIVE_DROWNING",

        source_name="BLENDER_DROWNING",

        device=device,
    )

    # =========================================================================
    # FLOATING
    # =========================================================================

    (
        float_seq,
        float_win,
        float_summary,
    ) = process_class(
        model=model,

        root=FLOATING_ROOT,

        class_name="FLOATING",

        source_name="BLENDER_FLOATING",

        device=device,
    )

    # =========================================================================
    # SAVE
    # =========================================================================

    active_seq.to_csv(
        ACTIVE_SEQUENCE_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    active_win.to_csv(
        ACTIVE_WINDOW_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    float_seq.to_csv(
        FLOAT_SEQUENCE_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    float_win.to_csv(
        FLOAT_WINDOW_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    combined_seq = pd.concat(
        [
            active_seq,
            float_seq,
        ],
        ignore_index=True,
    )

    combined_win = pd.concat(
        [
            active_win,
            float_win,
        ],
        ignore_index=True,
    )

    combined_summary = pd.concat(
        [
            active_summary,
            float_summary,
        ],
        ignore_index=True,
    )

    combined_seq.to_csv(
        COMBINED_SEQUENCE_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    combined_win.to_csv(
        COMBINED_WINDOW_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    combined_summary.to_csv(
        VIDEO_SUMMARY_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # =========================================================================
    # CHECK
    # =========================================================================

    validate(
        active_seq,
        "ACTIVE"
    )

    validate(
        float_seq,
        "FLOATING"
    )

    # =========================================================================
    # FINAL
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
        f"ACTIVE WINDOWS     : {len(active_win)}"
    )

    print(
        f"ACTIVE SEQUENCES   : {len(active_seq)}"
    )

    print(
        f"FLOAT WINDOWS      : {len(float_win)}"
    )

    print(
        f"FLOAT SEQUENCES    : {len(float_seq)}"
    )

    print(
        f"TOTAL WINDOWS      : {len(combined_win)}"
    )

    print(
        f"TOTAL SEQUENCES    : {len(combined_seq)}"
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
