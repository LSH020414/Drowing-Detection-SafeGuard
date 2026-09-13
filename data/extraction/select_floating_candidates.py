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

VIDEO_ROOT = Path(
    r"D:\임베디드 경진대회\머리 추적\CVAT용 영상\converted_10fps"
)

MODEL_PATH = Path(
    r"D:\임베디드 경진대회\머리 추적\models\pool_head_best.pt"
)

OUTPUT_ROOT = Path(
    r"D:\임베디드 경진대회\머리 추적\익수자 판별 학습\FLOATING_SELECTION"
)

TRACKER_PATH = Path(
    r"C:\Users\이승희\Desktop\bytetrack_pool.yaml"
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
# FLOATING SELECTION SETTINGS
# =============================================================================

# 최소 추적 지속시간
MIN_TRACK_SECONDS = 5.0

# 최소 검출 비율
MIN_VISIBLE_RATIO = 0.60

# -------------------------------------------------------------------------
# 머리 크기
#
# sqrt(head_box_area) / sqrt(frame_area)
#
# 1280x720 영상이면 대략 head scale 24px 정도부터 통과하는 수준.
# -------------------------------------------------------------------------

MIN_HEAD_SCALE_NORM = 0.025


# -------------------------------------------------------------------------
# 움직임
#
# 단위 = head-size / second
#
# 부영 후보를 비교적 보수적으로 선별한다.
# -------------------------------------------------------------------------

MAX_MEAN_SPEED = 1.20

MAX_SPEED_STD = 1.50

MAX_VERTICAL_SPEED = 0.80

MAX_TOTAL_SPEED_P95 = 3.50


# 너무 큰 순간 점프는 tracking 오류 가능성
MAX_SINGLE_STEP_HEADS = 4.0

MAX_JUMP_RATIO = 0.15


# =============================================================================
# DATA
# =============================================================================

def find_videos(root: Path):

    videos = []

    for path in root.rglob("*"):

        if (
            path.is_file()
            and path.suffix.lower() in VIDEO_EXTENSIONS
        ):
            videos.append(path)

    return sorted(videos)


def head_scale(box):

    x1, y1, x2, y2 = box

    width = max(
        1.0,
        x2 - x1
    )

    height = max(
        1.0,
        y2 - y1
    )

    return math.sqrt(
        width * height
    )


def head_center(box):

    x1, y1, x2, y2 = box

    return (
        (x1 + x2) / 2.0,
        (y1 + y2) / 2.0,
    )


# =============================================================================
# TRACK ANALYSIS
# =============================================================================

def analyze_track(
    records,
    fps,
    frame_width,
    frame_height,
):

    records = sorted(
        records,
        key=lambda x: x["frame"]
    )

    if not records:

        return None

    first_frame = records[0]["frame"]
    last_frame = records[-1]["frame"]

    span_frames = (
        last_frame
        - first_frame
        + 1
    )

    duration_sec = (
        span_frames / fps
    )

    visible_frames = len(
        records
    )

    visible_ratio = (
        visible_frames
        /
        max(
            1,
            span_frames
        )
    )

    # =========================================================================
    # HEAD SIZE
    # =========================================================================

    scales = np.asarray(
        [
            record["scale"]
            for record in records
        ],
        dtype=np.float32,
    )

    median_scale = float(
        np.median(
            scales
        )
    )

    frame_scale = math.sqrt(
        max(
            1.0,
            frame_width
            * frame_height
        )
    )

    head_scale_norm = (
        median_scale
        /
        frame_scale
    )

    # =========================================================================
    # MOTION
    # =========================================================================

    speeds = []
    vertical_speeds = []

    jump_count = 0
    valid_pairs = 0

    for previous, current in zip(
        records,
        records[1:]
    ):

        frame_gap = (
            current["frame"]
            -
            previous["frame"]
        )

        if frame_gap <= 0:
            continue

        # LOST 구간을 건너뛰어서 속도를 계산하면 안 됨.
        # 너무 긴 gap은 연결하지 않는다.
        if frame_gap > max(
            2,
            round(fps * 0.30)
        ):
            continue

        dt = (
            frame_gap / fps
        )

        dx = (
            current["cx"]
            -
            previous["cx"]
        )

        dy = (
            current["cy"]
            -
            previous["cy"]
        )

        local_scale = max(
            1.0,
            (
                current["scale"]
                +
                previous["scale"]
            ) / 2.0
        )

        distance_heads = (
            math.hypot(
                dx,
                dy
            )
            /
            local_scale
        )

        vertical_heads = (
            abs(dy)
            /
            local_scale
        )

        speed = (
            distance_heads
            /
            dt
        )

        vertical_speed = (
            vertical_heads
            /
            dt
        )

        speeds.append(
            speed
        )

        vertical_speeds.append(
            vertical_speed
        )

        valid_pairs += 1

        if (
            distance_heads
            >
            MAX_SINGLE_STEP_HEADS
        ):
            jump_count += 1

    if speeds:

        mean_speed = float(
            np.mean(
                speeds
            )
        )

        speed_std = float(
            np.std(
                speeds
            )
        )

        speed_p95 = float(
            np.percentile(
                speeds,
                95
            )
        )

    else:

        mean_speed = 0.0
        speed_std = 0.0
        speed_p95 = 0.0

    if vertical_speeds:

        mean_vertical_speed = float(
            np.mean(
                vertical_speeds
            )
        )

    else:

        mean_vertical_speed = 0.0

    jump_ratio = (
        jump_count
        /
        valid_pairs
        if valid_pairs > 0
        else 0.0
    )

    # =========================================================================
    # REASONS
    # =========================================================================

    reject_reasons = []

    if (
        duration_sec
        <
        MIN_TRACK_SECONDS
    ):

        reject_reasons.append(
            "too_short"
        )

    if (
        visible_ratio
        <
        MIN_VISIBLE_RATIO
    ):

        reject_reasons.append(
            "low_visibility"
        )

    if (
        head_scale_norm
        <
        MIN_HEAD_SCALE_NORM
    ):

        reject_reasons.append(
            "small_head"
        )

    if (
        mean_speed
        >
        MAX_MEAN_SPEED
    ):

        reject_reasons.append(
            "high_mean_speed"
        )

    if (
        speed_std
        >
        MAX_SPEED_STD
    ):

        reject_reasons.append(
            "unstable_speed"
        )

    if (
        mean_vertical_speed
        >
        MAX_VERTICAL_SPEED
    ):

        reject_reasons.append(
            "high_vertical_motion"
        )

    if (
        speed_p95
        >
        MAX_TOTAL_SPEED_P95
    ):

        reject_reasons.append(
            "aggressive_motion"
        )

    if (
        jump_ratio
        >
        MAX_JUMP_RATIO
    ):

        reject_reasons.append(
            "tracking_jump"
        )

    if reject_reasons:

        quality = "REJECT"

    else:

        quality = "FLOATING_CANDIDATE"

    return {
        "quality":
            quality,

        "reject_reason":
            "+".join(
                reject_reasons
            ),

        "first_frame":
            first_frame,

        "last_frame":
            last_frame,

        "duration_sec":
            duration_sec,

        "visible_frames":
            visible_frames,

        "visible_ratio":
            visible_ratio,

        "median_head_scale":
            median_scale,

        "head_scale_norm":
            head_scale_norm,

        "mean_speed":
            mean_speed,

        "speed_std":
            speed_std,

        "speed_p95":
            speed_p95,

        "mean_vertical_speed":
            mean_vertical_speed,

        "jump_ratio":
            jump_ratio,
    }


# =============================================================================
# PROCESS VIDEO
# =============================================================================

def process_video(
    model,
    video_path,
    device,
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

    frame_count = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    cap.release()

    if fps <= 0:
        fps = 10.0

    track_records = defaultdict(
        list
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

        boxes = result.boxes

        if (
            boxes is None
            or boxes.id is None
            or len(boxes) == 0
        ):
            continue

        ids = (
            boxes.id
            .int()
            .detach()
            .cpu()
            .tolist()
        )

        xyxy_values = (
            boxes.xyxy
            .detach()
            .cpu()
            .numpy()
        )

        confidences = (
            boxes.conf
            .detach()
            .cpu()
            .tolist()
        )

        for (
            track_id,
            xyxy,
            confidence
        ) in zip(
            ids,
            xyxy_values,
            confidences
        ):

            box = [
                float(
                    value
                )
                for value in xyxy
            ]

            cx, cy = head_center(
                box
            )

            scale = head_scale(
                box
            )

            track_records[
                int(
                    track_id
                )
            ].append(
                {
                    "frame":
                        frame_index,

                    "cx":
                        cx,

                    "cy":
                        cy,

                    "scale":
                        scale,

                    "confidence":
                        float(
                            confidence
                        ),

                    "x1":
                        box[0],

                    "y1":
                        box[1],

                    "x2":
                        box[2],

                    "y2":
                        box[3],
                }
            )

    track_results = []

    for (
        track_id,
        records
    ) in track_records.items():

        analysis = analyze_track(
            records=records,
            fps=fps,
            frame_width=width,
            frame_height=height,
        )

        if analysis is None:
            continue

        track_results.append(
            {
                "video_id":
                    video_path.name,

                "relative_path":
                    str(
                        video_path.relative_to(
                            VIDEO_ROOT
                        )
                    ),

                "track_id":
                    track_id,

                "fps":
                    fps,

                "width":
                    width,

                "height":
                    height,

                "video_frames":
                    frame_count,

                **analysis,
            }
        )

    return track_results


# =============================================================================
# MAIN
# =============================================================================

def main():

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    videos = find_videos(
        VIDEO_ROOT
    )

    print(
        "=" * 80
    )

    print(
        "FLOATING CANDIDATE SELECTION"
    )

    print(
        "=" * 80
    )

    print(
        f"[VIDEO ROOT] {VIDEO_ROOT}"
    )

    print(
        f"[FOUND VIDEOS] {len(videos)}"
    )

    print(
        f"[MODEL] {MODEL_PATH}"
    )

    if not videos:

        print(
            "영상 없음"
        )

        return

    if not MODEL_PATH.is_file():

        raise FileNotFoundError(
            MODEL_PATH
        )

    if not TRACKER_PATH.is_file():

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

    all_rows = []

    for (
        video_index,
        video_path
    ) in enumerate(
        videos,
        start=1
    ):

        print()

        print(
            "=" * 80
        )

        print(
            f"[{video_index}/{len(videos)}]"
        )

        print(
            video_path
        )

        try:

            rows = process_video(
                model=model,
                video_path=video_path,
                device=device,
            )

        except Exception as error:

            print(
                f"[ERROR] {error}"
            )

            continue

        all_rows.extend(
            rows
        )

        candidate_count = sum(
            row["quality"]
            ==
            "FLOATING_CANDIDATE"

            for row in rows
        )

        reject_count = (
            len(rows)
            -
            candidate_count
        )

        print(
            f"TRACKS={len(rows)}"
            f" | CANDIDATE={candidate_count}"
            f" | REJECT={reject_count}"
        )

    # =============================================================================
    # SAVE
    # =============================================================================

    df = pd.DataFrame(
        all_rows
    )

    all_path = (
        OUTPUT_ROOT
        /
        "FLOATING_ALL_tracks.csv"
    )

    candidate_path = (
        OUTPUT_ROOT
        /
        "FLOATING_CANDIDATES.csv"
    )

    reject_path = (
        OUTPUT_ROOT
        /
        "FLOATING_REJECTED.csv"
    )

    df.to_csv(
        all_path,
        index=False,
        encoding="utf-8-sig"
    )

    if not df.empty:

        candidate_df = df[
            df["quality"]
            ==
            "FLOATING_CANDIDATE"
        ].copy()

        reject_df = df[
            df["quality"]
            ==
            "REJECT"
        ].copy()

        candidate_df.to_csv(
            candidate_path,
            index=False,
            encoding="utf-8-sig"
        )

        reject_df.to_csv(
            reject_path,
            index=False,
            encoding="utf-8-sig"
        )

    else:

        candidate_df = pd.DataFrame()
        reject_df = pd.DataFrame()

    # =============================================================================
    # REPORT
    # =============================================================================

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
        f"VIDEOS       : "
        f"{len(videos)}"
    )

    print(
        f"TRACKS       : "
        f"{len(df)}"
    )

    print(
        f"CANDIDATES   : "
        f"{len(candidate_df)}"
    )

    print(
        f"REJECTED     : "
        f"{len(reject_df)}"
    )

    if not reject_df.empty:

        print()

        print(
            "[REJECT REASONS]"
        )

        reason_counts = (
            reject_df[
                "reject_reason"
            ]
            .value_counts()
        )

        print(
            reason_counts
            .head(20)
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
