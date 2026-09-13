from __future__ import annotations

import csv
import math
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO


# ============================================================
# 1. PATH
# ============================================================

VIDEO_ROOT = Path(
    r"D:\임베디드 경진대회\머리 추적\Swim xyz 영상 모음\videos"
)

MODEL_PATH = Path(
    r"D:\임베디드 경진대회\머리 추적\models\pool_head_best.pt"
)

OUTPUT_ROOT = Path(
    r"D:\임베디드 경진대회\머리 추적\익수자 판별 학습\SwimXYZ_NORMAL_CLEAN"
)


# ============================================================
# 2. VIDEO SETTINGS
# ============================================================

VIDEO_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".webm",
}

# 5초 window
WINDOW_SECONDS = 5.0

# 1초 간격 sliding window
WINDOW_STRIDE_SECONDS = 1.0

# Detector는 매 프레임 돌리지 않고 최대 10Hz
TARGET_SAMPLE_HZ = 10.0

# 너무 많은 유사 window가 한 영상에서 생성되는 것 방지
MAX_WINDOWS_PER_VIDEO = 8


# ============================================================
# 3. DETECTOR SETTINGS
# ============================================================

DETECT_CONF = 0.35
DETECT_IOU = 0.45
IMAGE_SIZE = 960

# 이전 머리와 다음 머리를 연결할 때 사용
CONTINUITY_DISTANCE_WEIGHT = 0.45

# 이전 머리 중심으로부터 너무 멀리 있는 검출은 제외
MAX_CENTER_DISTANCE_NORM = 0.35


# ============================================================
# 4. QUALITY FILTER
# ============================================================

# ---------------------------
# CORE
# ---------------------------

CORE_MIN_VISIBLE_RATIO = 0.75
CORE_MAX_MISSING_SEC = 1.0
CORE_MAX_BORDER_RATIO = 0.15
CORE_MAX_JUMP_RATIO = 0.05
CORE_MIN_MEAN_CONF = 0.40


# ---------------------------
# HARD
# ---------------------------

HARD_MIN_VISIBLE_RATIO = 0.50
HARD_MAX_MISSING_SEC = 2.50
HARD_MAX_BORDER_RATIO = 0.35
HARD_MAX_JUMP_RATIO = 0.15
HARD_MIN_MEAN_CONF = 0.32


# ---------------------------
# INVALID MOTION
# ---------------------------

# 연속된 두 visible frame 사이에서
# 머리 크기 대비 중심점 이동량이 이것보다 크면 jump
JUMP_HEAD_SIZE_THRESHOLD = 3.0

# 화면 가장자리 판정
BORDER_MARGIN_RATIO = 0.05


# ============================================================
# 5. DATA CLASS
# ============================================================

@dataclass
class HeadObservation:
    time_sec: float
    frame_idx: int

    detected: int

    confidence: float

    cx: float
    cy: float
    w: float
    h: float

    cx_norm: float
    cy_norm: float
    w_norm: float
    h_norm: float


# ============================================================
# 6. UTILITY
# ============================================================

def safe_nan():
    return float("nan")


def find_videos(root: Path) -> list[Path]:
    videos = []

    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS:
            videos.append(p)

    return sorted(videos)


def infer_metadata(video_path: Path) -> dict:
    text = str(video_path).lower()

    stroke = "UNKNOWN"

    if "freestyle" in text:
        stroke = "FREESTYLE"
    elif "backstroke" in text:
        stroke = "BACKSTROKE"
    elif "breaststroke" in text:
        stroke = "BREASTSTROKE"
    elif "butterfly" in text:
        stroke = "BUTTERFLY"

    view = "UNKNOWN"

    if "front" in text:
        view = "FRONT"
    elif "aerial" in text:
        view = "AERIAL"
    elif "side" in text:
        view = "SIDE"
    elif "under" in text:
        view = "UNDER"

    speed = None

    m = re.search(r"speed[_\- ]?([0-9,.]+)", text)

    if m:
        speed = m.group(1)

    return {
        "stroke": stroke,
        "view": view,
        "speed": speed,
    }


# ============================================================
# 7. DETECTION SELECTION
# ============================================================

def choose_best_detection(
    boxes,
    frame_w: int,
    frame_h: int,
    previous: Optional[HeadObservation],
):
    """
    SwimXYZ는 기본적으로 수영자 1명이라고 가정.

    첫 detection:
        confidence가 가장 높은 머리

    이후:
        confidence + 이전 위치와의 연속성
    """

    if boxes is None or len(boxes) == 0:
        return None

    candidates = []

    for box in boxes:

        conf = float(box.conf[0])

        x1, y1, x2, y2 = map(
            float,
            box.xyxy[0].tolist()
        )

        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)

        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        cx_norm = cx / frame_w
        cy_norm = cy / frame_h

        if previous is None or previous.detected == 0:

            score = conf

        else:

            dx = cx_norm - previous.cx_norm
            dy = cy_norm - previous.cy_norm

            distance = math.sqrt(dx * dx + dy * dy)

            if distance > MAX_CENTER_DISTANCE_NORM:
                continue

            continuity_score = max(
                0.0,
                1.0 - distance / MAX_CENTER_DISTANCE_NORM
            )

            score = (
                conf
                + CONTINUITY_DISTANCE_WEIGHT * continuity_score
            )

        candidates.append(
            {
                "score": score,
                "conf": conf,
                "cx": cx,
                "cy": cy,
                "w": w,
                "h": h,
                "cx_norm": cx_norm,
                "cy_norm": cy_norm,
                "w_norm": w / frame_w,
                "h_norm": h / frame_h,
            }
        )

    if not candidates:
        return None

    candidates.sort(
        key=lambda x: x["score"],
        reverse=True,
    )

    return candidates[0]


# ============================================================
# 8. VIDEO → FRAME SEQUENCE
# ============================================================

def process_video(
    model: YOLO,
    video_path: Path,
) -> tuple[list[HeadObservation], dict]:

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(
            f"영상 열기 실패: {video_path}"
        )

    fps = float(cap.get(cv2.CAP_PROP_FPS))

    frame_count = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    frame_w = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    frame_h = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    if fps <= 0:
        fps = 30.0

    # 목표 10Hz에 맞춰 sampling
    frame_step = max(
        1,
        round(fps / TARGET_SAMPLE_HZ)
    )

    actual_sample_hz = fps / frame_step

    observations: list[HeadObservation] = []

    previous: Optional[HeadObservation] = None

    frame_idx = 0

    while True:

        ok, frame = cap.read()

        if not ok:
            break

        if frame_idx % frame_step != 0:
            frame_idx += 1
            continue

        time_sec = frame_idx / fps

        result = model.predict(
            frame,
            imgsz=IMAGE_SIZE,
            conf=DETECT_CONF,
            iou=DETECT_IOU,
            verbose=False,
        )[0]

        det = choose_best_detection(
            result.boxes,
            frame_w,
            frame_h,
            previous,
        )

        if det is None:

            obs = HeadObservation(
                time_sec=time_sec,
                frame_idx=frame_idx,

                detected=0,

                confidence=0.0,

                cx=safe_nan(),
                cy=safe_nan(),
                w=safe_nan(),
                h=safe_nan(),

                cx_norm=safe_nan(),
                cy_norm=safe_nan(),
                w_norm=safe_nan(),
                h_norm=safe_nan(),
            )

        else:

            obs = HeadObservation(
                time_sec=time_sec,
                frame_idx=frame_idx,

                detected=1,

                confidence=det["conf"],

                cx=det["cx"],
                cy=det["cy"],
                w=det["w"],
                h=det["h"],

                cx_norm=det["cx_norm"],
                cy_norm=det["cy_norm"],
                w_norm=det["w_norm"],
                h_norm=det["h_norm"],
            )

            previous = obs

        observations.append(obs)

        frame_idx += 1

    cap.release()

    metadata = {
        "fps": fps,
        "frame_count": frame_count,
        "width": frame_w,
        "height": frame_h,
        "sample_hz": actual_sample_hz,
        "duration_sec": frame_count / fps,
    }

    return observations, metadata


# ============================================================
# 9. WINDOW FEATURE
# ============================================================

def longest_missing_duration(
    detected: np.ndarray,
    sample_hz: float,
) -> float:

    longest = 0
    current = 0

    for x in detected:

        if x == 0:
            current += 1
            longest = max(longest, current)

        else:
            current = 0

    return longest / sample_hz


def count_submerge_cycles(
    detected: np.ndarray,
) -> int:
    """
    visible → missing → visible
    패턴 횟수
    """

    cycles = 0
    in_missing = False

    for x in detected:

        if x == 0:
            in_missing = True

        elif x == 1 and in_missing:
            cycles += 1
            in_missing = False

    return cycles


def calculate_features(
    window: list[HeadObservation],
    sample_hz: float,
) -> dict:

    detected = np.array(
        [x.detected for x in window],
        dtype=np.int32,
    )

    visible_ratio = float(
        detected.mean()
    )

    disappear_ratio = 1.0 - visible_ratio

    visible = [
        x for x in window
        if x.detected == 1
    ]

    max_missing_sec = longest_missing_duration(
        detected,
        sample_hz,
    )

    submerge_cycles = count_submerge_cycles(
        detected
    )

    if not visible:

        return {
            "visible_ratio": visible_ratio,
            "disappear_ratio": disappear_ratio,

            "max_missing_sec": max_missing_sec,
            "submerge_cycles": submerge_cycles,

            "mean_conf": 0.0,

            "border_ratio": 1.0,
            "jump_ratio": 1.0,

            "total_distance_norm": np.nan,
            "net_distance_norm": np.nan,

            "vertical_motion_norm": np.nan,

            "progress_ratio": np.nan,

            "head_scale_mean": np.nan,
            "head_scale_change": np.nan,
        }

    confs = np.array(
        [x.confidence for x in visible],
        dtype=np.float32,
    )

    mean_conf = float(
        confs.mean()
    )

    # ------------------------------------
    # Border ratio
    # ------------------------------------

    border_count = 0

    for x in visible:

        if (
            x.cx_norm <= BORDER_MARGIN_RATIO
            or x.cx_norm >= 1.0 - BORDER_MARGIN_RATIO
            or x.cy_norm <= BORDER_MARGIN_RATIO
            or x.cy_norm >= 1.0 - BORDER_MARGIN_RATIO
        ):
            border_count += 1

    border_ratio = (
        border_count / len(visible)
    )

    # ------------------------------------
    # Median head size
    # ------------------------------------

    head_sizes = np.array(
        [
            math.sqrt(
                max(
                    1e-8,
                    x.w_norm * x.h_norm
                )
            )
            for x in visible
        ],
        dtype=np.float32,
    )

    median_head_size = float(
        np.median(head_sizes)
    )

    mean_head_scale = float(
        np.mean(head_sizes)
    )

    if mean_head_scale > 0:

        head_scale_change = float(
            (
                np.max(head_sizes)
                - np.min(head_sizes)
            )
            / mean_head_scale
        )

    else:
        head_scale_change = 0.0

    # ------------------------------------
    # Motion
    # ------------------------------------

    total_distance = 0.0
    vertical_motion = 0.0

    jump_count = 0
    pair_count = 0

    first_visible = None
    last_visible = None

    previous_visible = None

    for x in window:

        if x.detected == 0:
            continue

        if first_visible is None:
            first_visible = x

        last_visible = x

        if previous_visible is not None:

            dx = (
                x.cx_norm
                - previous_visible.cx_norm
            )

            dy = (
                x.cy_norm
                - previous_visible.cy_norm
            )

            dist = math.sqrt(
                dx * dx + dy * dy
            )

            total_distance += dist
            vertical_motion += abs(dy)

            previous_size = math.sqrt(
                max(
                    1e-8,
                    previous_visible.w_norm
                    * previous_visible.h_norm
                )
            )

            current_size = math.sqrt(
                max(
                    1e-8,
                    x.w_norm
                    * x.h_norm
                )
            )

            local_head_size = max(
                1e-6,
                (
                    previous_size
                    + current_size
                ) / 2
            )

            dist_head_units = (
                dist / local_head_size
            )

            pair_count += 1

            if (
                dist_head_units
                > JUMP_HEAD_SIZE_THRESHOLD
            ):
                jump_count += 1

        previous_visible = x

    if median_head_size <= 0:
        median_head_size = 1e-6

    total_distance_norm = (
        total_distance
        / median_head_size
    )

    vertical_motion_norm = (
        vertical_motion
        / median_head_size
    )

    if (
        first_visible is not None
        and last_visible is not None
    ):

        dx = (
            last_visible.cx_norm
            - first_visible.cx_norm
        )

        dy = (
            last_visible.cy_norm
            - first_visible.cy_norm
        )

        net_distance = math.sqrt(
            dx * dx + dy * dy
        )

        net_distance_norm = (
            net_distance
            / median_head_size
        )

    else:

        net_distance_norm = 0.0

    if total_distance_norm > 1e-6:

        progress_ratio = (
            net_distance_norm
            / total_distance_norm
        )

    else:

        progress_ratio = 0.0

    if pair_count > 0:

        jump_ratio = (
            jump_count / pair_count
        )

    else:

        jump_ratio = 0.0

    return {
        "visible_ratio": visible_ratio,
        "disappear_ratio": disappear_ratio,

        "max_missing_sec": max_missing_sec,
        "submerge_cycles": submerge_cycles,

        "mean_conf": mean_conf,

        "border_ratio": border_ratio,
        "jump_ratio": jump_ratio,

        "total_distance_norm":
            total_distance_norm,

        "net_distance_norm":
            net_distance_norm,

        "vertical_motion_norm":
            vertical_motion_norm,

        "progress_ratio":
            progress_ratio,

        "head_scale_mean":
            mean_head_scale,

        "head_scale_change":
            head_scale_change,
    }


# ============================================================
# 10. QUALITY CLASSIFICATION
# ============================================================

def classify_window(
    features: dict,
) -> tuple[str, str]:

    visible = features["visible_ratio"]
    missing = features["max_missing_sec"]

    border = features["border_ratio"]
    jump = features["jump_ratio"]

    conf = features["mean_conf"]

    # ------------------------------------
    # HARD REJECT
    # ------------------------------------

    if visible < HARD_MIN_VISIBLE_RATIO:

        return (
            "REJECT",
            "visible_ratio_too_low"
        )

    if missing > HARD_MAX_MISSING_SEC:

        return (
            "REJECT",
            "missing_too_long"
        )

    if border > HARD_MAX_BORDER_RATIO:

        return (
            "REJECT",
            "too_close_to_border"
        )

    if jump > HARD_MAX_JUMP_RATIO:

        return (
            "REJECT",
            "unstable_tracking_jump"
        )

    if conf < HARD_MIN_MEAN_CONF:

        return (
            "REJECT",
            "low_detector_confidence"
        )

    # ------------------------------------
    # CORE
    # ------------------------------------

    core = (
        visible >= CORE_MIN_VISIBLE_RATIO
        and missing <= CORE_MAX_MISSING_SEC
        and border <= CORE_MAX_BORDER_RATIO
        and jump <= CORE_MAX_JUMP_RATIO
        and conf >= CORE_MIN_MEAN_CONF
    )

    if core:

        return (
            "CORE",
            "clean_normal_sequence"
        )

    # ------------------------------------
    # HARD NORMAL
    # ------------------------------------

    reasons = []

    if visible < CORE_MIN_VISIBLE_RATIO:
        reasons.append("lower_visibility")

    if missing > CORE_MAX_MISSING_SEC:
        reasons.append("normal_submersion")

    if features["submerge_cycles"] >= 2:
        reasons.append("repeated_submerge")

    if border > CORE_MAX_BORDER_RATIO:
        reasons.append("border_case")

    if jump > CORE_MAX_JUMP_RATIO:
        reasons.append("motion_difficult")

    if conf < CORE_MIN_MEAN_CONF:
        reasons.append("lower_confidence")

    if not reasons:
        reasons.append("difficult_normal")

    return (
        "HARD",
        "+".join(reasons)
    )


# ============================================================
# 11. CREATE WINDOWS
# ============================================================

def generate_windows(
    observations: list[HeadObservation],
    sample_hz: float,
):

    window_samples = max(
        1,
        round(
            WINDOW_SECONDS
            * sample_hz
        )
    )

    stride_samples = max(
        1,
        round(
            WINDOW_STRIDE_SECONDS
            * sample_hz
        )
    )

    if len(observations) < window_samples:
        return

    generated = 0

    for start in range(
        0,
        len(observations) - window_samples + 1,
        stride_samples,
    ):

        if generated >= MAX_WINDOWS_PER_VIDEO:
            break

        end = start + window_samples

        window = observations[
            start:end
        ]

        yield start, end, window

        generated += 1


# ============================================================
# 12. SAVE FRAME SEQUENCE
# ============================================================

def observation_to_row(
    video_id: str,
    window_id: str,
    quality: str,
    label: str,
    obs: HeadObservation,
):

    return {
        "video_id": video_id,
        "window_id": window_id,

        "quality": quality,
        "label": label,

        "time_sec": obs.time_sec,
        "frame_idx": obs.frame_idx,

        "detected": obs.detected,

        "confidence": obs.confidence,

        "cx": obs.cx,
        "cy": obs.cy,

        "w": obs.w,
        "h": obs.h,

        "cx_norm": obs.cx_norm,
        "cy_norm": obs.cy_norm,

        "w_norm": obs.w_norm,
        "h_norm": obs.h_norm,
    }


# ============================================================
# 13. MAIN
# ============================================================

def main():

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print("SwimXYZ NORMAL CLEAN DATA EXTRACTION")
    print("=" * 70)

    print(f"[VIDEO ROOT]")
    print(VIDEO_ROOT)

    print()

    print(f"[MODEL]")
    print(MODEL_PATH)

    print()

    videos = find_videos(
        VIDEO_ROOT
    )

    print(
        f"[FOUND VIDEOS] {len(videos)}"
    )

    if len(videos) == 0:

        print(
            "영상이 없습니다."
        )

        return

    model = YOLO(
        str(MODEL_PATH)
    )

    summary_rows = []

    frame_rows = []

    video_summary = []

    counts = {
        "CORE": 0,
        "HARD": 0,
        "REJECT": 0,
    }

    for video_index, video_path in enumerate(
        videos,
        start=1,
    ):

        print()
        print("=" * 70)

        print(
            f"[{video_index}/{len(videos)}]"
        )

        print(video_path)

        metadata = infer_metadata(
            video_path
        )

        try:

            observations, video_info = process_video(
                model,
                video_path,
            )

        except Exception as e:

            print(
                f"[ERROR] {e}"
            )

            continue

        video_id = str(
            video_path.relative_to(
                VIDEO_ROOT
            )
        )

        video_counts = {
            "CORE": 0,
            "HARD": 0,
            "REJECT": 0,
        }

        for (
            start,
            end,
            window,
        ) in generate_windows(
            observations,
            video_info["sample_hz"],
        ):

            features = calculate_features(
                window,
                video_info["sample_hz"],
            )

            quality, reason = classify_window(
                features
            )

            counts[quality] += 1
            video_counts[quality] += 1

            start_sec = window[0].time_sec
            end_sec = window[-1].time_sec

            window_id = (
                f"{video_index:05d}_"
                f"{start_sec:.2f}_"
                f"{end_sec:.2f}"
            )

            row = {
                "video_id": video_id,
                "window_id": window_id,

                "stroke":
                    metadata["stroke"],

                "view":
                    metadata["view"],

                "speed":
                    metadata["speed"],

                "start_sec":
                    start_sec,

                "end_sec":
                    end_sec,

                "quality":
                    quality,

                "quality_reason":
                    reason,

                "label":
                    "NORMAL_SWIMMING",

                **features,
            }

            summary_rows.append(
                row
            )

            # CORE / HARD만 frame sequence 저장
            if quality != "REJECT":

                for obs in window:

                    frame_rows.append(
                        observation_to_row(
                            video_id,
                            window_id,
                            quality,
                            "NORMAL_SWIMMING",
                            obs,
                        )
                    )

        video_summary.append(
            {
                "video_id":
                    video_id,

                "stroke":
                    metadata["stroke"],

                "view":
                    metadata["view"],

                "speed":
                    metadata["speed"],

                "duration_sec":
                    video_info["duration_sec"],

                "sample_hz":
                    video_info["sample_hz"],

                "CORE":
                    video_counts["CORE"],

                "HARD":
                    video_counts["HARD"],

                "REJECT":
                    video_counts["REJECT"],
            }
        )

        print(
            f"CORE={video_counts['CORE']} "
            f"HARD={video_counts['HARD']} "
            f"REJECT={video_counts['REJECT']}"
        )

    # ========================================================
    # SAVE
    # ========================================================

    summary_df = pd.DataFrame(
        summary_rows
    )

    frame_df = pd.DataFrame(
        frame_rows
    )

    video_df = pd.DataFrame(
        video_summary
    )

    all_path = (
        OUTPUT_ROOT
        / "NORMAL_ALL_windows.csv"
    )

    core_path = (
        OUTPUT_ROOT
        / "NORMAL_CORE_windows.csv"
    )

    hard_path = (
        OUTPUT_ROOT
        / "NORMAL_HARD_windows.csv"
    )

    reject_path = (
        OUTPUT_ROOT
        / "NORMAL_REJECT_windows.csv"
    )

    frame_path = (
        OUTPUT_ROOT
        / "NORMAL_frame_sequences.csv"
    )

    video_path = (
        OUTPUT_ROOT
        / "NORMAL_video_summary.csv"
    )

    summary_df.to_csv(
        all_path,
        index=False,
        encoding="utf-8-sig",
    )

    if not summary_df.empty:

        summary_df[
            summary_df["quality"]
            == "CORE"
        ].to_csv(
            core_path,
            index=False,
            encoding="utf-8-sig",
        )

        summary_df[
            summary_df["quality"]
            == "HARD"
        ].to_csv(
            hard_path,
            index=False,
            encoding="utf-8-sig",
        )

        summary_df[
            summary_df["quality"]
            == "REJECT"
        ].to_csv(
            reject_path,
            index=False,
            encoding="utf-8-sig",
        )

    frame_df.to_csv(
        frame_path,
        index=False,
        encoding="utf-8-sig",
    )

    video_df.to_csv(
        video_path,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # FINAL REPORT
    # ========================================================

    print()
    print("=" * 70)

    print("FINISHED")

    print("=" * 70)

    print(
        f"CORE   : {counts['CORE']}"
    )

    print(
        f"HARD   : {counts['HARD']}"
    )

    print(
        f"REJECT : {counts['REJECT']}"
    )

    total = sum(
        counts.values()
    )

    print(
        f"TOTAL  : {total}"
    )

    if total > 0:

        print()

        print(
            f"CORE Ratio   : "
            f"{counts['CORE']/total*100:.1f}%"
        )

        print(
            f"HARD Ratio   : "
            f"{counts['HARD']/total*100:.1f}%"
        )

        print(
            f"REJECT Ratio : "
            f"{counts['REJECT']/total*100:.1f}%"
        )

    print()

    print(
        "[OUTPUT]"
    )

    print(OUTPUT_ROOT)


if __name__ == "__main__":
    main()