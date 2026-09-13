from __future__ import annotations

import math
import shutil
import xml.etree.ElementTree as ET
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
    r"D:\임베디드 경진대회\머리 추적\Swim2_ONLY\Swim2 ONLY\Swim2 ONLY"
)

MODEL_PATH = Path(
    r"D:\임베디드 경진대회\머리 추적\models\pool_head_best.pt"
)

OUTPUT_ROOT = Path(
    r"D:\임베디드 경진대회\머리 추적\SWIM2_CVAT_SINGLE_10FPS"
)

IMAGE_ROOT = OUTPUT_ROOT / "images"

ANNOTATION_XML = OUTPUT_ROOT / "annotations.xml"

MANIFEST_CSV = OUTPUT_ROOT / "manifest.csv"


# =============================================================================
# SOURCE
# =============================================================================

# 최종 NORMAL 학습용이므로 우선 Above만 사용.
# Under까지 만들고 싶으면 False로 바꾸면 됨.
ABOVE_ONLY = True


VIDEO_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".webm",
    ".m4v",
}


# =============================================================================
# SAMPLING
# =============================================================================

SOURCE_EXPECTED_FPS = 60.0

TARGET_FPS = 10.0


# =============================================================================
# YOLO
# =============================================================================

IMG_SIZE = 960

CONF = 0.05

IOU = 0.70


# =============================================================================
# SINGLE-SWIMMER TRACK RECOVERY
# =============================================================================

# 이전 머리 위치와 비교할 때 기본 허용 이동거리
# 단위 = 머리 크기
BASE_MAX_DISTANCE_HEADS = 2.5

# 검출이 오래 끊겼을수록 위치 gate를 완화한다.
DISTANCE_GROWTH_PER_SECOND = 1.5

# 최대 거리 gate
MAX_DISTANCE_HEADS = 8.0

# 머리 크기 변화 허용 범위
MIN_SCALE_RATIO = 0.30
MAX_SCALE_RATIO = 3.50

# 오래 사라진 뒤에는 위치보다 confidence를 더 믿음.
LONG_GAP_SECONDS = 2.0

# 너무 작은 검출은 노이즈일 가능성이 높으므로 제거
MIN_HEAD_SCALE_NORM = 0.010

JPEG_QUALITY = 92


# =============================================================================
# HELPERS
# =============================================================================

def find_videos(root: Path):

    videos = []

    for path in root.rglob("*"):

        if (
            path.is_file()
            and
            path.suffix.lower() in VIDEO_EXTENSIONS
        ):

            text = str(path).lower()

            if ABOVE_ONLY:

                if "above" not in text:
                    continue

            videos.append(path)

    return sorted(videos)


def safe_name(value: str):

    value = str(value)

    for char in [
        "\\",
        "/",
        ":",
        "*",
        "?",
        '"',
        "<",
        ">",
        "|",
        " ",
    ]:

        value = value.replace(
            char,
            "_"
        )

    return value


def box_center(box):

    x1, y1, x2, y2 = box

    return (
        (x1 + x2) / 2.0,
        (y1 + y2) / 2.0,
    )


def box_scale(box):

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
# SINGLE SWIMMER TRACKER
# =============================================================================

class SingleSwimmerTracker:

    """
    영상 안에는 수영자가 1명뿐이라는 가정.

    따라서 ByteTrack ID가 아니라,
    프레임마다 검출된 머리 후보 중
    이전 수영자 위치/크기와 가장 자연스러운 후보 하나를 선택한다.

    검출이 끊겨도 track 자체는 삭제하지 않는다.
    다시 검출되면 같은 swimmer로 복구한다.
    """

    def __init__(self):

        self.initialized = False

        self.last_sample_index = None

        self.last_cx = None
        self.last_cy = None
        self.last_scale = None

        self.prev_sample_index = None

        self.prev_cx = None
        self.prev_cy = None

    def predict_position(
        self,
        current_sample_index,
    ):

        if not self.initialized:

            return None

        pred_x = self.last_cx
        pred_y = self.last_cy

        if (
            self.prev_sample_index is not None
            and
            self.last_sample_index is not None
            and
            self.prev_cx is not None
            and
            self.prev_cy is not None
        ):

            dt = (
                self.last_sample_index
                -
                self.prev_sample_index
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
                    current_sample_index
                    -
                    self.last_sample_index
                )

                # 긴 gap에 velocity extrapolation이 폭주하지 않도록 제한
                future = min(
                    future,
                    int(
                        TARGET_FPS
                        *
                        1.0
                    )
                )

                pred_x = (
                    self.last_cx
                    +
                    vx * future
                )

                pred_y = (
                    self.last_cy
                    +
                    vy * future
                )

        return (
            pred_x,
            pred_y
        )

    def select_detection(
        self,
        detections,
        sample_index,
    ):

        if len(detections) == 0:

            return None

        # ---------------------------------------------------------------------
        # 아직 swimmer 초기화 안 된 경우
        #
        # 단일 swimmer 영상이므로 가장 높은 confidence 후보부터 시작.
        # ---------------------------------------------------------------------

        if not self.initialized:

            detection = max(
                detections,
                key=lambda x: x["confidence"]
            )

            self.update(
                detection,
                sample_index
            )

            return detection

        # ---------------------------------------------------------------------
        # gap 계산
        # ---------------------------------------------------------------------

        gap_samples = (
            sample_index
            -
            self.last_sample_index
        )

        gap_seconds = (
            gap_samples
            /
            TARGET_FPS
        )

        predicted = (
            self.predict_position(
                sample_index
            )
        )

        pred_x, pred_y = (
            predicted
        )

        # ---------------------------------------------------------------------
        # 긴 gap
        #
        # 단일 swimmer밖에 없으므로
        # 지나치게 강한 위치 gate를 사용하지 않는다.
        # ---------------------------------------------------------------------

        if (
            gap_seconds
            >=
            LONG_GAP_SECONDS
        ):

            # 이전 머리 크기와 너무 비정상적으로 다른 것만 제외
            plausible = []

            for det in detections:

                scale_ratio = (
                    det["scale"]
                    /
                    max(
                        1.0,
                        self.last_scale
                    )
                )

                if (
                    MIN_SCALE_RATIO
                    <=
                    scale_ratio
                    <=
                    MAX_SCALE_RATIO
                ):

                    plausible.append(
                        det
                    )

            if plausible:

                detection = max(
                    plausible,
                    key=lambda x: x["confidence"]
                )

            else:

                detection = max(
                    detections,
                    key=lambda x: x["confidence"]
                )

            self.update(
                detection,
                sample_index
            )

            return detection

        # ---------------------------------------------------------------------
        # 일반 gap
        # ---------------------------------------------------------------------

        allowed_distance = (
            BASE_MAX_DISTANCE_HEADS
            +
            DISTANCE_GROWTH_PER_SECOND
            *
            gap_seconds
        )

        allowed_distance = min(
            allowed_distance,
            MAX_DISTANCE_HEADS
        )

        candidates = []

        for detection in detections:

            scale_ratio = (
                detection["scale"]
                /
                max(
                    1.0,
                    self.last_scale
                )
            )

            if not (
                MIN_SCALE_RATIO
                <=
                scale_ratio
                <=
                MAX_SCALE_RATIO
            ):
                continue

            local_scale = max(
                1.0,
                (
                    detection["scale"]
                    +
                    self.last_scale
                ) / 2.0
            )

            distance_heads = (
                math.hypot(
                    detection["cx"] - pred_x,
                    detection["cy"] - pred_y,
                )
                /
                local_scale
            )

            if (
                distance_heads
                >
                allowed_distance
            ):
                continue

            scale_penalty = abs(
                math.log(
                    max(
                        1e-6,
                        scale_ratio
                    )
                )
            )

            # 낮을수록 좋은 score
            score = (
                distance_heads
                +
                0.35 * scale_penalty
                -
                0.20 * detection["confidence"]
            )

            candidates.append(
                (
                    score,
                    detection
                )
            )

        # ---------------------------------------------------------------------
        # continuity candidate가 있으면 최저 score
        # ---------------------------------------------------------------------

        if candidates:

            candidates.sort(
                key=lambda x: x[0]
            )

            detection = (
                candidates[0][1]
            )

            self.update(
                detection,
                sample_index
            )

            return detection

        # ---------------------------------------------------------------------
        # gate를 통과하는 detection 없음
        #
        # 이 경우 잘못된 detection을 억지로 연결하지 않고
        # LOST로 남긴다.
        # ---------------------------------------------------------------------

        return None

    def update(
        self,
        detection,
        sample_index,
    ):

        if self.initialized:

            self.prev_sample_index = (
                self.last_sample_index
            )

            self.prev_cx = (
                self.last_cx
            )

            self.prev_cy = (
                self.last_cy
            )

        self.last_sample_index = (
            sample_index
        )

        self.last_cx = (
            detection["cx"]
        )

        self.last_cy = (
            detection["cy"]
        )

        self.last_scale = (
            detection["scale"]
        )

        self.initialized = True


# =============================================================================
# YOLO DETECTIONS
# =============================================================================

def detect_heads(
    model,
    frame,
    device,
):

    result = model.predict(
        source=frame,

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
        boxes is None
        or
        len(boxes) == 0
    ):

        return detections

    h, w = frame.shape[:2]

    frame_scale = math.sqrt(
        max(
            1.0,
            w * h
        )
    )

    xyxy_values = (
        boxes.xyxy
        .detach()
        .cpu()
        .numpy()
    )

    confidence_values = (
        boxes.conf
        .detach()
        .cpu()
        .tolist()
    )

    for (
        xyxy,
        confidence
    ) in zip(
        xyxy_values,
        confidence_values
    ):

        box = tuple(
            float(v)
            for v in xyxy
        )

        cx, cy = box_center(
            box
        )

        scale = box_scale(
            box
        )

        scale_norm = (
            scale
            /
            frame_scale
        )

        if (
            scale_norm
            <
            MIN_HEAD_SCALE_NORM
        ):

            continue

        detections.append(
            {
                "box":
                    box,

                "confidence":
                    float(
                        confidence
                    ),

                "cx":
                    cx,

                "cy":
                    cy,

                "scale":
                    scale,

                "scale_norm":
                    scale_norm,
            }
        )

    return detections


# =============================================================================
# CVAT XML
# =============================================================================

def create_cvat_xml(
    image_records,
    video_track_records,
):

    root = ET.Element(
        "annotations"
    )

    ET.SubElement(
        root,
        "version"
    ).text = "1.1"

    meta = ET.SubElement(
        root,
        "meta"
    )

    task = ET.SubElement(
        meta,
        "task"
    )

    ET.SubElement(
        task,
        "name"
    ).text = (
        "SWIM2_ABOVE_NORMAL_10FPS"
    )

    ET.SubElement(
        task,
        "size"
    ).text = str(
        len(image_records)
    )

    ET.SubElement(
        task,
        "mode"
    ).text = "interpolation"

    labels = ET.SubElement(
        task,
        "labels"
    )

    label = ET.SubElement(
        labels,
        "label"
    )

    ET.SubElement(
        label,
        "name"
    ).text = "head"

    ET.SubElement(
        label,
        "color"
    ).text = "#ff0000"

    ET.SubElement(
        label,
        "type"
    ).text = "rectangle"

    # =========================================================================
    # 각 원본 영상 = 별도 CVAT Track
    #
    # 영상마다 swimmer가 1명이므로
    # track 하나만 생성.
    # =========================================================================

    track_id = 0

    for (
        video_key,
        records
    ) in video_track_records.items():

        if len(records) == 0:
            continue

        track_element = ET.SubElement(
            root,
            "track",
            {
                "id":
                    str(track_id),

                "label":
                    "head",

                "source":
                    "auto",
            }
        )

        track_id += 1

        sorted_frames = sorted(
            records.keys()
        )

        previous_detected = False

        previous_box = None

        last_global_frame = None

        for global_frame in sorted_frames:

            record = records[
                global_frame
            ]

            detected = (
                record["detected"]
            )

            box = (
                record["box"]
            )

            if detected:

                x1, y1, x2, y2 = (
                    box
                )

                ET.SubElement(
                    track_element,
                    "box",
                    {
                        "frame":
                            str(
                                global_frame
                            ),

                        "outside":
                            "0",

                        "occluded":
                            "0",

                        "keyframe":
                            "1",

                        "xtl":
                            f"{x1:.2f}",

                        "ytl":
                            f"{y1:.2f}",

                        "xbr":
                            f"{x2:.2f}",

                        "ybr":
                            f"{y2:.2f}",

                        "z_order":
                            "0",
                    }
                )

                previous_detected = True

                previous_box = box

            else:

                # detected → missing으로 바뀌는 첫 프레임에만 outside=1
                if (
                    previous_detected
                    and
                    previous_box is not None
                ):

                    x1, y1, x2, y2 = (
                        previous_box
                    )

                    ET.SubElement(
                        track_element,
                        "box",
                        {
                            "frame":
                                str(
                                    global_frame
                                ),

                            "outside":
                                "1",

                            "occluded":
                                "0",

                            "keyframe":
                                "1",

                            "xtl":
                                f"{x1:.2f}",

                            "ytl":
                                f"{y1:.2f}",

                            "xbr":
                                f"{x2:.2f}",

                            "ybr":
                                f"{y2:.2f}",

                            "z_order":
                                "0",
                        }
                    )

                previous_detected = False

            last_global_frame = (
                global_frame
            )

        # 영상이 detection으로 끝났으면
        # 다음 영상으로 interpolation되지 않게 마지막에서 outside 종료
        if (
            previous_detected
            and
            previous_box is not None
            and
            last_global_frame is not None
        ):

            x1, y1, x2, y2 = (
                previous_box
            )

            ET.SubElement(
                track_element,
                "box",
                {
                    "frame":
                        str(
                            last_global_frame
                        ),

                    "outside":
                        "1",

                    "occluded":
                        "0",

                    "keyframe":
                        "1",

                    "xtl":
                        f"{x1:.2f}",

                    "ytl":
                        f"{y1:.2f}",

                    "xbr":
                        f"{x2:.2f}",

                    "ybr":
                        f"{y2:.2f}",

                    "z_order":
                        "0",
                }
            )

    tree = ET.ElementTree(
        root
    )

    ET.indent(
        tree,
        space="  "
    )

    tree.write(
        ANNOTATION_XML,
        encoding="utf-8",
        xml_declaration=True,
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    print(
        "=" * 80
    )

    print(
        "SWIM2 SINGLE SWIMMER -> CVAT 10FPS"
    )

    print(
        "=" * 80
    )

    if not VIDEO_ROOT.exists():

        raise FileNotFoundError(
            f"영상 경로 없음:\n{VIDEO_ROOT}"
        )

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"모델 없음:\n{MODEL_PATH}"
        )

    videos = find_videos(
        VIDEO_ROOT
    )

    print(
        f"[VIDEOS] {len(videos)}"
    )

    if not videos:

        return

    # 기존 출력 삭제 후 새로 생성
    if OUTPUT_ROOT.exists():

        print(
            "[INFO] 기존 CVAT 출력 폴더 삭제"
        )

        shutil.rmtree(
            OUTPUT_ROOT
        )

    IMAGE_ROOT.mkdir(
        parents=True,
        exist_ok=True
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

    global_frame = 0

    image_records = []

    video_track_records = {}

    manifest_rows = []

    # =========================================================================
    # VIDEO
    # =========================================================================

    for video_index, video_path in enumerate(
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

        cap = cv2.VideoCapture(
            str(
                video_path
            )
        )

        if not cap.isOpened():

            print(
                "[ERROR] 영상 열기 실패"
            )

            continue

        fps = float(
            cap.get(
                cv2.CAP_PROP_FPS
            )
        )

        total_frames = int(
            cap.get(
                cv2.CAP_PROP_FRAME_COUNT
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

        if fps <= 0:

            fps = (
                SOURCE_EXPECTED_FPS
            )

        # Swim2 ONLY가 60fps이므로
        # 6프레임마다 정확히 10fps
        sample_interval = int(
            round(
                fps
                /
                TARGET_FPS
            )
        )

        actual_fps = (
            fps
            /
            sample_interval
        )

        print(
            f"FPS={fps:.3f}"
            f" | sample every {sample_interval}"
            f" | output={actual_fps:.3f}fps"
        )

        if (
            abs(
                actual_fps
                -
                TARGET_FPS
            )
            >
            0.05
        ):

            print(
                "[WARNING] 정확히 10fps가 아님."
            )

        video_key = str(
            video_path.relative_to(
                VIDEO_ROOT
            )
        )

        video_safe = safe_name(
            video_key
        )

        video_folder = (
            IMAGE_ROOT
            /
            f"{video_index:04d}_{video_safe}"
        )

        video_folder.mkdir(
            parents=True,
            exist_ok=True
        )

        tracker = (
            SingleSwimmerTracker()
        )

        track_records = {}

        source_frame = 0

        sample_index = 0

        detected_count = 0

        # =====================================================================
        # FRAME
        # =====================================================================

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            if (
                source_frame
                %
                sample_interval
                != 0
            ):

                source_frame += 1
                continue

            detections = detect_heads(
                model=model,
                frame=frame,
                device=device,
            )

            selected = (
                tracker.select_detection(
                    detections=detections,
                    sample_index=sample_index,
                )
            )

            image_name = (
                f"{global_frame:08d}.jpg"
            )

            image_path = (
                video_folder
                /
                image_name
            )

            cv2.imwrite(
                str(
                    image_path
                ),
                frame,
                [
                    cv2.IMWRITE_JPEG_QUALITY,
                    JPEG_QUALITY,
                ],
            )

            if selected is not None:

                detected = 1
                box = selected["box"]

                detected_count += 1

                confidence = (
                    selected[
                        "confidence"
                    ]
                )

            else:

                detected = 0
                box = None
                confidence = 0.0

            track_records[
                global_frame
            ] = {
                "detected":
                    detected,

                "box":
                    box,
            }

            relative_image = str(
                image_path.relative_to(
                    IMAGE_ROOT
                )
            )

            image_records.append(
                {
                    "global_frame":
                        global_frame,

                    "name":
                        relative_image,

                    "width":
                        width,

                    "height":
                        height,
                }
            )

            manifest_rows.append(
                {
                    "global_frame":
                        global_frame,

                    "video_id":
                        video_key,

                    "source_frame":
                        source_frame,

                    "sample_index":
                        sample_index,

                    "time_sec":
                        source_frame
                        /
                        fps,

                    "detected":
                        detected,

                    "confidence":
                        confidence,

                    "image_path":
                        relative_image,
                }
            )

            global_frame += 1

            sample_index += 1

            source_frame += 1

        cap.release()

        video_track_records[
            video_key
        ] = track_records

        visible_ratio = (
            detected_count
            /
            max(
                1,
                sample_index
            )
        )

        print(
            f"SAMPLES={sample_index}"
            f" | DETECTED={detected_count}"
            f" | visible={visible_ratio:.3f}"
        )

    # =========================================================================
    # MANIFEST
    # =========================================================================

    manifest_df = pd.DataFrame(
        manifest_rows
    )

    manifest_df.to_csv(
        MANIFEST_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    # =========================================================================
    # CVAT XML
    # =========================================================================

    create_cvat_xml(
        image_records=image_records,
        video_track_records=video_track_records,
    )

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
        f"VIDEOS       : {len(video_track_records)}"
    )

    print(
        f"10FPS IMAGES : {len(image_records)}"
    )

    print(
        f"MANIFEST     : {MANIFEST_CSV}"
    )

    print(
        f"ANNOTATIONS  : {ANNOTATION_XML}"
    )

    print(
        f"IMAGE ROOT   : {IMAGE_ROOT}"
    )

    print()
    print(
        "CVAT 업로드용 데이터 생성 완료"
    )


if __name__ == "__main__":
    main()