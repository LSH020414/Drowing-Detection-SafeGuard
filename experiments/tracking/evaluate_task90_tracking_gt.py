from __future__ import annotations

import csv
import importlib.util
import math
import sys
import tempfile
import zipfile
import xml.etree.ElementTree as ET

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO


# =============================================================================
# PATHS
# =============================================================================

VIDEO_PATH = Path(
    r"C:\Users\이승희\Desktop\Video Project 7.mp4"
)

GT_ZIP_PATH = Path(
    r"C:\Users\이승희\Desktop"
    r"\task_90_annotations_2026_08_28_14_00_38_cvat for video 1.1.zip"
)

BASE_TRACKING_SCRIPT = Path(
    r"D:\임베디드 경진대회\머리 추적\트레킹 테스트"
    r"\compare_V10_V11_tracking.py"
)

OUTPUT_DIR = Path(
    r"D:\임베디드 경진대회\머리 추적\트레킹 테스트"
    r"\Task90_GT_Evaluation"
)

TRACKER_YAML = (
    OUTPUT_DIR
    / "bytetrack_pool_10fps.yaml"
)


# =============================================================================
# EVALUATION SETTINGS
# =============================================================================

# GT와 Prediction bbox가 이 IoU 이상이면 detection match
IOU_MATCH_THRESHOLD = 0.50

# 필요하면 나중에 0.30에서도 sensitivity test 가능
SECONDARY_IOU_THRESHOLD = 0.30

REID_THRESHOLD = 0.84

SAVE_RESULT_VIDEO = True

# CVAT의 occluded=True 박스도 GT로 포함
INCLUDE_OCCLUDED = True

# CVAT label이 여러 개 있을 경우.
# None이면 모든 rectangle track 사용.
GT_LABEL_FILTER = None

DEVICE = 0 if torch.cuda.is_available() else "cpu"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class BoxRecord:
    frame: int
    object_id: int

    x1: float
    y1: float
    x2: float
    y2: float

    confidence: float = 1.0
    occluded: bool = False


# =============================================================================
# BASIC FUNCTIONS
# =============================================================================

def iou(
    first: BoxRecord,
    second: BoxRecord,
) -> float:

    ix1 = max(
        first.x1,
        second.x1,
    )

    iy1 = max(
        first.y1,
        second.y1,
    )

    ix2 = min(
        first.x2,
        second.x2,
    )

    iy2 = min(
        first.y2,
        second.y2,
    )

    iw = max(
        0.0,
        ix2 - ix1,
    )

    ih = max(
        0.0,
        iy2 - iy1,
    )

    intersection = (
        iw * ih
    )

    first_area = max(
        0.0,
        first.x2 - first.x1,
    ) * max(
        0.0,
        first.y2 - first.y1,
    )

    second_area = max(
        0.0,
        second.x2 - second.x1,
    ) * max(
        0.0,
        second.y2 - second.y1,
    )

    union = (
        first_area
        +
        second_area
        -
        intersection
    )

    if union <= 0:
        return 0.0

    return float(
        intersection / union
    )


def get_video_info(
    path: Path,
):

    cap = cv2.VideoCapture(
        str(path)
    )

    if not cap.isOpened():

        raise RuntimeError(
            f"영상 열기 실패:\n{path}"
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

    frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    cap.release()

    return (
        fps,
        width,
        height,
        frames,
    )


# =============================================================================
# LOAD EXISTING V10/V11 TRACKING CODE
# =============================================================================

def load_tracking_module():

    if not BASE_TRACKING_SCRIPT.is_file():

        raise FileNotFoundError(
            "기존 tracking 코드가 없습니다:\n"
            f"{BASE_TRACKING_SCRIPT}"
        )

    module_name = (
        "task90_tracking_compare"
    )

    spec = (
        importlib.util
        .spec_from_file_location(
            module_name,
            BASE_TRACKING_SCRIPT,
        )
    )

    if (
        spec is None
        or
        spec.loader is None
    ):

        raise RuntimeError(
            "compare_V10_V11_tracking.py "
            "module 생성 실패"
        )

    module = (
        importlib.util
        .module_from_spec(
            spec
        )
    )

    # Python 3.13 dataclass 대응
    sys.modules[
        module_name
    ] = module

    spec.loader.exec_module(
        module
    )

    return module


# =============================================================================
# CVAT XML
# =============================================================================

def find_annotations_xml(
    directory: Path,
) -> Path:

    candidates = list(
        directory.rglob(
            "annotations.xml"
        )
    )

    if not candidates:

        candidates = list(
            directory.rglob(
                "*.xml"
            )
        )

    if not candidates:

        raise FileNotFoundError(
            "CVAT ZIP 내부에서 XML을 찾지 못했습니다."
        )

    return candidates[0]


def parse_bool(
    value,
) -> bool:

    return str(
        value
    ).strip().lower() in {
        "1",
        "true",
        "yes",
    }


def interpolate_value(
    first: float,
    second: float,
    ratio: float,
) -> float:

    return (
        first
        +
        (
            second
            -
            first
        )
        *
        ratio
    )


def parse_cvat_gt(
    zip_path: Path,
):

    print()
    print("=" * 80)
    print("CVAT GT LOAD")
    print("=" * 80)

    with tempfile.TemporaryDirectory() as temp:

        temp_dir = Path(
            temp
        )

        with zipfile.ZipFile(
            zip_path,
            "r",
        ) as archive:

            archive.extractall(
                temp_dir
            )

        xml_path = (
            find_annotations_xml(
                temp_dir
            )
        )

        print(
            "XML:",
            xml_path,
        )

        tree = ET.parse(
            xml_path
        )

        root = tree.getroot()

        gt_by_frame = defaultdict(
            list
        )

        track_count = 0
        box_count = 0

        for track_element in root.findall(
            "track"
        ):

            track_id = int(
                track_element.attrib[
                    "id"
                ]
            )

            label = (
                track_element.attrib
                .get(
                    "label",
                    "",
                )
            )

            if (
                GT_LABEL_FILTER is not None
                and
                label != GT_LABEL_FILTER
            ):

                continue

            boxes = []

            for box in track_element.findall(
                "box"
            ):

                frame = int(
                    box.attrib[
                        "frame"
                    ]
                )

                outside = parse_bool(
                    box.attrib.get(
                        "outside",
                        "0",
                    )
                )

                occluded = parse_bool(
                    box.attrib.get(
                        "occluded",
                        "0",
                    )
                )

                boxes.append(
                    {
                        "frame": frame,
                        "outside": outside,
                        "occluded": occluded,

                        "x1": float(
                            box.attrib[
                                "xtl"
                            ]
                        ),

                        "y1": float(
                            box.attrib[
                                "ytl"
                            ]
                        ),

                        "x2": float(
                            box.attrib[
                                "xbr"
                            ]
                        ),

                        "y2": float(
                            box.attrib[
                                "ybr"
                            ]
                        ),
                    }
                )

            if not boxes:
                continue

            boxes.sort(
                key=lambda item:
                    item["frame"]
            )

            track_count += 1

            # -----------------------------------------------------------------
            # Exact keyframes + interpolation
            # -----------------------------------------------------------------

            for index in range(
                len(boxes)
            ):

                current = boxes[
                    index
                ]

                frame = current[
                    "frame"
                ]

                if not current[
                    "outside"
                ]:

                    if (
                        INCLUDE_OCCLUDED
                        or
                        not current[
                            "occluded"
                        ]
                    ):

                        gt_by_frame[
                            frame
                        ].append(
                            BoxRecord(
                                frame=frame,
                                object_id=track_id,

                                x1=current[
                                    "x1"
                                ],

                                y1=current[
                                    "y1"
                                ],

                                x2=current[
                                    "x2"
                                ],

                                y2=current[
                                    "y2"
                                ],

                                occluded=current[
                                    "occluded"
                                ],
                            )
                        )

                        box_count += 1

                if (
                    index
                    ==
                    len(boxes) - 1
                ):
                    continue

                next_box = boxes[
                    index + 1
                ]

                start_frame = current[
                    "frame"
                ]

                end_frame = next_box[
                    "frame"
                ]

                gap = (
                    end_frame
                    -
                    start_frame
                )

                if gap <= 1:
                    continue

                # CVAT interpolation:
                # 시작 box가 outside면 그 이후 구간에는 객체 없음.
                if current[
                    "outside"
                ]:
                    continue

                for intermediate_frame in range(
                    start_frame + 1,
                    end_frame,
                ):

                    ratio = (
                        intermediate_frame
                        -
                        start_frame
                    ) / gap

                    x1 = interpolate_value(
                        current["x1"],
                        next_box["x1"],
                        ratio,
                    )

                    y1 = interpolate_value(
                        current["y1"],
                        next_box["y1"],
                        ratio,
                    )

                    x2 = interpolate_value(
                        current["x2"],
                        next_box["x2"],
                        ratio,
                    )

                    y2 = interpolate_value(
                        current["y2"],
                        next_box["y2"],
                        ratio,
                    )

                    occluded = (
                        current[
                            "occluded"
                        ]
                    )

                    if (
                        not INCLUDE_OCCLUDED
                        and
                        occluded
                    ):
                        continue

                    gt_by_frame[
                        intermediate_frame
                    ].append(
                        BoxRecord(
                            frame=intermediate_frame,
                            object_id=track_id,

                            x1=x1,
                            y1=y1,
                            x2=x2,
                            y2=y2,

                            occluded=occluded,
                        )
                    )

                    box_count += 1

        print(
            "GT tracks:",
            track_count,
        )

        print(
            "GT boxes:",
            box_count,
        )

        if gt_by_frame:

            print(
                "GT frame range:",
                min(
                    gt_by_frame.keys()
                ),
                "~",
                max(
                    gt_by_frame.keys()
                ),
            )

        return dict(
            gt_by_frame
        )


# =============================================================================
# TRACKING
# =============================================================================

def create_observations(
    module,
    result,
):

    observations = {}

    boxes = result.boxes

    if (
        boxes is None
        or
        boxes.id is None
        or
        len(boxes) == 0
    ):

        return observations

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

    for (
        bbox,
        local_id,
        confidence,
    ) in zip(
        coords,
        ids,
        confidences,
    ):

        local_id = int(
            local_id
        )

        observations[
            local_id
        ] = module.HeadObservation(
            local_id=local_id,

            x1=float(
                bbox[0]
            ),

            y1=float(
                bbox[1]
            ),

            x2=float(
                bbox[2]
            ),

            y2=float(
                bbox[3]
            ),

            confidence=float(
                confidence
            ),
        )

    return observations


def run_tracking(
    module,
    mode: str,
    fps: float,
    width: int,
    height: int,
    total_frames: int,
):

    use_reid = (
        mode
        ==
        "V11_REID_084"
    )

    print()
    print("=" * 80)
    print(
        "TRACKING:",
        mode,
    )
    print("=" * 80)

    # -------------------------------------------------------------------------
    # Fresh detector/tracker each run
    # -------------------------------------------------------------------------

    detector = YOLO(
        str(
            module.HEAD_MODEL
        )
    )

    config = (
        module.make_runtime_config(
            fps
        )
    )

    manager = (
        module.StableIDManager(
            config=config,
            use_reid=use_reid,
        )
    )

    reid_encoder = None

    if use_reid:

        module.REID_MIN_SIMILARITY = (
            REID_THRESHOLD
        )

        reid_encoder = (
            module.PoolReIDEncoder(
                module.POOL_REID_MODEL,
                DEVICE,
            )
        )

    prediction_by_frame = defaultdict(
        list
    )

    raw_by_frame = defaultdict(
        list
    )

    csv_rows = []

    if SAVE_RESULT_VIDEO:

        video_path = (
            OUTPUT_DIR
            /
            f"{mode}_GT_eval.mp4"
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

    else:

        writer = None

    results = detector.track(
        source=str(
            VIDEO_PATH
        ),
        stream=True,
        persist=False,
        tracker=str(
            TRACKER_YAML
        ),
        conf=module.CONF,
        iou=module.IOU,
        imgsz=module.IMG_SIZE,
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

        observations = (
            create_observations(
                module,
                result,
            )
        )

        # ---------------------------------------------------------------------
        # Save raw ByteTrack
        # ---------------------------------------------------------------------

        for (
            local_id,
            observation,
        ) in observations.items():

            raw_by_frame[
                frame_index
            ].append(
                BoxRecord(
                    frame=frame_index,
                    object_id=local_id,

                    x1=observation.x1,
                    y1=observation.y1,
                    x2=observation.x2,
                    y2=observation.y2,

                    confidence=(
                        observation.confidence
                    ),
                )
            )

        # ---------------------------------------------------------------------
        # ReID embedding
        # ---------------------------------------------------------------------

        if (
            use_reid
            and
            reid_encoder is not None
        ):

            reid_encoder.encode_observations(
                frame,
                observations,
            )

        # ---------------------------------------------------------------------
        # Stable ID
        # ---------------------------------------------------------------------

        assignments, events = (
            manager.update(
                frame_index,
                observations,
            )
        )

        for (
            local_id,
            stable_id,
        ) in assignments.items():

            observation = (
                observations.get(
                    local_id
                )
            )

            if observation is None:
                continue

            stable_id = (
                manager.canonical_id(
                    stable_id
                )
            )

            record = BoxRecord(
                frame=frame_index,
                object_id=stable_id,

                x1=observation.x1,
                y1=observation.y1,
                x2=observation.x2,
                y2=observation.y2,

                confidence=(
                    observation.confidence
                ),
            )

            prediction_by_frame[
                frame_index
            ].append(
                record
            )

            csv_rows.append(
                {
                    "frame":
                        frame_index,

                    "time_sec":
                        frame_index
                        /
                        fps,

                    "stable_id":
                        stable_id,

                    "local_id":
                        local_id,

                    "confidence":
                        observation.confidence,

                    "x1":
                        observation.x1,

                    "y1":
                        observation.y1,

                    "x2":
                        observation.x2,

                    "y2":
                        observation.y2,
                }
            )

            if writer is not None:

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
                    (
                        x1,
                        y1,
                    ),
                    (
                        x2,
                        y2,
                    ),
                    (
                        0,
                        255,
                        255,
                    ),
                    2,
                )

                cv2.putText(
                    frame,
                    f"ID {stable_id}",
                    (
                        x1,
                        max(
                            20,
                            y1 - 5,
                        ),
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (
                        0,
                        255,
                        255,
                    ),
                    2,
                    cv2.LINE_AA,
                )

        if writer is not None:

            cv2.putText(
                frame,
                mode,
                (
                    20,
                    35,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
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
        ) % 50 == 0:

            print(
                f"{frame_index + 1}"
                f"/{total_frames}"
            )

    if writer is not None:

        writer.release()

    # -------------------------------------------------------------------------
    # Alias canonicalization
    # -------------------------------------------------------------------------

    canonical_predictions = (
        defaultdict(
            list
        )
    )

    for frame_index, records in (
        prediction_by_frame.items()
    ):

        for record in records:

            canonical_id = (
                manager.canonical_id(
                    record.object_id
                )
            )

            canonical_predictions[
                frame_index
            ].append(
                BoxRecord(
                    frame=record.frame,
                    object_id=canonical_id,

                    x1=record.x1,
                    y1=record.y1,
                    x2=record.x2,
                    y2=record.y2,

                    confidence=(
                        record.confidence
                    ),
                )
            )

    write_prediction_csv(
        OUTPUT_DIR
        /
        f"{mode}_frame_predictions.csv",
        csv_rows,
    )

    print(
        "Created IDs:",
        manager.created_count,
    )

    print(
        "Recovery:",
        manager.recovered_count,
    )

    print(
        "Alias:",
        manager.alias_count,
    )

    if use_reid:

        print(
            "ReID checks:",
            manager.reid_candidate_checks,
        )

        print(
            "ReID rejected:",
            manager.reid_gate_rejections,
        )

    return (
        dict(
            canonical_predictions
        ),
        dict(
            raw_by_frame
        ),
    )


# =============================================================================
# CSV
# =============================================================================

def write_prediction_csv(
    path: Path,
    rows,
):

    if not rows:
        return

    with path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


def write_dict_csv(
    path: Path,
    rows,
):

    if not rows:
        return

    keys = []
    seen = set()

    for row in rows:

        for key in row:

            if key not in seen:

                keys.append(
                    key
                )

                seen.add(
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
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


# =============================================================================
# FRAME MATCHING
# =============================================================================

def match_frame(
    gt_boxes: list[BoxRecord],
    pred_boxes: list[BoxRecord],
    threshold: float,
):

    if (
        not gt_boxes
        or
        not pred_boxes
    ):

        return []

    cost = np.full(
        (
            len(gt_boxes),
            len(pred_boxes),
        ),
        1e6,
        dtype=np.float64,
    )

    similarities = np.zeros(
        cost.shape,
        dtype=np.float64,
    )

    for gt_index, gt in enumerate(
        gt_boxes
    ):

        for pred_index, pred in enumerate(
            pred_boxes
        ):

            value = iou(
                gt,
                pred,
            )

            similarities[
                gt_index,
                pred_index
            ] = value

            if value >= threshold:

                cost[
                    gt_index,
                    pred_index
                ] = (
                    1.0 - value
                )

    row_indices, column_indices = (
        linear_sum_assignment(
            cost
        )
    )

    matches = []

    for gt_index, pred_index in zip(
        row_indices,
        column_indices,
    ):

        value = similarities[
            gt_index,
            pred_index
        ]

        if value < threshold:
            continue

        matches.append(
            (
                gt_index,
                pred_index,
                float(
                    value
                ),
            )
        )

    return matches


# =============================================================================
# MOT METRICS
# =============================================================================

def evaluate_tracking(
    name: str,
    gt_by_frame,
    pred_by_frame,
    total_frames: int,
    iou_threshold: float,
):

    print()
    print(
        f"[EVALUATE] "
        f"{name}"
        f" @ IoU {iou_threshold}"
    )

    total_gt = 0
    total_pred = 0

    tp = 0
    fp = 0
    fn = 0

    id_switches = 0
    fragmentations = 0

    # GT ID -> 마지막으로 matched된 prediction ID
    last_pred_for_gt = {}

    # GT ID -> 이전 frame의 matched 여부
    previous_frame_matched = {}

    # GT ID -> 과거에 한 번이라도 matched된 적 있는지
    ever_matched = {}

    # IDF1 global association
    pair_match_counts = defaultdict(
        int
    )

    per_frame_rows = []

    for frame_index in range(
        total_frames
    ):

        gt_boxes = gt_by_frame.get(
            frame_index,
            [],
        )

        pred_boxes = pred_by_frame.get(
            frame_index,
            [],
        )

        total_gt += len(
            gt_boxes
        )

        total_pred += len(
            pred_boxes
        )

        matches = match_frame(
            gt_boxes,
            pred_boxes,
            iou_threshold,
        )

        matched_gt_indices = {
            item[0]
            for item in matches
        }

        matched_pred_indices = {
            item[1]
            for item in matches
        }

        tp += len(
            matches
        )

        fn += (
            len(gt_boxes)
            -
            len(matches)
        )

        fp += (
            len(pred_boxes)
            -
            len(matches)
        )

        current_matched_gt_ids = set()

        # ---------------------------------------------------------------------
        # Matched identities
        # ---------------------------------------------------------------------

        for (
            gt_index,
            pred_index,
            match_iou,
        ) in matches:

            gt = gt_boxes[
                gt_index
            ]

            pred = pred_boxes[
                pred_index
            ]

            gt_id = gt.object_id
            pred_id = pred.object_id

            current_matched_gt_ids.add(
                gt_id
            )

            pair_match_counts[
                (
                    gt_id,
                    pred_id,
                )
            ] += 1

            # -------------------------------------------------------------
            # ID switch
            # -------------------------------------------------------------

            if gt_id in last_pred_for_gt:

                if (
                    last_pred_for_gt[
                        gt_id
                    ]
                    !=
                    pred_id
                ):

                    id_switches += 1

            last_pred_for_gt[
                gt_id
            ] = pred_id

            # -------------------------------------------------------------
            # Fragmentation
            #
            # 과거 matched
            # → 이전 frame unmatched
            # → 현재 다시 matched
            # -------------------------------------------------------------

            if (
                ever_matched.get(
                    gt_id,
                    False,
                )
                and
                not previous_frame_matched.get(
                    gt_id,
                    False,
                )
            ):

                fragmentations += 1

            ever_matched[
                gt_id
            ] = True

            previous_frame_matched[
                gt_id
            ] = True

            per_frame_rows.append(
                {
                    "model": name,
                    "frame": frame_index,

                    "gt_id": gt_id,
                    "pred_id": pred_id,

                    "iou": match_iou,

                    "matched": 1,
                }
            )

        # ---------------------------------------------------------------------
        # Unmatched GT
        # ---------------------------------------------------------------------

        for gt_index, gt in enumerate(
            gt_boxes
        ):

            if (
                gt_index
                in
                matched_gt_indices
            ):
                continue

            previous_frame_matched[
                gt.object_id
            ] = False

            per_frame_rows.append(
                {
                    "model": name,
                    "frame": frame_index,

                    "gt_id":
                        gt.object_id,

                    "pred_id":
                        "",

                    "iou":
                        "",

                    "matched":
                        0,
                }
            )

    # =========================================================================
    # Detection metrics
    # =========================================================================

    precision = (
        tp
        /
        (
            tp + fp
        )
        if (
            tp + fp
        ) > 0
        else 0.0
    )

    recall = (
        tp
        /
        (
            tp + fn
        )
        if (
            tp + fn
        ) > 0
        else 0.0
    )

    mota = (
        1.0
        -
        (
            fn
            +
            fp
            +
            id_switches
        )
        /
        total_gt
        if total_gt > 0
        else 0.0
    )

    # =========================================================================
    # IDF1 global identity matching
    # =========================================================================

    gt_ids = sorted(
        {
            gt.object_id

            for frame_boxes
            in gt_by_frame.values()

            for gt
            in frame_boxes
        }
    )

    pred_ids = sorted(
        {
            pred.object_id

            for frame_boxes
            in pred_by_frame.values()

            for pred
            in frame_boxes
        }
    )

    if (
        gt_ids
        and
        pred_ids
    ):

        gt_index_map = {
            gt_id: index
            for index, gt_id
            in enumerate(
                gt_ids
            )
        }

        pred_index_map = {
            pred_id: index
            for index, pred_id
            in enumerate(
                pred_ids
            )
        }

        identity_matrix = np.zeros(
            (
                len(gt_ids),
                len(pred_ids),
            ),
            dtype=np.float64,
        )

        for (
            gt_id,
            pred_id,
        ), count in (
            pair_match_counts.items()
        ):

            identity_matrix[
                gt_index_map[
                    gt_id
                ],
                pred_index_map[
                    pred_id
                ],
            ] = count

        row_indices, column_indices = (
            linear_sum_assignment(
                -identity_matrix
            )
        )

        idtp = int(
            sum(
                identity_matrix[
                    row,
                    column,
                ]

                for row, column
                in zip(
                    row_indices,
                    column_indices,
                )
            )
        )

    else:

        idtp = 0

    idfn = (
        total_gt
        -
        idtp
    )

    idfp = (
        total_pred
        -
        idtp
    )

    idp = (
        idtp
        /
        (
            idtp + idfp
        )
        if (
            idtp + idfp
        ) > 0
        else 0.0
    )

    idr = (
        idtp
        /
        (
            idtp + idfn
        )
        if (
            idtp + idfn
        ) > 0
        else 0.0
    )

    idf1 = (
        2
        *
        idtp
        /
        (
            2 * idtp
            +
            idfp
            +
            idfn
        )
        if (
            2 * idtp
            +
            idfp
            +
            idfn
        ) > 0
        else 0.0
    )

    unique_gt = len(
        gt_ids
    )

    unique_pred = len(
        pred_ids
    )

    result = {
        "model": name,

        "iou_threshold":
            iou_threshold,

        "gt_detections":
            total_gt,

        "prediction_detections":
            total_pred,

        "unique_gt_ids":
            unique_gt,

        "unique_pred_ids":
            unique_pred,

        "TP":
            tp,

        "FP":
            fp,

        "FN":
            fn,

        "precision":
            precision,

        "recall":
            recall,

        "IDTP":
            idtp,

        "IDFP":
            idfp,

        "IDFN":
            idfn,

        "IDP":
            idp,

        "IDR":
            idr,

        "IDF1":
            idf1,

        "ID_switches":
            id_switches,

        "fragmentations":
            fragmentations,

        "MOTA":
            mota,
    }

    return (
        result,
        per_frame_rows,
    )


# =============================================================================
# PRINT RESULT
# =============================================================================

def print_comparison(
    rows,
    threshold,
):

    selected = [
        row
        for row in rows
        if math.isclose(
            row[
                "iou_threshold"
            ],
            threshold,
            abs_tol=1e-6,
        )
    ]

    print()
    print("=" * 125)

    print(
        f"FINAL TRACKING COMPARISON "
        f"@ IoU {threshold:.2f}"
    )

    print("=" * 125)

    header = (
        f"{'MODEL':<20}"
        f"{'IDF1':>9}"
        f"{'IDP':>9}"
        f"{'IDR':>9}"
        f"{'IDSW':>9}"
        f"{'FRAG':>9}"
        f"{'MOTA':>9}"
        f"{'PREC':>9}"
        f"{'RECALL':>9}"
        f"{'PRED ID':>10}"
    )

    print(
        header
    )

    print(
        "-" * len(
            header
        )
    )

    for row in selected:

        print(
            f"{row['model']:<20}"

            f"{row['IDF1']:>9.4f}"
            f"{row['IDP']:>9.4f}"
            f"{row['IDR']:>9.4f}"

            f"{row['ID_switches']:>9}"
            f"{row['fragmentations']:>9}"

            f"{row['MOTA']:>9.4f}"
            f"{row['precision']:>9.4f}"
            f"{row['recall']:>9.4f}"

            f"{row['unique_pred_ids']:>10}"
        )


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 80)
    print("TASK 90 GT TRACKING EVALUATION")
    print("=" * 80)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------------------
    # File checks
    # -------------------------------------------------------------------------

    for name, path in [
        (
            "VIDEO",
            VIDEO_PATH,
        ),
        (
            "GT ZIP",
            GT_ZIP_PATH,
        ),
        (
            "BASE TRACKER",
            BASE_TRACKING_SCRIPT,
        ),
    ]:

        print()
        print(
            name,
            ":",
            path,
        )

        if not path.is_file():

            raise FileNotFoundError(
                f"{name} 파일 없음:\n{path}"
            )

    # -------------------------------------------------------------------------
    # Video
    # -------------------------------------------------------------------------

    (
        fps,
        width,
        height,
        total_frames,
    ) = get_video_info(
        VIDEO_PATH
    )

    print()
    print(
        "FPS:",
        fps,
    )

    print(
        "Frames:",
        total_frames,
    )

    print(
        "Resolution:",
        f"{width}x{height}",
    )

    if abs(
        fps - 10.0
    ) > 0.1:

        print()
        print(
            "[WARN] 영상 FPS가 10이 아닙니다."
        )

    # -------------------------------------------------------------------------
    # CVAT GT
    # -------------------------------------------------------------------------

    gt_by_frame = (
        parse_cvat_gt(
            GT_ZIP_PATH
        )
    )

    # -------------------------------------------------------------------------
    # Tracking module
    # -------------------------------------------------------------------------

    module = (
        load_tracking_module()
    )

    module.SOURCE = (
        VIDEO_PATH
    )

    module.OUTPUT_ROOT = (
        OUTPUT_DIR
    )

    module.TRACKER_10FPS = (
        TRACKER_YAML
    )

    module.REID_MIN_SIMILARITY = (
        REID_THRESHOLD
    )

    # -------------------------------------------------------------------------
    # Create FPS-correct ByteTrack yaml
    # -------------------------------------------------------------------------

    module.create_tracker_yaml(
        fps
    )

    # -------------------------------------------------------------------------
    # V10
    # -------------------------------------------------------------------------

    (
        v10_predictions,
        raw_predictions,
    ) = run_tracking(
        module=module,
        mode="V10_GEOMETRY",
        fps=fps,
        width=width,
        height=height,
        total_frames=total_frames,
    )

    # -------------------------------------------------------------------------
    # V11 0.84
    # -------------------------------------------------------------------------

    (
        v11_predictions,
        _,
    ) = run_tracking(
        module=module,
        mode="V11_REID_084",
        fps=fps,
        width=width,
        height=height,
        total_frames=total_frames,
    )

    # -------------------------------------------------------------------------
    # Evaluation
    # -------------------------------------------------------------------------

    all_results = []
    all_matches = []

    for threshold in [
        IOU_MATCH_THRESHOLD,
        SECONDARY_IOU_THRESHOLD,
    ]:

        for (
            model_name,
            predictions,
        ) in [
            (
                "Raw_ByteTrack",
                raw_predictions,
            ),
            (
                "V10_Geometry",
                v10_predictions,
            ),
            (
                "V11_ReID_0.84",
                v11_predictions,
            ),
        ]:

            result, matches = (
                evaluate_tracking(
                    name=model_name,
                    gt_by_frame=gt_by_frame,
                    pred_by_frame=predictions,
                    total_frames=total_frames,
                    iou_threshold=threshold,
                )
            )

            all_results.append(
                result
            )

            for row in matches:

                row[
                    "iou_threshold"
                ] = threshold

            all_matches.extend(
                matches
            )

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------

    write_dict_csv(
        OUTPUT_DIR
        /
        "tracking_metrics.csv",
        all_results,
    )

    write_dict_csv(
        OUTPUT_DIR
        /
        "frame_gt_prediction_matches.csv",
        all_matches,
    )

    # -------------------------------------------------------------------------
    # V10 → V11 improvement
    # -------------------------------------------------------------------------

    main_rows = {
        row["model"]: row

        for row in all_results

        if math.isclose(
            row[
                "iou_threshold"
            ],
            IOU_MATCH_THRESHOLD,
            abs_tol=1e-6,
        )
    }

    v10 = main_rows.get(
        "V10_Geometry"
    )

    v11 = main_rows.get(
        "V11_ReID_0.84"
    )

    if (
        v10 is not None
        and
        v11 is not None
    ):

        improvement = [
            {
                "comparison":
                    "V11_ReID_0.84_vs_V10",

                "IDF1_absolute_change":
                    v11["IDF1"]
                    -
                    v10["IDF1"],

                "IDF1_relative_percent":
                    (
                        (
                            v11["IDF1"]
                            -
                            v10["IDF1"]
                        )
                        /
                        v10["IDF1"]
                        *
                        100
                    )
                    if v10["IDF1"] > 0
                    else 0.0,

                "ID_switch_change":
                    v11["ID_switches"]
                    -
                    v10["ID_switches"],

                "ID_switch_reduction_percent":
                    (
                        (
                            v10["ID_switches"]
                            -
                            v11["ID_switches"]
                        )
                        /
                        v10["ID_switches"]
                        *
                        100
                    )
                    if (
                        v10["ID_switches"]
                        >
                        0
                    )
                    else 0.0,

                "fragmentation_change":
                    v11["fragmentations"]
                    -
                    v10["fragmentations"],

                "fragmentation_reduction_percent":
                    (
                        (
                            v10["fragmentations"]
                            -
                            v11["fragmentations"]
                        )
                        /
                        v10["fragmentations"]
                        *
                        100
                    )
                    if (
                        v10["fragmentations"]
                        >
                        0
                    )
                    else 0.0,

                "MOTA_change":
                    v11["MOTA"]
                    -
                    v10["MOTA"],
            }
        ]

        write_dict_csv(
            OUTPUT_DIR
            /
            "v10_vs_v11_improvement.csv",
            improvement,
        )

    # -------------------------------------------------------------------------
    # Console result
    # -------------------------------------------------------------------------

    print_comparison(
        all_results,
        IOU_MATCH_THRESHOLD,
    )

    print_comparison(
        all_results,
        SECONDARY_IOU_THRESHOLD,
    )

    print()
    print("=" * 80)
    print("DONE")
    print("=" * 80)

    print(
        "결과 폴더:"
    )

    print(
        OUTPUT_DIR
    )

    print()
    print(
        "중요 결과:"
    )

    print(
        OUTPUT_DIR
        /
        "tracking_metrics.csv"
    )

    print(
        OUTPUT_DIR
        /
        "v10_vs_v11_improvement.csv"
    )


if __name__ == "__main__":
    main()
