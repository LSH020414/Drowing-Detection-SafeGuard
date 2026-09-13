from pathlib import Path
import re
import math

import cv2
import torch
from ultralytics import YOLO


# =============================================================================
# ROOT
# =============================================================================

IMAGE_ROOT = Path(
    r"D:\임베디드 경진대회\머리 추적\SWIM2_CVAT_SINGLE_10FPS\images"
)


# =============================================================================
# TEST STROKES
# =============================================================================

# Breaststroke는 이미 테스트했으므로 제외
STROKES = [
    "Backstroke",
    "Freestyle",
    "Butterfly",
]


# =============================================================================
# MODELS
# =============================================================================

PERSON_MODEL = "yolo11s.pt"

HEAD_MODEL = Path(
    r"D:\임베디드 경진대회\머리 추적\models\pool_head_best.pt"
)

HEAD_TRACKER = Path(
    r"C:\Users\이승희\Desktop\bytetrack_pool.yaml"
)


# =============================================================================
# OUTPUT
# =============================================================================

OUTPUT_ROOT = Path(
    r"D:\임베디드 경진대회\머리 추적\SWIM2_RETRACK_TEST\THREE_STROKES_V2"
)


# =============================================================================
# VIDEO
# =============================================================================

FPS = 10.0


# =============================================================================
# PERSON DETECTOR
# =============================================================================

PERSON_IMGSZ = 1280

PERSON_CONF = 0.08

PERSON_IOU = 0.60


# =============================================================================
# HEAD DETECTOR
# =============================================================================

HEAD_IMGSZ = 960

HEAD_CONF = 0.03

HEAD_IOU = 0.70


# =============================================================================
# PERSON ROI
# =============================================================================

EXPAND_X = 0.70

EXPAND_TOP = 0.80

EXPAND_BOTTOM = 0.50


# person 검출 잠깐 끊겨도
# 이전 ROI 1.5초 유지
ROI_MEMORY_FRAMES = 15


# =============================================================================
# HEAD FILTER
# =============================================================================

MIN_HEAD_PIXEL = 18.0

MAX_HEAD_PIXEL = 350.0


# head area / person area
MIN_HEAD_AREA_RATIO = 0.015

MAX_HEAD_AREA_RATIO = 0.22


# person box 기준 머리 중심의 최대 상대 Y
MAX_HEAD_RELATIVE_Y = 0.48


# 예상 머리 중심 위치
EXPECTED_HEAD_Y_RATIO = 0.28


# 위치 점수 penalty
HEAD_DISTANCE_WEIGHT = 1.5


# =============================================================================
# IMAGE
# =============================================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


# =============================================================================
# HELPERS
# =============================================================================

def natural_key(value):

    if isinstance(value, Path):
        text = value.name
    else:
        text = str(value)

    return [
        int(x)
        if x.isdigit()
        else x.lower()

        for x in re.split(
            r"(\d+)",
            text
        )
    ]


def clamp(
    value,
    minimum,
    maximum,
):

    return max(
        minimum,
        min(
            value,
            maximum
        )
    )


def find_images(
    folder: Path,
):

    return sorted(
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


def expand_person_box(
    box,
    frame_w,
    frame_h,
):

    x1, y1, x2, y2 = [
        float(v)
        for v in box
    ]

    w = max(
        1.0,
        x2 - x1
    )

    h = max(
        1.0,
        y2 - y1
    )

    ex1 = (
        x1
        -
        w
        *
        EXPAND_X
    )

    ex2 = (
        x2
        +
        w
        *
        EXPAND_X
    )

    ey1 = (
        y1
        -
        h
        *
        EXPAND_TOP
    )

    ey2 = (
        y2
        +
        h
        *
        EXPAND_BOTTOM
    )

    return (
        int(
            clamp(
                ex1,
                0,
                frame_w - 1
            )
        ),

        int(
            clamp(
                ey1,
                0,
                frame_h - 1
            )
        ),

        int(
            clamp(
                ex2,
                1,
                frame_w
            )
        ),

        int(
            clamp(
                ey2,
                1,
                frame_h
            )
        ),
    )


def calculate_head_scale(
    x1,
    y1,
    x2,
    y2,
):

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
# AUTO SELECT MIDDLE VIDEO
# =============================================================================

def select_middle_folder(
    stroke,
):

    candidates = []

    for folder in IMAGE_ROOT.iterdir():

        if not folder.is_dir():
            continue

        name_lower = (
            folder.name.lower()
        )

        if (
            stroke.lower()
            in
            name_lower
        ):

            candidates.append(
                folder
            )

    candidates = sorted(
        candidates,
        key=natural_key,
    )

    if not candidates:

        raise RuntimeError(
            f"{stroke} 폴더를 찾지 못했습니다."
        )

    middle_index = (
        len(candidates)
        //
        2
    )

    selected = (
        candidates[
            middle_index
        ]
    )

    print()
    print(
        f"[{stroke}]"
    )

    print(
        f"TOTAL VIDEOS : {len(candidates)}"
    )

    print(
        f"SELECT INDEX : {middle_index + 1}/{len(candidates)}"
    )

    print(
        f"SELECTED     : {selected.name}"
    )

    return selected


# =============================================================================
# PROCESS ONE STROKE
# =============================================================================

def process_video(
    stroke,
    image_folder,
    person_model,
    head_model,
    device,
):

    print()
    print(
        "=" * 100
    )

    print(
        f"{stroke.upper()} TEST"
    )

    print(
        "=" * 100
    )

    print(
        f"FOLDER : {image_folder}"
    )

    images = (
        find_images(
            image_folder
        )
    )

    print(
        f"IMAGES : {len(images)}"
    )

    if not images:

        print(
            "[SKIP] 이미지 없음"
        )

        return

    first = cv2.imread(
        str(
            images[0]
        )
    )

    if first is None:

        print(
            "[SKIP] 첫 이미지 읽기 실패"
        )

        return

    frame_h, frame_w = (
        first.shape[
            :2
        ]
    )

    print(
        f"FRAME  : {frame_w} x {frame_h}"
    )


    # =========================================================================
    # OUTPUT
    # =========================================================================

    output_path = (
        OUTPUT_ROOT
        /
        f"{stroke}_cascade_v2_preview.mp4"
    )

    fourcc = (
        cv2.VideoWriter_fourcc(
            *"mp4v"
        )
    )

    writer = cv2.VideoWriter(
        str(
            output_path
        ),

        fourcc,

        FPS,

        (
            frame_w,
            frame_h
        ),
    )


    # =========================================================================
    # RESET HEAD TRACKER
    # =========================================================================

    head_model.predictor = None


    # =========================================================================
    # STATE
    # =========================================================================

    last_person_roi = None

    last_person_frame = -9999


    # 마지막 person box도 저장
    #
    # person detector가 잠깐 끊겼을 때
    # head filtering 기준으로 사용할 수 있도록 함.
    last_person_box = None


    # =========================================================================
    # STATS
    # =========================================================================

    total_person_frames = 0

    total_head_raw = 0

    total_head_filtered = 0

    total_head_final = 0

    head_ids = set()


    # =========================================================================
    # FRAME LOOP
    # =========================================================================

    for frame_idx, image_path in enumerate(
        images
    ):

        frame = cv2.imread(
            str(
                image_path
            )
        )

        if frame is None:
            continue

        preview = frame.copy()


        # =====================================================================
        # 1. PERSON DETECTION
        # =====================================================================

        person_result = (
            person_model.predict(
                source=frame,

                imgsz=PERSON_IMGSZ,

                conf=PERSON_CONF,

                iou=PERSON_IOU,

                classes=[0],

                device=device,

                verbose=False,
            )[0]
        )


        person_candidates = []


        if (
            person_result.boxes
            is not None
            and
            len(
                person_result.boxes
            ) > 0
        ):

            person_xyxy = (
                person_result.boxes.xyxy
                .detach()
                .cpu()
                .numpy()
            )

            person_confs = (
                person_result.boxes.conf
                .detach()
                .cpu()
                .tolist()
            )


            for (
                xyxy,
                confidence,
            ) in zip(
                person_xyxy,
                person_confs,
            ):

                px1, py1, px2, py2 = [
                    float(v)
                    for v in xyxy
                ]


                person_cy = (
                    py1
                    +
                    py2
                ) / 2.0


                # 너무 위쪽 배경 제거
                if (
                    person_cy
                    <
                    frame_h
                    *
                    0.45
                ):

                    continue


                person_candidates.append(
                    (
                        float(
                            confidence
                        ),

                        px1,
                        py1,
                        px2,
                        py2,
                    )
                )


        # =====================================================================
        # 2. MAIN SWIMMER
        # =====================================================================

        selected_person = None


        if person_candidates:

            selected_person = max(
                person_candidates,
                key=lambda x:
                    x[0]
            )


        if (
            selected_person
            is not None
        ):

            (
                person_conf,
                px1,
                py1,
                px2,
                py2,
            ) = selected_person


            last_person_box = (
                px1,
                py1,
                px2,
                py2,
            )


            last_person_roi = (
                expand_person_box(
                    last_person_box,

                    frame_w,
                    frame_h,
                )
            )


            last_person_frame = (
                frame_idx
            )


            total_person_frames += 1


            # -------------------------------------------------------------
            # GREEN = PERSON
            # -------------------------------------------------------------

            cv2.rectangle(
                preview,

                (
                    int(
                        px1
                    ),
                    int(
                        py1
                    )
                ),

                (
                    int(
                        px2
                    ),
                    int(
                        py2
                    )
                ),

                (
                    0,
                    255,
                    0
                ),

                3,
            )


            cv2.putText(
                preview,

                f"PERSON {person_conf:.2f}",

                (
                    int(
                        px1
                    ),

                    max(
                        30,
                        int(
                            py1
                        )
                        -
                        10
                    )
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.75,

                (
                    0,
                    255,
                    0
                ),

                2,

                cv2.LINE_AA,
            )


        # =====================================================================
        # 3. ROI MEMORY
        # =====================================================================

        use_roi = None

        use_person_box = None


        if (
            last_person_roi
            is not None
            and
            (
                frame_idx
                -
                last_person_frame
            )
            <=
            ROI_MEMORY_FRAMES
        ):

            use_roi = (
                last_person_roi
            )

            use_person_box = (
                last_person_box
            )


        # =====================================================================
        # 4. HEAD DETECTOR
        # =====================================================================

        if (
            use_roi
            is not None
            and
            use_person_box
            is not None
        ):

            (
                rx1,
                ry1,
                rx2,
                ry2,
            ) = use_roi


            (
                px1,
                py1,
                px2,
                py2,
            ) = use_person_box


            roi = frame[
                ry1:ry2,
                rx1:rx2
            ]


            if (
                roi.shape[0] > 40
                and
                roi.shape[1] > 40
            ):

                # -------------------------------------------------------------
                # MAGENTA = HEAD SEARCH ROI
                # -------------------------------------------------------------

                cv2.rectangle(
                    preview,

                    (
                        rx1,
                        ry1
                    ),

                    (
                        rx2,
                        ry2
                    ),

                    (
                        255,
                        0,
                        255
                    ),

                    2,
                )


                # -------------------------------------------------------------
                # HEAD DETECTOR + BYTETRACK
                # -------------------------------------------------------------

                head_result = (
                    head_model.track(
                        source=roi,

                        persist=True,

                        tracker=str(
                            HEAD_TRACKER
                        ),

                        imgsz=HEAD_IMGSZ,

                        conf=HEAD_CONF,

                        iou=HEAD_IOU,

                        classes=[0],

                        device=device,

                        verbose=False,
                    )[0]
                )


                boxes = (
                    head_result.boxes
                )


                if (
                    boxes is not None
                    and
                    len(
                        boxes
                    ) > 0
                ):

                    total_head_raw += (
                        len(
                            boxes
                        )
                    )


                    head_xyxy = (
                        boxes.xyxy
                        .detach()
                        .cpu()
                        .numpy()
                    )


                    head_confs = (
                        boxes.conf
                        .detach()
                        .cpu()
                        .tolist()
                    )


                    if (
                        boxes.id
                        is not None
                    ):

                        head_track_ids = (
                            boxes.id
                            .int()
                            .detach()
                            .cpu()
                            .tolist()
                        )

                    else:

                        head_track_ids = [
                            -1
                            for _ in range(
                                len(
                                    head_xyxy
                                )
                            )
                        ]


                    # =========================================================
                    # PERSON GEOMETRY
                    # =========================================================

                    person_w = max(
                        1.0,
                        px2 - px1
                    )

                    person_h = max(
                        1.0,
                        py2 - py1
                    )

                    person_area = (
                        person_w
                        *
                        person_h
                    )


                    expected_head_x = (
                        px1
                        +
                        px2
                    ) / 2.0


                    expected_head_y = (
                        py1
                        +
                        person_h
                        *
                        EXPECTED_HEAD_Y_RATIO
                    )


                    # =========================================================
                    # HEAD CANDIDATES
                    # =========================================================

                    candidates = []


                    for (
                        track_id,
                        xyxy,
                        confidence,
                    ) in zip(
                        head_track_ids,
                        head_xyxy,
                        head_confs,
                    ):

                        if (
                            track_id
                            <
                            0
                        ):

                            continue


                        hx1, hy1, hx2, hy2 = [
                            float(v)
                            for v in xyxy
                        ]


                        # -----------------------------------------------------
                        # ROI → ORIGINAL
                        # -----------------------------------------------------

                        ox1_f = (
                            hx1
                            +
                            rx1
                        )

                        oy1_f = (
                            hy1
                            +
                            ry1
                        )

                        ox2_f = (
                            hx2
                            +
                            rx1
                        )

                        oy2_f = (
                            hy2
                            +
                            ry1
                        )


                        head_w = max(
                            1.0,
                            ox2_f
                            -
                            ox1_f
                        )

                        head_h = max(
                            1.0,
                            oy2_f
                            -
                            oy1_f
                        )


                        head_area = (
                            head_w
                            *
                            head_h
                        )


                        head_cx = (
                            ox1_f
                            +
                            ox2_f
                        ) / 2.0


                        head_cy = (
                            oy1_f
                            +
                            oy2_f
                        ) / 2.0


                        # =====================================================
                        # FILTER 1:
                        # ABSOLUTE SIZE
                        # =====================================================

                        scale = (
                            calculate_head_scale(
                                ox1_f,
                                oy1_f,
                                ox2_f,
                                oy2_f,
                            )
                        )


                        if (
                            scale
                            <
                            MIN_HEAD_PIXEL
                            or
                            scale
                            >
                            MAX_HEAD_PIXEL
                        ):

                            continue


                        # =====================================================
                        # FILTER 2:
                        # HEAD/PERSON AREA
                        # =====================================================

                        area_ratio = (
                            head_area
                            /
                            person_area
                        )


                        if not (
                            MIN_HEAD_AREA_RATIO
                            <=
                            area_ratio
                            <=
                            MAX_HEAD_AREA_RATIO
                        ):

                            continue


                        # =====================================================
                        # FILTER 3:
                        # POSITION
                        # =====================================================

                        relative_y = (
                            head_cy
                            -
                            py1
                        ) / person_h


                        if (
                            relative_y
                            >
                            MAX_HEAD_RELATIVE_Y
                        ):

                            continue


                        total_head_filtered += 1


                        # =====================================================
                        # SCORE
                        # =====================================================

                        distance = math.hypot(
                            head_cx
                            -
                            expected_head_x,

                            head_cy
                            -
                            expected_head_y,
                        )


                        normalized_distance = (
                            distance
                            /
                            max(
                                person_w,
                                person_h,
                                1.0
                            )
                        )


                        score = (
                            float(
                                confidence
                            )
                            -
                            HEAD_DISTANCE_WEIGHT
                            *
                            normalized_distance
                        )


                        candidates.append(
                            (
                                score,

                                float(
                                    confidence
                                ),

                                int(
                                    track_id
                                ),

                                hx1,
                                hy1,
                                hx2,
                                hy2,
                            )
                        )


                    # =========================================================
                    # FINAL: ONE HEAD ONLY
                    # =========================================================

                    candidates.sort(
                        key=lambda x:
                            x[0],
                        reverse=True,
                    )


                    if candidates:

                        (
                            score,
                            confidence,
                            track_id,
                            hx1,
                            hy1,
                            hx2,
                            hy2,
                        ) = (
                            candidates[0]
                        )


                        total_head_final += 1


                        head_ids.add(
                            track_id
                        )


                        ox1 = int(
                            hx1
                            +
                            rx1
                        )

                        oy1 = int(
                            hy1
                            +
                            ry1
                        )

                        ox2 = int(
                            hx2
                            +
                            rx1
                        )

                        oy2 = int(
                            hy2
                            +
                            ry1
                        )


                        # -----------------------------------------------------
                        # ORANGE = FINAL HEAD
                        # -----------------------------------------------------

                        cv2.rectangle(
                            preview,

                            (
                                ox1,
                                oy1
                            ),

                            (
                                ox2,
                                oy2
                            ),

                            (
                                0,
                                165,
                                255
                            ),

                            4,
                        )


                        cv2.putText(
                            preview,

                            (
                                f"HEAD ID {track_id} "
                                f"{confidence:.2f}"
                            ),

                            (
                                ox1,

                                max(
                                    30,
                                    oy1 - 10
                                )
                            ),

                            cv2.FONT_HERSHEY_SIMPLEX,

                            0.75,

                            (
                                0,
                                165,
                                255
                            ),

                            2,

                            cv2.LINE_AA,
                        )


        # =====================================================================
        # FRAME INFO
        # =====================================================================

        cv2.putText(
            preview,

            f"{stroke} | Frame {frame_idx}",

            (
                30,
                50
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            1.0,

            (
                255,
                255,
                255
            ),

            2,

            cv2.LINE_AA,
        )


        cv2.putText(
            preview,

            "GREEN=PERSON  MAGENTA=ROI  ORANGE=FINAL HEAD",

            (
                30,
                90
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.75,

            (
                255,
                255,
                255
            ),

            2,

            cv2.LINE_AA,
        )


        writer.write(
            preview
        )


        if (
            frame_idx
            %
            50
            ==
            0
        ):

            print(
                f"[{frame_idx}/{len(images)}]"
            )


    # =========================================================================
    # FINISH
    # =========================================================================

    writer.release()


    print()
    print(
        f"[{stroke} RESULT]"
    )

    print(
        f"IMAGES            : {len(images)}"
    )

    print(
        f"PERSON FRAMES     : {total_person_frames}"
    )

    print(
        f"PERSON RATIO      : "
        f"{total_person_frames / len(images):.3f}"
    )

    print(
        f"HEAD RAW          : {total_head_raw}"
    )

    print(
        f"HEAD FILTERED     : {total_head_filtered}"
    )

    print(
        f"FINAL HEAD FRAMES : {total_head_final}"
    )

    print(
        f"FINAL HEAD RATIO  : "
        f"{total_head_final / len(images):.3f}"
    )

    print(
        f"HEAD TRACK IDS    : {len(head_ids)}"
    )

    print(
        f"OUTPUT            : {output_path}"
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    print(
        "=" * 100
    )

    print(
        "SWIM2 THREE STROKES CASCADE V2 TEST"
    )

    print(
        "=" * 100
    )

    if not IMAGE_ROOT.exists():

        raise FileNotFoundError(
            f"IMAGE ROOT 없음:\n{IMAGE_ROOT}"
        )

    if not HEAD_MODEL.exists():

        raise FileNotFoundError(
            f"HEAD MODEL 없음:\n{HEAD_MODEL}"
        )

    if not HEAD_TRACKER.exists():

        raise FileNotFoundError(
            f"TRACKER 없음:\n{HEAD_TRACKER}"
        )


    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )


    device = (
        0
        if torch.cuda.is_available()
        else "cpu"
    )


    print(
        f"DEVICE : {device}"
    )


    # =========================================================================
    # SELECT TEST VIDEOS
    # =========================================================================

    selected_folders = {}


    for stroke in STROKES:

        selected_folders[
            stroke
        ] = (
            select_middle_folder(
                stroke
            )
        )


    print()
    print(
        "=" * 100
    )

    print(
        "SELECTED TEST VIDEOS"
    )

    print(
        "=" * 100
    )


    for stroke, folder in selected_folders.items():

        print(
            f"{stroke:12s} : {folder.name}"
        )


    # =========================================================================
    # LOAD MODELS ONCE
    # =========================================================================

    print()
    print(
        "Loading person model..."
    )

    person_model = YOLO(
        PERSON_MODEL
    )


    print(
        "Loading head model..."
    )

    head_model = YOLO(
        str(
            HEAD_MODEL
        )
    )


    # =========================================================================
    # RUN THREE STROKES
    # =========================================================================

    for stroke in STROKES:

        process_video(
            stroke=stroke,

            image_folder=selected_folders[
                stroke
            ],

            person_model=person_model,

            head_model=head_model,

            device=device,
        )


    print()
    print(
        "=" * 100
    )

    print(
        "ALL TESTS FINISHED"
    )

    print(
        "=" * 100
    )

    print(
        f"OUTPUT ROOT:\n{OUTPUT_ROOT}"
    )


if __name__ == "__main__":

    main()
