from pathlib import Path
import re
import math

import cv2
import torch

from ultralytics import YOLO


# =============================================================================
# INPUT
# =============================================================================

IMAGE_FOLDER = Path(
    r"D:\임베디드 경진대회\머리 추적\SWIM2_CVAT_SINGLE_10FPS\images\0039_Breaststroke_Above_DIVE_P018_BREAST_ABOVE_20250630_003._SWIM2.MP4"
)


# =============================================================================
# MODELS
# =============================================================================

# 기본 YOLO:
# 수영자 위치를 대략 찾는 용도
PERSON_MODEL = "yolo11s.pt"


# 우리가 만든 머리 detector
HEAD_MODEL = Path(
    r"D:\임베디드 경진대회\머리 추적\models\pool_head_best.pt"
)


# 기존 ByteTrack 설정
HEAD_TRACKER = Path(
    r"C:\Users\이승희\Desktop\bytetrack_pool.yaml"
)


# =============================================================================
# OUTPUT
# =============================================================================

OUTPUT_PATH = Path(
    r"D:\임베디드 경진대회\머리 추적\SWIM2_RETRACK_TEST\0039_person_head_cascade_v2_preview.mp4"
)


# =============================================================================
# VIDEO
# =============================================================================

FPS = 10.0


# =============================================================================
# PERSON DETECTOR SETTINGS
# =============================================================================

PERSON_IMGSZ = 1280

# recall 우선
PERSON_CONF = 0.08

PERSON_IOU = 0.60


# =============================================================================
# HEAD DETECTOR SETTINGS
# =============================================================================

HEAD_IMGSZ = 960

# Swim2에서 domain mismatch가 있으므로 낮게 유지
HEAD_CONF = 0.03

HEAD_IOU = 0.70


# =============================================================================
# PERSON ROI EXPANSION
# =============================================================================

# 사람 박스가 수영 자세에서 매우 작거나 찌그러질 수 있으므로
# 주변을 넓혀 head detector에게 보여준다.

EXPAND_X = 0.70

EXPAND_TOP = 0.80

EXPAND_BOTTOM = 0.50


# =============================================================================
# PERSON ROI MEMORY
# =============================================================================

# person detector가 잠시 수영자를 놓쳐도
# 이전 ROI에서 head detector를 계속 돌림.
#
# 10fps 기준 15 frame = 1.5초
ROI_MEMORY_FRAMES = 15


# =============================================================================
# HEAD SIZE FILTER
# =============================================================================

# 절대 박스 크기
# sqrt(width * height)

MIN_HEAD_PIXEL = 18.0

MAX_HEAD_PIXEL = 350.0


# =============================================================================
# HEAD / PERSON RELATION FILTER
# =============================================================================

# head box 면적 / person box 면적
#
# 몸통이나 상체 전체가 head로 잡히는 것을 제거
MIN_HEAD_AREA_RATIO = 0.015

MAX_HEAD_AREA_RATIO = 0.22


# person box의 위에서 몇 %까지를
# 머리 후보 허용 영역으로 볼 것인지
#
# 0.48 = 상단 약 48%
MAX_HEAD_RELATIVE_Y = 0.48


# person 박스 내에서 예상 머리 위치
#
# 일반 직립 사람처럼 맨 위가 아니라
# 수영자 자세를 고려해 28% 지점
EXPECTED_HEAD_Y_RATIO = 0.28


# 예상 머리 위치와 거리 penalty
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

def natural_key(path: Path):

    return [
        int(x)
        if x.isdigit()
        else x.lower()

        for x in re.split(
            r"(\d+)",
            path.name
        )
    ]


def find_images():

    return sorted(
        [
            p
            for p in IMAGE_FOLDER.iterdir()

            if (
                p.is_file()
                and
                p.suffix.lower()
                in IMAGE_EXTENSIONS
            )
        ],
        key=natural_key,
    )


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

    ex1 = int(
        clamp(
            ex1,
            0,
            frame_w - 1
        )
    )

    ex2 = int(
        clamp(
            ex2,
            1,
            frame_w
        )
    )

    ey1 = int(
        clamp(
            ey1,
            0,
            frame_h - 1
        )
    )

    ey2 = int(
        clamp(
            ey2,
            1,
            frame_h
        )
    )

    return (
        ex1,
        ey1,
        ex2,
        ey2,
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
# MAIN
# =============================================================================

def main():

    print(
        "=" * 80
    )

    print(
        "SWIM2 PERSON -> HEAD CASCADE V2"
    )

    print(
        "=" * 80
    )

    # =========================================================================
    # CHECK
    # =========================================================================

    if not IMAGE_FOLDER.exists():

        raise FileNotFoundError(
            f"IMAGE FOLDER 없음:\n{IMAGE_FOLDER}"
        )

    if not HEAD_MODEL.exists():

        raise FileNotFoundError(
            f"HEAD MODEL 없음:\n{HEAD_MODEL}"
        )

    if not HEAD_TRACKER.exists():

        raise FileNotFoundError(
            f"TRACKER 없음:\n{HEAD_TRACKER}"
        )

    images = find_images()

    print(
        f"IMAGES : {len(images)}"
    )

    if not images:

        raise RuntimeError(
            "이미지가 없습니다."
        )

    first = cv2.imread(
        str(
            images[0]
        )
    )

    if first is None:

        raise RuntimeError(
            "첫 이미지 읽기 실패"
        )

    frame_h, frame_w = (
        first.shape[
            :2
        ]
    )

    print(
        f"FRAME : {frame_w} x {frame_h}"
    )


    # =========================================================================
    # DEVICE
    # =========================================================================

    device = (
        0
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"DEVICE : {device}"
    )


    # =========================================================================
    # LOAD MODELS
    # =========================================================================

    print(
        f"PERSON MODEL : {PERSON_MODEL}"
    )

    print(
        f"HEAD MODEL   : {HEAD_MODEL}"
    )

    person_model = YOLO(
        PERSON_MODEL
    )

    head_model = YOLO(
        str(
            HEAD_MODEL
        )
    )

    # tracker 초기화
    head_model.predictor = None


    # =========================================================================
    # OUTPUT
    # =========================================================================

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        str(
            OUTPUT_PATH
        ),
        fourcc,
        FPS,
        (
            frame_w,
            frame_h
        ),
    )


    # =========================================================================
    # STATE
    # =========================================================================

    last_person_roi = None

    last_person_frame = -9999


    # =========================================================================
    # STATISTICS
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

            print(
                f"[WARNING] 읽기 실패: {image_path}"
            )

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


        # =====================================================================
        # PERSON BOX COLLECTION
        # =====================================================================

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
                    py1 + py2
                ) / 2.0


                # -------------------------------------------------------------
                # 너무 위쪽 배경 사람 제거
                # -------------------------------------------------------------

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
        # 2. SELECT MAIN SWIMMER
        # =====================================================================

        selected_person = None

        if person_candidates:

            # 현재는 가장 confidence 높은 person
            selected_person = max(
                person_candidates,
                key=lambda x:
                    x[0]
            )


        # person coordinates
        person_conf = None

        px1 = None
        py1 = None
        px2 = None
        py2 = None


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


            # -------------------------------------------------------------
            # 확대 ROI 생성
            # -------------------------------------------------------------

            last_person_roi = (
                expand_person_box(
                    (
                        px1,
                        py1,
                        px2,
                        py2,
                    ),

                    frame_w,
                    frame_h,
                )
            )

            last_person_frame = (
                frame_idx
            )

            total_person_frames += 1


            # -------------------------------------------------------------
            # PERSON BOX = GREEN
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


        # =====================================================================
        # 4. HEAD DETECTION
        # =====================================================================

        if (
            use_roi
            is not None
        ):

            (
                rx1,
                ry1,
                rx2,
                ry2,
            ) = use_roi


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
                # ROI BOX = MAGENTA
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
                # HEAD DETECTION + BYTETRACK
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

                    total_head_raw += len(
                        boxes
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
                    # HEAD CANDIDATES
                    # =========================================================

                    candidates = []


                    # ---------------------------------------------------------
                    # person box가 현재 프레임에서 있을 경우
                    # ---------------------------------------------------------

                    if (
                        px1 is not None
                        and
                        py1 is not None
                        and
                        px2 is not None
                        and
                        py2 is not None
                    ):

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


                        # =====================================================
                        # LOOP HEAD BOX
                        # =====================================================

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


                            # -------------------------------------------------
                            # ROI -> ORIGINAL coordinates
                            # -------------------------------------------------

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


                            # =================================================
                            # FILTER 1:
                            # ABSOLUTE SIZE
                            # =================================================

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


                            # =================================================
                            # FILTER 2:
                            # HEAD AREA / PERSON AREA
                            # =================================================

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


                            # =================================================
                            # FILTER 3:
                            # HEAD POSITION
                            # =================================================

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


                            # =================================================
                            # FILTER 4:
                            # EXPECTED HEAD DISTANCE
                            # =================================================

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


                            # -------------------------------------------------
                            # FINAL SCORE
                            #
                            # confidence 높을수록 좋음
                            # 예상 머리 위치와 가까울수록 좋음
                            # -------------------------------------------------

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
                    # FINAL HEAD SELECTION
                    # =========================================================

                    candidates.sort(
                        key=lambda x:
                            x[0],
                        reverse=True,
                    )


                    # 한 사람당 가장 좋은 머리 하나만 선택
                    if candidates:

                        best = (
                            candidates[0]
                        )

                        (
                            score,
                            confidence,
                            track_id,
                            hx1,
                            hy1,
                            hx2,
                            hy2,
                        ) = best


                        total_head_final += 1


                        head_ids.add(
                            int(
                                track_id
                            )
                        )


                        # -----------------------------------------------------
                        # ROI -> ORIGINAL
                        # -----------------------------------------------------

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
                        # FINAL HEAD BOX = ORANGE
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
        # FRAME NUMBER
        # =====================================================================

        cv2.putText(
            preview,

            f"Frame {frame_idx}",

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


        # =====================================================================
        # LEGEND
        # =====================================================================

        cv2.putText(
            preview,

            "GREEN=PERSON  MAGENTA=HEAD ROI  ORANGE=FINAL HEAD",

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


        # =====================================================================
        # SAVE FRAME
        # =====================================================================

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
                f" "
                f"person={selected_person is not None}"
            )


    # =========================================================================
    # FINISH
    # =========================================================================

    writer.release()


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
        f"IMAGES                : {len(images)}"
    )


    print(
        f"PERSON FRAMES         : {total_person_frames}"
    )


    print(
        f"PERSON RATIO          : "
        f"{total_person_frames / len(images):.3f}"
    )


    print(
        f"HEAD RAW              : {total_head_raw}"
    )


    print(
        f"HEAD AFTER FILTER     : {total_head_filtered}"
    )


    print(
        f"FINAL HEAD FRAMES     : {total_head_final}"
    )


    print(
        f"FINAL HEAD RATIO      : "
        f"{total_head_final / len(images):.3f}"
    )


    print(
        f"HEAD TRACK IDS        : {len(head_ids)}"
    )


    print()
    print(
        f"OUTPUT:\n{OUTPUT_PATH}"
    )


# =============================================================================
# START
# =============================================================================

if __name__ == "__main__":

    main()
