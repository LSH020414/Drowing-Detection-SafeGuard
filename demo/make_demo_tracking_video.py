from pathlib import Path
import cv2
import torch
from ultralytics import YOLO


# =============================================================================
# ★★★★★ 여기만 수정하면 됨 ★★★★★
# =============================================================================

# videos 폴더 안에 있는 영상 파일명
VIDEO_NAME = "수동적 익수자 발사.mp4"


# -----------------------------------------------------------------------------
# 시간별 상태 설정
#
# 사용 가능한 상태:
# SWIMMING
# FLOATING
# ACTIVE_DROWNING
# PASSIVE_DROWNING
#
# 형식:
# (시작시간, 종료시간, 상태)
# -----------------------------------------------------------------------------

STATE_TIMELINE = [
    (0.0, 18.0, "FLOATING"),
    (18.0, 30.0, "PASSIVE_DROWNING"),
    (30.2, 99999.0, "FLOATING"),
]


# =============================================================================
# PATH
# =============================================================================

VIDEO_ROOT = Path(
    r"C:\Users\이승희\Desktop\videos"
)

INPUT_VIDEO = (
    VIDEO_ROOT
    /
    VIDEO_NAME
)

MODEL_PATH = Path(
    r"D:\임베디드 경진대회\머리 추적\models\pool_head_best.pt"
)

TRACKER_PATH = Path(
    r"C:\Users\이승희\Desktop\bytetrack_pool.yaml"
)

OUTPUT_ROOT = (
    VIDEO_ROOT
    /
    "tracked_output"
)

OUTPUT_VIDEO = (
    OUTPUT_ROOT
    /
    f"{Path(VIDEO_NAME).stem}_ID1_state.mp4"
)


# =============================================================================
# DETECTOR SETTINGS
# =============================================================================

IMG_SIZE = 960

CONF = 0.05

IOU = 0.70


# =============================================================================
# DISPLAY SETTINGS
# =============================================================================

DISPLAY_ID = 1

BOX_COLOR = (0, 165, 255)

BOX_THICKNESS = 4

LABEL_FONT = cv2.FONT_HERSHEY_SIMPLEX

LABEL_SCALE = 0.85

LABEL_THICKNESS = 3

CONF_SCALE = 0.65

CONF_THICKNESS = 2

SCREEN_MARGIN = 8


# =============================================================================
# STATE
# =============================================================================

VALID_STATES = {
    "SWIMMING",
    "FLOATING",
    "ACTIVE_DROWNING",
    "PASSIVE_DROWNING",
}


def get_state(current_time):

    for start_time, end_time, state in STATE_TIMELINE:

        if start_time <= current_time < end_time:
            return state

    return "SWIMMING"


def get_state_text(state):

    if state == "SWIMMING":
        return "SWIMMING"

    elif state == "FLOATING":
        return "FLOATING"

    elif state == "ACTIVE_DROWNING":
        return "ACTIVE DROWNING"

    elif state == "PASSIVE_DROWNING":
        return "PASSIVE DROWNING"

    return state


# =============================================================================
# SAFE LABEL POSITION
# =============================================================================

def get_safe_label_position(
    text,
    x1,
    y1,
    x2,
    y2,
    frame_width,
    frame_height,
    font,
    font_scale,
    thickness,
):

    (text_w, text_h), baseline = cv2.getTextSize(
        text,
        font,
        font_scale,
        thickness
    )

    # X 위치
    text_x = x1

    if text_x + text_w + SCREEN_MARGIN > frame_width:
        text_x = frame_width - text_w - SCREEN_MARGIN

    text_x = max(
        SCREEN_MARGIN,
        text_x
    )

    # Y 위치
    candidate_y = y1 - 12

    # 박스 위쪽에 공간이 있으면 위에 표시
    if (
        candidate_y
        -
        text_h
        -
        baseline
        >=
        SCREEN_MARGIN
    ):
        text_y = candidate_y

    else:
        # 위에 공간이 없으면 박스 아래
        text_y = (
            y2
            +
            text_h
            +
            12
        )

        # 아래도 넘치면 화면 안쪽으로 제한
        if (
            text_y
            +
            baseline
            +
            SCREEN_MARGIN
            >
            frame_height
        ):
            text_y = (
                frame_height
                -
                baseline
                -
                SCREEN_MARGIN
            )

    return text_x, text_y


# =============================================================================
# SAFE CONFIDENCE POSITION
# =============================================================================

def get_safe_conf_position(
    text,
    x1,
    y1,
    x2,
    y2,
    frame_width,
    frame_height,
):

    (text_w, text_h), baseline = cv2.getTextSize(
        text,
        LABEL_FONT,
        CONF_SCALE,
        CONF_THICKNESS
    )

    # X
    text_x = x1

    if (
        text_x
        +
        text_w
        +
        SCREEN_MARGIN
        >
        frame_width
    ):
        text_x = (
            frame_width
            -
            text_w
            -
            SCREEN_MARGIN
        )

    text_x = max(
        SCREEN_MARGIN,
        text_x
    )

    # Y
    candidate_y = (
        y2
        +
        text_h
        +
        8
    )

    # 아래 공간 충분
    if (
        candidate_y
        +
        baseline
        +
        SCREEN_MARGIN
        <
        frame_height
    ):
        text_y = candidate_y

    else:
        # 아래 공간 부족하면 위쪽
        text_y = (
            y1
            -
            8
        )

        if (
            text_y
            -
            text_h
            <
            SCREEN_MARGIN
        ):
            text_y = (
                text_h
                +
                SCREEN_MARGIN
            )

    return text_x, text_y


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 80)

    print(
        "HEAD DETECTOR + BYTETRACK + MANUAL STATE"
    )

    print("=" * 80)


    # =========================================================================
    # CHECK
    # =========================================================================

    if not INPUT_VIDEO.exists():

        raise FileNotFoundError(
            f"\n영상이 없습니다:\n{INPUT_VIDEO}"
        )


    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"\n모델이 없습니다:\n{MODEL_PATH}"
        )


    if not TRACKER_PATH.exists():

        raise FileNotFoundError(
            f"\nTracker 설정이 없습니다:\n{TRACKER_PATH}"
        )


    for start_time, end_time, state in STATE_TIMELINE:

        if state not in VALID_STATES:

            raise ValueError(
                f"잘못된 상태 이름: {state}"
            )


    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )


    # =========================================================================
    # VIDEO INFO
    # =========================================================================

    cap = cv2.VideoCapture(
        str(INPUT_VIDEO)
    )


    if not cap.isOpened():

        raise RuntimeError(
            f"영상 열기 실패:\n{INPUT_VIDEO}"
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


    duration = (
        total_frames / fps
        if fps > 0
        else 0
    )


    print(
        f"VIDEO    : {INPUT_VIDEO}"
    )

    print(
        f"SIZE     : {width} x {height}"
    )

    print(
        f"FPS      : {fps:.3f}"
    )

    print(
        f"FRAMES   : {total_frames}"
    )

    print(
        f"DURATION : {duration:.2f} sec"
    )


    # =========================================================================
    # TIMELINE
    # =========================================================================

    print()

    print(
        "[STATE TIMELINE]"
    )


    for start, end, state in STATE_TIMELINE:

        print(
            f"{start:6.1f}s ~ {end:6.1f}s"
            f" : {state}"
        )


    # =========================================================================
    # DEVICE
    # =========================================================================

    device = (
        0
        if torch.cuda.is_available()
        else "cpu"
    )


    print()

    print(
        f"DEVICE : {device}"
    )


    # =========================================================================
    # MODEL
    # =========================================================================

    model = YOLO(
        str(MODEL_PATH)
    )

    model.predictor = None


    # =========================================================================
    # WRITER
    # =========================================================================

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )


    writer = cv2.VideoWriter(
        str(OUTPUT_VIDEO),
        fourcc,
        fps,
        (
            width,
            height
        ),
    )


    # =========================================================================
    # VARIABLES
    # =========================================================================

    frame_idx = 0

    detection_frames = 0


    # =========================================================================
    # LOOP
    # =========================================================================

    while True:

        ret, frame = cap.read()


        if not ret:
            break


        current_time = (
            frame_idx
            /
            fps
        )


        # =====================================================================
        # STATE
        # =====================================================================

        state = get_state(
            current_time
        )


        state_text = get_state_text(
            state
        )


        # =====================================================================
        # DETECTOR + TRACKER
        # =====================================================================

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


        # =====================================================================
        # SELECT BEST HEAD
        # =====================================================================

        best_detection = None


        if (
            boxes is not None
            and
            len(boxes) > 0
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


            best_index = max(
                range(
                    len(conf_values)
                ),
                key=lambda i:
                    conf_values[i],
            )


            best_detection = (
                xyxy_values[best_index],
                float(
                    conf_values[best_index]
                ),
            )


        # =====================================================================
        # DRAW
        # =====================================================================

        if best_detection is not None:

            xyxy, confidence = best_detection


            x1, y1, x2, y2 = [
                int(v)
                for v in xyxy
            ]


            # 좌표 화면 안으로 제한
            x1 = max(
                0,
                min(
                    width - 1,
                    x1
                )
            )

            x2 = max(
                0,
                min(
                    width - 1,
                    x2
                )
            )

            y1 = max(
                0,
                min(
                    height - 1,
                    y1
                )
            )

            y2 = max(
                0,
                min(
                    height - 1,
                    y2
                )
            )


            detection_frames += 1


            # -----------------------------------------------------------------
            # HEAD BOX
            # -----------------------------------------------------------------

            cv2.rectangle(
                frame,
                (
                    x1,
                    y1
                ),
                (
                    x2,
                    y2
                ),
                BOX_COLOR,
                BOX_THICKNESS,
            )


            # -----------------------------------------------------------------
            # ID + STATE
            # -----------------------------------------------------------------

            label = (
                f"ID {DISPLAY_ID} | "
                f"{state_text}"
            )


            label_x, label_y = (
                get_safe_label_position(
                    text=label,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    frame_width=width,
                    frame_height=height,
                    font=LABEL_FONT,
                    font_scale=LABEL_SCALE,
                    thickness=LABEL_THICKNESS,
                )
            )


            cv2.putText(
                frame,
                label,
                (
                    label_x,
                    label_y
                ),
                LABEL_FONT,
                LABEL_SCALE,
                BOX_COLOR,
                LABEL_THICKNESS,
                cv2.LINE_AA,
            )


            # -----------------------------------------------------------------
            # CONFIDENCE
            # -----------------------------------------------------------------

            conf_label = (
                f"HEAD {confidence:.2f}"
            )


            conf_x, conf_y = (
                get_safe_conf_position(
                    text=conf_label,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    frame_width=width,
                    frame_height=height,
                )
            )


            cv2.putText(
                frame,
                conf_label,
                (
                    conf_x,
                    conf_y
                ),
                LABEL_FONT,
                CONF_SCALE,
                BOX_COLOR,
                CONF_THICKNESS,
                cv2.LINE_AA,
            )


        # =====================================================================
        # STATUS PANEL
        # =====================================================================

        overlay = frame.copy()


        panel_x1 = 20
        panel_y1 = 20

        panel_x2 = min(
            630,
            width - 20
        )

        panel_y2 = min(
            145,
            height - 20
        )


        cv2.rectangle(
            overlay,
            (
                panel_x1,
                panel_y1
            ),
            (
                panel_x2,
                panel_y2
            ),
            (
                0,
                0,
                0
            ),
            -1,
        )


        frame = cv2.addWeighted(
            overlay,
            0.55,
            frame,
            0.45,
            0,
        )


        # TIME
        cv2.putText(
            frame,
            f"TIME: {current_time:05.1f}s",
            (
                40,
                60
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (
                255,
                255,
                255
            ),
            2,
            cv2.LINE_AA,
        )


        # ID
        cv2.putText(
            frame,
            f"TRACK ID: {DISPLAY_ID}",
            (
                40,
                100
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (
                255,
                255,
                255
            ),
            2,
            cv2.LINE_AA,
        )


        # STATE
        cv2.putText(
            frame,
            f"STATE: {state_text}",
            (
                min(
                    250,
                    max(
                        40,
                        width // 3
                    )
                ),
                100
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            BOX_COLOR,
            3,
            cv2.LINE_AA,
        )


        # =====================================================================
        # WRITE
        # =====================================================================

        writer.write(
            frame
        )


        frame_idx += 1


        if (
            frame_idx
            %
            100
            ==
            0
        ):

            print(
                f"[{frame_idx}/{total_frames}] "
                f"time={current_time:.1f}s "
                f"state={state}"
            )


    # =========================================================================
    # FINISH
    # =========================================================================

    cap.release()

    writer.release()


    print()

    print("=" * 80)

    print(
        "FINISHED"
    )

    print("=" * 80)


    print(
        f"TOTAL FRAMES     : {frame_idx}"
    )


    print(
        f"DETECTION FRAMES : {detection_frames}"
    )


    if frame_idx > 0:

        print(
            f"DETECTION RATIO  : "
            f"{detection_frames / frame_idx:.3f}"
        )


    print()

    print(
        f"OUTPUT:\n{OUTPUT_VIDEO}"
    )


# =============================================================================
# START
# =============================================================================

if __name__ == "__main__":
    main()
