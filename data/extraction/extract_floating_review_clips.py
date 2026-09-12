from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path

import cv2
import pandas as pd
import torch
from ultralytics import YOLO


# =============================================================================
# PATH
# =============================================================================

VIDEO_ROOT = Path(
    r"D:\임베디드 경진대회\머리 추적\CVAT용 영상\converted_10fps"
)

CANDIDATE_CSV = Path(
    r"D:\임베디드 경진대회\머리 추적\익수자 판별 학습\FLOATING_SELECTION\FLOATING_CANDIDATES.csv"
)

MODEL_PATH = Path(
    r"D:\임베디드 경진대회\머리 추적\models\pool_head_best.pt"
)

TRACKER_PATH = Path(
    r"C:\Users\이승희\Desktop\bytetrack_pool.yaml"
)

OUTPUT_ROOT = Path(
    r"D:\임베디드 경진대회\머리 추적\익수자 판별 학습\FLOATING_REVIEW"
)

CLIP_DIR = OUTPUT_ROOT / "clips"

MANIFEST_PATH = OUTPUT_ROOT / "FLOATING_REVIEW_manifest.csv"


# =============================================================================
# DETECTOR / TRACKER
# =============================================================================

IMG_SIZE = 960

CONF = 0.05

IOU = 0.70


# =============================================================================
# REVIEW CLIP
# =============================================================================

CLIP_SECONDS = 5.0

# 원본이 10fps이므로 출력도 원본 fps 그대로.
# 필요하면 나중에 강제로 10fps로 바꿀 수 있음.

VIDEO_CODEC = "mp4v"


# =============================================================================
# HELPER
# =============================================================================

def safe_filename(value: str) -> str:

    value = str(value)

    replacements = {
        "\\": "__",
        "/": "__",
        ":": "_",
        "*": "_",
        "?": "_",
        '"': "_",
        "<": "_",
        ">": "_",
        "|": "_",
        " ": "_",
    }

    for old, new in replacements.items():

        value = value.replace(
            old,
            new
        )

    return value


def find_video(relative_path: str) -> Path | None:

    relative_path = str(relative_path)

    # CSV의 relative_path를 그대로 우선 사용
    candidate = (
        VIDEO_ROOT
        /
        Path(relative_path)
    )

    if candidate.exists():
        return candidate

    # Windows 경로 separator 보정
    normalized = (
        relative_path
        .replace("\\", "/")
    )

    candidate = (
        VIDEO_ROOT
        /
        Path(normalized)
    )

    if candidate.exists():
        return candidate

    # 마지막 수단: 파일명으로 검색
    filename = Path(
        normalized
    ).name

    matches = list(
        VIDEO_ROOT.rglob(
            filename
        )
    )

    if len(matches) == 1:
        return matches[0]

    return None


def choose_clip_range(
    first_frame: int,
    last_frame: int,
    fps: float,
    total_frames: int,
):

    clip_frames = max(
        1,
        int(
            round(
                CLIP_SECONDS
                *
                fps
            )
        )
    )

    # Track 가운데를 기준으로 5초
    middle = int(
        round(
            (
                first_frame
                +
                last_frame
            )
            /
            2
        )
    )

    start = (
        middle
        -
        clip_frames // 2
    )

    start = max(
        0,
        start
    )

    end = (
        start
        +
        clip_frames
        -
        1
    )

    if end >= total_frames:

        end = (
            total_frames
            -
            1
        )

        start = max(
            0,
            end
            -
            clip_frames
            +
            1
        )

    return (
        start,
        end
    )


def draw_target_box(
    frame,
    box,
    track_id,
):

    x1, y1, x2, y2 = [
        int(round(v))
        for v in box
    ]

    # 사용자가 어떤 머리를 봐야 하는지 명확하게
    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (0, 0, 255),
        3,
    )

    label = (
        f"FLOATING CANDIDATE | ID {track_id}"
    )

    text_y = max(
        30,
        y1 - 10
    )

    cv2.putText(
        frame,
        label,
        (x1, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    CLIP_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    if not CANDIDATE_CSV.exists():

        raise FileNotFoundError(
            f"후보 CSV 없음:\n{CANDIDATE_CSV}"
        )

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"모델 없음:\n{MODEL_PATH}"
        )

    if not TRACKER_PATH.exists():

        raise FileNotFoundError(
            f"Tracker YAML 없음:\n{TRACKER_PATH}"
        )

    df = pd.read_csv(
        CANDIDATE_CSV
    )

    required_columns = {
        "video_id",
        "relative_path",
        "track_id",
        "first_frame",
        "last_frame",
    }

    missing_columns = (
        required_columns
        -
        set(df.columns)
    )

    if missing_columns:

        raise RuntimeError(
            "FLOATING_CANDIDATES.csv에 "
            "필요한 컬럼이 없습니다:\n"
            f"{sorted(missing_columns)}"
        )

    print(
        "=" * 80
    )

    print(
        "FLOATING REVIEW CLIP EXTRACTION"
    )

    print(
        "=" * 80
    )

    print(
        f"CANDIDATES : {len(df)}"
    )

    print(
        f"VIDEOS     : "
        f"{df['relative_path'].nunique()}"
    )

    print(
        f"OUTPUT     : {CLIP_DIR}"
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

    # =============================================================================
    # Candidate를 영상별로 묶음
    # =============================================================================

    grouped = df.groupby(
        "relative_path",
        sort=False
    )

    manifest_rows = []

    global_candidate_index = 0

    total_videos = grouped.ngroups

    # =============================================================================
    # VIDEO LOOP
    # =============================================================================

    for video_index, (
        relative_path,
        video_candidates
    ) in enumerate(
        grouped,
        start=1
    ):

        print()
        print(
            "=" * 80
        )

        print(
            f"[VIDEO {video_index}/{total_videos}]"
        )

        print(
            relative_path
        )

        video_path = find_video(
            relative_path
        )

        if video_path is None:

            print(
                "[ERROR] 원본 영상 못 찾음"
            )

            for _, row in video_candidates.iterrows():

                manifest_rows.append(
                    {
                        **row.to_dict(),

                        "review_index":
                            None,

                        "clip_path":
                            "",

                        "clip_start_frame":
                            None,

                        "clip_end_frame":
                            None,

                        "matched_frames":
                            0,

                        "status":
                            "VIDEO_NOT_FOUND",

                        "manual_label":
                            "",

                        "manual_note":
                            "",
                    }
                )

            continue

        # =========================================================================
        # Video metadata
        # =========================================================================

        cap_meta = cv2.VideoCapture(
            str(
                video_path
            )
        )

        if not cap_meta.isOpened():

            print(
                "[ERROR] 영상 열기 실패"
            )

            continue

        fps = float(
            cap_meta.get(
                cv2.CAP_PROP_FPS
            )
        )

        width = int(
            cap_meta.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
        )

        height = int(
            cap_meta.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
        )

        total_frames = int(
            cap_meta.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )

        cap_meta.release()

        if fps <= 0:

            fps = 10.0

        print(
            f"FPS={fps:.2f}"
            f" | SIZE={width}x{height}"
            f" | FRAMES={total_frames}"
            f" | CANDIDATES={len(video_candidates)}"
        )

        # =========================================================================
        # Candidate 정보 구성
        # =========================================================================

        candidate_info = {}

        for _, row in video_candidates.iterrows():

            global_candidate_index += 1

            track_id = int(
                row["track_id"]
            )

            first_frame = int(
                row["first_frame"]
            )

            last_frame = int(
                row["last_frame"]
            )

            (
                clip_start,
                clip_end
            ) = choose_clip_range(
                first_frame=first_frame,
                last_frame=last_frame,
                fps=fps,
                total_frames=total_frames,
            )

            clip_name = (
                f"{global_candidate_index:04d}"
                f"__T{track_id}"
                f"__{safe_filename(Path(relative_path).stem)}"
                f"__F{clip_start}-{clip_end}"
                f".mp4"
            )

            clip_path = (
                CLIP_DIR
                /
                clip_name
            )

            candidate_info[
                track_id
            ] = {
                "row":
                    row.to_dict(),

                "review_index":
                    global_candidate_index,

                "track_id":
                    track_id,

                "clip_start":
                    clip_start,

                "clip_end":
                    clip_end,

                "clip_path":
                    clip_path,

                "writer":
                    None,

                "matched_frames":
                    0,

                "frames_written":
                    0,
            }

        # =========================================================================
        # Tracker
        # =========================================================================

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

        fourcc = cv2.VideoWriter_fourcc(
            *VIDEO_CODEC
        )

        # =========================================================================
        # FRAME LOOP
        # =========================================================================

        for frame_index, result in enumerate(
            results
        ):

            # 이 프레임에 필요한 candidate가 하나라도 있는지 먼저 확인
            active_candidates = []

            for info in candidate_info.values():

                if (
                    info["clip_start"]
                    <=
                    frame_index
                    <=
                    info["clip_end"]
                ):

                    active_candidates.append(
                        info
                    )

            if not active_candidates:

                continue

            original_frame = result.orig_img

            if original_frame is None:

                continue

            # 현재 frame의 track boxes를 dictionary로
            current_tracks = {}

            boxes = result.boxes

            if (
                boxes is not None
                and
                boxes.id is not None
                and
                len(boxes) > 0
            ):

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

                for (
                    track_id,
                    xyxy
                ) in zip(
                    ids,
                    xyxy_values
                ):

                    current_tracks[
                        int(
                            track_id
                        )
                    ] = [
                        float(v)
                        for v in xyxy
                    ]

            # Candidate 별 clip
            for info in active_candidates:

                if info["writer"] is None:

                    info["writer"] = cv2.VideoWriter(
                        str(
                            info["clip_path"]
                        ),
                        fourcc,
                        fps,
                        (
                            width,
                            height
                        ),
                    )

                frame = (
                    original_frame
                    .copy()
                )

                track_id = info[
                    "track_id"
                ]

                # 후보 track이 현재 검출된 경우 박스 표시
                if track_id in current_tracks:

                    draw_target_box(
                        frame=frame,
                        box=current_tracks[
                            track_id
                        ],
                        track_id=track_id,
                    )

                    info[
                        "matched_frames"
                    ] += 1

                else:

                    # LOST인 경우에도 사용자가 알아볼 수 있게 표시
                    cv2.putText(
                        frame,
                        (
                            f"TARGET ID {track_id}"
                            " | LOST"
                        ),
                        (
                            30,
                            45
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (
                            0,
                            0,
                            255
                        ),
                        2,
                        cv2.LINE_AA,
                    )

                # Review 번호
                cv2.putText(
                    frame,
                    (
                        f"REVIEW "
                        f"{info['review_index']:04d}"
                    ),
                    (
                        30,
                        height - 30
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (
                        255,
                        255,
                        255
                    ),
                    2,
                    cv2.LINE_AA,
                )

                info[
                    "writer"
                ].write(
                    frame
                )

                info[
                    "frames_written"
                ] += 1

        # =========================================================================
        # Writers close
        # =========================================================================

        for info in candidate_info.values():

            if info["writer"] is not None:

                info[
                    "writer"
                ].release()

            expected_frames = (
                info["clip_end"]
                -
                info["clip_start"]
                +
                1
            )

            matched_ratio = (
                info["matched_frames"]
                /
                max(
                    1,
                    expected_frames
                )
            )

            if (
                info["frames_written"]
                ==
                0
            ):

                status = (
                    "NO_CLIP"
                )

            elif (
                info["matched_frames"]
                ==
                0
            ):

                status = (
                    "TRACK_ID_NOT_MATCHED"
                )

            elif (
                matched_ratio
                <
                0.30
            ):

                status = (
                    "LOW_TRACK_MATCH"
                )

            else:

                status = (
                    "OK"
                )

            manifest_rows.append(
                {
                    **info["row"],

                    "review_index":
                        info[
                            "review_index"
                        ],

                    "clip_path":
                        str(
                            info[
                                "clip_path"
                            ]
                        ),

                    "clip_start_frame":
                        info[
                            "clip_start"
                        ],

                    "clip_end_frame":
                        info[
                            "clip_end"
                        ],

                    "frames_written":
                        info[
                            "frames_written"
                        ],

                    "matched_frames":
                        info[
                            "matched_frames"
                        ],

                    "matched_ratio":
                        matched_ratio,

                    "status":
                        status,

                    # 사람이 나중에 입력
                    "manual_label":
                        "",

                    "manual_note":
                        "",
                }
            )

        # 진행 상황
        ok_count = sum(
            1
            for info in candidate_info.values()
            if info["matched_frames"] > 0
        )

        print(
            f"CREATED={len(candidate_info)}"
            f" | MATCHED={ok_count}"
        )

        # 중간 저장
        pd.DataFrame(
            manifest_rows
        ).to_csv(
            MANIFEST_PATH,
            index=False,
            encoding="utf-8-sig",
        )

    # =============================================================================
    # FINAL
    # =============================================================================

    manifest = pd.DataFrame(
        manifest_rows
    )

    manifest.to_csv(
        MANIFEST_PATH,
        index=False,
        encoding="utf-8-sig",
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
        f"TOTAL CANDIDATES : {len(manifest)}"
    )

    if not manifest.empty:

        print()
        print(
            "[STATUS]"
        )

        print(
            manifest[
                "status"
            ]
            .value_counts()
            .to_string()
        )

    print()
    print(
        "[CLIPS]"
    )

    print(
        CLIP_DIR
    )

    print()
    print(
        "[MANIFEST]"
    )

    print(
        MANIFEST_PATH
    )


if __name__ == "__main__":

    main()