from __future__ import annotations

import argparse
import csv
import io
import math
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import requests


# ============================================================
# 기본 설정
# ============================================================

CVAT_URL = "http://localhost:8080"

# 처음에는 Task 86 하나만 테스트
DEFAULT_TASK_START = 86
DEFAULT_TASK_END = 86

OUTPUT_ROOT = Path(
    r"D:\임베디드 경진대회\머리 추적\ReID_dataset"
)

# ------------------------------------------------------------
# ReID crop 설정
# ------------------------------------------------------------

# 10fps 영상에서 3프레임마다 1장
# 약 3.33 crop/sec
FRAME_STRIDE = 3

# 최소 Track 길이
MIN_VISIBLE_FRAMES = 30

# bbox 주변 여백
# 0.20 = 각 방향으로 bbox 크기의 20% 추가
CROP_MARGIN = 0.20

# 너무 작은 머리 제외
MIN_WIDTH_RATIO = 0.005
MIN_HEIGHT_RATIO = 0.009
MIN_AREA_RATIO = 0.000040

# JPEG 저장 품질
JPEG_QUALITY = 95

# occluded=True인 프레임은 1차 데이터셋에서 제외
EXCLUDE_OCCLUDED = True


# ============================================================
# Train / Val / Test Task 분리
#
# 전체 86~106 추출할 때 사용
#
# 처음 Task86만 실행하면 자동으로 train으로 들어감.
# ============================================================

TRAIN_TASKS = set(range(86, 102))   # 86~101
VAL_TASKS = set(range(102, 105))   # 102~104
TEST_TASKS = set(range(105, 107))  # 105~106


# ============================================================
# 유틸
# ============================================================

def get_split(task_id: int) -> str:
    if task_id in TRAIN_TASKS:
        return "train"

    if task_id in VAL_TASKS:
        return "val"

    if task_id in TEST_TASKS:
        return "test"

    return "train"


def safe_int(value, default=0):
    try:
        return int(value)
    except Exception:
        return default


# ============================================================
# CVAT 로그인
# ============================================================

def login_cvat(
    server: str,
    username: str | None,
    password: str | None,
    token: str | None,
) -> requests.Session:

    session = requests.Session()

    server = server.rstrip("/")

    # --------------------------------------------------------
    # Token 인증
    # --------------------------------------------------------

    if token:
        session.headers.update(
            {
                "Authorization": f"Token {token}"
            }
        )

        response = session.get(
            server + "/api/users/self",
            timeout=60,
        )

        if not response.ok:
            print("[ERROR] Token 인증 실패")
            print(
                f"HTTP {response.status_code}"
            )
            print(
                response.text[:2000]
            )

            response.raise_for_status()

        data = response.json()

        print(
            "[OK] CVAT Token 로그인:",
            data.get("username", ""),
        )

        return session

    # --------------------------------------------------------
    # ID/PW 인증
    # --------------------------------------------------------

    if not username or password is None:
        raise RuntimeError(
            "--username/--password 또는 --token이 필요합니다."
        )

    response = session.post(
        server + "/api/auth/login",
        json={
            "username": username,
            "password": password,
        },
        timeout=60,
    )

    if not response.ok:
        print("[ERROR] CVAT 로그인 실패")
        print(
            f"HTTP {response.status_code}"
        )
        print(
            response.text[:2000]
        )

        response.raise_for_status()

    data = response.json()

    auth_token = data.get("key")

    if auth_token:
        session.headers.update(
            {
                "Authorization":
                    f"Token {auth_token}"
            }
        )

    print(
        f"[OK] CVAT 로그인: {username}"
    )

    return session


# ============================================================
# JSON GET
# ============================================================

def get_json(
    session: requests.Session,
    server: str,
    endpoint: str,
):

    url = (
        server.rstrip("/")
        +
        endpoint
    )

    response = session.get(
        url,
        timeout=600,
    )

    if not response.ok:
        print()
        print(
            f"[ERROR] GET {endpoint}"
        )
        print(
            f"HTTP {response.status_code}"
        )
        print(
            response.text[:3000]
        )

        response.raise_for_status()

    return response.json()


# ============================================================
# Task frame 가져오기
# ============================================================

def get_task_frame(
    session: requests.Session,
    server: str,
    task_id: int,
    frame_number: int,
):

    url = (
        server.rstrip("/")
        +
        f"/api/tasks/{task_id}/data"
    )

    response = session.get(
        url,
        params={
            "type": "frame",
            "number": frame_number,
            "quality": "original",
        },
        timeout=600,
    )

    if not response.ok:
        print()
        print(
            f"[ERROR] Task {task_id} "
            f"frame {frame_number} 다운로드 실패"
        )

        print(
            f"HTTP {response.status_code}"
        )

        print(
            response.text[:1000]
        )

        response.raise_for_status()

    image_array = np.frombuffer(
        response.content,
        dtype=np.uint8,
    )

    frame = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR,
    )

    if frame is None:
        raise RuntimeError(
            f"Task {task_id} frame {frame_number} "
            "이미지 decode 실패"
        )

    return frame


# ============================================================
# 실제 Visible frame 수 계산
# ============================================================

def count_visible_frames(
    track: dict,
    total_frames: int,
) -> int:

    shapes = sorted(
        track.get("shapes", []),
        key=lambda s:
            safe_int(
                s.get("frame", 0)
            ),
    )

    if not shapes:
        return 0

    visible_count = 0

    for i, shape in enumerate(shapes):

        start_frame = safe_int(
            shape.get("frame", 0)
        )

        if i + 1 < len(shapes):
            end_frame = (
                safe_int(
                    shapes[
                        i + 1
                    ].get(
                        "frame",
                        start_frame + 1,
                    )
                )
                - 1
            )

        else:
            end_frame = (
                total_frames - 1
            )

        start_frame = max(
            0,
            start_frame,
        )

        end_frame = min(
            total_frames - 1,
            end_frame,
        )

        if end_frame < start_frame:
            continue

        if not bool(
            shape.get(
                "outside",
                False,
            )
        ):
            visible_count += (
                end_frame
                -
                start_frame
                +
                1
            )

    return visible_count


# ============================================================
# Rectangle interpolation
#
# CVAT Track은 keyframe 사이 bbox를 interpolation해서 보여준다.
#
# 따라서 수동 Merge 후 keyframe이 듬성듬성해도
# 중간 frame bbox를 계산할 수 있도록 만든다.
# ============================================================

def interpolate_points(
    p1,
    p2,
    alpha: float,
):

    if (
        not isinstance(p1, list)
        or
        not isinstance(p2, list)
        or
        len(p1) != 4
        or
        len(p2) != 4
    ):
        return None

    return [
        float(p1[i])
        +
        (
            float(p2[i])
            -
            float(p1[i])
        )
        *
        alpha

        for i in range(4)
    ]


# ============================================================
# Track을 frame별 bbox로 변환
# ============================================================

def build_track_frame_boxes(
    track: dict,
    total_frames: int,
    stride: int,
):

    shapes = sorted(
        track.get(
            "shapes",
            [],
        ),
        key=lambda s:
            safe_int(
                s.get(
                    "frame",
                    0,
                )
            ),
    )

    if not shapes:
        return []

    output = []

    # --------------------------------------------------------
    # shape 구간별 처리
    # --------------------------------------------------------

    for i, current in enumerate(shapes):

        current_frame = safe_int(
            current.get(
                "frame",
                0,
            )
        )

        current_outside = bool(
            current.get(
                "outside",
                False,
            )
        )

        current_occluded = bool(
            current.get(
                "occluded",
                False,
            )
        )

        current_points = (
            current.get(
                "points",
                None,
            )
        )

        # outside면 해당 구간 사용 안 함
        if current_outside:
            continue

        # ----------------------------------------------------
        # 마지막 shape
        # ----------------------------------------------------

        if i + 1 >= len(shapes):

            end_frame = (
                total_frames - 1
            )

            for frame_number in range(
                current_frame,
                end_frame + 1,
            ):

                if (
                    frame_number
                    %
                    stride
                    !=
                    0
                ):
                    continue

                if (
                    EXCLUDE_OCCLUDED
                    and
                    current_occluded
                ):
                    continue

                output.append(
                    {
                        "frame":
                            frame_number,

                        "points":
                            current_points,

                        "occluded":
                            current_occluded,
                    }
                )

            continue

        # ----------------------------------------------------
        # 다음 shape
        # ----------------------------------------------------

        nxt = shapes[
            i + 1
        ]

        next_frame = safe_int(
            nxt.get(
                "frame",
                current_frame + 1,
            )
        )

        next_points = (
            nxt.get(
                "points",
                current_points,
            )
        )

        next_outside = bool(
            nxt.get(
                "outside",
                False,
            )
        )

        # ----------------------------------------------------
        # current부터 next 직전까지
        # ----------------------------------------------------

        end_frame = (
            next_frame - 1
        )

        interval = (
            next_frame
            -
            current_frame
        )

        for frame_number in range(
            current_frame,
            end_frame + 1,
        ):

            if (
                frame_number
                %
                stride
                !=
                0
            ):
                continue

            if (
                EXCLUDE_OCCLUDED
                and
                current_occluded
            ):
                continue

            if interval > 0:

                alpha = (
                    frame_number
                    -
                    current_frame
                ) / interval

            else:
                alpha = 0.0

            points = interpolate_points(
                current_points,
                next_points,
                alpha,
            )

            if points is None:
                continue

            output.append(
                {
                    "frame":
                        frame_number,

                    "points":
                        points,

                    "occluded":
                        current_occluded,
                }
            )

        # ----------------------------------------------------
        # 다음 shape 자체가 visible인 경우
        # 다음 반복에서 처리되므로 여기서는 넣지 않음.
        #
        # next_outside는 다음 loop에서 자동으로 skip.
        # ----------------------------------------------------

    return output


# ============================================================
# Bounding box 정리
# ============================================================

def normalize_bbox(
    points,
):

    if (
        not isinstance(points, list)
        or
        len(points) != 4
    ):
        return None

    x1, y1, x2, y2 = map(
        float,
        points,
    )

    left = min(
        x1,
        x2,
    )

    right = max(
        x1,
        x2,
    )

    top = min(
        y1,
        y2,
    )

    bottom = max(
        y1,
        y2,
    )

    if (
        right <= left
        or
        bottom <= top
    ):
        return None

    return (
        left,
        top,
        right,
        bottom,
    )


# ============================================================
# 작은 머리 필터
# ============================================================

def passes_size_filter(
    bbox,
    frame_width: int,
    frame_height: int,
):

    x1, y1, x2, y2 = bbox

    width = (
        x2 - x1
    )

    height = (
        y2 - y1
    )

    frame_area = (
        frame_width
        *
        frame_height
    )

    bbox_area = (
        width
        *
        height
    )

    width_ratio = (
        width
        /
        frame_width
    )

    height_ratio = (
        height
        /
        frame_height
    )

    area_ratio = (
        bbox_area
        /
        frame_area
    )

    passed = (
        width_ratio
        >=
        MIN_WIDTH_RATIO

        and

        height_ratio
        >=
        MIN_HEIGHT_RATIO

        and

        area_ratio
        >=
        MIN_AREA_RATIO
    )

    return (
        passed,
        width_ratio,
        height_ratio,
        area_ratio,
    )


# ============================================================
# Margin 포함 crop
# ============================================================

def crop_with_margin(
    frame,
    bbox,
    margin: float,
):

    frame_height, frame_width = (
        frame.shape[:2]
    )

    x1, y1, x2, y2 = bbox

    box_width = (
        x2 - x1
    )

    box_height = (
        y2 - y1
    )

    margin_x = (
        box_width
        *
        margin
    )

    margin_y = (
        box_height
        *
        margin
    )

    crop_x1 = int(
        math.floor(
            x1 - margin_x
        )
    )

    crop_y1 = int(
        math.floor(
            y1 - margin_y
        )
    )

    crop_x2 = int(
        math.ceil(
            x2 + margin_x
        )
    )

    crop_y2 = int(
        math.ceil(
            y2 + margin_y
        )
    )

    crop_x1 = max(
        0,
        crop_x1,
    )

    crop_y1 = max(
        0,
        crop_y1,
    )

    crop_x2 = min(
        frame_width,
        crop_x2,
    )

    crop_y2 = min(
        frame_height,
        crop_y2,
    )

    if (
        crop_x2 <= crop_x1
        or
        crop_y2 <= crop_y1
    ):
        return None, None

    crop = frame[
        crop_y1:crop_y2,
        crop_x1:crop_x2
    ]

    if crop.size == 0:
        return None, None

    return (
        crop,
        (
            crop_x1,
            crop_y1,
            crop_x2,
            crop_y2,
        )
    )


# ============================================================
# 이미지 저장
# ============================================================

def save_jpeg(
    image,
    path: Path,
):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    success = cv2.imwrite(
        str(path),
        image,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            JPEG_QUALITY,
        ],
    )

    if not success:
        raise RuntimeError(
            f"이미지 저장 실패: {path}"
        )


# ============================================================
# Task 하나 처리
# ============================================================

def process_task(
    session,
    server,
    task_id,
    output_root,
    metadata_rows,
    identity_stats,
):

    print()
    print("=" * 100)
    print(
        f"TASK #{task_id}"
    )
    print("=" * 100)

    # --------------------------------------------------------
    # Task info
    # --------------------------------------------------------

    task_info = get_json(
        session,
        server,
        f"/api/tasks/{task_id}",
    )

    task_name = task_info.get(
        "name",
        "",
    )

    print(
        f"Name: {task_name}"
    )

    # --------------------------------------------------------
    # frame 수
    # --------------------------------------------------------

    meta = get_json(
        session,
        server,
        f"/api/tasks/{task_id}/data/meta",
    )

    total_frames = safe_int(
        meta.get(
            "size",
            0,
        )
    )

    print(
        f"Frames: {total_frames}"
    )

    if total_frames <= 0:
        print(
            "[SKIP] Frame 없음"
        )
        return

    # --------------------------------------------------------
    # annotation
    # --------------------------------------------------------

    annotations = get_json(
        session,
        server,
        f"/api/tasks/{task_id}/annotations/",
    )

    tracks = annotations.get(
        "tracks",
        [],
    )

    print(
        f"Tracks: {len(tracks)}"
    )

    split = get_split(
        task_id
    )

    # --------------------------------------------------------
    # Track -> frame assignments
    #
    # frame_assignments[frame] =
    # [
    #   {
    #      track_id,
    #      identity,
    #      bbox,
    #      ...
    #   }
    # ]
    # --------------------------------------------------------

    frame_assignments = defaultdict(
        list
    )

    eligible_tracks = 0
    short_tracks = 0

    for track in tracks:

        track_id = track.get(
            "id",
            None,
        )

        if track_id is None:
            continue

        visible_frames = (
            count_visible_frames(
                track,
                total_frames,
            )
        )

        if (
            visible_frames
            <
            MIN_VISIBLE_FRAMES
        ):
            short_tracks += 1
            continue

        eligible_tracks += 1

        identity = (
            f"task{task_id:03d}"
            f"_id{int(track_id):06d}"
        )

        frame_boxes = (
            build_track_frame_boxes(
                track,
                total_frames,
                FRAME_STRIDE,
            )
        )

        identity_stats[
            identity
        ] = {
            "identity":
                identity,

            "split":
                split,

            "task_id":
                task_id,

            "track_id":
                int(track_id),

            "visible_frames":
                visible_frames,

            "candidate_crops":
                len(frame_boxes),

            "saved_crops":
                0,

            "small_filtered":
                0,

            "invalid_bbox":
                0,
        }

        for item in frame_boxes:

            bbox = normalize_bbox(
                item.get(
                    "points"
                )
            )

            if bbox is None:
                identity_stats[
                    identity
                ][
                    "invalid_bbox"
                ] += 1

                continue

            frame_assignments[
                item["frame"]
            ].append(
                {
                    "track_id":
                        int(track_id),

                    "identity":
                        identity,

                    "bbox":
                        bbox,

                    "visible_frames":
                        visible_frames,

                    "occluded":
                        bool(
                            item.get(
                                "occluded",
                                False,
                            )
                        ),
                }
            )

    print(
        f"30 frame 이상 Track: "
        f"{eligible_tracks}"
    )

    print(
        f"30 frame 미만 제외: "
        f"{short_tracks}"
    )

    print(
        f"추출 대상 frame 수: "
        f"{len(frame_assignments)}"
    )

    if not frame_assignments:
        print(
            "[SKIP] 추출 대상 없음"
        )
        return

    # --------------------------------------------------------
    # 실제 frame 다운로드 및 crop
    # --------------------------------------------------------

    saved_total = 0
    filtered_small_total = 0

    sorted_frames = sorted(
        frame_assignments.keys()
    )

    for frame_index, frame_number in enumerate(
        sorted_frames,
        start=1,
    ):

        print(
            f"\r"
            f"Task {task_id} "
            f"frame {frame_index}/"
            f"{len(sorted_frames)} "
            f"(#{frame_number})",
            end="",
            flush=True,
        )

        try:
            frame = get_task_frame(
                session,
                server,
                task_id,
                frame_number,
            )

        except Exception as exc:
            print()
            print(
                f"[WARN] frame {frame_number} "
                f"건너뜀: {exc}"
            )
            continue

        frame_height, frame_width = (
            frame.shape[:2]
        )

        assignments = (
            frame_assignments[
                frame_number
            ]
        )

        for assignment in assignments:

            identity = assignment[
                "identity"
            ]

            bbox = assignment[
                "bbox"
            ]

            (
                passed,
                width_ratio,
                height_ratio,
                area_ratio,
            ) = passes_size_filter(
                bbox,
                frame_width,
                frame_height,
            )

            if not passed:

                filtered_small_total += 1

                identity_stats[
                    identity
                ][
                    "small_filtered"
                ] += 1

                continue

            crop, crop_bbox = (
                crop_with_margin(
                    frame,
                    bbox,
                    CROP_MARGIN,
                )
            )

            if crop is None:
                continue

            # ------------------------------------------------
            # 저장 경로
            # ------------------------------------------------

            identity_dir = (
                output_root
                /
                split
                /
                identity
            )

            filename = (
                f"f{frame_number:06d}.jpg"
            )

            output_path = (
                identity_dir
                /
                filename
            )

            save_jpeg(
                crop,
                output_path,
            )

            identity_stats[
                identity
            ][
                "saved_crops"
            ] += 1

            saved_total += 1

            x1, y1, x2, y2 = bbox

            (
                crop_x1,
                crop_y1,
                crop_x2,
                crop_y2,
            ) = crop_bbox

            metadata_rows.append(
                {
                    "split":
                        split,

                    "identity":
                        identity,

                    "task_id":
                        task_id,

                    "track_id":
                        assignment[
                            "track_id"
                        ],

                    "frame":
                        frame_number,

                    "image_path":
                        str(
                            output_path
                        ),

                    "x1":
                        round(
                            x1,
                            3,
                        ),

                    "y1":
                        round(
                            y1,
                            3,
                        ),

                    "x2":
                        round(
                            x2,
                            3,
                        ),

                    "y2":
                        round(
                            y2,
                            3,
                        ),

                    "bbox_width":
                        round(
                            x2 - x1,
                            3,
                        ),

                    "bbox_height":
                        round(
                            y2 - y1,
                            3,
                        ),

                    "width_ratio":
                        round(
                            width_ratio,
                            8,
                        ),

                    "height_ratio":
                        round(
                            height_ratio,
                            8,
                        ),

                    "area_ratio":
                        round(
                            area_ratio,
                            10,
                        ),

                    "crop_x1":
                        crop_x1,

                    "crop_y1":
                        crop_y1,

                    "crop_x2":
                        crop_x2,

                    "crop_y2":
                        crop_y2,

                    "occluded":
                        assignment[
                            "occluded"
                        ],

                    "track_visible_frames":
                        assignment[
                            "visible_frames"
                        ],
                }
            )

    print()

    print(
        f"[DONE] Task {task_id}"
    )

    print(
        f"  저장 crop: {saved_total}"
    )

    print(
        f"  작은 머리 제외: "
        f"{filtered_small_total}"
    )


# ============================================================
# CSV 저장
# ============================================================

def save_metadata_csv(
    rows,
    output_root: Path,
):

    path = (
        output_root
        /
        "metadata.csv"
    )

    fieldnames = [
        "split",
        "identity",
        "task_id",
        "track_id",
        "frame",
        "image_path",
        "x1",
        "y1",
        "x2",
        "y2",
        "bbox_width",
        "bbox_height",
        "width_ratio",
        "height_ratio",
        "area_ratio",
        "crop_x1",
        "crop_y1",
        "crop_x2",
        "crop_y2",
        "occluded",
        "track_visible_frames",
    ]

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            rows
        )

    return path


def save_identity_stats_csv(
    identity_stats,
    output_root: Path,
):

    path = (
        output_root
        /
        "identity_stats.csv"
    )

    rows = list(
        identity_stats.values()
    )

    rows.sort(
        key=lambda x:
            (
                x[
                    "split"
                ],
                x[
                    "task_id"
                ],
                x[
                    "track_id"
                ],
            )
    )

    fieldnames = [
        "identity",
        "split",
        "task_id",
        "track_id",
        "visible_frames",
        "candidate_crops",
        "saved_crops",
        "small_filtered",
        "invalid_bbox",
    ]

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            rows
        )

    return path


# ============================================================
# 빈 identity 폴더 제거
# ============================================================

def remove_empty_identity_dirs(
    output_root: Path,
):

    for split in [
        "train",
        "val",
        "test",
    ]:

        split_dir = (
            output_root
            /
            split
        )

        if not split_dir.exists():
            continue

        for child in split_dir.iterdir():

            if not child.is_dir():
                continue

            has_files = any(
                child.iterdir()
            )

            if not has_files:
                child.rmdir()


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "CVAT Head Tracking annotation을 "
            "Pool Head ReID 학습용 crop dataset으로 변환"
        )
    )

    parser.add_argument(
        "--server",
        default=CVAT_URL,
    )

    parser.add_argument(
        "--username",
        default=None,
    )

    parser.add_argument(
        "--password",
        default=None,
    )

    parser.add_argument(
        "--token",
        default=None,
    )

    parser.add_argument(
        "--task-start",
        type=int,
        default=DEFAULT_TASK_START,
    )

    parser.add_argument(
        "--task-end",
        type=int,
        default=DEFAULT_TASK_END,
    )

    parser.add_argument(
        "--output",
        default=str(
            OUTPUT_ROOT
        ),
    )

    args = parser.parse_args()

    if (
        args.task_start
        >
        args.task_end
    ):
        raise ValueError(
            "--task-start는 "
            "--task-end보다 작거나 같아야 합니다."
        )

    output_root = Path(
        args.output
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    for split in [
        "train",
        "val",
        "test",
    ]:
        (
            output_root
            /
            split
        ).mkdir(
            parents=True,
            exist_ok=True,
        )

    print()
    print("#" * 100)
    print("CVAT -> Pool Head ReID Dataset Extractor")
    print("#" * 100)

    print(
        f"Server       : {args.server}"
    )

    print(
        f"Tasks        : "
        f"{args.task_start}"
        f" ~ "
        f"{args.task_end}"
    )

    print(
        f"Output       : {output_root}"
    )

    print(
        f"Frame stride : {FRAME_STRIDE}"
    )

    print(
        f"Min visible  : "
        f"{MIN_VISIBLE_FRAMES}"
    )

    print(
        f"Crop margin  : "
        f"{CROP_MARGIN}"
    )

    print(
        f"Exclude occ. : "
        f"{EXCLUDE_OCCLUDED}"
    )

    print("#" * 100)
    print()

    session = login_cvat(
        server=args.server,
        username=args.username,
        password=args.password,
        token=args.token,
    )

    metadata_rows = []

    identity_stats = {}

    for task_id in range(
        args.task_start,
        args.task_end + 1,
    ):

        try:

            process_task(
                session=session,
                server=args.server,
                task_id=task_id,
                output_root=output_root,
                metadata_rows=metadata_rows,
                identity_stats=identity_stats,
            )

        except KeyboardInterrupt:

            print()
            print(
                "[STOP] 사용자 중단"
            )
            raise

        except Exception as exc:

            print()
            print(
                f"[ERROR] Task #{task_id}"
            )
            print(
                repr(exc)
            )

            print(
                "이 Task에서 작업을 중단합니다."
            )

            raise

    remove_empty_identity_dirs(
        output_root
    )

    metadata_path = (
        save_metadata_csv(
            metadata_rows,
            output_root,
        )
    )

    stats_path = (
        save_identity_stats_csv(
            identity_stats,
            output_root,
        )
    )

    # ========================================================
    # 결과 요약
    # ========================================================

    total_ids = len(
        identity_stats
    )

    total_saved = sum(
        item[
            "saved_crops"
        ]
        for item
        in identity_stats.values()
    )

    train_ids = sum(
        1
        for item
        in identity_stats.values()
        if item[
            "split"
        ] == "train"
        and item[
            "saved_crops"
        ] > 0
    )

    val_ids = sum(
        1
        for item
        in identity_stats.values()
        if item[
            "split"
        ] == "val"
        and item[
            "saved_crops"
        ] > 0
    )

    test_ids = sum(
        1
        for item
        in identity_stats.values()
        if item[
            "split"
        ] == "test"
        and item[
            "saved_crops"
        ] > 0
    )

    print()
    print("#" * 100)
    print("추출 완료")
    print("#" * 100)

    print(
        f"Identity 총합 : "
        f"{total_ids}"
    )

    print(
        f"Train IDs     : "
        f"{train_ids}"
    )

    print(
        f"Val IDs       : "
        f"{val_ids}"
    )

    print(
        f"Test IDs      : "
        f"{test_ids}"
    )

    print(
        f"저장 crop     : "
        f"{total_saved}"
    )

    print()

    print(
        f"metadata.csv  : "
        f"{metadata_path}"
    )

    print(
        f"ID 통계       : "
        f"{stats_path}"
    )

    print()

    print(
        "이미지 위치:"
    )

    print(
        output_root
    )

    print("#" * 100)


if __name__ == "__main__":
    main()