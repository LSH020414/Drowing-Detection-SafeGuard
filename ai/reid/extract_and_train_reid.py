from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path

from ultralytics import YOLO


# ============================================================
# 기본 경로
# ============================================================

BASE_DIR = Path(
    r"D:\임베디드 경진대회\머리 추적\CVAT용 영상"
)

EXTRACT_SCRIPT = (
    BASE_DIR
    /
    "extract_cvat_reid_dataset.py"
)

DATASET_ROOT = Path(
    r"D:\임베디드 경진대회\머리 추적\ReID_dataset"
)

RUNS_ROOT = Path(
    r"D:\임베디드 경진대회\머리 추적\ReID_runs"
)

RUN_NAME = "pool_head_reid_yolo11n_cls"


# ============================================================
# 학습 설정
# ============================================================

BASE_MODEL = "yolo11n-cls.pt"

EPOCHS = 50

# 머리 crop은 작은 이미지가 많으므로
# 224가 우선적으로 무난함
IMAGE_SIZE = 224

BATCH_SIZE = 64

DEVICE = 0

WORKERS = 4

PATIENCE = 10

SEED = 42


# ============================================================
# 데이터 검증 기준
# ============================================================

# Identity 하나당 최소 crop 수
MIN_IMAGES_PER_ID = 5

# split별 최소 identity 수
MIN_TRAIN_IDS = 20
MIN_VAL_IDS = 5
MIN_TEST_IDS = 5

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


# ============================================================
# 출력
# ============================================================

def section(title: str):
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


# ============================================================
# 이미지 수 계산
# ============================================================

def count_images(folder: Path) -> int:

    if not folder.exists():
        return 0

    return sum(
        1
        for path in folder.iterdir()
        if (
            path.is_file()
            and
            path.suffix.lower()
            in IMAGE_EXTENSIONS
        )
    )


# ============================================================
# Split 검사
# ============================================================

def analyze_split(
    dataset_root: Path,
    split: str,
):

    split_dir = (
        dataset_root
        /
        split
    )

    result = {
        "split": split,
        "identities": 0,
        "images": 0,
        "valid_identities": 0,
        "invalid_identities": [],
    }

    if not split_dir.exists():
        return result

    identity_dirs = sorted(
        [
            path
            for path in split_dir.iterdir()
            if path.is_dir()
        ]
    )

    result[
        "identities"
    ] = len(identity_dirs)

    for identity_dir in identity_dirs:

        image_count = count_images(
            identity_dir
        )

        result[
            "images"
        ] += image_count

        if (
            image_count
            >=
            MIN_IMAGES_PER_ID
        ):
            result[
                "valid_identities"
            ] += 1

        else:
            result[
                "invalid_identities"
            ].append(
                (
                    identity_dir.name,
                    image_count,
                )
            )

    return result


# ============================================================
# 너무 적은 ID 폴더 제거
#
# ReID/classification 학습에서
# 1~2장짜리 identity는 학습 가치가 매우 낮으므로
# 자동으로 제외한다.
# ============================================================

def remove_too_small_identities(
    dataset_root: Path,
):

    removed = []

    for split in (
        "train",
        "val",
        "test",
    ):

        split_dir = (
            dataset_root
            /
            split
        )

        if not split_dir.exists():
            continue

        for identity_dir in list(
            split_dir.iterdir()
        ):

            if not identity_dir.is_dir():
                continue

            image_count = count_images(
                identity_dir
            )

            if (
                image_count
                <
                MIN_IMAGES_PER_ID
            ):

                removed.append(
                    (
                        split,
                        identity_dir.name,
                        image_count,
                    )
                )

                shutil.rmtree(
                    identity_dir
                )

    return removed


# ============================================================
# 데이터셋 검증
# ============================================================

def validate_dataset(
    dataset_root: Path,
):

    section(
        "[2/3] ReID Dataset Validation"
    )

    if not dataset_root.exists():

        raise RuntimeError(
            f"데이터셋 폴더가 없습니다:\n"
            f"{dataset_root}"
        )

    metadata_path = (
        dataset_root
        /
        "metadata.csv"
    )

    stats_path = (
        dataset_root
        /
        "identity_stats.csv"
    )

    if not metadata_path.exists():

        raise RuntimeError(
            "metadata.csv가 없습니다. "
            "CVAT 추출이 정상적으로 끝나지 않은 것 같습니다."
        )

    if not stats_path.exists():

        raise RuntimeError(
            "identity_stats.csv가 없습니다."
        )

    # --------------------------------------------------------
    # 너무 적은 crop ID 제거
    # --------------------------------------------------------

    removed = (
        remove_too_small_identities(
            dataset_root
        )
    )

    if removed:

        print()
        print(
            f"[INFO] crop {MIN_IMAGES_PER_ID}장 미만 "
            f"Identity {len(removed)}개 제외"
        )

        for (
            split,
            identity,
            count,
        ) in removed[:30]:

            print(
                f"  {split:5s} | "
                f"{identity} | "
                f"{count} images"
            )

        if len(removed) > 30:

            print(
                f"  ... "
                f"{len(removed) - 30}개 추가"
            )

    # --------------------------------------------------------
    # 다시 분석
    # --------------------------------------------------------

    train = analyze_split(
        dataset_root,
        "train",
    )

    val = analyze_split(
        dataset_root,
        "val",
    )

    test = analyze_split(
        dataset_root,
        "test",
    )

    print()
    print(
        f"{'Split':<10}"
        f"{'IDs':>10}"
        f"{'Images':>15}"
    )

    print(
        "-" * 40
    )

    for info in (
        train,
        val,
        test,
    ):

        print(
            f"{info['split']:<10}"
            f"{info['valid_identities']:>10}"
            f"{info['images']:>15}"
        )

    print()

    # --------------------------------------------------------
    # 최소 기준 검사
    # --------------------------------------------------------

    errors = []

    if (
        train[
            "valid_identities"
        ]
        <
        MIN_TRAIN_IDS
    ):
        errors.append(
            f"Train Identity가 너무 적음: "
            f"{train['valid_identities']} "
            f"< {MIN_TRAIN_IDS}"
        )

    if (
        val[
            "valid_identities"
        ]
        <
        MIN_VAL_IDS
    ):
        errors.append(
            f"Val Identity가 너무 적음: "
            f"{val['valid_identities']} "
            f"< {MIN_VAL_IDS}"
        )

    if (
        test[
            "valid_identities"
        ]
        <
        MIN_TEST_IDS
    ):
        errors.append(
            f"Test Identity가 너무 적음: "
            f"{test['valid_identities']} "
            f"< {MIN_TEST_IDS}"
        )

    if errors:

        print(
            "[ERROR] 학습 전 데이터 검증 실패"
        )

        for error in errors:
            print(
                "  -",
                error,
            )

        raise RuntimeError(
            "데이터셋 검증 실패. "
            "자동 학습을 시작하지 않습니다."
        )

    print(
        "[OK] 데이터셋 기본 검증 통과"
    )

    print()
    print(
        "주의: 이 검증은 ID 폴더의 수와 이미지 수만 검사합니다."
    )

    print(
        "한 ID 폴더에 서로 다른 사람이 섞였는지는 "
        "CVAT 보정 품질에 따라 결정됩니다."
    )

    return {
        "train": train,
        "val": val,
        "test": test,
    }


# ============================================================
# CVAT -> ReID crop 추출
# ============================================================

def run_extraction(
    username: str,
    password: str,
    task_start: int,
    task_end: int,
    server: str,
    clean_dataset: bool,
):

    section(
        "[1/3] CVAT -> ReID Crop Extraction"
    )

    if not EXTRACT_SCRIPT.exists():

        raise FileNotFoundError(
            f"추출 스크립트를 찾을 수 없습니다:\n"
            f"{EXTRACT_SCRIPT}"
        )

    # --------------------------------------------------------
    # 기존 dataset 삭제 옵션
    # --------------------------------------------------------

    if (
        clean_dataset
        and
        DATASET_ROOT.exists()
    ):

        print(
            "[INFO] 기존 ReID_dataset 삭제:"
        )

        print(
            DATASET_ROOT
        )

        shutil.rmtree(
            DATASET_ROOT
        )

    command = [
        sys.executable,
        str(
            EXTRACT_SCRIPT
        ),
        "--server",
        server,
        "--username",
        username,
        "--password",
        password,
        "--task-start",
        str(
            task_start
        ),
        "--task-end",
        str(
            task_end
        ),
        "--output",
        str(
            DATASET_ROOT
        ),
    ]

    print()
    print(
        f"Task 범위: "
        f"{task_start} ~ {task_end}"
    )

    print(
        f"Output: {DATASET_ROOT}"
    )

    print()
    print(
        "[START] Crop 추출 시작"
    )
    print()

    result = subprocess.run(
        command,
        check=False,
    )

    if (
        result.returncode
        !=
        0
    ):

        raise RuntimeError(
            "CVAT crop 추출 중 오류가 발생했습니다. "
            "ReID 학습을 시작하지 않습니다."
        )

    print()
    print(
        "[OK] CVAT crop 추출 완료"
    )


# ============================================================
# ReID 학습
# ============================================================

def train_reid(
    dataset_root: Path,
):

    section(
        "[3/3] Pool Head ReID Training"
    )

    RUNS_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    expected_run_dir = (
        RUNS_ROOT
        /
        RUN_NAME
    )

    # --------------------------------------------------------
    # 기존 동일 run 삭제
    # --------------------------------------------------------

    if expected_run_dir.exists():

        print(
            "[INFO] 기존 학습 결과 삭제:"
        )

        print(
            expected_run_dir
        )

        shutil.rmtree(
            expected_run_dir
        )

    print()
    print(
        f"Base model : {BASE_MODEL}"
    )

    print(
        f"Dataset    : {dataset_root}"
    )

    print(
        f"Epochs     : {EPOCHS}"
    )

    print(
        f"Image size : {IMAGE_SIZE}"
    )

    print(
        f"Batch      : {BATCH_SIZE}"
    )

    print(
        f"Device     : {DEVICE}"
    )

    print(
        f"Patience   : {PATIENCE}"
    )

    print()

    print(
        "[START] Pool Head ReID fine-tuning"
    )

    # --------------------------------------------------------
    # YOLO classification model
    #
    # 각 ID 폴더 = 하나의 classification identity
    #
    # 학습 후 classifier의 backbone/embedding을
    # BoT-SORT ReID feature extractor로 사용한다.
    # --------------------------------------------------------

    model = YOLO(
        BASE_MODEL
    )

    results = model.train(
        data=str(
            dataset_root
        ),

        epochs=EPOCHS,

        imgsz=IMAGE_SIZE,

        batch=BATCH_SIZE,

        device=DEVICE,

        workers=WORKERS,

        patience=PATIENCE,

        seed=SEED,

        project=str(
            RUNS_ROOT
        ),

        name=RUN_NAME,

        exist_ok=True,

        pretrained=True,

        verbose=True,
    )

    print()
    print(
        "[OK] ReID training 완료"
    )

    # --------------------------------------------------------
    # 결과 경로 확인
    # --------------------------------------------------------

    best_path = (
        expected_run_dir
        /
        "weights"
        /
        "best.pt"
    )

    last_path = (
        expected_run_dir
        /
        "weights"
        /
        "last.pt"
    )

    print()
    print(
        "Best model:"
    )

    print(
        best_path
    )

    print()
    print(
        "Last model:"
    )

    print(
        last_path
    )

    if not best_path.exists():

        print()
        print(
            "[WARN] 예상 위치에서 best.pt를 찾지 못했습니다."
        )

        print(
            "Ultralytics 출력 로그의 save_dir을 확인하세요."
        )

    return best_path


# ============================================================
# 결과 요약 저장
# ============================================================

def save_pipeline_summary(
    dataset_stats,
    best_model: Path,
):

    summary_path = (
        RUNS_ROOT
        /
        RUN_NAME
        /
        "pipeline_summary.csv"
    )

    summary_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []

    for split in (
        "train",
        "val",
        "test",
    ):

        info = (
            dataset_stats[
                split
            ]
        )

        rows.append(
            {
                "split":
                    split,

                "identities":
                    info[
                        "valid_identities"
                    ],

                "images":
                    info[
                        "images"
                    ],

                "best_model":
                    str(
                        best_model
                    ),
            }
        )

    with open(
        summary_path,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "split",
                "identities",
                "images",
                "best_model",
            ],
        )

        writer.writeheader()

        writer.writerows(
            rows
        )

    return summary_path


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "CVAT Tracking GT 추출 -> "
            "Pool Head ReID dataset 검증 -> "
            "YOLO classification ReID fine-tuning"
        )
    )

    parser.add_argument(
        "--server",
        default="http://localhost:8080",
    )

    parser.add_argument(
        "--username",
        required=True,
    )

    parser.add_argument(
        "--password",
        required=True,
    )

    parser.add_argument(
        "--task-start",
        type=int,
        default=86,
    )

    parser.add_argument(
        "--task-end",
        type=int,
        default=106,
    )

    parser.add_argument(
        "--clean",
        action="store_true",
        help=(
            "기존 ReID_dataset을 삭제하고 "
            "처음부터 다시 추출"
        ),
    )

    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help=(
            "이미 추출된 ReID_dataset을 사용하고 "
            "추출 단계 생략"
        ),
    )

    args = parser.parse_args()

    print()
    print("#" * 100)
    print(
        "CVAT -> Pool Head ReID Auto Pipeline"
    )
    print("#" * 100)

    print(
        f"CVAT server : {args.server}"
    )

    print(
        f"Task range  : "
        f"{args.task_start} ~ {args.task_end}"
    )

    print(
        f"Dataset     : {DATASET_ROOT}"
    )

    print(
        f"Runs        : {RUNS_ROOT}"
    )

    print(
        f"Epochs      : {EPOCHS}"
    )

    print("#" * 100)

    # ========================================================
    # 1. Extract
    # ========================================================

    if not args.skip_extract:

        run_extraction(
            username=args.username,
            password=args.password,
            task_start=args.task_start,
            task_end=args.task_end,
            server=args.server,
            clean_dataset=args.clean,
        )

    else:

        section(
            "[1/3] CVAT Extraction SKIPPED"
        )

        print(
            "기존 ReID_dataset을 사용합니다:"
        )

        print(
            DATASET_ROOT
        )

    # ========================================================
    # 2. Validate
    # ========================================================

    dataset_stats = (
        validate_dataset(
            DATASET_ROOT
        )
    )

    # ========================================================
    # 3. Train
    # ========================================================

    best_model = (
        train_reid(
            DATASET_ROOT
        )
    )

    # ========================================================
    # Summary
    # ========================================================

    summary_path = (
        save_pipeline_summary(
            dataset_stats,
            best_model,
        )
    )

    section(
        "PIPELINE COMPLETE"
    )

    print(
        "CVAT crop 추출:"
    )

    print(
        DATASET_ROOT
    )

    print()
    print(
        "Pool Head ReID best model:"
    )

    print(
        best_model
    )

    print()
    print(
        "Summary:"
    )

    print(
        summary_path
    )

    print()
    print(
        "다음 단계:"
    )

    print(
        "best.pt를 BoT-SORT ReID 모델에 연결하여 "
        "generic ReID와 tracking 성능을 비교합니다."
    )


if __name__ == "__main__":
    main()