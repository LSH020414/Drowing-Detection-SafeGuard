from __future__ import annotations

import csv
import random
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from ultralytics import YOLO


# =============================================================================
# PATHS
# =============================================================================

REID_MODEL = Path(
    r"D:\임베디드 경진대회\머리 추적\ReID_runs"
    r"\pool_head_reid_yolo11n_cls\weights\best.pt"
)

DATASET_ROOT = Path(
    r"D:\임베디드 경진대회\머리 추적\ReID_dataset"
)

TEST_ROOT = DATASET_ROOT / "test"

OUTPUT_DIR = Path(
    r"D:\임베디드 경진대회\머리 추적\트레킹 테스트"
    r"\Pool_ReID_similarity_test"
)


# =============================================================================
# SETTINGS
# =============================================================================

DEVICE = 0 if torch.cuda.is_available() else "cpu"

IMG_SIZE = 224

SEED = 42

MAX_IMAGES_PER_ID = 30

MAX_SAME_PAIRS_PER_ID = 100

MAX_DIFF_PAIRS_PER_ID_PAIR = 20

MAX_TOTAL_DIFF_PAIRS = 30000


IMAGE_SUFFIXES = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


# =============================================================================
# BASIC UTILITIES
# =============================================================================

def l2_normalize(
    vector: np.ndarray,
) -> np.ndarray:

    vector = np.asarray(
        vector,
        dtype=np.float32,
    ).reshape(-1)

    norm = float(
        np.linalg.norm(vector)
    )

    if norm <= 1e-12:
        return vector

    return vector / norm


def cosine_similarity(
    first: np.ndarray,
    second: np.ndarray,
) -> float:

    first = l2_normalize(first)
    second = l2_normalize(second)

    similarity = np.dot(
        first,
        second,
    )

    return float(
        np.clip(
            similarity,
            -1.0,
            1.0,
        )
    )


def percentile(
    values: list[float],
    q: float,
) -> float:

    if not values:
        return float("nan")

    return float(
        np.percentile(
            np.asarray(
                values,
                dtype=np.float32,
            ),
            q,
        )
    )


# =============================================================================
# DATASET HELPERS
# =============================================================================

def parse_task_id(
    identity_name: str,
) -> str:

    match = re.search(
        r"task(\d+)",
        identity_name,
        re.IGNORECASE,
    )

    if match:
        return match.group(1)

    return "UNKNOWN"


def collect_dataset():

    if not TEST_ROOT.is_dir():

        raise FileNotFoundError(
            f"test 폴더가 없습니다:\n{TEST_ROOT}"
        )

    identities = {}

    for identity_dir in sorted(
        TEST_ROOT.iterdir()
    ):

        if not identity_dir.is_dir():
            continue

        images = []

        for path in identity_dir.rglob("*"):

            if not path.is_file():
                continue

            if (
                path.suffix.lower()
                not in IMAGE_SUFFIXES
            ):
                continue

            images.append(
                path
            )

        if len(images) < 2:
            continue

        random.shuffle(
            images
        )

        images = images[
            :MAX_IMAGES_PER_ID
        ]

        identities[
            identity_dir.name
        ] = images

    return identities


# =============================================================================
# EMBEDDING EXTRACTION
# =============================================================================

def extract_embedding(
    model: YOLO,
    image_path: Path,
):

    image = cv2.imread(
        str(image_path)
    )

    if image is None:

        print(
            f"[WARN] 이미지 읽기 실패: "
            f"{image_path}"
        )

        return None

    try:

        outputs = model.embed(
            source=image,
            imgsz=IMG_SIZE,
            device=DEVICE,
            verbose=False,
        )

        outputs = list(
            outputs
        )

        if not outputs:

            print(
                f"[WARN] embedding 없음: "
                f"{image_path}"
            )

            return None

        embedding = outputs[0]

        if isinstance(
            embedding,
            torch.Tensor,
        ):

            vector = (
                embedding
                .detach()
                .float()
                .cpu()
                .numpy()
                .reshape(-1)
            )

        else:

            vector = np.asarray(
                embedding,
                dtype=np.float32,
            ).reshape(-1)

        return l2_normalize(
            vector
        )

    except Exception as error:

        print(
            f"[ERROR] Embedding 실패:"
        )

        print(
            image_path
        )

        print(
            error
        )

        return None


def extract_all_embeddings(
    model: YOLO,
    identities,
):

    embeddings = {}

    total_images = sum(
        len(paths)
        for paths in identities.values()
    )

    processed = 0

    for identity_name, image_paths in identities.items():

        identity_embeddings = []

        for image_path in image_paths:

            embedding = extract_embedding(
                model,
                image_path,
            )

            processed += 1

            if embedding is not None:

                identity_embeddings.append(
                    (
                        image_path,
                        embedding,
                    )
                )

            if (
                processed % 100 == 0
                or
                processed == total_images
            ):

                print(
                    f"Embedding: "
                    f"{processed}/{total_images}"
                )

        if len(identity_embeddings) >= 2:

            embeddings[
                identity_name
            ] = identity_embeddings

    return embeddings


# =============================================================================
# SAME PERSON PAIRS
# =============================================================================

def create_same_pairs(
    embeddings,
):

    rows = []

    for identity_name, items in embeddings.items():

        all_pairs = []

        for first_index in range(
            len(items)
        ):

            for second_index in range(
                first_index + 1,
                len(items),
            ):

                all_pairs.append(
                    (
                        first_index,
                        second_index,
                    )
                )

        random.shuffle(
            all_pairs
        )

        all_pairs = all_pairs[
            :MAX_SAME_PAIRS_PER_ID
        ]

        task_id = parse_task_id(
            identity_name
        )

        for first_index, second_index in all_pairs:

            path_a, embedding_a = (
                items[first_index]
            )

            path_b, embedding_b = (
                items[second_index]
            )

            similarity = cosine_similarity(
                embedding_a,
                embedding_b,
            )

            rows.append(
                {
                    "pair_type": "SAME",
                    "task_a": task_id,
                    "task_b": task_id,
                    "identity_a": identity_name,
                    "identity_b": identity_name,
                    "image_a": str(path_a),
                    "image_b": str(path_b),
                    "similarity": similarity,
                }
            )

    return rows


# =============================================================================
# DIFFERENT PERSON PAIRS
#
# 같은 task 내부에서 다른 identity끼리 비교.
# 같은 배경/조명/카메라 조건이라 hard negative 역할을 한다.
# =============================================================================

def create_different_pairs(
    embeddings,
):

    rows = []

    task_groups = defaultdict(
        list
    )

    for identity_name in embeddings:

        task_id = parse_task_id(
            identity_name
        )

        task_groups[
            task_id
        ].append(
            identity_name
        )

    for task_id, identity_names in task_groups.items():

        if len(identity_names) < 2:
            continue

        for first_index in range(
            len(identity_names)
        ):

            for second_index in range(
                first_index + 1,
                len(identity_names),
            ):

                identity_a = (
                    identity_names[
                        first_index
                    ]
                )

                identity_b = (
                    identity_names[
                        second_index
                    ]
                )

                images_a = embeddings[
                    identity_a
                ]

                images_b = embeddings[
                    identity_b
                ]

                all_pairs = []

                for image_a_index in range(
                    len(images_a)
                ):

                    for image_b_index in range(
                        len(images_b)
                    ):

                        all_pairs.append(
                            (
                                image_a_index,
                                image_b_index,
                            )
                        )

                random.shuffle(
                    all_pairs
                )

                all_pairs = all_pairs[
                    :MAX_DIFF_PAIRS_PER_ID_PAIR
                ]

                for image_a_index, image_b_index in all_pairs:

                    path_a, embedding_a = (
                        images_a[
                            image_a_index
                        ]
                    )

                    path_b, embedding_b = (
                        images_b[
                            image_b_index
                        ]
                    )

                    similarity = cosine_similarity(
                        embedding_a,
                        embedding_b,
                    )

                    rows.append(
                        {
                            "pair_type": "DIFFERENT",
                            "task_a": task_id,
                            "task_b": task_id,
                            "identity_a": identity_a,
                            "identity_b": identity_b,
                            "image_a": str(path_a),
                            "image_b": str(path_b),
                            "similarity": similarity,
                        }
                    )

                    if (
                        len(rows)
                        >=
                        MAX_TOTAL_DIFF_PAIRS
                    ):

                        return rows

    return rows


# =============================================================================
# ROC / THRESHOLD
# =============================================================================

def calculate_threshold_metrics(
    same_values,
    different_values,
):

    same_array = np.asarray(
        same_values,
        dtype=np.float32,
    )

    diff_array = np.asarray(
        different_values,
        dtype=np.float32,
    )

    values = np.concatenate(
        [
            same_array,
            diff_array,
        ]
    )

    thresholds = np.linspace(
        float(values.min()),
        float(values.max()),
        1001,
    )

    best = None

    roc_points = []

    for threshold in thresholds:

        # SAME인데 similarity가 threshold 이상
        true_positive_rate = float(
            np.mean(
                same_array
                >=
                threshold
            )
        )

        false_reject_rate = (
            1.0
            -
            true_positive_rate
        )

        # DIFFERENT인데 threshold 이상이라
        # 같은 사람으로 잘못 받아들이는 비율
        false_accept_rate = float(
            np.mean(
                diff_array
                >=
                threshold
            )
        )

        true_negative_rate = (
            1.0
            -
            false_accept_rate
        )

        balanced_accuracy = (
            true_positive_rate
            +
            true_negative_rate
        ) / 2.0

        youden_j = (
            true_positive_rate
            -
            false_accept_rate
        )

        eer_difference = abs(
            false_accept_rate
            -
            false_reject_rate
        )

        record = {
            "threshold": float(
                threshold
            ),

            "tpr": true_positive_rate,

            "far": false_accept_rate,

            "frr": false_reject_rate,

            "balanced_accuracy":
                balanced_accuracy,

            "youden_j":
                youden_j,

            "eer_difference":
                eer_difference,
        }

        roc_points.append(
            record
        )

        if (
            best is None
            or
            record["youden_j"]
            >
            best["youden_j"]
        ):

            best = record

    # -------------------------------------------------------------------------
    # ROC AUC
    # -------------------------------------------------------------------------

    ordered = sorted(
        roc_points,
        key=lambda item:
            item["far"],
    )

    x = np.asarray(
        [
            item["far"]
            for item in ordered
        ],
        dtype=np.float64,
    )

    y = np.asarray(
        [
            item["tpr"]
            for item in ordered
        ],
        dtype=np.float64,
    )

    auc = float(
        np.trapezoid(
            y,
            x,
        )
    )

    eer_record = min(
        roc_points,
        key=lambda item:
            item["eer_difference"],
    )

    return (
        best,
        eer_record,
        auc,
        roc_points,
    )


# =============================================================================
# CSV
# =============================================================================

def write_csv(
    path: Path,
    rows,
):

    if not rows:
        return

    fieldnames = list(
        rows[0].keys()
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


# =============================================================================
# PLOTS
# =============================================================================

def save_histogram(
    same_values,
    different_values,
    threshold,
):

    plt.figure(
        figsize=(
            10,
            6,
        )
    )

    plt.hist(
        same_values,
        bins=50,
        alpha=0.55,
        density=True,
        label="Same person",
    )

    plt.hist(
        different_values,
        bins=50,
        alpha=0.55,
        density=True,
        label="Different person",
    )

    plt.axvline(
        threshold,
        linestyle="--",
        linewidth=2,
        label=f"Best threshold = {threshold:.3f}",
    )

    plt.xlabel(
        "Cosine similarity"
    )

    plt.ylabel(
        "Density"
    )

    plt.title(
        "Pool ReID Similarity Distribution"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        /
        "similarity_distribution.png",
        dpi=180,
    )

    plt.close()


def save_roc_curve(
    roc_rows,
):

    ordered = sorted(
        roc_rows,
        key=lambda item:
            item["far"],
    )

    fars = [
        item["far"]
        for item in ordered
    ]

    tprs = [
        item["tpr"]
        for item in ordered
    ]

    plt.figure(
        figsize=(
            7,
            7,
        )
    )

    plt.plot(
        fars,
        tprs,
        label="Pool ReID",
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random",
    )

    plt.xlabel(
        "False Accept Rate"
    )

    plt.ylabel(
        "True Positive Rate"
    )

    plt.title(
        "Pool ReID ROC Curve"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        /
        "roc_curve.png",
        dpi=180,
    )

    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def main():

    random.seed(
        SEED
    )

    np.random.seed(
        SEED
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 80)
    print("POOL HEAD ReID SIMILARITY EVALUATION")
    print("=" * 80)

    print()

    print(
        "Model:"
    )

    print(
        REID_MODEL
    )

    print()

    print(
        "Test dataset:"
    )

    print(
        TEST_ROOT
    )

    print()

    print(
        "Device:",
        DEVICE
    )

    # -------------------------------------------------------------------------
    # Check files
    # -------------------------------------------------------------------------

    if not REID_MODEL.is_file():

        raise FileNotFoundError(
            f"ReID 모델이 없습니다:\n"
            f"{REID_MODEL}"
        )

    if not TEST_ROOT.is_dir():

        raise FileNotFoundError(
            f"ReID test 데이터셋이 없습니다:\n"
            f"{TEST_ROOT}"
        )

    # -------------------------------------------------------------------------
    # Dataset
    # -------------------------------------------------------------------------

    identities = collect_dataset()

    print()
    print(
        "Test identities:",
        len(identities),
    )

    print(
        "Test images:",
        sum(
            len(items)
            for items in identities.values()
        ),
    )

    if len(identities) < 2:

        raise RuntimeError(
            "평가 가능한 identity가 2개 미만입니다."
        )

    # -------------------------------------------------------------------------
    # Load model
    # -------------------------------------------------------------------------

    print()
    print(
        "ReID 모델 로딩..."
    )

    model = YOLO(
        str(REID_MODEL)
    )

    # -------------------------------------------------------------------------
    # Embeddings
    # -------------------------------------------------------------------------

    print()
    print(
        "[1/4] Embedding 추출"
    )

    embeddings = extract_all_embeddings(
        model,
        identities,
    )

    print()
    print(
        "Embedding identities:",
        len(embeddings),
    )

    if len(embeddings) < 2:

        raise RuntimeError(
            "Embedding 생성 가능한 identity가 부족합니다."
        )

    # -------------------------------------------------------------------------
    # SAME
    # -------------------------------------------------------------------------

    print()
    print(
        "[2/4] Same-person pair 생성"
    )

    same_rows = create_same_pairs(
        embeddings
    )

    print(
        "Same pairs:",
        len(same_rows),
    )

    # -------------------------------------------------------------------------
    # DIFFERENT
    # -------------------------------------------------------------------------

    print()
    print(
        "[3/4] Different-person pair 생성"
    )

    different_rows = create_different_pairs(
        embeddings
    )

    print(
        "Different pairs:",
        len(different_rows),
    )

    if not same_rows:

        raise RuntimeError(
            "Same-person pair가 생성되지 않았습니다."
        )

    if not different_rows:

        raise RuntimeError(
            "Different-person pair가 생성되지 않았습니다.\n"
            "test dataset에서 같은 task 안에 "
            "2명 이상의 identity가 있는지 확인하세요."
        )

    # -------------------------------------------------------------------------
    # Similarities
    # -------------------------------------------------------------------------

    same_values = [
        float(
            row["similarity"]
        )
        for row in same_rows
    ]

    different_values = [
        float(
            row["similarity"]
        )
        for row in different_rows
    ]

    # -------------------------------------------------------------------------
    # ROC / threshold
    # -------------------------------------------------------------------------

    print()
    print(
        "[4/4] Threshold / ROC 계산"
    )

    (
        best,
        eer,
        auc,
        roc_rows,
    ) = calculate_threshold_metrics(
        same_values,
        different_values,
    )

    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------

    same_summary = {
        "group": "SAME",

        "count":
            len(
                same_values
            ),

        "mean":
            mean(
                same_values
            ),

        "median":
            median(
                same_values
            ),

        "p05":
            percentile(
                same_values,
                5,
            ),

        "p95":
            percentile(
                same_values,
                95,
            ),

        "min":
            min(
                same_values
            ),

        "max":
            max(
                same_values
            ),
    }

    different_summary = {
        "group": "DIFFERENT",

        "count":
            len(
                different_values
            ),

        "mean":
            mean(
                different_values
            ),

        "median":
            median(
                different_values
            ),

        "p05":
            percentile(
                different_values,
                5,
            ),

        "p95":
            percentile(
                different_values,
                95,
            ),

        "min":
            min(
                different_values
            ),

        "max":
            max(
                different_values
            ),
    }

    summary_rows = [
        same_summary,
        different_summary,
    ]

    threshold_summary = [
        {
            "roc_auc": auc,

            "best_threshold":
                best["threshold"],

            "best_tpr":
                best["tpr"],

            "best_far":
                best["far"],

            "best_frr":
                best["frr"],

            "best_balanced_accuracy":
                best[
                    "balanced_accuracy"
                ],

            "best_youden_j":
                best["youden_j"],

            "eer_threshold":
                eer["threshold"],

            "eer_far":
                eer["far"],

            "eer_frr":
                eer["frr"],
        }
    ]

    # -------------------------------------------------------------------------
    # Save files
    # -------------------------------------------------------------------------

    write_csv(
        OUTPUT_DIR
        /
        "same_pairs.csv",
        same_rows,
    )

    write_csv(
        OUTPUT_DIR
        /
        "different_pairs.csv",
        different_rows,
    )

    write_csv(
        OUTPUT_DIR
        /
        "similarity_summary.csv",
        summary_rows,
    )

    write_csv(
        OUTPUT_DIR
        /
        "threshold_summary.csv",
        threshold_summary,
    )

    write_csv(
        OUTPUT_DIR
        /
        "roc_thresholds.csv",
        roc_rows,
    )

    save_histogram(
        same_values,
        different_values,
        best["threshold"],
    )

    save_roc_curve(
        roc_rows,
    )

    # -------------------------------------------------------------------------
    # Console result
    # -------------------------------------------------------------------------

    print()
    print("=" * 80)
    print("RESULT")
    print("=" * 80)

    print()
    print(
        "[SAME PERSON]"
    )

    print(
        f"count  = "
        f"{len(same_values)}"
    )

    print(
        f"mean   = "
        f"{mean(same_values):.4f}"
    )

    print(
        f"median = "
        f"{median(same_values):.4f}"
    )

    print(
        f"p05    = "
        f"{percentile(same_values, 5):.4f}"
    )

    print(
        f"p95    = "
        f"{percentile(same_values, 95):.4f}"
    )

    print(
        f"min    = "
        f"{min(same_values):.4f}"
    )

    print(
        f"max    = "
        f"{max(same_values):.4f}"
    )

    print()
    print(
        "[DIFFERENT PERSON]"
    )

    print(
        f"count  = "
        f"{len(different_values)}"
    )

    print(
        f"mean   = "
        f"{mean(different_values):.4f}"
    )

    print(
        f"median = "
        f"{median(different_values):.4f}"
    )

    print(
        f"p05    = "
        f"{percentile(different_values, 5):.4f}"
    )

    print(
        f"p95    = "
        f"{percentile(different_values, 95):.4f}"
    )

    print(
        f"min    = "
        f"{min(different_values):.4f}"
    )

    print(
        f"max    = "
        f"{max(different_values):.4f}"
    )

    print()
    print(
        "[DISCRIMINATION]"
    )

    print(
        f"ROC-AUC = "
        f"{auc:.4f}"
    )

    print()
    print(
        "Best threshold "
        "(Youden J)"
    )

    print(
        f"threshold = "
        f"{best['threshold']:.4f}"
    )

    print(
        f"TPR       = "
        f"{best['tpr']:.4f}"
    )

    print(
        f"FAR       = "
        f"{best['far']:.4f}"
    )

    print(
        f"FRR       = "
        f"{best['frr']:.4f}"
    )

    print(
        f"Balanced accuracy = "
        f"{best['balanced_accuracy']:.4f}"
    )

    print()
    print(
        "Approx EER"
    )

    print(
        f"threshold = "
        f"{eer['threshold']:.4f}"
    )

    print(
        f"FAR       = "
        f"{eer['far']:.4f}"
    )

    print(
        f"FRR       = "
        f"{eer['frr']:.4f}"
    )

    print()
    print(
        "결과 저장:"
    )

    print(
        OUTPUT_DIR
    )

    print()
    print(
        "생성 파일:"
    )

    print(
        "same_pairs.csv"
    )

    print(
        "different_pairs.csv"
    )

    print(
        "similarity_summary.csv"
    )

    print(
        "threshold_summary.csv"
    )

    print(
        "roc_thresholds.csv"
    )

    print(
        "similarity_distribution.png"
    )

    print(
        "roc_curve.png"
    )


if __name__ == "__main__":
    main()