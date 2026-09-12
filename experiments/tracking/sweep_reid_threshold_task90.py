from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import cv2
import torch


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

COMPARE_SCRIPT = Path(
    r"D:\임베디드 경진대회\머리 추적\트레킹 테스트"
    r"\compare_V10_V11_tracking.py"
)

EVALUATION_SCRIPT = Path(
    r"D:\임베디드 경진대회\머리 추적\트레킹 테스트"
    r"\evaluate_task90_tracking_gt.py"
)

OUTPUT_ROOT = Path(
    r"D:\임베디드 경진대회\머리 추적\트레킹 테스트"
    r"\Task90_ReID_Threshold_Sweep"
)


# =============================================================================
# FINAL GEOMETRY SETTING
# =============================================================================

GEOMETRY_MAX_SCORE = 0.45


# =============================================================================
# REID SWEEP
# =============================================================================

REID_THRESHOLDS = [
    0.80,
    0.82,
    0.84,
    0.86,
]


# =============================================================================
# GT EVALUATION
# =============================================================================

PRIMARY_IOU = 0.50
SECONDARY_IOU = 0.30

DEVICE = 0 if torch.cuda.is_available() else "cpu"


# =============================================================================
# DYNAMIC IMPORT
# =============================================================================

def load_python_module(
    path: Path,
    module_name: str,
):

    if not path.is_file():
        raise FileNotFoundError(
            f"파일 없음:\n{path}"
        )

    spec = importlib.util.spec_from_file_location(
        module_name,
        path,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"모듈 로딩 실패:\n{path}"
        )

    module = importlib.util.module_from_spec(
        spec
    )

    # Python 3.13 dataclass 대응
    sys.modules[module_name] = module

    spec.loader.exec_module(
        module
    )

    return module


# =============================================================================
# CSV
# =============================================================================

def write_csv(
    path: Path,
    rows: list[dict],
):

    if not rows:
        return

    keys = []
    seen = set()

    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                keys.append(key)

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
        writer.writerows(rows)


# =============================================================================
# VIDEO INFO
# =============================================================================

def get_video_info(
    path: Path,
):

    capture = cv2.VideoCapture(
        str(path)
    )

    if not capture.isOpened():
        raise RuntimeError(
            f"영상 열기 실패:\n{path}"
        )

    fps = float(
        capture.get(
            cv2.CAP_PROP_FPS
        )
    )

    width = int(
        capture.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        capture.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    frames = int(
        capture.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    capture.release()

    return fps, width, height, frames


# =============================================================================
# GEOMETRY SCORE
# =============================================================================

def calculate_geometry_score(
    manager,
    track,
    details,
    compare_module,
):

    distance_ratio = float(
        details["distance_ratio"]
    )

    normalized_distance = (
        distance_ratio
        /
        compare_module.ID_RECOVERY_DISTANCE_FACTOR
    )

    gap_frames = float(
        details["gap_frames"]
    )

    gap_ratio = (
        gap_frames
        /
        max(
            manager.cfg.max_gap_frames,
            1,
        )
    )

    scale_ratio = float(
        details["scale_ratio"]
    )

    scale_difference = abs(
        1.0 - scale_ratio
    )

    short_penalty = manager.short_penalty(
        track.visible_frames
    )

    score = (
        compare_module.V10_DISTANCE_WEIGHT
        *
        normalized_distance

        +

        compare_module.V10_GAP_WEIGHT
        *
        gap_ratio

        +

        compare_module.V10_SCALE_WEIGHT
        *
        scale_difference

        +

        short_penalty
    )

    return float(score)


# =============================================================================
# INSTALL GEOMETRY GATE
# =============================================================================

def install_geometry_gate(
    compare_module,
):

    manager_class = (
        compare_module.StableIDManager
    )

    def gated_find_recovery(
        self,
        tentative,
        used,
    ):

        candidates = []

        for stable_id, track in self.tracks.items():

            canonical = self.canonical_id(
                stable_id
            )

            if canonical != stable_id:
                continue

            if stable_id in used:
                continue

            details = self.score_candidate(
                track,
                tentative,
            )

            if details is None:
                continue

            geometry_score = (
                calculate_geometry_score(
                    self,
                    track,
                    details,
                    compare_module,
                )
            )

            if (
                geometry_score
                >
                GEOMETRY_MAX_SCORE
            ):
                continue

            details[
                "geometry_score"
            ] = geometry_score

            details[
                "geometry_max_score"
            ] = GEOMETRY_MAX_SCORE

            candidates.append(
                (
                    details["score"],
                    stable_id,
                    details,
                )
            )

        if not candidates:
            return None, None

        candidates.sort(
            key=lambda item:
                item[0]
        )

        _, stable_id, details = (
            candidates[0]
        )

        return stable_id, details

    manager_class.find_recovery = (
        gated_find_recovery
    )


# =============================================================================
# CONFIGURE ONE RUN
# =============================================================================

def configure_run(
    compare_module,
    evaluation_module,
    run_dir: Path,
    reid_threshold: float,
):

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    tracker_yaml = (
        run_dir
        /
        "bytetrack_pool_10fps.yaml"
    )

    compare_module.SOURCE = (
        VIDEO_PATH
    )

    compare_module.OUTPUT_ROOT = (
        run_dir
    )

    compare_module.TRACKER_10FPS = (
        tracker_yaml
    )

    compare_module.REID_MIN_SIMILARITY = (
        reid_threshold
    )

    evaluation_module.VIDEO_PATH = (
        VIDEO_PATH
    )

    evaluation_module.GT_ZIP_PATH = (
        GT_ZIP_PATH
    )

    evaluation_module.OUTPUT_DIR = (
        run_dir
    )

    evaluation_module.TRACKER_YAML = (
        tracker_yaml
    )

    evaluation_module.REID_THRESHOLD = (
        reid_threshold
    )

    evaluation_module.SAVE_RESULT_VIDEO = (
        False
    )


# =============================================================================
# RUN ONE
# =============================================================================

def run_one(
    compare_module,
    evaluation_module,
    gt_by_frame,
    reid_threshold: float,
    fps: float,
    width: int,
    height: int,
    total_frames: int,
):

    label = (
        f"V12_ReID_{reid_threshold:.2f}"
    )

    run_dir = (
        OUTPUT_ROOT
        /
        label
    )

    configure_run(
        compare_module,
        evaluation_module,
        run_dir,
        reid_threshold,
    )

    compare_module.create_tracker_yaml(
        fps
    )

    print()
    print("#" * 100)
    print(
        f"{label}"
        f" | GeoMax={GEOMETRY_MAX_SCORE}"
    )
    print("#" * 100)

    predictions, _ = (
        evaluation_module.run_tracking(
            module=compare_module,
            mode="V11_REID_084",
            fps=fps,
            width=width,
            height=height,
            total_frames=total_frames,
        )
    )

    rows = []

    for iou_threshold in [
        PRIMARY_IOU,
        SECONDARY_IOU,
    ]:

        metrics, _ = (
            evaluation_module.evaluate_tracking(
                name=label,
                gt_by_frame=gt_by_frame,
                pred_by_frame=predictions,
                total_frames=total_frames,
                iou_threshold=iou_threshold,
            )
        )

        metrics[
            "reid_threshold"
        ] = reid_threshold

        metrics[
            "geometry_max_score"
        ] = GEOMETRY_MAX_SCORE

        rows.append(
            metrics
        )

    return rows


# =============================================================================
# PRINT TABLE
# =============================================================================

def print_table(
    rows: list[dict],
    iou_threshold: float,
):

    selected = [
        row
        for row in rows
        if abs(
            float(
                row["iou_threshold"]
            )
            -
            iou_threshold
        )
        <
        1e-8
    ]

    print()
    print("=" * 140)
    print(
        f"FINAL RESULT @ IoU {iou_threshold:.2f}"
    )
    print("=" * 140)

    header = (
        f"{'MODEL':<22}"
        f"{'REID':>8}"
        f"{'IDF1':>9}"
        f"{'IDP':>9}"
        f"{'IDR':>9}"
        f"{'IDSW':>9}"
        f"{'FRAG':>9}"
        f"{'MOTA':>9}"
        f"{'PRED_ID':>10}"
    )

    print(header)
    print("-" * len(header))

    for row in selected:

        print(
            f"{row['model']:<22}"
            f"{float(row['reid_threshold']):>8.2f}"
            f"{row['IDF1']:>9.4f}"
            f"{row['IDP']:>9.4f}"
            f"{row['IDR']:>9.4f}"
            f"{row['ID_switches']:>9}"
            f"{row['fragmentations']:>9}"
            f"{row['MOTA']:>9.4f}"
            f"{row['unique_pred_ids']:>10}"
        )


# =============================================================================
# RANKING
# =============================================================================

def build_ranking(
    rows,
):

    selected = [
        row.copy()
        for row in rows
        if abs(
            float(
                row["iou_threshold"]
            )
            -
            PRIMARY_IOU
        )
        <
        1e-8
    ]

    selected.sort(
        key=lambda row:
            (
                -row["IDF1"],
                row["ID_switches"],
                row["fragmentations"],
            )
    )

    for rank, row in enumerate(
        selected,
        start=1,
    ):
        row["rank"] = rank

    return selected


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 100)
    print("TASK 90 - FINAL ReID THRESHOLD SWEEP")
    print("=" * 100)

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

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
            "COMPARE SCRIPT",
            COMPARE_SCRIPT,
        ),
        (
            "EVALUATION SCRIPT",
            EVALUATION_SCRIPT,
        ),
    ]:

        print()
        print(
            name,
            "=",
            path,
        )

        if not path.is_file():
            raise FileNotFoundError(
                f"{name} 없음:\n{path}"
            )

    compare_module = load_python_module(
        COMPARE_SCRIPT,
        "final_reid_compare",
    )

    evaluation_module = load_python_module(
        EVALUATION_SCRIPT,
        "final_reid_eval",
    )

    # -------------------------------------------------------------------------
    # Install GeoMax=0.45 once
    # -------------------------------------------------------------------------

    install_geometry_gate(
        compare_module
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
        "FPS =",
        fps,
    )

    print(
        "Frames =",
        total_frames,
    )

    print(
        "Resolution =",
        f"{width}x{height}",
    )

    print(
        "Device =",
        DEVICE,
    )

    print(
        "Geometry Max Score =",
        GEOMETRY_MAX_SCORE,
    )

    print(
        "ReID thresholds =",
        REID_THRESHOLDS,
    )

    # -------------------------------------------------------------------------
    # GT
    # -------------------------------------------------------------------------

    gt_by_frame = (
        evaluation_module.parse_cvat_gt(
            GT_ZIP_PATH
        )
    )

    # -------------------------------------------------------------------------
    # Sweep
    # -------------------------------------------------------------------------

    all_results = []

    for threshold in REID_THRESHOLDS:

        rows = run_one(
            compare_module,
            evaluation_module,
            gt_by_frame,
            threshold,
            fps,
            width,
            height,
            total_frames,
        )

        all_results.extend(
            rows
        )

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------

    write_csv(
        OUTPUT_ROOT
        /
        "reid_threshold_sweep_metrics.csv",
        all_results,
    )

    ranking = build_ranking(
        all_results
    )

    write_csv(
        OUTPUT_ROOT
        /
        "reid_threshold_sweep_ranking.csv",
        ranking,
    )

    (
        OUTPUT_ROOT
        /
        "reid_threshold_sweep_results.json"
    ).write_text(
        json.dumps(
            all_results,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # -------------------------------------------------------------------------
    # Console
    # -------------------------------------------------------------------------

    print_table(
        all_results,
        PRIMARY_IOU,
    )

    print_table(
        all_results,
        SECONDARY_IOU,
    )

    print()
    print("=" * 100)
    print("FINAL RANKING @ IoU 0.50")
    print("=" * 100)

    for row in ranking:

        print(
            f"{row['rank']:>2}. "
            f"ReID={float(row['reid_threshold']):.2f}"
            f" | IDF1={row['IDF1']:.4f}"
            f" | IDSW={row['ID_switches']}"
            f" | FRAG={row['fragmentations']}"
            f" | MOTA={row['MOTA']:.4f}"
            f" | PredID={row['unique_pred_ids']}"
        )

    print()
    print("=" * 100)
    print("DONE")
    print("=" * 100)

    print(
        "결과 폴더:"
    )

    print(
        OUTPUT_ROOT
    )

    print()
    print(
        "중요 파일:"
    )

    print(
        OUTPUT_ROOT
        /
        "reid_threshold_sweep_ranking.csv"
    )

    print(
        OUTPUT_ROOT
        /
        "reid_threshold_sweep_metrics.csv"
    )


if __name__ == "__main__":
    main()