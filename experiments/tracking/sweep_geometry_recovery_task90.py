from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np
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
    r"\Task90_Geometry_Score_Sweep"
)


# =============================================================================
# TEST SETTINGS
# =============================================================================

REID_THRESHOLD = 0.84

GEOMETRY_SCORE_THRESHOLDS = [
    0.45,
    0.50,
    0.55,
    0.60,
]

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
            f"module load 실패:\n{path}"
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
    video_path: Path,
):

    capture = cv2.VideoCapture(
        str(video_path)
    )

    if not capture.isOpened():
        raise RuntimeError(
            f"영상 열기 실패:\n{video_path}"
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
#
# V10 원래 geometry score:
#
# 0.70 * normalized distance
# + 0.20 * gap
# + 0.10 * scale difference
# + short-track penalty
#
# 낮을수록 좋은 후보
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
# PATCH RECOVERY
#
# 기존 score_candidate()는 그대로 사용.
#
# 즉:
#   geometry gate
#   scale/gap gate
#   ReID >= 0.84
#
# 는 기존대로 작동.
#
# 그 후 geometry score가 너무 나쁜 후보를 추가로 거부.
#
# alias merge에는 영향을 주지 않는다.
# =============================================================================

def install_geometry_recovery_gate(
    compare_module,
    max_geometry_score: float,
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

            # -------------------------------------------------------------
            # NEW V12 GATE
            # -------------------------------------------------------------

            if (
                geometry_score
                >
                max_geometry_score
            ):
                continue

            details[
                "geometry_score"
            ] = geometry_score

            details[
                "geometry_max_threshold"
            ] = max_geometry_score

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
# RESTORE ORIGINAL RECOVERY
# =============================================================================

def restore_recovery(
    compare_module,
    original_find_recovery,
):

    compare_module.StableIDManager.find_recovery = (
        original_find_recovery
    )


# =============================================================================
# CONFIGURE OUTPUT
# =============================================================================

def configure_run_paths(
    compare_module,
    evaluation_module,
    run_dir: Path,
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
        REID_THRESHOLD
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
        REID_THRESHOLD
    )

    evaluation_module.SAVE_RESULT_VIDEO = (
        False
    )


# =============================================================================
# RUN ONE TRACKER
# =============================================================================

def run_one_tracking(
    compare_module,
    evaluation_module,
    mode: str,
    label: str,
    run_dir: Path,
    fps: float,
    width: int,
    height: int,
    total_frames: int,
):

    configure_run_paths(
        compare_module,
        evaluation_module,
        run_dir,
    )

    compare_module.create_tracker_yaml(
        fps
    )

    print()
    print("#" * 100)
    print(label)
    print("#" * 100)

    predictions, raw_predictions = (
        evaluation_module.run_tracking(
            module=compare_module,
            mode=mode,
            fps=fps,
            width=width,
            height=height,
            total_frames=total_frames,
        )
    )

    return (
        predictions,
        raw_predictions,
    )


# =============================================================================
# EVALUATE ONE
# =============================================================================

def evaluate_one(
    evaluation_module,
    label: str,
    gt_by_frame,
    predictions,
    total_frames: int,
    iou_threshold: float,
):

    metrics, match_rows = (
        evaluation_module.evaluate_tracking(
            name=label,
            gt_by_frame=gt_by_frame,
            pred_by_frame=predictions,
            total_frames=total_frames,
            iou_threshold=iou_threshold,
        )
    )

    return metrics, match_rows


# =============================================================================
# PRINT TABLE
# =============================================================================

def print_table(
    results: list[dict],
    iou_threshold: float,
):

    rows = [
        row
        for row in results
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
        f"{'MODEL':<25}"
        f"{'IDF1':>9}"
        f"{'IDP':>9}"
        f"{'IDR':>9}"
        f"{'IDSW':>9}"
        f"{'FRAG':>9}"
        f"{'MOTA':>9}"
        f"{'PREC':>9}"
        f"{'RECALL':>9}"
        f"{'PRED_ID':>10}"
    )

    print(header)
    print("-" * len(header))

    for row in rows:

        print(
            f"{row['model']:<25}"
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
# RANKING
# =============================================================================

def build_ranking(
    all_results: list[dict],
):

    primary = [
        row.copy()
        for row in all_results
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

    raw = next(
        (
            row
            for row in primary
            if row["model"]
            ==
            "Raw_ByteTrack"
        ),
        None,
    )

    v10 = next(
        (
            row
            for row in primary
            if row["model"]
            ==
            "V10_Geometry"
        ),
        None,
    )

    for row in primary:

        if raw is not None:

            row[
                "IDF1_change_vs_raw"
            ] = (
                row["IDF1"]
                -
                raw["IDF1"]
            )

            row[
                "IDSW_change_vs_raw"
            ] = (
                row["ID_switches"]
                -
                raw["ID_switches"]
            )

        if v10 is not None:

            row[
                "IDF1_change_vs_v10"
            ] = (
                row["IDF1"]
                -
                v10["IDF1"]
            )

            row[
                "IDF1_relative_vs_v10_percent"
            ] = (
                (
                    row["IDF1"]
                    -
                    v10["IDF1"]
                )
                /
                v10["IDF1"]
                *
                100.0
                if v10["IDF1"] > 0
                else 0.0
            )

            row[
                "IDSW_reduction_vs_v10"
            ] = (
                v10["ID_switches"]
                -
                row["ID_switches"]
            )

    primary.sort(
        key=lambda row:
            (
                -row["IDF1"],
                row["ID_switches"],
                row["fragmentations"],
            )
    )

    for index, row in enumerate(
        primary,
        start=1,
    ):
        row["rank"] = index

    return primary


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 100)
    print("TASK 90 - V12 GEOMETRY RECOVERY SCORE SWEEP")
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

    # -------------------------------------------------------------------------
    # Load modules
    # -------------------------------------------------------------------------

    compare_module = load_python_module(
        COMPARE_SCRIPT,
        "geometry_sweep_compare",
    )

    evaluation_module = load_python_module(
        EVALUATION_SCRIPT,
        "geometry_sweep_evaluation",
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
        "ReID threshold =",
        REID_THRESHOLD,
    )

    print(
        "Geometry thresholds =",
        GEOMETRY_SCORE_THRESHOLDS,
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
    # Store original method
    # -------------------------------------------------------------------------

    original_find_recovery = (
        compare_module
        .StableIDManager
        .find_recovery
    )

    all_results = []

    # =========================================================================
    # 1. V10 BASELINE
    # =========================================================================

    restore_recovery(
        compare_module,
        original_find_recovery,
    )

    v10_dir = (
        OUTPUT_ROOT
        /
        "V10_Geometry"
    )

    v10_predictions, raw_predictions = (
        run_one_tracking(
            compare_module,
            evaluation_module,
            mode="V10_GEOMETRY",
            label="V10 Geometry",
            run_dir=v10_dir,
            fps=fps,
            width=width,
            height=height,
            total_frames=total_frames,
        )
    )

    # -------------------------------------------------------------------------
    # Raw + V10 evaluation
    # -------------------------------------------------------------------------

    for iou_threshold in [
        PRIMARY_IOU,
        SECONDARY_IOU,
    ]:

        raw_metrics, _ = (
            evaluate_one(
                evaluation_module,
                "Raw_ByteTrack",
                gt_by_frame,
                raw_predictions,
                total_frames,
                iou_threshold,
            )
        )

        v10_metrics, _ = (
            evaluate_one(
                evaluation_module,
                "V10_Geometry",
                gt_by_frame,
                v10_predictions,
                total_frames,
                iou_threshold,
            )
        )

        all_results.extend(
            [
                raw_metrics,
                v10_metrics,
            ]
        )

    # =========================================================================
    # 2. V11 ORIGINAL - ReID 0.84, no geometry max gate
    # =========================================================================

    restore_recovery(
        compare_module,
        original_find_recovery,
    )

    v11_dir = (
        OUTPUT_ROOT
        /
        "V11_ReID_084_NoGeometryGate"
    )

    v11_predictions, _ = (
        run_one_tracking(
            compare_module,
            evaluation_module,
            mode="V11_REID_084",
            label="V11 ReID 0.84 - No Geometry Gate",
            run_dir=v11_dir,
            fps=fps,
            width=width,
            height=height,
            total_frames=total_frames,
        )
    )

    for iou_threshold in [
        PRIMARY_IOU,
        SECONDARY_IOU,
    ]:

        metrics, _ = (
            evaluate_one(
                evaluation_module,
                "V11_ReID_0.84",
                gt_by_frame,
                v11_predictions,
                total_frames,
                iou_threshold,
            )
        )

        all_results.append(
            metrics
        )

    # =========================================================================
    # 3. V12 GEOMETRY SCORE SWEEP
    # =========================================================================

    for geometry_threshold in (
        GEOMETRY_SCORE_THRESHOLDS
    ):

        restore_recovery(
            compare_module,
            original_find_recovery,
        )

        install_geometry_recovery_gate(
            compare_module,
            geometry_threshold,
        )

        label = (
            f"V12_GeoMax_{geometry_threshold:.2f}"
        )

        run_dir = (
            OUTPUT_ROOT
            /
            label
        )

        predictions, _ = (
            run_one_tracking(
                compare_module,
                evaluation_module,
                mode="V11_REID_084",
                label=(
                    f"V12 ReID 0.84 + "
                    f"Geometry <= "
                    f"{geometry_threshold:.2f}"
                ),
                run_dir=run_dir,
                fps=fps,
                width=width,
                height=height,
                total_frames=total_frames,
            )
        )

        for iou_threshold in [
            PRIMARY_IOU,
            SECONDARY_IOU,
        ]:

            metrics, _ = (
                evaluate_one(
                    evaluation_module,
                    label,
                    gt_by_frame,
                    predictions,
                    total_frames,
                    iou_threshold,
                )
            )

            metrics[
                "reid_threshold"
            ] = (
                REID_THRESHOLD
            )

            metrics[
                "geometry_max_score"
            ] = (
                geometry_threshold
            )

            all_results.append(
                metrics
            )

    # -------------------------------------------------------------------------
    # Restore
    # -------------------------------------------------------------------------

    restore_recovery(
        compare_module,
        original_find_recovery,
    )

    # -------------------------------------------------------------------------
    # Save all metrics
    # -------------------------------------------------------------------------

    write_csv(
        OUTPUT_ROOT
        /
        "geometry_sweep_tracking_metrics.csv",
        all_results,
    )

    ranking = build_ranking(
        all_results
    )

    write_csv(
        OUTPUT_ROOT
        /
        "geometry_sweep_ranking.csv",
        ranking,
    )

    (
        OUTPUT_ROOT
        /
        "geometry_sweep_results.json"
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
    print("RANKING - IoU 0.50")
    print("=" * 100)

    for row in ranking:

        print(
            f"{row['rank']:>2}. "
            f"{row['model']:<25}"
            f" IDF1={row['IDF1']:.4f}"
            f" IDSW={row['ID_switches']}"
            f" FRAG={row['fragmentations']}"
            f" MOTA={row['MOTA']:.4f}"
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
        "가장 중요한 파일:"
    )

    print(
        OUTPUT_ROOT
        /
        "geometry_sweep_ranking.csv"
    )

    print(
        OUTPUT_ROOT
        /
        "geometry_sweep_tracking_metrics.csv"
    )


if __name__ == "__main__":
    main()
