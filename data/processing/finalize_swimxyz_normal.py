from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(
    r"D:\임베디드 경진대회\머리 추적\익수자 판별 학습\SwimXYZ_NORMAL_CLEAN"
)

WINDOWS_CSV = ROOT / "NORMAL_ALL_windows.csv"
SEQUENCES_CSV = ROOT / "NORMAL_frame_sequences.csv"

OUT_WINDOWS = ROOT / "NORMAL_FINAL_USABLE_windows.csv"
OUT_SEQUENCES = ROOT / "NORMAL_FINAL_sequences.csv"
OUT_SUMMARY = ROOT / "NORMAL_FINAL_summary.txt"


def calculate_speed_features(group: pd.DataFrame):
    """
    ACTIVE와 같은 개념:
    속도 = head-size / second
    가속도 = head-size / second^2

    LOST를 사이에 두고 속도/가속도를 연결하지 않는다.
    """

    g = group.sort_values("time_sec").reset_index(drop=True)

    speeds = []
    acceleration_runs = []

    current_run_speeds = []

    for i in range(1, len(g)):

        prev = g.iloc[i - 1]
        cur = g.iloc[i]

        # 둘 다 visible인 연속 sample에서만 속도 계산
        if int(prev["detected"]) != 1 or int(cur["detected"]) != 1:

            if current_run_speeds:
                acceleration_runs.append(current_run_speeds)
                current_run_speeds = []

            continue

        dt = float(cur["time_sec"]) - float(prev["time_sec"])

        if dt <= 0:
            continue

        # raw pixel 좌표 사용
        dx = float(cur["cx"]) - float(prev["cx"])
        dy = float(cur["cy"]) - float(prev["cy"])

        prev_scale = np.sqrt(
            max(
                1e-8,
                float(prev["w"]) * float(prev["h"])
            )
        )

        cur_scale = np.sqrt(
            max(
                1e-8,
                float(cur["w"]) * float(cur["h"])
            )
        )

        head_scale = max(
            1.0,
            (prev_scale + cur_scale) / 2.0
        )

        distance_heads = (
            np.hypot(dx, dy)
            / head_scale
        )

        speed = (
            distance_heads
            / dt
        )

        speeds.append(speed)
        current_run_speeds.append((speed, dt))

    if current_run_speeds:
        acceleration_runs.append(current_run_speeds)

    if speeds:

        mean_speed = float(np.mean(speeds))
        max_speed = float(np.max(speeds))
        speed_std = float(np.std(speeds))

    else:

        mean_speed = 0.0
        max_speed = 0.0
        speed_std = 0.0

    accelerations = []

    for run in acceleration_runs:

        if len(run) < 2:
            continue

        for i in range(1, len(run)):

            prev_speed = run[i - 1][0]
            current_speed = run[i][0]

            # 현재 interval의 dt 사용
            dt = run[i][1]

            if dt <= 0:
                continue

            acceleration = abs(
                (current_speed - prev_speed)
                / dt
            )

            accelerations.append(acceleration)

    if accelerations:

        mean_acceleration = float(
            np.mean(accelerations)
        )

        max_acceleration = float(
            np.max(accelerations)
        )

        acceleration_std = float(
            np.std(accelerations)
        )

    else:

        mean_acceleration = 0.0
        max_acceleration = 0.0
        acceleration_std = 0.0

    return {
        "mean_speed": mean_speed,
        "max_speed": max_speed,
        "speed_std": speed_std,
        "mean_acceleration": mean_acceleration,
        "max_acceleration": max_acceleration,
        "acceleration_std": acceleration_std,
    }


def main():

    print("=" * 70)
    print("SwimXYZ NORMAL FINALIZATION")
    print("=" * 70)

    if not WINDOWS_CSV.exists():
        raise FileNotFoundError(WINDOWS_CSV)

    if not SEQUENCES_CSV.exists():
        raise FileNotFoundError(SEQUENCES_CSV)

    print("CSV 읽는 중...")

    windows = pd.read_csv(WINDOWS_CSV)
    sequences = pd.read_csv(SEQUENCES_CSV)

    print(f"전체 windows : {len(windows)}")
    print(f"sequence rows: {len(sequences)}")

    # ============================================================
    # 최종 NORMAL 기준
    #
    # CORE 전부
    # +
    # HARD 중 border_case가 없는 것
    # ============================================================

    reason = (
        windows["quality_reason"]
        .fillna("")
        .astype(str)
    )

    usable_mask = (
        (windows["quality"] == "CORE")
        |
        (
            (windows["quality"] == "HARD")
            &
            (~reason.str.contains("border_case", regex=False))
        )
    )

    final_windows = (
        windows[usable_mask]
        .copy()
        .reset_index(drop=True)
    )

    print()
    print(
        f"최종 usable windows: {len(final_windows)}"
    )

    # ============================================================
    # 해당 567개 window의 sequence만 남김
    # ============================================================

    usable_window_ids = set(
        final_windows["window_id"].astype(str)
    )

    final_sequences = sequences[
        sequences["window_id"]
        .astype(str)
        .isin(usable_window_ids)
    ].copy()

    print(
        f"최종 sequence rows: {len(final_sequences)}"
    )

    # ============================================================
    # 속도 / 가속도 계산
    # ============================================================

    print()
    print("속도/가속도 계산 중...")

    feature_rows = []

    groups = final_sequences.groupby(
        "window_id",
        sort=False
    )

    total = groups.ngroups

    for idx, (window_id, group) in enumerate(
        groups,
        start=1
    ):

        features = calculate_speed_features(group)

        feature_rows.append(
            {
                "window_id": str(window_id),
                **features,
            }
        )

        if idx % 100 == 0 or idx == total:

            print(
                f"{idx}/{total}"
            )

    speed_df = pd.DataFrame(feature_rows)

    # 기존에 같은 이름의 컬럼이 있다면 제거 후 새 값 사용
    speed_columns = [
        "mean_speed",
        "max_speed",
        "speed_std",
        "mean_acceleration",
        "max_acceleration",
        "acceleration_std",
    ]

    for column in speed_columns:

        if column in final_windows.columns:
            final_windows.drop(
                columns=[column],
                inplace=True
            )

    final_windows["window_id"] = (
        final_windows["window_id"]
        .astype(str)
    )

    final_windows = final_windows.merge(
        speed_df,
        on="window_id",
        how="left",
    )

    for column in speed_columns:

        final_windows[column] = (
            final_windows[column]
            .fillna(0.0)
        )

    # ============================================================
    # label 확정
    # ============================================================

    final_windows["label"] = "NORMAL_SWIMMING"

    final_sequences["label"] = "NORMAL_SWIMMING"

    # ============================================================
    # 저장
    # ============================================================

    final_windows.to_csv(
        OUT_WINDOWS,
        index=False,
        encoding="utf-8-sig",
    )

    final_sequences.to_csv(
        OUT_SEQUENCES,
        index=False,
        encoding="utf-8-sig",
    )

    core_count = int(
        (final_windows["quality"] == "CORE").sum()
    )

    hard_count = int(
        (final_windows["quality"] == "HARD").sum()
    )

    summary = (
        "SwimXYZ NORMAL FINAL\n"
        "====================\n"
        f"CORE: {core_count}\n"
        f"HARD(no border): {hard_count}\n"
        f"TOTAL: {len(final_windows)}\n"
        f"SEQUENCE ROWS: {len(final_sequences)}\n"
        "\n"
        "Added features:\n"
        "mean_speed\n"
        "max_speed\n"
        "speed_std\n"
        "mean_acceleration\n"
        "max_acceleration\n"
        "acceleration_std\n"
    )

    OUT_SUMMARY.write_text(
        summary,
        encoding="utf-8"
    )

    print()
    print("=" * 70)
    print("FINISHED")
    print("=" * 70)

    print(f"CORE            : {core_count}")
    print(f"HARD(no border) : {hard_count}")
    print(f"TOTAL           : {len(final_windows)}")

    print()

    print(
        "Mean speed       : "
        f"{final_windows['mean_speed'].mean():.3f}"
    )

    print(
        "Mean speed std   : "
        f"{final_windows['speed_std'].mean():.3f}"
    )

    print(
        "Mean acceleration: "
        f"{final_windows['mean_acceleration'].mean():.3f}"
    )

    print()

    print("[OUTPUT]")
    print(OUT_WINDOWS)
    print(OUT_SEQUENCES)
    print(OUT_SUMMARY)


if __name__ == "__main__":
    main()