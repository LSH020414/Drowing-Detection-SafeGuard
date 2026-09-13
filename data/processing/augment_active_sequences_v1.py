from pathlib import Path
import math
import random

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

INPUT_CSV = Path(
    r"D:\임베디드 경진대회\머리 추적\익수자 판별 학습"
    r"\ACTIVE_DROWNING_FINAL\ACTIVE_FINAL_sequences.csv"
)

OUTPUT_DIR = INPUT_CSV.parent / "ACTIVE_AUGMENTED_V1"

SEED = 42

# 최종 TRAIN ACTIVE window 목표
TARGET_TRAIN_WINDOWS = 1000

# video_id 기준 split
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# 50 timestep 기준
EXPECTED_STEPS = 50

# ------------------------------------------------------------
# Augmentation 강도
# ------------------------------------------------------------

# feature 전체 표준편차 대비 Gaussian noise
GAUSSIAN_STD_RATIO_MIN = 0.010
GAUSSIAN_STD_RATIO_MAX = 0.040

# 한 window 전체를 약간 확대/축소
SCALE_JITTER_MIN = 0.97
SCALE_JITTER_MAX = 1.03

# 시간축 이동
MAX_TEMPORAL_SHIFT = 2

# 일부 timestep을 주변값으로 interpolation
TIMESTEP_DROPOUT_PROB = 0.04


# ============================================================
# RANDOM
# ============================================================

random.seed(SEED)
np.random.seed(SEED)


# ============================================================
# COLUMN RULES
# ============================================================

# 절대로 Gaussian noise를 주지 않을 컬럼
EXCLUDE_EXACT = {
    "video_id",
    "window_id",
    "class",
    "label",
    "state",
    "track_id",
    "id",
}

# 이름에 이것들이 포함되면 보통 metadata/time index
EXCLUDE_CONTAINS = (
    "frame",
    "timestep",
    "step",
    "time",
    "timestamp",
)

# 0/1 같은 상태 feature는 연속형 noise 제외
BINARY_HINTS = (
    "detected",
    "visible",
    "missing",
    "lost",
    "outside",
    "valid",
)


# ============================================================
# UTIL
# ============================================================

def ensure_output_dir():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


def get_window_count(df):
    return df["window_id"].nunique()


def split_video_ids(df):
    """
    같은 원본 video_id가 train/val/test에 동시에 들어가는
    leakage 방지.
    """

    video_ids = sorted(
        df["video_id"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    rng = np.random.default_rng(SEED)
    rng.shuffle(video_ids)

    n = len(video_ids)

    n_train = int(round(n * TRAIN_RATIO))
    n_val = int(round(n * VAL_RATIO))

    # 최소한 test가 남도록 조정
    if n >= 3:
        n_train = min(
            max(1, n_train),
            n - 2
        )

        n_val = min(
            max(1, n_val),
            n - n_train - 1
        )

    train_ids = set(
        video_ids[:n_train]
    )

    val_ids = set(
        video_ids[
            n_train:
            n_train + n_val
        ]
    )

    test_ids = set(
        video_ids[
            n_train + n_val:
        ]
    )

    return (
        train_ids,
        val_ids,
        test_ids
    )


def is_binary_column(series, col_name):
    name = col_name.lower()

    if any(
        hint in name
        for hint in BINARY_HINTS
    ):
        return True

    values = (
        pd.to_numeric(
            series,
            errors="coerce"
        )
        .dropna()
        .unique()
    )

    if len(values) <= 2:
        rounded = set(
            np.round(
                values.astype(float),
                6
            )
        )

        if rounded.issubset({0.0, 1.0}):
            return True

    return False


def find_augment_columns(df):
    """
    실제 classifier feature 후보 중
    연속형 numerical feature만 선택.
    """

    augment_cols = []

    for col in df.columns:

        lower = col.lower()

        if lower in EXCLUDE_EXACT:
            continue

        if any(
            token in lower
            for token in EXCLUDE_CONTAINS
        ):
            continue

        if not pd.api.types.is_numeric_dtype(
            df[col]
        ):
            continue

        if is_binary_column(
            df[col],
            col
        ):
            continue

        augment_cols.append(col)

    return augment_cols


def global_feature_std(
    train_df,
    augment_cols
):
    result = {}

    for col in augment_cols:

        values = pd.to_numeric(
            train_df[col],
            errors="coerce"
        )

        std = float(
            values.std()
        )

        if (
            not np.isfinite(std)
            or std <= 1e-12
        ):
            std = 0.0

        result[col] = std

    return result


# ============================================================
# AUGMENTATION
# ============================================================

def temporal_shift(window, shift):
    """
    timestep 순서를 ±1~2 frame 정도 이동.

    np.roll처럼 반대쪽 값이 넘어오지 않게
    끝값으로 padding.
    """

    if shift == 0:
        return window.copy()

    result = window.copy()

    if shift > 0:

        result.iloc[shift:] = (
            window.iloc[:-shift]
            .to_numpy()
        )

        for i in range(shift):
            result.iloc[i] = (
                window.iloc[0]
            )

    else:

        s = abs(shift)

        result.iloc[:-s] = (
            window.iloc[s:]
            .to_numpy()
        )

        for i in range(s):
            result.iloc[-(i + 1)] = (
                window.iloc[-1]
            )

    return result


def interpolate_random_timesteps(
    window,
    augment_cols
):
    """
    아주 일부 timestep feature를
    앞뒤 timestep 평균으로 대체.
    """

    result = window.copy()

    n = len(result)

    if n < 3:
        return result

    for i in range(1, n - 1):

        if (
            np.random.random()
            < TIMESTEP_DROPOUT_PROB
        ):

            for col in augment_cols:

                a = result.iloc[i - 1][col]
                b = result.iloc[i + 1][col]

                if (
                    pd.notna(a)
                    and pd.notna(b)
                ):
                    result.iat[
                        i,
                        result.columns.get_loc(col)
                    ] = (
                        float(a)
                        + float(b)
                    ) / 2.0

    return result


def augment_window(
    original_window,
    augment_cols,
    feature_std,
    aug_index
):
    """
    하나의 50-step ACTIVE window에서
    새로운 variation 생성.
    """

    w = (
        original_window
        .copy()
        .reset_index(drop=True)
    )

    original_video_id = str(
        w.loc[0, "video_id"]
    )

    original_window_id = str(
        w.loc[0, "window_id"]
    )

    # --------------------------------------------------------
    # 1. Temporal shift
    # --------------------------------------------------------

    shift = np.random.randint(
        -MAX_TEMPORAL_SHIFT,
        MAX_TEMPORAL_SHIFT + 1
    )

    if shift != 0:

        feature_only = (
            w[augment_cols]
            .copy()
        )

        shifted = temporal_shift(
            feature_only,
            shift
        )

        w.loc[:, augment_cols] = (
            shifted.to_numpy()
        )

    # --------------------------------------------------------
    # 2. Scale jitter
    # --------------------------------------------------------

    scale = np.random.uniform(
        SCALE_JITTER_MIN,
        SCALE_JITTER_MAX
    )

    # center 좌표는 단순 scale하면
    # 영상 좌표 자체가 이동할 수 있으므로
    # 주로 scale/size 관련 feature에 적용.
    for col in augment_cols:

        lower = col.lower()

        if (
            "scale" in lower
            or "width" in lower
            or "height" in lower
            or "area" in lower
            or "size" in lower
        ):

            numeric = pd.to_numeric(
                w[col],
                errors="coerce"
            )

            w[col] = numeric * scale

    # --------------------------------------------------------
    # 3. Gaussian feature noise
    # --------------------------------------------------------

    noise_ratio = np.random.uniform(
        GAUSSIAN_STD_RATIO_MIN,
        GAUSSIAN_STD_RATIO_MAX
    )

    for col in augment_cols:

        std = feature_std.get(
            col,
            0.0
        )

        if std <= 0:
            continue

        original_numeric = pd.to_numeric(
            w[col],
            errors="coerce"
        )

        noise = np.random.normal(
            loc=0.0,
            scale=std * noise_ratio,
            size=len(w)
        )

        mask = original_numeric.notna()

        new_values = (
            original_numeric.copy()
        )

        new_values.loc[mask] = (
            original_numeric.loc[mask]
            + noise[mask.to_numpy()]
        )

        w[col] = new_values

    # --------------------------------------------------------
    # 4. Random timestep interpolation
    # --------------------------------------------------------

    w = interpolate_random_timesteps(
        w,
        augment_cols
    )

    # --------------------------------------------------------
    # Normalized feature 범위 보호
    # --------------------------------------------------------

    for col in augment_cols:

        lower = col.lower()

        if lower.endswith("_norm"):

            # head_scale_norm,
            # center_x_norm,
            # center_y_norm 등이 일반적으로 0~1 범위
            w[col] = (
                pd.to_numeric(
                    w[col],
                    errors="coerce"
                )
                .clip(0.0, 1.0)
            )

    # --------------------------------------------------------
    # metadata
    # --------------------------------------------------------

    new_window_id = (
        f"{original_window_id}"
        f"_AUG{aug_index:04d}"
    )

    w["window_id"] = new_window_id

    # 원본 video_id는 유지.
    # 그래야 lineage를 추적할 수 있음.
    w["source_video_id"] = (
        original_video_id
    )

    w["source_window_id"] = (
        original_window_id
    )

    w["is_augmented"] = 1

    w["augmentation_shift"] = shift
    w["augmentation_scale"] = scale
    w["augmentation_noise_ratio"] = (
        noise_ratio
    )

    return w


# ============================================================
# WINDOW VALIDATION
# ============================================================

def valid_windows_only(df):
    """
    50-step window만 사용.
    """

    sizes = (
        df.groupby("window_id")
        .size()
    )

    good_ids = sizes[
        sizes == EXPECTED_STEPS
    ].index

    bad = sizes[
        sizes != EXPECTED_STEPS
    ]

    if len(bad) > 0:

        print()
        print(
            "[WARNING] 50-step이 아닌 window:",
            len(bad)
        )

        print(
            bad.head(20)
        )

    return df[
        df["window_id"].isin(
            good_ids
        )
    ].copy()


# ============================================================
# MAIN
# ============================================================

def main():

    ensure_output_dir()

    print("=" * 70)
    print("ACTIVE SEQUENCE AUGMENTATION V1")
    print("=" * 70)

    print()
    print("Input:")
    print(INPUT_CSV)

    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            INPUT_CSV
        )

    df = pd.read_csv(
        INPUT_CSV
    )

    print()
    print(
        "Rows       :",
        len(df)
    )

    print(
        "Windows    :",
        get_window_count(df)
    )

    print(
        "Videos     :",
        df["video_id"].nunique()
    )

    print()
    print("Columns:")

    for col in df.columns:
        print(" -", col)

    # --------------------------------------------------------
    # 50 timestep window만
    # --------------------------------------------------------

    df = valid_windows_only(
        df
    )

    # --------------------------------------------------------
    # Video-level split
    # --------------------------------------------------------

    (
        train_video_ids,
        val_video_ids,
        test_video_ids
    ) = split_video_ids(df)

    train_real = df[
        df["video_id"]
        .astype(str)
        .isin(train_video_ids)
    ].copy()

    val_real = df[
        df["video_id"]
        .astype(str)
        .isin(val_video_ids)
    ].copy()

    test_real = df[
        df["video_id"]
        .astype(str)
        .isin(test_video_ids)
    ].copy()

    # --------------------------------------------------------
    # 원본 metadata 추가
    # --------------------------------------------------------

    for part in (
        train_real,
        val_real,
        test_real
    ):

        part["source_video_id"] = (
            part["video_id"]
        )

        part["source_window_id"] = (
            part["window_id"]
        )

        part["is_augmented"] = 0

        part[
            "augmentation_shift"
        ] = 0

        part[
            "augmentation_scale"
        ] = 1.0

        part[
            "augmentation_noise_ratio"
        ] = 0.0

    train_window_count = (
        get_window_count(
            train_real
        )
    )

    val_window_count = (
        get_window_count(
            val_real
        )
    )

    test_window_count = (
        get_window_count(
            test_real
        )
    )

    print()
    print("=" * 70)
    print("VIDEO-LEVEL SPLIT")
    print("=" * 70)

    print(
        f"TRAIN real : "
        f"{train_window_count} windows / "
        f"{len(train_video_ids)} videos"
    )

    print(
        f"VAL real   : "
        f"{val_window_count} windows / "
        f"{len(val_video_ids)} videos"
    )

    print(
        f"TEST real  : "
        f"{test_window_count} windows / "
        f"{len(test_video_ids)} videos"
    )

    # --------------------------------------------------------
    # continuous feature 선택
    # --------------------------------------------------------

    augment_cols = find_augment_columns(
        train_real
    )

    print()
    print("=" * 70)
    print("AUGMENTED FEATURE COLUMNS")
    print("=" * 70)

    for col in augment_cols:
        print(" -", col)

    feature_std = global_feature_std(
        train_real,
        augment_cols
    )

    # --------------------------------------------------------
    # 필요한 augmentation window 수
    # --------------------------------------------------------

    need_aug = max(
        0,
        TARGET_TRAIN_WINDOWS
        - train_window_count
    )

    print()
    print(
        "Target TRAIN windows :",
        TARGET_TRAIN_WINDOWS
    )

    print(
        "Real TRAIN windows   :",
        train_window_count
    )

    print(
        "Need augmented       :",
        need_aug
    )

    # --------------------------------------------------------
    # 원본 window 목록
    # --------------------------------------------------------

    grouped = {
        window_id: group.copy()
        for window_id, group
        in train_real.groupby(
            "window_id",
            sort=False
        )
    }

    original_window_ids = list(
        grouped.keys()
    )

    if not original_window_ids:
        raise RuntimeError(
            "TRAIN window가 없습니다."
        )

    # --------------------------------------------------------
    # Augmentation 생성
    # --------------------------------------------------------

    augmented_windows = []

    for aug_index in range(
        1,
        need_aug + 1
    ):

        source_id = random.choice(
            original_window_ids
        )

        source_window = grouped[
            source_id
        ]

        aug = augment_window(
            source_window,
            augment_cols,
            feature_std,
            aug_index
        )

        augmented_windows.append(
            aug
        )

        if (
            aug_index % 100 == 0
            or aug_index == need_aug
        ):
            print(
                f"Augmented "
                f"{aug_index}/{need_aug}"
            )

    # --------------------------------------------------------
    # Train 결합
    # --------------------------------------------------------

    if augmented_windows:

        train_aug = pd.concat(
            augmented_windows,
            ignore_index=True
        )

        train_final = pd.concat(
            [
                train_real,
                train_aug
            ],
            ignore_index=True
        )

    else:

        train_aug = pd.DataFrame(
            columns=train_real.columns
        )

        train_final = (
            train_real.copy()
        )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    train_real_path = (
        OUTPUT_DIR
        / "ACTIVE_TRAIN_REAL.csv"
    )

    train_aug_path = (
        OUTPUT_DIR
        / "ACTIVE_TRAIN_AUGMENTED_ONLY.csv"
    )

    train_final_path = (
        OUTPUT_DIR
        / "ACTIVE_TRAIN_FINAL.csv"
    )

    val_path = (
        OUTPUT_DIR
        / "ACTIVE_VAL_REAL.csv"
    )

    test_path = (
        OUTPUT_DIR
        / "ACTIVE_TEST_REAL.csv"
    )

    split_path = (
        OUTPUT_DIR
        / "ACTIVE_VIDEO_SPLIT.csv"
    )

    summary_path = (
        OUTPUT_DIR
        / "ACTIVE_AUGMENT_SUMMARY.csv"
    )

    train_real.to_csv(
        train_real_path,
        index=False,
        encoding="utf-8-sig"
    )

    train_aug.to_csv(
        train_aug_path,
        index=False,
        encoding="utf-8-sig"
    )

    train_final.to_csv(
        train_final_path,
        index=False,
        encoding="utf-8-sig"
    )

    val_real.to_csv(
        val_path,
        index=False,
        encoding="utf-8-sig"
    )

    test_real.to_csv(
        test_path,
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # split record
    # --------------------------------------------------------

    split_rows = []

    for vid in sorted(
        train_video_ids
    ):
        split_rows.append(
            {
                "video_id": vid,
                "split": "train"
            }
        )

    for vid in sorted(
        val_video_ids
    ):
        split_rows.append(
            {
                "video_id": vid,
                "split": "val"
            }
        )

    for vid in sorted(
        test_video_ids
    ):
        split_rows.append(
            {
                "video_id": vid,
                "split": "test"
            }
        )

    pd.DataFrame(
        split_rows
    ).to_csv(
        split_path,
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # summary
    # --------------------------------------------------------

    summary = pd.DataFrame(
        [
            {
                "split": "train_real",
                "windows":
                    get_window_count(
                        train_real
                    ),
                "rows":
                    len(train_real)
            },
            {
                "split":
                    "train_augmented",
                "windows":
                    get_window_count(
                        train_aug
                    )
                    if len(train_aug)
                    else 0,
                "rows":
                    len(train_aug)
            },
            {
                "split": "train_final",
                "windows":
                    get_window_count(
                        train_final
                    ),
                "rows":
                    len(train_final)
            },
            {
                "split": "val_real",
                "windows":
                    get_window_count(
                        val_real
                    ),
                "rows":
                    len(val_real)
            },
            {
                "split": "test_real",
                "windows":
                    get_window_count(
                        test_real
                    ),
                "rows":
                    len(test_real)
            }
        ]
    )

    summary.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)

    print()
    print(
        "TRAIN REAL      :",
        get_window_count(
            train_real
        )
    )

    print(
        "TRAIN AUGMENTED :",
        get_window_count(
            train_aug
        )
        if len(train_aug)
        else 0
    )

    print(
        "TRAIN FINAL     :",
        get_window_count(
            train_final
        )
    )

    print(
        "VAL REAL        :",
        get_window_count(
            val_real
        )
    )

    print(
        "TEST REAL       :",
        get_window_count(
            test_real
        )
    )

    print()
    print("Output:")
    print(OUTPUT_DIR)

    print()
    print(
        "IMPORTANT: "
        "VAL/TEST에는 augmentation을 "
        "적용하지 않았습니다."
    )


if __name__ == "__main__":
    main()