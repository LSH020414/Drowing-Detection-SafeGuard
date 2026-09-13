from pathlib import Path
import random

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

INPUT_CSV = Path(
    r"D:\임베디드 경진대회\머리 추적\수영학습"
    r"\NORMAL_4STROKES_FULL_V3"
    r"\NORMAL_4STROKES_ALL_sequences.csv"
)

OUTPUT_DIR = Path(
    r"D:\임베디드 경진대회\머리 추적\수영학습"
    r"\NORMAL_4STROKES_AUGMENTED_V1"
)

RANDOM_SEED = 42

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

WINDOW_SIZE = 50

# 원본 train window 하나당 생성할 augmentation 수
AUG_PER_REAL_WINDOW = 3


# ============================================================
# UTILS
# ============================================================

def smooth_noise(n, std, rng, kernel=5):
    raw = rng.normal(
        0.0,
        std,
        n + kernel - 1
    )

    k = np.ones(kernel) / kernel

    return np.convolve(
        raw,
        k,
        mode="valid"
    )[:n]


def split_sources(source_ids):

    ids = list(source_ids)

    random.Random(
        RANDOM_SEED
    ).shuffle(ids)

    n = len(ids)

    n_train = int(
        n * TRAIN_RATIO
    )

    n_val = int(
        n * VAL_RATIO
    )

    train = set(
        ids[:n_train]
    )

    val = set(
        ids[
            n_train:
            n_train + n_val
        ]
    )

    test = set(
        ids[
            n_train + n_val:
        ]
    )

    return train, val, test


# ============================================================
# MISSING FEATURE RECOMPUTE
# ============================================================

def recompute_temporal_features(df):

    df = df.copy()

    detected = (
        df["detected"]
        .astype(int)
        .to_numpy()
    )

    change = np.diff(
        detected,
        prepend=detected[0]
    )

    df["lost_event"] = (
        change < 0
    ).astype(int)

    df["redetect_event"] = (
        change > 0
    ).astype(int)

    consecutive = []

    c = 0

    for d in detected:

        if d == 0:
            c += 1
        else:
            c = 0

        consecutive.append(c)

    df[
        "consecutive_missing"
    ] = consecutive

    visible_ratio = float(
        detected.mean()
    )

    missing_ratio = (
        1.0
        -
        visible_ratio
    )

    df[
        "visible_ratio"
    ] = visible_ratio

    df[
        "missing_ratio"
    ] = missing_ratio

    df[
        "max_consecutive_missing"
    ] = max(consecutive)

    df[
        "lost_count"
    ] = int(
        df["lost_event"].sum()
    )

    df[
        "redetect_count"
    ] = int(
        df["redetect_event"].sum()
    )

    return df


# ============================================================
# MOTION FEATURES
# ============================================================

def recompute_motion_features(df):

    df = df.copy()

    df["head_scale"] = np.sqrt(
        np.maximum(
            df["w_norm"]
            *
            df["h_norm"],
            0
        )
    )

    df["dx"] = (
        df["cx_norm"]
        .diff()
        .fillna(0)
    )

    df["dy"] = (
        df["cy_norm"]
        .diff()
        .fillna(0)
    )

    df["speed"] = np.sqrt(
        df["dx"] ** 2
        +
        df["dy"] ** 2
    )

    df[
        "vertical_speed"
    ] = (
        df["cy_norm"]
        .diff()
        .fillna(0)
    )

    df[
        "scale_change"
    ] = (
        df["head_scale"]
        .diff()
        .fillna(0)
    )

    return df


# ============================================================
# AUG 1 : POSITION / SCALE JITTER
# ============================================================

def augment_jitter(df, rng):

    out = df.copy()

    n = len(out)

    dx = smooth_noise(
        n,
        0.004,
        rng,
        7
    )

    dy = smooth_noise(
        n,
        0.006,
        rng,
        7
    )

    scale_noise = smooth_noise(
        n,
        0.025,
        rng,
        7
    )

    out["cx_norm"] += dx
    out["cy_norm"] += dy

    out["w_norm"] *= (
        1.0
        +
        scale_noise
    )

    out["h_norm"] *= (
        1.0
        +
        scale_noise
    )

    return out


# ============================================================
# AUG 2 : TEMPORAL SPEED
# ============================================================

def augment_tempo(df, rng):

    out = df.copy()

    n = len(out)

    original_idx = np.arange(
        n
    )

    speed_factor = rng.uniform(
        0.90,
        1.10
    )

    center = (
        n - 1
    ) / 2

    warped = (
        center
        +
        (
            original_idx
            -
            center
        )
        *
        speed_factor
    )

    warped = np.clip(
        warped,
        0,
        n - 1
    )

    continuous_cols = [
        "cx_norm",
        "cy_norm",
        "w_norm",
        "h_norm",
        "confidence",
    ]

    for col in continuous_cols:

        values = (
            out[col]
            .astype(float)
            .to_numpy()
        )

        out[col] = np.interp(
            original_idx,
            warped,
            values
        )

    detected = (
        out["detected"]
        .astype(float)
        .to_numpy()
    )

    detected_interp = np.interp(
        original_idx,
        warped,
        detected
    )

    out["detected"] = (
        detected_interp
        >=
        0.5
    ).astype(int)

    return out


# ============================================================
# AUG 3 : REALISTIC HEAD MISSING VARIATION
# ============================================================

def augment_missing(df, rng):

    out = df.copy()

    n = len(out)

    detected = (
        out["detected"]
        .astype(int)
        .to_numpy()
        .copy()
    )

    # 정상 수영 특성을 파괴하지 않도록
    # 짧은 추가 missing만 생성
    num_segments = rng.integers(
        1,
        3
    )

    for _ in range(
        num_segments
    ):

        seg_len = int(
            rng.integers(
                1,
                4
            )
        )

        start = int(
            rng.integers(
                0,
                max(
                    1,
                    n - seg_len
                )
            )
        )

        end = min(
            n,
            start + seg_len
        )

        detected[
            start:end
        ] = 0

        out.loc[
            out.index[
                start:end
            ],
            "confidence"
        ] = 0.0

    out["detected"] = (
        detected
    )

    return out


# ============================================================
# MIXED
# ============================================================

def augment_mixed(df, rng):

    out = augment_jitter(
        df,
        rng
    )

    out = augment_tempo(
        out,
        rng
    )

    # mixed에서는 missing 증강 확률 50%
    if rng.random() < 0.5:

        out = augment_missing(
            out,
            rng
        )

    return out


# ============================================================
# FINALIZE AUG
# ============================================================

def finalize_augmented(
    df,
    aug_name,
    source_window_id
):

    out = df.copy()

    out["cx_norm"] = np.clip(
        out["cx_norm"],
        0,
        1
    )

    out["cy_norm"] = np.clip(
        out["cy_norm"],
        0,
        1
    )

    out["w_norm"] = np.clip(
        out["w_norm"],
        0.001,
        1
    )

    out["h_norm"] = np.clip(
        out["h_norm"],
        0.001,
        1
    )

    out["confidence"] = np.clip(
        out["confidence"],
        0,
        1
    )

    out = recompute_motion_features(
        out
    )

    out = recompute_temporal_features(
        out
    )

    out[
        "source_window_id"
    ] = source_window_id

    out[
        "augmentation_type"
    ] = aug_name

    out[
        "is_augmented"
    ] = 1

    out[
        "window_id"
    ] = (
        str(
            source_window_id
        )
        +
        "__"
        +
        aug_name
    )

    return out


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print(
        "Loading:",
        INPUT_CSV
    )

    df = pd.read_csv(
        INPUT_CSV
    )

    print()
    print(
        "rows:",
        len(df)
    )

    print(
        "windows:",
        df[
            "window_id"
        ].nunique()
    )

    print(
        "sources:",
        df[
            "video_id"
        ].nunique()
    )

    print()

    print(
        "styles:"
    )

    print(
        df[
            "swim_style"
        ]
        .value_counts()
    )

    # ========================================================
    # SPLIT BY SOURCE VIDEO
    # ========================================================

    source_ids = (
        df[
            "video_id"
        ]
        .astype(str)
        .unique()
    )

    train_ids, val_ids, test_ids = (
        split_sources(
            source_ids
        )
    )

    train_real = df[
        df["video_id"]
        .astype(str)
        .isin(train_ids)
    ].copy()

    val_real = df[
        df["video_id"]
        .astype(str)
        .isin(val_ids)
    ].copy()

    test_real = df[
        df["video_id"]
        .astype(str)
        .isin(test_ids)
    ].copy()

    for part in [
        train_real,
        val_real,
        test_real
    ]:

        part[
            "source_window_id"
        ] = part[
            "window_id"
        ].astype(str)

        part[
            "augmentation_type"
        ] = "REAL"

        part[
            "is_augmented"
        ] = 0

    print(
        "\nREAL SPLIT"
    )

    print(
        "TRAIN:",
        train_real[
            "window_id"
        ].nunique()
    )

    print(
        "VAL:",
        val_real[
            "window_id"
        ].nunique()
    )

    print(
        "TEST:",
        test_real[
            "window_id"
        ].nunique()
    )

    # ========================================================
    # AUGMENT TRAIN
    # ========================================================

    augmented = []

    groups = list(
        train_real.groupby(
            "window_id",
            sort=False
        )
    )

    print(
        "\nAugmenting..."
    )

    for i, (
        window_id,
        window_df
    ) in enumerate(groups):

        window_df = (
            window_df
            .sort_values(
                "timestep"
            )
            .reset_index(
                drop=True
            )
        )

        if len(
            window_df
        ) != WINDOW_SIZE:

            print(
                "SKIP bad window:",
                window_id,
                len(window_df)
            )

            continue

        for aug_idx in range(
            AUG_PER_REAL_WINDOW
        ):

            rng = np.random.default_rng(
                RANDOM_SEED
                +
                i * 100
                +
                aug_idx
            )

            if aug_idx % 3 == 0:

                aug = augment_jitter(
                    window_df,
                    rng
                )

                aug_name = (
                    f"AUG_JITTER_{aug_idx}"
                )

            elif aug_idx % 3 == 1:

                aug = augment_tempo(
                    window_df,
                    rng
                )

                aug_name = (
                    f"AUG_TEMPO_{aug_idx}"
                )

            else:

                aug = augment_mixed(
                    window_df,
                    rng
                )

                aug_name = (
                    f"AUG_MIXED_{aug_idx}"
                )

            aug = finalize_augmented(
                aug,
                aug_name,
                window_id
            )

            augmented.append(
                aug
            )

        if (
            (i + 1) % 50 == 0
            or
            i + 1 == len(groups)
        ):

            print(
                f"{i + 1}/"
                f"{len(groups)}"
            )

    aug_df = pd.concat(
        augmented,
        ignore_index=True
    )

    train_final = pd.concat(
        [
            train_real,
            aug_df
        ],
        ignore_index=True
    )

    # ========================================================
    # SAVE
    # ========================================================

    train_real.to_csv(
        OUTPUT_DIR
        /
        "NORMAL_TRAIN_REAL.csv",
        index=False,
        encoding="utf-8-sig"
    )

    aug_df.to_csv(
        OUTPUT_DIR
        /
        "NORMAL_TRAIN_AUGMENTED.csv",
        index=False,
        encoding="utf-8-sig"
    )

    train_final.to_csv(
        OUTPUT_DIR
        /
        "NORMAL_TRAIN_FINAL.csv",
        index=False,
        encoding="utf-8-sig"
    )

    val_real.to_csv(
        OUTPUT_DIR
        /
        "NORMAL_VAL_REAL.csv",
        index=False,
        encoding="utf-8-sig"
    )

    test_real.to_csv(
        OUTPUT_DIR
        /
        "NORMAL_TEST_REAL.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    summary = []

    for name, part in [
        (
            "TRAIN_REAL",
            train_real
        ),
        (
            "TRAIN_AUG",
            aug_df
        ),
        (
            "TRAIN_FINAL",
            train_final
        ),
        (
            "VAL_REAL",
            val_real
        ),
        (
            "TEST_REAL",
            test_real
        ),
    ]:

        summary.append({
            "split": name,

            "windows":
                part[
                    "window_id"
                ].nunique(),

            "rows":
                len(part),

            "sources":
                part[
                    "video_id"
                ].nunique(),

            "mean_visible_ratio":
                part[
                    "visible_ratio"
                ].mean(),

            "mean_missing_ratio":
                part[
                    "missing_ratio"
                ].mean(),
        })

    summary_df = pd.DataFrame(
        summary
    )

    summary_df.to_csv(
        OUTPUT_DIR
        /
        "NORMAL_AUGMENT_SUMMARY.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # ========================================================
    # STYLE COUNTS
    # ========================================================

    style_summary = (
        train_final
        .groupby(
            "swim_style"
        )[
            "window_id"
        ]
        .nunique()
        .reset_index(
            name="train_final_windows"
        )
    )

    style_summary.to_csv(
        OUTPUT_DIR
        /
        "NORMAL_STYLE_COUNTS.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print()
    print(
        "=" * 60
    )

    print(
        "DONE"
    )

    print(
        "=" * 60
    )

    print(
        "TRAIN REAL:",
        train_real[
            "window_id"
        ].nunique()
    )

    print(
        "TRAIN AUG:",
        aug_df[
            "window_id"
        ].nunique()
    )

    print(
        "TRAIN FINAL:",
        train_final[
            "window_id"
        ].nunique()
    )

    print(
        "VAL REAL:",
        val_real[
            "window_id"
        ].nunique()
    )

    print(
        "TEST REAL:",
        test_real[
            "window_id"
        ].nunique()
    )

    print()

    print(
        "TRAIN FINAL STYLE:"
    )

    print(
        style_summary
    )

    print()

    print(
        "Output:",
        OUTPUT_DIR
    )


if __name__ == "__main__":
    main()