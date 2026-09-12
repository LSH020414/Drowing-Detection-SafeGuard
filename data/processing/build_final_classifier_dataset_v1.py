from pathlib import Path
import random
import numpy as np
import pandas as pd


ROOT = Path(r"D:\임베디드 경진대회\머리 추적")

OUTPUT = (
    ROOT
    / "익수자 판별 학습"
    / "FINAL_CLASSIFIER_DATASET_V1"
)
OUTPUT.mkdir(parents=True, exist_ok=True)

WINDOW_SIZE = 50
SEED = 42

FEATURES = [
    "detected",
    "confidence",
    "cx_norm",
    "cy_norm",
    "head_scale",
    "dx",
    "dy",
    "speed",
    "scale_change",
    "lost_event",
    "redetect_event",
    "consecutive_missing",
    "visible_ratio",
    "missing_ratio",
]


# ============================================================
# PATHS
# ============================================================

SWIMXYZ = (
    ROOT
    / "익수자 판별 학습"
    / "SwimXYZ_NORMAL_CLEAN"
    / "NORMAL_FINAL_sequences.csv"
)

SWIM4_ROOT = (
    ROOT
    / "수영학습"
    / "NORMAL_4STROKES_AUGMENTED_V1"
)

SWIM4_TRAIN = SWIM4_ROOT / "NORMAL_TRAIN_FINAL.csv"
SWIM4_VAL = SWIM4_ROOT / "NORMAL_VAL_REAL.csv"
SWIM4_TEST = SWIM4_ROOT / "NORMAL_TEST_REAL.csv"

FLOATING = (
    ROOT
    / "익수자 판별 학습"
    / "FLOATING_FINAL"
    / "FLOATING_FINAL_sequences.csv"
)

ACTIVE_ROOT = (
    ROOT
    / "익수자 판별 학습"
    / "ACTIVE_DROWNING_FINAL"
    / "ACTIVE_AUGMENTED_V1"
)

ACTIVE_TRAIN = ACTIVE_ROOT / "ACTIVE_TRAIN_FINAL.csv"
ACTIVE_VAL = ACTIVE_ROOT / "ACTIVE_VAL_REAL.csv"
ACTIVE_TEST = ACTIVE_ROOT / "ACTIVE_TEST_REAL.csv"


# ============================================================
# HELPERS
# ============================================================

def first_existing(df, names, default=np.nan):
    for n in names:
        if n in df.columns:
            return pd.to_numeric(df[n], errors="coerce")
    return pd.Series([default] * len(df), index=df.index)


def build_features(g):
    detected = first_existing(
        g,
        ["detected", "visible"],
        1.0
    ).fillna(0).clip(0, 1)

    confidence = first_existing(
        g,
        ["confidence", "conf", "score"],
        1.0
    ).fillna(0).clip(0, 1)

    cx = first_existing(
        g,
        ["cx_norm", "center_x_norm"],
        np.nan
    )

    cy = first_existing(
        g,
        ["cy_norm", "center_y_norm"],
        np.nan
    )

    cx = cx.interpolate(limit_direction="both").ffill().bfill().fillna(0.5)
    cy = cy.interpolate(limit_direction="both").ffill().bfill().fillna(0.5)

    if "head_scale_norm" in g.columns:
        scale = pd.to_numeric(
            g["head_scale_norm"],
            errors="coerce"
        )
    elif "head_scale" in g.columns:
        scale = pd.to_numeric(
            g["head_scale"],
            errors="coerce"
        )
    else:
        w = first_existing(
            g,
            ["w_norm", "width_norm"],
            np.nan
        )
        h = first_existing(
            g,
            ["h_norm", "height_norm"],
            np.nan
        )

        scale = np.sqrt(
            np.maximum(
                w * h,
                0
            )
        )

    scale = (
        pd.Series(scale)
        .interpolate(limit_direction="both")
        .ffill()
        .bfill()
        .fillna(0.0)
    )

    dx = cx.diff().fillna(0)
    dy = cy.diff().fillna(0)

    speed = np.sqrt(
        dx ** 2
        +
        dy ** 2
    )

    scale_change = (
        scale.diff().fillna(0)
    )

    detected_np = (
        detected.round()
        .astype(int)
        .to_numpy()
    )

    change = np.diff(
        detected_np,
        prepend=detected_np[0]
    )

    lost_event = (
        change < 0
    ).astype(np.float32)

    redetect_event = (
        change > 0
    ).astype(np.float32)

    consecutive_missing = []

    count = 0

    for d in detected_np:
        if d == 0:
            count += 1
        else:
            count = 0

        consecutive_missing.append(
            count / WINDOW_SIZE
        )

    visible_ratio = float(
        detected_np.mean()
    )

    missing_ratio = (
        1.0
        -
        visible_ratio
    )

    return pd.DataFrame({
        "detected":
            detected_np.astype(np.float32),

        "confidence":
            confidence.to_numpy(dtype=np.float32),

        "cx_norm":
            cx.to_numpy(dtype=np.float32),

        "cy_norm":
            cy.to_numpy(dtype=np.float32),

        "head_scale":
            np.asarray(scale, dtype=np.float32),

        "dx":
            dx.to_numpy(dtype=np.float32),

        "dy":
            dy.to_numpy(dtype=np.float32),

        "speed":
            np.asarray(speed, dtype=np.float32),

        "scale_change":
            scale_change.to_numpy(dtype=np.float32),

        "lost_event":
            lost_event,

        "redetect_event":
            redetect_event,

        "consecutive_missing":
            np.asarray(
                consecutive_missing,
                dtype=np.float32
            ),

        "visible_ratio":
            np.full(
                WINDOW_SIZE,
                visible_ratio,
                dtype=np.float32
            ),

        "missing_ratio":
            np.full(
                WINDOW_SIZE,
                missing_ratio,
                dtype=np.float32
            ),
    })


def process_csv(path, label, prefix):
    print("\nLOAD:", path)

    if not path.exists():
        print("NOT FOUND")
        return pd.DataFrame()

    df = pd.read_csv(path)

    if "window_id" not in df.columns:
        raise RuntimeError(
            f"window_id 없음: {path}"
        )

    results = []

    for wid, g in df.groupby(
        "window_id",
        sort=False
    ):
        if len(g) != WINDOW_SIZE:
            continue

        if "timestep" in g.columns:
            g = g.sort_values("timestep")
        elif "frame_idx" in g.columns:
            g = g.sort_values("frame_idx")
        elif "time_sec" in g.columns:
            g = g.sort_values("time_sec")

        feat = build_features(
            g.reset_index(drop=True)
        )

        feat["window_id"] = (
            prefix
            +
            "__"
            +
            str(wid)
        )

        if "video_id" in g.columns:
            video_id = str(
                g.iloc[0]["video_id"]
            )
        else:
            video_id = str(wid)

        feat["video_id"] = (
            prefix
            +
            "__"
            +
            video_id
        )

        feat["label"] = label

        feat["class_id"] = {
            "SWIMMING": 0,
            "FLOATING": 1,
            "ACTIVE": 2
        }[label]

        feat["timestep"] = np.arange(
            WINDOW_SIZE
        )

        results.append(feat)

    if not results:
        return pd.DataFrame()

    result = pd.concat(
        results,
        ignore_index=True
    )

    print(
        label,
        "windows:",
        result["window_id"].nunique()
    )

    return result


def split_by_video(df):
    videos = list(
        df["video_id"].unique()
    )

    random.Random(
        SEED
    ).shuffle(videos)

    n = len(videos)

    n_train = int(
        round(n * 0.70)
    )

    n_val = int(
        round(n * 0.15)
    )

    train_ids = set(
        videos[:n_train]
    )

    val_ids = set(
        videos[
            n_train:
            n_train + n_val
        ]
    )

    test_ids = set(
        videos[
            n_train + n_val:
        ]
    )

    return (
        df[
            df["video_id"].isin(
                train_ids
            )
        ].copy(),

        df[
            df["video_id"].isin(
                val_ids
            )
        ].copy(),

        df[
            df["video_id"].isin(
                test_ids
            )
        ].copy(),
    )


def main():

    # --------------------------------------------------------
    # SWIMXYZ
    # --------------------------------------------------------

    swimxyz = process_csv(
        SWIMXYZ,
        "SWIMMING",
        "SWIMXYZ"
    )

    sx_train, sx_val, sx_test = (
        split_by_video(swimxyz)
    )

    # --------------------------------------------------------
    # REAL 4 STROKES
    # --------------------------------------------------------

    sw4_train = process_csv(
        SWIM4_TRAIN,
        "SWIMMING",
        "SWIM4"
    )

    sw4_val = process_csv(
        SWIM4_VAL,
        "SWIMMING",
        "SWIM4VAL"
    )

    sw4_test = process_csv(
        SWIM4_TEST,
        "SWIMMING",
        "SWIM4TEST"
    )

    swimming_train = pd.concat(
        [sx_train, sw4_train],
        ignore_index=True
    )

    swimming_val = pd.concat(
        [sx_val, sw4_val],
        ignore_index=True
    )

    swimming_test = pd.concat(
        [sx_test, sw4_test],
        ignore_index=True
    )

    # --------------------------------------------------------
    # FLOATING
    # --------------------------------------------------------

    floating = process_csv(
        FLOATING,
        "FLOATING",
        "FLOATING"
    )

    fl_train, fl_val, fl_test = (
        split_by_video(floating)
    )

    # --------------------------------------------------------
    # ACTIVE
    # --------------------------------------------------------

    ac_train = process_csv(
        ACTIVE_TRAIN,
        "ACTIVE",
        "ACTIVE"
    )

    ac_val = process_csv(
        ACTIVE_VAL,
        "ACTIVE",
        "ACTIVEVAL"
    )

    ac_test = process_csv(
        ACTIVE_TEST,
        "ACTIVE",
        "ACTIVETEST"
    )

    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------

    train = pd.concat(
        [
            swimming_train,
            fl_train,
            ac_train
        ],
        ignore_index=True
    )

    val = pd.concat(
        [
            swimming_val,
            fl_val,
            ac_val
        ],
        ignore_index=True
    )

    test = pd.concat(
        [
            swimming_test,
            fl_test,
            ac_test
        ],
        ignore_index=True
    )

    train.to_csv(
        OUTPUT / "CLASSIFIER_TRAIN.csv",
        index=False,
        encoding="utf-8-sig"
    )

    val.to_csv(
        OUTPUT / "CLASSIFIER_VAL.csv",
        index=False,
        encoding="utf-8-sig"
    )

    test.to_csv(
        OUTPUT / "CLASSIFIER_TEST.csv",
        index=False,
        encoding="utf-8-sig"
    )

    pd.DataFrame({
        "feature": FEATURES
    }).to_csv(
        OUTPUT / "FEATURE_SCHEMA.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("\n==============================")
    print("FINAL COUNTS")
    print("==============================")

    for name, part in [
        ("TRAIN", train),
        ("VAL", val),
        ("TEST", test),
    ]:
        print("\n", name)

        print(
            part.groupby(
                "label"
            )["window_id"]
            .nunique()
        )

    print("\nOUTPUT:", OUTPUT)


if __name__ == "__main__":
    main()
