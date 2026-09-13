from pathlib import Path
import numpy as np
import pandas as pd


TRAIN_CSV = Path(
    r"D:\임베디드 경진대회\머리 추적"
    r"\익수자 판별 학습"
    r"\FINAL_CLASSIFIER_DATASET_V1"
    r"\CLASSIFIER_TRAIN.csv"
)

OUT_NPY = Path(
    r"D:\임베디드 경진대회\머리 추적"
    r"\models"
    r"\classifier_calib.npy"
)

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

WINDOW_SIZE = 50
CALIB_WINDOWS = 128
SEED = 42


def main():
    df = pd.read_csv(TRAIN_CSV)

    windows = []

    for wid, g in df.groupby("window_id", sort=False):
        if "timestep" in g.columns:
            g = g.sort_values("timestep")

        if len(g) != WINDOW_SIZE:
            continue

        x = (
            g[FEATURES]
            .apply(pd.to_numeric, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .to_numpy(dtype=np.float32)
        )

        windows.append(x)

    print("Usable windows:", len(windows))

    rng = np.random.default_rng(SEED)

    idx = rng.choice(
        len(windows),
        size=min(CALIB_WINDOWS, len(windows)),
        replace=False
    )

    selected = np.stack(
        [windows[i] for i in idx],
        axis=0
    )

    # 학습 입력: [B, 1, 50, 14]
    # Hailo calibration용 NHWC 형태:
    # [B, 50, 14, 1]
    selected = np.expand_dims(
        selected,
        axis=-1
    )

    OUT_NPY.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    np.save(
        OUT_NPY,
        selected
    )

    print("DONE")
    print("shape :", selected.shape)
    print("dtype :", selected.dtype)
    print("min   :", selected.min())
    print("max   :", selected.max())
    print("output:", OUT_NPY)


if __name__ == "__main__":
    main()
