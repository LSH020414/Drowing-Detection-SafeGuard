from pathlib import Path
import random
import shutil

import cv2
import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

SRC_DIR = Path(
    r"D:\임베디드 경진대회\머리 추적\CVAT용 영상\extracted_1fps"
)

DST_DIR = Path(
    r"D:\임베디드 경진대회\머리 추적\hailo_calib_head"
)

TARGET_COUNT = 400
SEED = 42

IMAGE_EXTS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


# ============================================================
# IMAGE ANALYSIS
# ============================================================

def analyze_image(path):
    img = cv2.imread(str(path))

    if img is None:
        return None

    h, w = img.shape[:2]

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    brightness = float(
        np.mean(gray)
    )

    contrast = float(
        np.std(gray)
    )

    return {
        "path": path,
        "width": w,
        "height": h,
        "pixels": w * h,
        "brightness": brightness,
        "contrast": contrast,
    }


# ============================================================
# SOURCE GROUP
# ============================================================

def get_source_group(path):
    """
    source 다양성을 확보하기 위한 그룹 분류.

    1순위:
        extracted_1fps 아래 첫 번째 하위 폴더

    하위 폴더가 없다면:
        파일명 앞쪽 토큰을 사용
    """

    rel = path.relative_to(SRC_DIR)

    if len(rel.parts) > 1:
        return rel.parts[0]

    stem = path.stem

    parts = (
        stem
        .replace("-", "_")
        .split("_")
    )

    if len(parts) >= 2:
        return "_".join(parts[:2])

    return stem


# ============================================================
# SAFE PERCENTILE BIN
# ============================================================

def percentile_bin(series, names):
    """
    qcut을 사용하지 않는다.

    동일값이 많거나 해상도 종류가 적어도
    절대 duplicate edge 에러가 발생하지 않도록
    percentile rank를 기반으로 분할한다.
    """

    numeric = pd.to_numeric(
        series,
        errors="coerce"
    )

    unique_count = numeric.nunique(
        dropna=True
    )

    if unique_count <= 1:
        return pd.Series(
            [names[0]] * len(series),
            index=series.index
        )

    percentile = numeric.rank(
        method="average",
        pct=True
    )

    if len(names) == 2:
        return pd.cut(
            percentile,
            bins=[
                0.0,
                0.5,
                1.0
            ],
            labels=names,
            include_lowest=True
        )

    if len(names) == 3:
        return pd.cut(
            percentile,
            bins=[
                0.0,
                1 / 3,
                2 / 3,
                1.0
            ],
            labels=names,
            include_lowest=True
        )

    if len(names) == 4:
        return pd.cut(
            percentile,
            bins=[
                0.0,
                0.25,
                0.50,
                0.75,
                1.0
            ],
            labels=names,
            include_lowest=True
        )

    raise ValueError(
        "지원하지 않는 bin 개수입니다."
    )


# ============================================================
# BINNING
# ============================================================

def make_bins(df):

    # --------------------------------------------------------
    # BRIGHTNESS
    # --------------------------------------------------------

    df["brightness_bin"] = pd.cut(
        df["brightness"],
        bins=[
            -1,
            70,
            120,
            180,
            256
        ],
        labels=[
            "dark",
            "mid_dark",
            "mid_bright",
            "bright"
        ],
        include_lowest=True
    )

    # --------------------------------------------------------
    # CONTRAST
    # --------------------------------------------------------

    df["contrast_bin"] = percentile_bin(
        df["contrast"],
        [
            "low_contrast",
            "mid_contrast",
            "high_contrast"
        ]
    )

    # --------------------------------------------------------
    # RESOLUTION
    # --------------------------------------------------------

    resolution_count = df["pixels"].nunique(
        dropna=True
    )

    print()
    print(
        "[INFO] 고유 해상도 개수:",
        resolution_count
    )

    if resolution_count <= 1:

        df["resolution_bin"] = (
            "same_res"
        )

    elif resolution_count == 2:

        unique_pixels = sorted(
            df["pixels"]
            .dropna()
            .unique()
            .tolist()
        )

        low_pixels = unique_pixels[0]
        high_pixels = unique_pixels[1]

        df["resolution_bin"] = np.where(
            df["pixels"] == low_pixels,
            "small_res",
            "large_res"
        )

        print(
            "[INFO] small resolution pixels:",
            low_pixels
        )

        print(
            "[INFO] large resolution pixels:",
            high_pixels
        )

    else:

        df["resolution_bin"] = percentile_bin(
            df["pixels"],
            [
                "small_res",
                "mid_res",
                "large_res"
            ]
        )

    return df


# ============================================================
# BALANCED SAMPLING
# ============================================================

def balanced_sample(df, target_count):
    """
    단순 랜덤이 아니라 아래를 최대한 균형 있게 뽑음.

    - source_group
    - brightness
    - contrast
    - resolution
    """

    rng = random.Random(SEED)

    selected_indices = []
    selected_set = set()

    # --------------------------------------------------------
    # 1. 각 source에서 최소한 일부 확보
    # --------------------------------------------------------

    source_groups = list(
        df.groupby(
            "source_group"
        )
    )

    rng.shuffle(
        source_groups
    )

    # source 수에 따라 균등 quota
    source_count = max(
        1,
        len(source_groups)
    )

    per_source_quota = max(
        1,
        target_count // source_count
    )

    for source_name, group in source_groups:

        indices = group.index.tolist()

        rng.shuffle(indices)

        take = min(
            per_source_quota,
            len(indices)
        )

        for idx in indices[:take]:

            if idx not in selected_set:

                selected_indices.append(
                    idx
                )

                selected_set.add(
                    idx
                )

            if (
                len(selected_indices)
                >= target_count
            ):
                break

        if (
            len(selected_indices)
            >= target_count
        ):
            break


    # --------------------------------------------------------
    # 2. brightness / contrast / resolution 조합 균형
    # --------------------------------------------------------

    if (
        len(selected_indices)
        < target_count
    ):

        combo_groups = list(
            df.groupby(
                [
                    "brightness_bin",
                    "contrast_bin",
                    "resolution_bin"
                ],
                observed=True
            )
        )

        rng.shuffle(
            combo_groups
        )

        while (
            len(selected_indices)
            < target_count
        ):

            added = 0

            for _, group in combo_groups:

                candidates = (
                    group.index.tolist()
                )

                rng.shuffle(
                    candidates
                )

                for idx in candidates:

                    if (
                        idx
                        not in selected_set
                    ):

                        selected_indices.append(
                            idx
                        )

                        selected_set.add(
                            idx
                        )

                        added += 1

                        break

                if (
                    len(selected_indices)
                    >= target_count
                ):
                    break

            if added == 0:
                break


    # --------------------------------------------------------
    # 3. 부족하면 전체에서 랜덤 보충
    # --------------------------------------------------------

    if (
        len(selected_indices)
        < target_count
    ):

        remaining = [
            idx
            for idx in df.index
            if idx not in selected_set
        ]

        rng.shuffle(
            remaining
        )

        need = (
            target_count
            - len(selected_indices)
        )

        selected_indices.extend(
            remaining[:need]
        )


    # --------------------------------------------------------
    # 결과
    # --------------------------------------------------------

    selected_indices = (
        selected_indices[
            :target_count
        ]
    )

    result = df.loc[
        selected_indices
    ].copy()

    result = result.reset_index(
        drop=True
    )

    return result


# ============================================================
# OUTPUT CLEAN
# ============================================================

def clear_output_images():
    DST_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    for path in DST_DIR.iterdir():

        if (
            path.is_file()
            and
            path.suffix.lower()
            in IMAGE_EXTS
        ):
            path.unlink()


# ============================================================
# MAIN
# ============================================================

def main():

    random.seed(SEED)
    np.random.seed(SEED)

    print("=" * 70)
    print(
        "HAILO DETECTOR CALIBRATION SET BUILDER"
    )
    print("=" * 70)

    print()
    print("SOURCE:")
    print(SRC_DIR)

    print()
    print("OUTPUT:")
    print(DST_DIR)

    # --------------------------------------------------------
    # SOURCE CHECK
    # --------------------------------------------------------

    if not SRC_DIR.exists():

        raise FileNotFoundError(
            f"Source folder not found:\n"
            f"{SRC_DIR}"
        )

    # --------------------------------------------------------
    # FIND IMAGES
    # --------------------------------------------------------

    files = [
        p
        for p in SRC_DIR.rglob("*")
        if (
            p.is_file()
            and
            p.suffix.lower()
            in IMAGE_EXTS
        )
    ]

    print()
    print(
        "전체 이미지:",
        len(files)
    )

    if len(files) == 0:

        raise RuntimeError(
            "이미지를 찾지 못했습니다."
        )

    # --------------------------------------------------------
    # ANALYZE
    # --------------------------------------------------------

    records = []

    for i, path in enumerate(
        files,
        start=1
    ):

        info = analyze_image(
            path
        )

        if info is None:
            continue

        info[
            "source_group"
        ] = get_source_group(
            path
        )

        records.append(
            info
        )

        if (
            i % 500 == 0
            or
            i == len(files)
        ):

            print(
                f"분석 중: "
                f"{i}/{len(files)}"
            )

    df = pd.DataFrame(
        records
    )

    print()
    print(
        "정상 이미지:",
        len(df)
    )

    print(
        "source 그룹:",
        df[
            "source_group"
        ].nunique()
    )

    # --------------------------------------------------------
    # BINNING
    # --------------------------------------------------------

    df = make_bins(
        df
    )

    # --------------------------------------------------------
    # TARGET
    # --------------------------------------------------------

    target = min(
        TARGET_COUNT,
        len(df)
    )

    print()
    print(
        "목표 calibration:",
        target
    )

    # --------------------------------------------------------
    # SELECT
    # --------------------------------------------------------

    selected = balanced_sample(
        df,
        target
    )

    print()
    print(
        "선택된 이미지:",
        len(selected)
    )

    # --------------------------------------------------------
    # OUTPUT CLEAN
    # --------------------------------------------------------

    clear_output_images()

    # --------------------------------------------------------
    # COPY
    # --------------------------------------------------------

    output_names = []

    for idx, row in selected.iterrows():

        src = row[
            "path"
        ]

        ext = (
            src.suffix.lower()
        )

        dst_name = (
            f"calib_"
            f"{idx:04d}"
            f"{ext}"
        )

        dst = (
            DST_DIR
            / dst_name
        )

        shutil.copy2(
            src,
            dst
        )

        output_names.append(
            dst_name
        )

    selected[
        "calibration_filename"
    ] = output_names

    # --------------------------------------------------------
    # SAVE MANIFEST
    # --------------------------------------------------------

    manifest_path = (
        DST_DIR
        / "calibration_manifest.csv"
    )

    # Path 객체를 문자열로 변환
    selected[
        "path"
    ] = selected[
        "path"
    ].astype(str)

    selected.to_csv(
        manifest_path,
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "CALIBRATION SUMMARY"
    )
    print("=" * 70)

    print()
    print(
        "총 calibration:",
        len(selected)
    )

    print(
        "출처 그룹 수:",
        selected[
            "source_group"
        ].nunique()
    )

    # --------------------------------------------------------
    # SOURCE SUMMARY
    # --------------------------------------------------------

    print()
    print("SOURCE GROUP:")

    print(
        selected[
            "source_group"
        ]
        .value_counts()
    )

    # --------------------------------------------------------
    # BRIGHTNESS SUMMARY
    # --------------------------------------------------------

    print()
    print("BRIGHTNESS:")

    print(
        selected[
            "brightness_bin"
        ]
        .value_counts(
            dropna=False
        )
        .sort_index()
    )

    # --------------------------------------------------------
    # CONTRAST SUMMARY
    # --------------------------------------------------------

    print()
    print("CONTRAST:")

    print(
        selected[
            "contrast_bin"
        ]
        .value_counts(
            dropna=False
        )
        .sort_index()
    )

    # --------------------------------------------------------
    # RESOLUTION SUMMARY
    # --------------------------------------------------------

    print()
    print("RESOLUTION:")

    print(
        selected[
            "resolution_bin"
        ]
        .value_counts(
            dropna=False
        )
        .sort_index()
    )

    # --------------------------------------------------------
    # RESOLUTION DETAIL
    # --------------------------------------------------------

    print()
    print("IMAGE SIZE:")

    size_summary = (
        selected
        .groupby(
            [
                "width",
                "height"
            ]
        )
        .size()
        .sort_values(
            ascending=False
        )
    )

    print(
        size_summary
    )

    # --------------------------------------------------------
    # BRIGHTNESS RANGE
    # --------------------------------------------------------

    print()
    print(
        "Brightness range:",
        f"{selected['brightness'].min():.2f}",
        "~",
        f"{selected['brightness'].max():.2f}"
    )

    print(
        "Contrast range:",
        f"{selected['contrast'].min():.2f}",
        "~",
        f"{selected['contrast'].max():.2f}"
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
        "Calibration images:"
    )

    print(
        DST_DIR
    )

    print()
    print(
        "Manifest:"
    )

    print(
        manifest_path
    )


# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":
    main()