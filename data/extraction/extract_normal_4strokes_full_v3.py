import re
import zipfile
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

ROOT_DIR = Path(
    r"D:\임베디드 경진대회\머리 추적\수영학습"
)

OUTPUT_DIR = ROOT_DIR / "NORMAL_4STROKES_FULL_V3"

TARGET_FPS = 10.0

WINDOW_SIZE = 50     # 5 sec
WINDOW_STRIDE = 25   # 2.5 sec overlap


# ============================================================
# STYLE
# ============================================================

def get_swim_style(name):
    n = name.upper()

    if "BACKSTROKE" in n:
        return "BACKSTROKE"

    if "BREASTSTROCK" in n or "BREASTSTROKE" in n:
        return "BREASTSTROKE"

    if "BUTTERFLY" in n:
        return "BUTTERFLY"

    if "FREESTYLE" in n:
        return "FREESTYLE"

    return None


def natural_key(text):
    return [
        int(x) if x.isdigit() else x.lower()
        for x in re.split(r"(\d+)", str(text))
    ]


def safe_float(value, default=np.nan):
    try:
        return float(value)
    except Exception:
        return default


def is_head_label(label):
    if not label:
        return False

    x = label.strip().lower()

    return (
        x == "head"
        or "head" in x
        or "머리" in x
    )


# ============================================================
# XML FIND
# ============================================================

def find_xml(root_dir):
    files = list(root_dir.rglob("annotations.xml"))

    if not files:
        files = list(root_dir.rglob("*.xml"))

    if not files:
        return None

    return files[0]


# ============================================================
# META
# ============================================================

def get_task_meta(root):
    """
    CVAT meta에서 원본 영상 크기 / frame count 등을 최대한 찾음
    """

    frame_count = None
    width = None
    height = None

    size_node = root.find(".//meta/task/size")

    if size_node is not None:
        try:
            frame_count = int(size_node.text)
        except Exception:
            pass

    width_node = root.find(".//meta/task/original_size/width")
    height_node = root.find(".//meta/task/original_size/height")

    if width_node is not None:
        width = safe_float(width_node.text)

    if height_node is not None:
        height = safe_float(height_node.text)

    return frame_count, width, height


# ============================================================
# IMAGE FORMAT
# ============================================================

def parse_image_format(root):
    """
    CVAT for images 형식

    모든 image element를 frame으로 사용.
    head box 없는 frame도 detected=0으로 유지.
    """

    records = []

    images = root.findall(".//image")

    for image in images:

        frame_idx = int(
            image.attrib.get("id", 0)
        )

        image_name = image.attrib.get(
            "name",
            ""
        )

        img_w = safe_float(
            image.attrib.get("width"),
            0
        )

        img_h = safe_float(
            image.attrib.get("height"),
            0
        )

        head_boxes = []

        for box in image.findall("box"):

            label = box.attrib.get(
                "label",
                ""
            )

            if not is_head_label(label):
                continue

            xtl = safe_float(
                box.attrib.get("xtl")
            )

            ytl = safe_float(
                box.attrib.get("ytl")
            )

            xbr = safe_float(
                box.attrib.get("xbr")
            )

            ybr = safe_float(
                box.attrib.get("ybr")
            )

            if np.isnan(
                [xtl, ytl, xbr, ybr]
            ).any():
                continue

            area = (
                max(0.0, xbr - xtl)
                *
                max(0.0, ybr - ytl)
            )

            head_boxes.append(
                (
                    area,
                    xtl,
                    ytl,
                    xbr,
                    ybr
                )
            )

        if not head_boxes:

            records.append({
                "frame_idx": frame_idx,
                "image_name": image_name,

                "detected": 0,
                "confidence": 0.0,

                "cx": np.nan,
                "cy": np.nan,
                "w": np.nan,
                "h": np.nan,

                "cx_norm": np.nan,
                "cy_norm": np.nan,
                "w_norm": np.nan,
                "h_norm": np.nan,
            })

            continue

        # 단일 수영자 데이터라고 보고 가장 큰 head 사용
        head_boxes.sort(
            key=lambda x: x[0],
            reverse=True
        )

        _, xtl, ytl, xbr, ybr = head_boxes[0]

        bw = xbr - xtl
        bh = ybr - ytl

        cx = (xtl + xbr) / 2.0
        cy = (ytl + ybr) / 2.0

        records.append({
            "frame_idx": frame_idx,
            "image_name": image_name,

            "detected": 1,
            "confidence": 1.0,

            "cx": cx,
            "cy": cy,
            "w": bw,
            "h": bh,

            "cx_norm":
                cx / img_w
                if img_w > 0
                else np.nan,

            "cy_norm":
                cy / img_h
                if img_h > 0
                else np.nan,

            "w_norm":
                bw / img_w
                if img_w > 0
                else np.nan,

            "h_norm":
                bh / img_h
                if img_h > 0
                else np.nan,
        })

    return records


# ============================================================
# TRACK FORMAT
# ============================================================

def interpolate_track_boxes(track_boxes, frame_count=None):
    """
    CVAT track keyframe 사이를 선형보간.

    outside=1 구간은 detected=0.

    중요:
    CVAT에서 box가 매 프레임 명시되지 않아도,
    두 keyframe 사이에는 실제 track이 존재하므로
    프레임을 전부 missing으로 처리하면 안 됨.
    """

    boxes = sorted(
        track_boxes,
        key=lambda b: b["frame_idx"]
    )

    if not boxes:
        return {}

    if frame_count is None:
        frame_count = (
            max(b["frame_idx"] for b in boxes)
            + 1
        )

    result = {}

    # 처음 keyframe 이전
    first = boxes[0]

    for f in range(
        0,
        min(first["frame_idx"], frame_count)
    ):
        result[f] = None

    # keyframe 사이 interpolation
    for i in range(len(boxes) - 1):

        a = boxes[i]
        b = boxes[i + 1]

        fa = a["frame_idx"]
        fb = b["frame_idx"]

        if fb <= fa:
            continue

        for f in range(
            fa,
            min(fb, frame_count)
        ):

            # outside이면 missing
            if a["outside"] == 1:
                result[f] = None
                continue

            t = (
                (f - fa)
                /
                float(fb - fa)
            )

            result[f] = {
                "xtl":
                    a["xtl"]
                    + t * (b["xtl"] - a["xtl"]),

                "ytl":
                    a["ytl"]
                    + t * (b["ytl"] - a["ytl"]),

                "xbr":
                    a["xbr"]
                    + t * (b["xbr"] - a["xbr"]),

                "ybr":
                    a["ybr"]
                    + t * (b["ybr"] - a["ybr"]),
            }

    # 마지막 keyframe
    last = boxes[-1]

    if last["frame_idx"] < frame_count:

        if last["outside"] == 0:

            result[last["frame_idx"]] = {
                "xtl": last["xtl"],
                "ytl": last["ytl"],
                "xbr": last["xbr"],
                "ybr": last["ybr"],
            }

        else:
            result[last["frame_idx"]] = None

    # 마지막 keyframe 이후
    for f in range(
        last["frame_idx"] + 1,
        frame_count
    ):
        result[f] = None

    return result


def parse_track_format(root):
    """
    CVAT track 형식.

    가장 긴 head track을 단일 수영자로 사용.
    outside=1 구간은 detected=0.
    """

    frame_count, img_w, img_h = (
        get_task_meta(root)
    )

    tracks = []

    for track in root.findall(".//track"):

        label = track.attrib.get(
            "label",
            ""
        )

        if not is_head_label(label):
            continue

        boxes = []

        for box in track.findall("box"):

            boxes.append({
                "frame_idx": int(
                    box.attrib.get(
                        "frame",
                        0
                    )
                ),

                "outside": int(
                    box.attrib.get(
                        "outside",
                        0
                    )
                ),

                "xtl": safe_float(
                    box.attrib.get("xtl")
                ),

                "ytl": safe_float(
                    box.attrib.get("ytl")
                ),

                "xbr": safe_float(
                    box.attrib.get("xbr")
                ),

                "ybr": safe_float(
                    box.attrib.get("ybr")
                ),
            })

        if boxes:
            tracks.append(boxes)

    if not tracks:
        return []

    # 단일 수영자 기준 가장 긴 track
    tracks.sort(
        key=lambda x: (
            max(b["frame_idx"] for b in x)
            -
            min(b["frame_idx"] for b in x)
        ),
        reverse=True
    )

    selected_track = tracks[0]

    if frame_count is None:
        frame_count = (
            max(
                b["frame_idx"]
                for b in selected_track
            )
            + 1
        )

    frame_map = interpolate_track_boxes(
        selected_track,
        frame_count=frame_count
    )

    records = []

    for frame_idx in range(frame_count):

        box = frame_map.get(
            frame_idx
        )

        if box is None:

            records.append({
                "frame_idx": frame_idx,
                "image_name": "",

                "detected": 0,
                "confidence": 0.0,

                "cx": np.nan,
                "cy": np.nan,
                "w": np.nan,
                "h": np.nan,

                "cx_norm": np.nan,
                "cy_norm": np.nan,
                "w_norm": np.nan,
                "h_norm": np.nan,
            })

            continue

        bw = (
            box["xbr"]
            -
            box["xtl"]
        )

        bh = (
            box["ybr"]
            -
            box["ytl"]
        )

        cx = (
            box["xtl"]
            +
            box["xbr"]
        ) / 2.0

        cy = (
            box["ytl"]
            +
            box["ybr"]
        ) / 2.0

        records.append({
            "frame_idx": frame_idx,
            "image_name": "",

            "detected": 1,
            "confidence": 1.0,

            "cx": cx,
            "cy": cy,
            "w": bw,
            "h": bh,

            "cx_norm":
                cx / img_w
                if img_w and img_w > 0
                else np.nan,

            "cy_norm":
                cy / img_h
                if img_h and img_h > 0
                else np.nan,

            "w_norm":
                bw / img_w
                if img_w and img_w > 0
                else np.nan,

            "h_norm":
                bh / img_h
                if img_h and img_h > 0
                else np.nan,
        })

    return records


# ============================================================
# XML PARSE
# ============================================================

def parse_cvat_xml(xml_path):

    tree = ET.parse(
        xml_path
    )

    root = tree.getroot()

    images = root.findall(
        ".//image"
    )

    tracks = root.findall(
        ".//track"
    )

    if images:

        return (
            parse_image_format(root),
            "IMAGE"
        )

    if tracks:

        return (
            parse_track_format(root),
            "TRACK"
        )

    return [], "UNKNOWN"


# ============================================================
# FEATURE PREP
# ============================================================

def prepare_frames(df):
    """
    detected=0은 절대 변경하지 않음.

    좌표는 temporal feature 계산을 위해서만
    보간/ffill/bfill 한다.

    즉:
      detected=0
      confidence=0
    정보는 그대로 보존.
    """

    df = df.sort_values(
        "frame_idx"
    ).reset_index(
        drop=True
    )

    df["detected"] = (
        pd.to_numeric(
            df["detected"],
            errors="coerce"
        )
        .fillna(0)
        .astype(int)
    )

    df["confidence"] = (
        pd.to_numeric(
            df["confidence"],
            errors="coerce"
        )
        .fillna(0.0)
    )

    coord_cols = [
        "cx",
        "cy",
        "w",
        "h",
        "cx_norm",
        "cy_norm",
        "w_norm",
        "h_norm",
    ]

    for col in coord_cols:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

        # feature 계산용 보간
        df[col] = (
            df[col]
            .interpolate(
                limit_direction="both"
            )
            .ffill()
            .bfill()
        )

    return df


# ============================================================
# LOST FEATURES
# ============================================================

def add_temporal_features(w):

    w = w.copy()

    w["head_scale"] = np.sqrt(
        np.maximum(
            w["w_norm"]
            *
            w["h_norm"],
            0.0
        )
    )

    w["dx"] = (
        w["cx_norm"]
        .diff()
        .fillna(0.0)
    )

    w["dy"] = (
        w["cy_norm"]
        .diff()
        .fillna(0.0)
    )

    w["speed"] = np.sqrt(
        w["dx"] ** 2
        +
        w["dy"] ** 2
    )

    w["vertical_speed"] = (
        w["cy_norm"]
        .diff()
        .fillna(0.0)
    )

    w["scale_change"] = (
        w["head_scale"]
        .diff()
        .fillna(0.0)
    )

    change = (
        w["detected"]
        .diff()
        .fillna(0)
    )

    w["lost_event"] = (
        change < 0
    ).astype(int)

    w["redetect_event"] = (
        change > 0
    ).astype(int)

    # ----------------------------
    # consecutive missing
    # ----------------------------

    consecutive_missing = []

    count = 0

    for detected in w["detected"]:

        if detected == 0:
            count += 1

        else:
            count = 0

        consecutive_missing.append(
            count
        )

    w["consecutive_missing"] = (
        consecutive_missing
    )

    # ----------------------------
    # Window-level temporal stats
    # repeated for all 50 steps
    # ----------------------------

    visible_ratio = float(
        w["detected"].mean()
    )

    missing_ratio = (
        1.0
        -
        visible_ratio
    )

    max_missing = int(
        max(
            consecutive_missing
        )
    )

    lost_count = int(
        w["lost_event"].sum()
    )

    redetect_count = int(
        w["redetect_event"].sum()
    )

    w["visible_ratio"] = (
        visible_ratio
    )

    w["missing_ratio"] = (
        missing_ratio
    )

    w["max_consecutive_missing"] = (
        max_missing
    )

    w["lost_count"] = (
        lost_count
    )

    w["redetect_count"] = (
        redetect_count
    )

    return w


# ============================================================
# WINDOWS
# ============================================================

def make_windows(
    df,
    source_name,
    style
):

    windows = []

    n = len(df)

    if n < WINDOW_SIZE:

        print(
            f"    too short: "
            f"{n} frames"
        )

        return windows

    window_idx = 0

    for start in range(
        0,
        n - WINDOW_SIZE + 1,
        WINDOW_STRIDE
    ):

        end = (
            start
            +
            WINDOW_SIZE
        )

        w = df.iloc[
            start:end
        ].copy()

        # 어떤 visible ratio든 허용.
        # 심지어 0이어도 영상이 수영 구간이면 NORMAL 데이터.
        visible_ratio = float(
            w["detected"].mean()
        )

        window_id = (
            f"NORMAL_"
            f"{style}_"
            f"{source_name}_"
            f"{window_idx:05d}"
        )

        w["video_id"] = (
            source_name
        )

        w["window_id"] = (
            window_id
        )

        w["swim_style"] = (
            style
        )

        w["label"] = (
            "NORMAL"
        )

        w["timestep"] = (
            np.arange(
                WINDOW_SIZE
            )
        )

        w["time_sec"] = (
            np.arange(
                WINDOW_SIZE
            )
            /
            TARGET_FPS
        )

        w = add_temporal_features(
            w
        )

        windows.append(
            w
        )

        window_idx += 1

    return windows


# ============================================================
# PROCESS ZIP
# ============================================================

def process_zip(zip_path):

    style = get_swim_style(
        zip_path.stem
    )

    if style is None:
        return None

    print()
    print("=" * 70)

    print(
        "PROCESS:",
        zip_path.name
    )

    print(
        "STYLE:",
        style
    )

    with tempfile.TemporaryDirectory() as td:

        td = Path(td)

        with zipfile.ZipFile(
            zip_path,
            "r"
        ) as z:

            z.extractall(
                td
            )

        xml_path = find_xml(
            td
        )

        if xml_path is None:

            print(
                "  [FAIL] XML 없음"
            )

            return None

        print(
            "  XML:",
            xml_path
        )

        records, fmt = (
            parse_cvat_xml(
                xml_path
            )
        )

        print(
            "  FORMAT:",
            fmt
        )

        print(
            "  TOTAL FRAMES:",
            len(records)
        )

        if not records:

            print(
                "  [FAIL] records 없음"
            )

            return None

        df = pd.DataFrame(
            records
        )

        df = prepare_frames(
            df
        )

        visible_ratio = float(
            df["detected"].mean()
        )

        missing_ratio = (
            1.0
            -
            visible_ratio
        )

        print(
            "  visible ratio:",
            round(
                visible_ratio,
                4
            )
        )

        print(
            "  missing ratio:",
            round(
                missing_ratio,
                4
            )
        )

        windows = make_windows(
            df,
            source_name=zip_path.stem,
            style=style
        )

        print(
            "  windows:",
            len(windows)
        )

        if not windows:
            return None

        return pd.concat(
            windows,
            ignore_index=True
        )


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    zip_files = sorted(
        ROOT_DIR.glob("*.zip"),
        key=natural_key
    )

    print(
        "TOTAL ZIP:",
        len(zip_files)
    )

    styles = {
        "BACKSTROKE": [],
        "BREASTSTROKE": [],
        "BUTTERFLY": [],
        "FREESTYLE": [],
    }

    source_summary = []

    for zip_path in zip_files:

        style = get_swim_style(
            zip_path.stem
        )

        if style is None:

            print(
                "[SKIP UNKNOWN]",
                zip_path.name
            )

            continue

        result = process_zip(
            zip_path
        )

        if result is None:
            continue

        styles[
            style
        ].append(
            result
        )

        source_summary.append({
            "source": zip_path.stem,
            "style": style,

            "rows":
                len(result),

            "windows":
                result[
                    "window_id"
                ].nunique(),

            "mean_visible_ratio":
                result[
                    "visible_ratio"
                ].mean(),

            "mean_missing_ratio":
                result[
                    "missing_ratio"
                ].mean(),

            "mean_max_missing":
                result[
                    "max_consecutive_missing"
                ].mean(),
        })

    all_results = []

    print()
    print("=" * 80)
    print("STYLE RESULT")
    print("=" * 80)

    style_summary = []

    for style, dfs in styles.items():

        if not dfs:

            print(
                f"{style:15s}: NO DATA"
            )

            continue

        style_df = pd.concat(
            dfs,
            ignore_index=True
        )

        out_path = (
            OUTPUT_DIR
            /
            f"NORMAL_{style}_sequences.csv"
        )

        style_df.to_csv(
            out_path,
            index=False,
            encoding="utf-8-sig"
        )

        windows = (
            style_df[
                "window_id"
            ].nunique()
        )

        sources = (
            style_df[
                "video_id"
            ].nunique()
        )

        rows = len(
            style_df
        )

        avg_visible = float(
            style_df[
                "visible_ratio"
            ].mean()
        )

        avg_missing = float(
            style_df[
                "missing_ratio"
            ].mean()
        )

        print(
            f"{style:15s}: "
            f"{windows:5d} windows / "
            f"{sources:3d} sources / "
            f"visible={avg_visible:.3f} / "
            f"missing={avg_missing:.3f}"
        )

        style_summary.append({
            "swim_style": style,
            "windows": windows,
            "sources": sources,
            "rows": rows,
            "mean_visible_ratio":
                avg_visible,
            "mean_missing_ratio":
                avg_missing,
        })

        all_results.append(
            style_df
        )

    if not all_results:

        raise RuntimeError(
            "추출된 데이터가 없습니다."
        )

    # ========================================================
    # ALL NORMAL
    # ========================================================

    final_df = pd.concat(
        all_results,
        ignore_index=True
    )

    final_path = (
        OUTPUT_DIR
        /
        "NORMAL_4STROKES_ALL_sequences.csv"
    )

    final_df.to_csv(
        final_path,
        index=False,
        encoding="utf-8-sig"
    )

    # ========================================================
    # WINDOW SUMMARY
    # ========================================================

    window_summary = (
        final_df
        .groupby(
            [
                "video_id",
                "window_id",
                "swim_style"
            ],
            as_index=False
        )
        .agg(
            visible_ratio=(
                "visible_ratio",
                "first"
            ),

            missing_ratio=(
                "missing_ratio",
                "first"
            ),

            max_consecutive_missing=(
                "max_consecutive_missing",
                "first"
            ),

            lost_count=(
                "lost_count",
                "first"
            ),

            redetect_count=(
                "redetect_count",
                "first"
            ),
        )
    )

    window_summary[
        "label"
    ] = "NORMAL"

    window_summary.to_csv(
        OUTPUT_DIR
        /
        "NORMAL_4STROKES_windows.csv",
        index=False,
        encoding="utf-8-sig"
    )

    pd.DataFrame(
        style_summary
    ).to_csv(
        OUTPUT_DIR
        /
        "NORMAL_4STROKES_summary.csv",
        index=False,
        encoding="utf-8-sig"
    )

    pd.DataFrame(
        source_summary
    ).to_csv(
        OUTPUT_DIR
        /
        "NORMAL_source_summary.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print()
    print("=" * 80)
    print("FINAL")
    print("=" * 80)

    print(
        "TOTAL WINDOWS:",
        final_df[
            "window_id"
        ].nunique()
    )

    print(
        "TOTAL SOURCES:",
        final_df[
            "video_id"
        ].nunique()
    )

    print(
        "TOTAL ROWS:",
        len(final_df)
    )

    print()

    print(
        "ALL SEQUENCES:",
        final_path
    )

    print()

    print(
        "DONE"
    )


if __name__ == "__main__":
    main()