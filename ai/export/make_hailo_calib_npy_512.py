from pathlib import Path
import random

import cv2
import numpy as np


CALIB_DIR = Path(
    r"D:\임베디드 경진대회\머리 추적\hailo_calib_head"
)

OUTPUT_NPY = Path(
    r"D:\임베디드 경진대회\머리 추적\models\pool_head_calib_512.npy"
)

IMG_SIZE = 512
CALIB_COUNT = 256
SEED = 42

EXTS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


def letterbox(image, size=512):
    h, w = image.shape[:2]

    scale = min(
        size / w,
        size / h
    )

    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    resized = cv2.resize(
        image,
        (new_w, new_h),
        interpolation=cv2.INTER_LINEAR
    )

    canvas = np.full(
        (size, size, 3),
        114,
        dtype=np.uint8
    )

    x = (size - new_w) // 2
    y = (size - new_h) // 2

    canvas[
        y:y + new_h,
        x:x + new_w
    ] = resized

    return canvas


def main():
    random.seed(SEED)

    files = [
        p
        for p in CALIB_DIR.iterdir()
        if p.is_file()
        and p.suffix.lower() in EXTS
    ]

    print("Calibration images found:", len(files))

    if len(files) == 0:
        raise RuntimeError(
            f"Calibration image가 없습니다: {CALIB_DIR}"
        )

    random.shuffle(files)

    files = files[
        :min(CALIB_COUNT, len(files))
    ]

    print("Calibration images selected:", len(files))

    data = np.empty(
        (
            len(files),
            IMG_SIZE,
            IMG_SIZE,
            3
        ),
        dtype=np.float32
    )

    for i, path in enumerate(files):
        image = cv2.imread(
            str(path),
            cv2.IMREAD_COLOR
        )

        if image is None:
            raise RuntimeError(
                f"이미지를 읽지 못했습니다: {path}"
            )

        # 512 x 512 letterbox
        image = letterbox(
            image,
            IMG_SIZE
        )

        # OpenCV BGR -> RGB
        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        # YOLO 입력 형태와 동일하게 0~1 정규화
        image = (
            image.astype(np.float32)
            / 255.0
        )

        data[i] = image

        if (
            (i + 1) % 32 == 0
            or
            (i + 1) == len(files)
        ):
            print(
                f"{i + 1}/{len(files)}"
            )

    OUTPUT_NPY.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    np.save(
        OUTPUT_NPY,
        data
    )

    print()
    print("========== DONE ==========")
    print("shape :", data.shape)
    print("dtype :", data.dtype)
    print("min   :", data.min())
    print("max   :", data.max())
    print("output:", OUTPUT_NPY)


if __name__ == "__main__":
    main()
