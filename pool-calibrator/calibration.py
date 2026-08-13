from __future__ import annotations

import cv2
import numpy as np


class CalibrationError(ValueError):
    pass


VALID_ROTATIONS = {0, 90, 180, 270}


def validate_pool_dimensions(length_value, width_value) -> tuple[float, float]:
    try:
        length_m = float(length_value)
        width_m = float(width_value)
    except (TypeError, ValueError):
        raise CalibrationError("수영장 길이와 폭을 숫자로 입력해 주세요.") from None
    if not (2 <= length_m <= 100):
        raise CalibrationError("수영장 길이는 2~100m 범위로 입력해 주세요.")
    if not (2 <= width_m <= 50):
        raise CalibrationError("수영장 폭은 2~50m 범위로 입력해 주세요.")
    return length_m, width_m


def order_points_by_image(array: np.ndarray) -> np.ndarray:
    sums = array[:, 0] + array[:, 1]
    differences = array[:, 1] - array[:, 0]
    indices = [
        int(np.argmin(sums)),
        int(np.argmin(differences)),
        int(np.argmax(sums)),
        int(np.argmax(differences)),
    ]
    if len(set(indices)) == 4:
        return array[indices]

    by_y = array[np.argsort(array[:, 1])]
    top = by_y[:2][np.argsort(by_y[:2, 0])]
    bottom = by_y[2:][np.argsort(by_y[2:, 0])]
    return np.array([top[0], top[1], bottom[1], bottom[0]], dtype=np.float32)


def validate_points(points, image_width: int, image_height: int) -> np.ndarray:
    if not isinstance(points, list) or len(points) != 4:
        raise CalibrationError("수영장 모서리 4점을 모두 지정해 주세요.")
    try:
        array = np.array([[float(point["x"]), float(point["y"])] for point in points], dtype=np.float32)
    except (KeyError, TypeError, ValueError):
        raise CalibrationError("모서리 좌표가 올바르지 않습니다.") from None
    if not np.isfinite(array).all():
        raise CalibrationError("모서리 좌표가 올바르지 않습니다.")
    if (
        (array[:, 0] < 0).any()
        or (array[:, 0] >= image_width).any()
        or (array[:, 1] < 0).any()
        or (array[:, 1] >= image_height).any()
    ):
        raise CalibrationError("모서리 점이 이미지 밖에 있습니다.")

    array = order_points_by_image(array)

    contour = array.reshape((-1, 1, 2))
    area = abs(cv2.contourArea(contour))
    if area < image_width * image_height * 0.01:
        raise CalibrationError("선택 영역이 너무 작습니다. 수영장 모서리를 다시 지정해 주세요.")
    if not cv2.isContourConvex(contour.astype(np.int32)):
        raise CalibrationError("네 모서리 위치를 구분할 수 없습니다. 점 위치를 확인해 주세요.")

    for index in range(4):
        if np.linalg.norm(array[index] - array[(index + 1) % 4]) < 10:
            raise CalibrationError("모서리 점 사이가 너무 가깝습니다.")
    return array


def validate_rotation(rotation_value) -> int:
    try:
        rotation_degrees = int(rotation_value or 0)
    except (TypeError, ValueError):
        raise CalibrationError("회전값은 0°, 90°, 180° 또는 270°여야 합니다.") from None
    if rotation_degrees not in VALID_ROTATIONS:
        raise CalibrationError("회전값은 0°, 90°, 180° 또는 270°여야 합니다.")
    return rotation_degrees


def orientation_transform(rotation_degrees: int, length_m: float, width_m: float):
    if rotation_degrees == 90:
        return np.array([[0, -1, width_m], [1, 0, 0], [0, 0, 1]], dtype=np.float64), width_m, length_m
    if rotation_degrees == 180:
        return np.array([[-1, 0, length_m], [0, -1, width_m], [0, 0, 1]], dtype=np.float64), length_m, width_m
    if rotation_degrees == 270:
        return np.array([[0, 1, 0], [-1, 0, length_m], [0, 0, 1]], dtype=np.float64), width_m, length_m
    return np.eye(3, dtype=np.float64), length_m, width_m


def rotate_preview(preview: np.ndarray, homography: np.ndarray, rotation_degrees: int):
    height, width = preview.shape[:2]
    if rotation_degrees == 90:
        transform = np.array([[0, -1, height - 1], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
        return cv2.rotate(preview, cv2.ROTATE_90_CLOCKWISE), transform @ homography
    if rotation_degrees == 180:
        transform = np.array([[-1, 0, width - 1], [0, -1, height - 1], [0, 0, 1]], dtype=np.float64)
        return cv2.rotate(preview, cv2.ROTATE_180), transform @ homography
    if rotation_degrees == 270:
        transform = np.array([[0, 1, 0], [-1, 0, width - 1], [0, 0, 1]], dtype=np.float64)
        return cv2.rotate(preview, cv2.ROTATE_90_COUNTERCLOCKWISE), transform @ homography
    return preview, homography


def flip_preview_horizontal(preview: np.ndarray, homography: np.ndarray, enabled: bool):
    if not enabled:
        return preview, homography
    width = preview.shape[1]
    transform = np.array([[-1, 0, width - 1], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    return cv2.flip(preview, 1), transform @ homography


def build_alignment_preview(
    previews: list[np.ndarray], oriented_width_m: float, oriented_height_m: float
):
    if len(previews) != 2 or any(preview is None or preview.size == 0 for preview in previews):
        raise CalibrationError("두 카메라 미리보기를 모두 생성해 주세요.")
    if previews[0].shape != previews[1].shape:
        raise CalibrationError("두 미리보기의 방향이 다릅니다. 회전 버튼으로 방향을 맞춰 주세요.")

    combined = cv2.addWeighted(previews[0], 0.5, previews[1], 0.5, 0)
    overlay = combined.copy()
    height, width = combined.shape[:2]
    longest_side = max(oriented_width_m, oriented_height_m)
    grid_step_m = 1 if longest_side <= 15 else 5 if longest_side <= 40 else 10

    def label(text: str, x: int, y: int):
        cv2.putText(overlay, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (7, 27, 38), 3, cv2.LINE_AA)
        cv2.putText(overlay, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 255, 255), 1, cv2.LINE_AA)

    meter = 0.0
    while meter <= oriented_width_m + 1e-6:
        x = min(width - 1, round(meter / oriented_width_m * (width - 1)))
        cv2.line(overlay, (x, 0), (x, height - 1), (236, 226, 73), 1, cv2.LINE_AA)
        label(f"{meter:g}", max(3, x + 4), 18)
        meter += grid_step_m

    meter = 0.0
    while meter <= oriented_height_m + 1e-6:
        y = min(height - 1, round(meter / oriented_height_m * (height - 1)))
        cv2.line(overlay, (0, y), (width - 1, y), (236, 226, 73), 1, cv2.LINE_AA)
        if y > 12:
            label(f"{meter:g}", 5, y - 4)
        meter += grid_step_m

    cv2.rectangle(overlay, (0, 0), (width - 1, height - 1), (66, 214, 197), 3)
    cv2.arrowedLine(overlay, (18, height - 20), (90, height - 20), (66, 214, 197), 3, tipLength=0.16)
    cv2.arrowedLine(overlay, (18, height - 20), (18, max(18, height - 92)), (66, 214, 197), 3, tipLength=0.16)
    label("X (m)", 96, height - 14)
    label("Y (m)", 25, max(20, height - 94))
    return overlay, grid_step_m


def _robust_pool_statistics(preview: np.ndarray) -> dict:
    if preview is None or preview.size == 0:
        raise CalibrationError("색상과 노출을 계산할 미리보기가 없습니다.")

    rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB).reshape(-1, 3).astype(np.float64)
    luma = rgb @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)
    low, high = np.percentile(luma, [10, 90])
    valid = (
        (luma >= low)
        & (luma <= high)
        & (rgb.min(axis=1) > 4)
        & (rgb.max(axis=1) < 251)
    )
    if int(valid.sum()) < max(100, round(len(rgb) * 0.02)):
        valid = luma > 1
    if not valid.any():
        raise CalibrationError("이미지가 너무 어둡거나 밝아 WB와 노출을 계산할 수 없습니다.")

    median_rgb = np.median(rgb[valid], axis=0)
    median_luma = float(np.median(luma[valid]))
    if median_luma < 1 or median_rgb[1] < 1:
        raise CalibrationError("이미지가 너무 어두워 WB와 노출을 계산할 수 없습니다.")
    return {
        "median_rgb": median_rgb,
        "median_luma": median_luma,
        "sample_count": int(valid.sum()),
    }


def calculate_color_calibration(previews: list[np.ndarray]) -> dict:
    """Calculate relative colour/exposure matching for two warped pool regions.

    These values match both cameras to their geometric-mean appearance. They are
    correction factors, not absolute sensor controls, because uploaded images do
    not include the live camera's AWB gains, exposure time, or analogue gain.
    """
    if len(previews) != 2 or any(preview is None or preview.size == 0 for preview in previews):
        raise CalibrationError("두 카메라 미리보기가 있어야 WB와 노출을 계산할 수 있습니다.")

    statistics = [_robust_pool_statistics(preview) for preview in previews]
    red_green = [item["median_rgb"][0] / item["median_rgb"][1] for item in statistics]
    blue_green = [item["median_rgb"][2] / item["median_rgb"][1] for item in statistics]
    target_red_green = float(np.sqrt(red_green[0] * red_green[1]))
    target_blue_green = float(np.sqrt(blue_green[0] * blue_green[1]))

    corrections = []
    balanced_luma = []
    for item, rg_ratio, bg_ratio in zip(statistics, red_green, blue_green, strict=True):
        red_gain = float(np.clip(target_red_green / max(rg_ratio, 1e-6), 0.5, 2.0))
        blue_gain = float(np.clip(target_blue_green / max(bg_ratio, 1e-6), 0.5, 2.0))
        red, green, blue = item["median_rgb"]
        corrected_luma = 0.2126 * red * red_gain + 0.7152 * green + 0.0722 * blue * blue_gain
        balanced_luma.append(float(max(corrected_luma, 1e-6)))
        corrections.append((red_gain, blue_gain))

    target_luma = float(np.sqrt(balanced_luma[0] * balanced_luma[1]))
    cameras = {}
    for index, (item, gains, camera_luma) in enumerate(
        zip(statistics, corrections, balanced_luma, strict=True), start=1
    ):
        raw_ev = float(np.log2(target_luma / camera_luma))
        exposure_ev = float(np.clip(raw_ev, -2.0, 2.0))
        cameras[f"camera{index}"] = {
            "white_balance": {
                "red_gain_multiplier": round(gains[0], 4),
                "green_gain_multiplier": 1.0,
                "blue_gain_multiplier": round(gains[1], 4),
            },
            "exposure": {
                "compensation_ev": round(exposure_ev, 4),
                "brightness_multiplier": round(float(2**exposure_ev), 4),
            },
            "measurement": {
                "median_rgb_0_255": {
                    "red": round(float(item["median_rgb"][0]), 2),
                    "green": round(float(item["median_rgb"][1]), 2),
                    "blue": round(float(item["median_rgb"][2]), 2),
                },
                "median_luma_0_255": round(float(item["median_luma"]), 2),
                "sample_count": item["sample_count"],
            },
        }

    return {
        "method": "relative_robust_pool_roi_v1",
        "scope": "homography_warped_pool_region",
        "reference": "geometric_mean_of_camera1_and_camera2",
        "cameras": cameras,
        "picamera2_mapping": {
            "white_balance": "multiply captured ColourGains red/blue values by these multipliers, then lock AWB",
            "exposure": "use compensation_ev as ExposureValue while auto exposure remains enabled",
        },
    }


def build_calibration(
    image: np.ndarray,
    points,
    length_m: float,
    width_m: float,
    rotation_degrees: int = 0,
    flip_horizontal: bool = False,
):
    if image is None or image.size == 0:
        raise CalibrationError("이미지를 읽을 수 없습니다.")
    image_height, image_width = image.shape[:2]
    source = validate_points(points, image_width, image_height)
    rotation_degrees = validate_rotation(rotation_degrees)
    if not isinstance(flip_horizontal, bool):
        raise CalibrationError("좌우 반전값이 올바르지 않습니다.")

    meters = np.array(
        [[0.0, 0.0], [length_m, 0.0], [length_m, width_m], [0.0, width_m]],
        dtype=np.float32,
    )
    meter_homography = cv2.getPerspectiveTransform(source, meters)
    if not np.isfinite(meter_homography).all() or abs(np.linalg.det(meter_homography)) < 1e-10:
        raise CalibrationError("호모그래피를 계산할 수 없습니다. 점 위치를 확인해 주세요.")

    pixels_per_meter = min(60.0, 1400.0 / length_m, 1000.0 / width_m)
    preview_width = max(320, round(length_m * pixels_per_meter))
    preview_height = max(180, round(width_m * pixels_per_meter))
    destination = np.array(
        [[0, 0], [preview_width - 1, 0], [preview_width - 1, preview_height - 1], [0, preview_height - 1]],
        dtype=np.float32,
    )
    preview_homography = cv2.getPerspectiveTransform(source, destination)
    preview = cv2.warpPerspective(image, preview_homography, (preview_width, preview_height))
    preview, preview_homography = rotate_preview(preview, preview_homography, rotation_degrees)
    preview, preview_homography = flip_preview_horizontal(
        preview, preview_homography, flip_horizontal
    )
    preview_height, preview_width = preview.shape[:2]

    orientation, oriented_width_m, oriented_height_m = orientation_transform(
        rotation_degrees, length_m, width_m
    )
    oriented_meter_homography = orientation @ meter_homography
    if flip_horizontal:
        horizontal_flip = np.array(
            [[-1, 0, oriented_width_m], [0, 1, 0], [0, 0, 1]], dtype=np.float64
        )
        oriented_meter_homography = horizontal_flip @ oriented_meter_homography

    calibration = {
        "image_size": {"width": image_width, "height": image_height},
        "source_points_px": [
            {"x": round(float(x), 3), "y": round(float(y), 3)} for x, y in source
        ],
        "rotation_degrees_clockwise": rotation_degrees,
        "flip_horizontal": flip_horizontal,
        "oriented_coordinate_size_m": {
            "width": oriented_width_m,
            "height": oriented_height_m,
        },
        "homography_pixel_to_meter": np.round(oriented_meter_homography, 10).tolist(),
        "homography_pixel_to_meter_unrotated": np.round(meter_homography, 10).tolist(),
        "preview": {
            "width": preview_width,
            "height": preview_height,
            "pixels_per_meter": round(float(pixels_per_meter), 4),
            "homography_pixel_to_preview": np.round(preview_homography, 10).tolist(),
        },
    }
    return calibration, preview
