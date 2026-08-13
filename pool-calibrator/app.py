from __future__ import annotations

import json
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request, send_file
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

from camera_capture import CameraCaptureError, capture_all_cameras

from calibration import (
    CalibrationError,
    build_alignment_preview,
    build_calibration,
    calculate_color_calibration,
    validate_pool_dimensions,
)


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("POOL_CALIBRATOR_DATA_DIR", BASE_DIR / "data"))
UPLOAD_DIR = DATA_DIR / "uploads"
PREVIEW_DIR = DATA_DIR / "previews"
CONFIG_PATH = DATA_DIR / "config.json"
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
CAPTURE_LOCK = threading.Lock()


def create_app(test_config: dict | None = None) -> Flask:
    app = Flask(__name__)
    app.config.update(
        MAX_CONTENT_LENGTH=16 * 1024 * 1024,
        DATA_DIR=DATA_DIR,
        UPLOAD_DIR=UPLOAD_DIR,
        PREVIEW_DIR=PREVIEW_DIR,
        CONFIG_PATH=CONFIG_PATH,
        SITE_NAME=os.environ.get("POOLSIGHT_SITE_NAME", "PoolSight"),
        CAMERA_CAPTURE_FUNCTION=capture_all_cameras,
    )
    if test_config:
        app.config.update(test_config)

    for key in ("DATA_DIR", "UPLOAD_DIR", "PREVIEW_DIR"):
        Path(app.config[key]).mkdir(parents=True, exist_ok=True)

    @app.get("/")
    def index():
        return render_template("index.html", site_name=configured_site_name(app))

    @app.get("/health")
    def health():
        return jsonify(status="ok")

    @app.get("/api/cameras/latest")
    def latest_camera_images():
        try:
            return jsonify(capture_manifest_response(app, load_capture_manifest(app)))
        except CameraCaptureError as exc:
            return api_error(str(exc), 404)

    @app.post("/api/cameras/capture")
    def capture_camera_images():
        if not CAPTURE_LOCK.acquire(blocking=False):
            return api_error("다른 카메라 촬영이 진행 중입니다. 잠시 후 다시 시도해 주세요.", 409)
        try:
            manifest = app.config["CAMERA_CAPTURE_FUNCTION"](Path(app.config["DATA_DIR"]))
            return jsonify(capture_manifest_response(app, manifest))
        except CameraCaptureError as exc:
            return api_error(str(exc), 503)
        finally:
            CAPTURE_LOCK.release()

    @app.post("/api/images")
    def upload_image():
        camera_id = request.form.get("camera_id", "")
        image = request.files.get("image")
        if camera_id not in {"camera1", "camera2"}:
            return api_error("카메라 번호가 올바르지 않습니다.", 400)
        if not image or not image.filename:
            return api_error("이미지 파일을 선택해 주세요.", 400)

        suffix = Path(secure_filename(image.filename)).suffix.lower()
        if suffix not in ALLOWED_EXTENSIONS:
            return api_error("JPG, PNG 또는 WEBP 이미지만 사용할 수 있습니다.", 400)

        raw = image.read()
        array = np.frombuffer(raw, dtype=np.uint8)
        decoded = cv2.imdecode(array, cv2.IMREAD_COLOR)
        if decoded is None:
            return api_error("이미지를 읽을 수 없습니다.", 400)
        height, width = decoded.shape[:2]
        if width < 320 or height < 240:
            return api_error("이미지 해상도는 최소 320×240이어야 합니다.", 400)

        image_id = uuid.uuid4().hex
        filename = f"{camera_id}-{image_id}.jpg"
        path = Path(app.config["UPLOAD_DIR"]) / filename
        if not cv2.imwrite(str(path), decoded, [cv2.IMWRITE_JPEG_QUALITY, 92]):
            return api_error("이미지를 저장하지 못했습니다.", 500)

        return jsonify(
            image_id=image_id,
            camera_id=camera_id,
            width=width,
            height=height,
            image_url=f"/api/images/{image_id}?camera_id={camera_id}",
        )

    @app.get("/api/images/<image_id>")
    def get_image(image_id: str):
        camera_id = request.args.get("camera_id", "")
        path = image_path(app, camera_id, image_id)
        if path is None or not path.exists():
            return api_error("이미지를 찾을 수 없습니다.", 404)
        return send_file(path, mimetype="image/jpeg")

    @app.post("/api/preview")
    def create_preview():
        payload = request.get_json(silent=True) or {}
        try:
            camera_id, image_id, points, length_m, width_m, rotation_degrees, flip_horizontal = parse_calibration_payload(payload)
            source = image_path(app, camera_id, image_id)
            if source is None or not source.exists():
                raise CalibrationError("업로드한 이미지를 다시 선택해 주세요.")
            image = cv2.imread(str(source))
            calibration, preview = build_calibration(
                image, points, length_m, width_m, rotation_degrees, flip_horizontal
            )
        except CalibrationError as exc:
            return api_error(str(exc), 400)

        preview_id = uuid.uuid4().hex
        preview_path = Path(app.config["PREVIEW_DIR"]) / f"{preview_id}.jpg"
        cv2.imwrite(str(preview_path), preview, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return jsonify(
            calibration=calibration,
            preview_url=f"/api/previews/{preview_id}",
        )

    @app.get("/api/previews/<preview_id>")
    def get_preview(preview_id: str):
        if not is_hex_id(preview_id):
            return api_error("미리보기를 찾을 수 없습니다.", 404)
        path = Path(app.config["PREVIEW_DIR"]) / f"{preview_id}.jpg"
        if not path.exists():
            return api_error("미리보기를 찾을 수 없습니다.", 404)
        return send_file(path, mimetype="image/jpeg")

    @app.post("/api/alignment-preview")
    def create_alignment_preview():
        payload = request.get_json(silent=True) or {}
        cameras = payload.get("cameras")
        if not isinstance(cameras, list) or len(cameras) != 2:
            return api_error("카메라 1과 카메라 2 보정을 모두 완료해 주세요.", 400)

        try:
            length_m, width_m = validate_pool_dimensions(
                payload.get("pool", {}).get("length_m"),
                payload.get("pool", {}).get("width_m"),
            )
            previews = []
            calibrations = []
            for expected_id, item in zip(("camera1", "camera2"), cameras, strict=True):
                camera_id, image_id, points, _, _, rotation_degrees, flip_horizontal = parse_calibration_payload(
                    {**item, "length_m": length_m, "width_m": width_m}
                )
                if camera_id != expected_id:
                    raise CalibrationError("카메라 보정 순서가 올바르지 않습니다.")
                source = image_path(app, camera_id, image_id)
                if source is None or not source.exists():
                    raise CalibrationError(f"{camera_label(camera_id)} 이미지를 다시 선택해 주세요.")
                calibration, preview = build_calibration(
                    cv2.imread(str(source)), points, length_m, width_m, rotation_degrees, flip_horizontal
                )
                calibrations.append(calibration)
                previews.append(preview)

            sizes = [calibration["oriented_coordinate_size_m"] for calibration in calibrations]
            if sizes[0] != sizes[1]:
                raise CalibrationError(
                    "두 미리보기의 가로·세로 방향이 다릅니다. 한쪽을 90° 돌려 맞춰 주세요."
                )
            combined, grid_step_m = build_alignment_preview(
                previews, sizes[0]["width"], sizes[0]["height"]
            )
            color_calibration = calculate_color_calibration(previews)
        except (CalibrationError, TypeError, AttributeError) as exc:
            return api_error(str(exc) or "입력값을 확인해 주세요.", 400)

        preview_id = uuid.uuid4().hex
        preview_path = Path(app.config["PREVIEW_DIR"]) / f"alignment-{preview_id}.jpg"
        cv2.imwrite(str(preview_path), combined, [cv2.IMWRITE_JPEG_QUALITY, 92])
        return jsonify(
            preview_url=f"/api/alignment-previews/{preview_id}",
            coordinate_size_m=sizes[0],
            grid_step_m=grid_step_m,
            color_calibration=color_calibration,
        )

    @app.get("/api/alignment-previews/<preview_id>")
    def get_alignment_preview(preview_id: str):
        if not is_hex_id(preview_id):
            return api_error("통합 미리보기를 찾을 수 없습니다.", 404)
        path = Path(app.config["PREVIEW_DIR"]) / f"alignment-{preview_id}.jpg"
        if not path.exists():
            return api_error("통합 미리보기를 찾을 수 없습니다.", 404)
        return send_file(path, mimetype="image/jpeg")

    @app.post("/api/config")
    def save_config():
        payload = request.get_json(silent=True) or {}
        cameras = payload.get("cameras")
        if not isinstance(cameras, list) or len(cameras) != 2:
            return api_error("카메라 1과 카메라 2 보정을 모두 완료해 주세요.", 400)

        try:
            length_m, width_m = validate_pool_dimensions(
                payload.get("pool", {}).get("length_m"),
                payload.get("pool", {}).get("width_m"),
            )
            saved_cameras = []
            previews = []
            for expected_id, item in zip(("camera1", "camera2"), cameras, strict=True):
                camera_id, image_id, points, item_length, item_width, rotation_degrees, flip_horizontal = parse_calibration_payload(
                    {**item, "length_m": length_m, "width_m": width_m}
                )
                if camera_id != expected_id:
                    raise CalibrationError("카메라 보정 순서가 올바르지 않습니다.")
                if item_length != length_m or item_width != width_m:
                    raise CalibrationError("수영장 크기가 카메라별로 일치하지 않습니다.")
                source = image_path(app, camera_id, image_id)
                if source is None or not source.exists():
                    raise CalibrationError(f"{camera_label(camera_id)} 이미지를 다시 선택해 주세요.")
                image = cv2.imread(str(source))
                calibration, preview = build_calibration(
                    image, points, length_m, width_m, rotation_degrees, flip_horizontal
                )
                previews.append(preview)
                saved_cameras.append(
                    {
                        "id": camera_id,
                        "label": camera_label(camera_id),
                        "image_file": str(source.relative_to(Path(app.config["DATA_DIR"]))).replace("\\", "/"),
                        **calibration,
                    }
                )
            coordinate_sizes = {
                (
                    camera["oriented_coordinate_size_m"]["width"],
                    camera["oriented_coordinate_size_m"]["height"],
                )
                for camera in saved_cameras
            }
            if len(coordinate_sizes) != 1:
                raise CalibrationError(
                    "두 미리보기의 가로·세로 방향이 다릅니다. 한쪽을 90° 돌려 맞춰 주세요."
                )
            color_calibration = calculate_color_calibration(previews)
            for camera in saved_cameras:
                camera["image_adjustments"] = color_calibration["cameras"][camera["id"]]
        except (CalibrationError, TypeError, AttributeError) as exc:
            return api_error(str(exc) or "입력값을 확인해 주세요.", 400)

        config = {
            "schema_version": 4,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "coordinate_system": {
                "origin": "top_left_after_preview_rotation",
                "x_axis": "preview_right",
                "y_axis": "preview_down",
                "unit": "meter",
            },
            "point_order": ["top_left", "top_right", "bottom_right", "bottom_left"],
            "pool": {"length_m": length_m, "width_m": width_m},
            "color_calibration": {
                key: value for key, value in color_calibration.items() if key != "cameras"
            },
            "cameras": saved_cameras,
        }
        config_path = Path(app.config["CONFIG_PATH"])
        temp_path = config_path.with_suffix(".json.tmp")
        temp_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
        temp_path.replace(config_path)
        return jsonify(message="설정이 저장되었습니다.", config=config, download_url="/api/config")

    @app.get("/api/config")
    def download_config():
        path = Path(app.config["CONFIG_PATH"])
        if not path.exists():
            return api_error("저장된 설정이 없습니다.", 404)
        return send_file(path, mimetype="application/json", as_attachment=True, download_name="config.json")

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(_error):
        return api_error("이미지 크기는 16MB 이하여야 합니다.", 413)

    return app


def api_error(message: str, status: int):
    return jsonify(error=message), status


def is_hex_id(value: str) -> bool:
    return len(value) == 32 and all(char in "0123456789abcdef" for char in value)


def configured_site_name(app: Flask) -> str:
    settings_path = Path(app.config["DATA_DIR"]) / "site-settings.json"
    if settings_path.exists():
        try:
            value = json.loads(settings_path.read_text(encoding="utf-8")).get("site_name")
            if isinstance(value, str) and 1 <= len(value.strip()) <= 60:
                return value.strip()
        except (OSError, json.JSONDecodeError, AttributeError):
            pass
    return str(app.config.get("SITE_NAME") or "PoolSight")


def load_capture_manifest(app: Flask) -> dict:
    path = Path(app.config["DATA_DIR"]) / "latest-captures.json"
    if not path.exists():
        raise CameraCaptureError("아직 자동 촬영된 이미지가 없습니다.")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CameraCaptureError("자동 촬영 정보를 읽지 못했습니다. 다시 촬영해 주세요.") from exc
    return manifest


def capture_manifest_response(app: Flask, manifest: dict) -> dict:
    items = manifest.get("cameras") if isinstance(manifest, dict) else None
    if not isinstance(items, list) or len(items) != 2:
        raise CameraCaptureError("카메라 1·2 자동 촬영 정보가 올바르지 않습니다.")
    cameras = []
    for expected_id, item in zip(("camera1", "camera2"), items, strict=True):
        camera_id = item.get("camera_id") if isinstance(item, dict) else None
        image_id = item.get("image_id") if isinstance(item, dict) else None
        path = image_path(app, str(camera_id), str(image_id))
        if camera_id != expected_id or path is None or not path.exists():
            raise CameraCaptureError(f"{camera_label(expected_id)} 자동 촬영 이미지를 찾지 못했습니다.")
        cameras.append(
            {
                "camera_id": camera_id,
                "camera_index": item.get("camera_index"),
                "image_id": image_id,
                "width": item.get("width"),
                "height": item.get("height"),
                "image_url": f"/api/images/{image_id}?camera_id={camera_id}",
            }
        )
    return {"captured_at": manifest.get("captured_at"), "cameras": cameras}


def image_path(app: Flask, camera_id: str, image_id: str) -> Path | None:
    if camera_id not in {"camera1", "camera2"} or not is_hex_id(image_id):
        return None
    return Path(app.config["UPLOAD_DIR"]) / f"{camera_id}-{image_id}.jpg"


def parse_calibration_payload(payload: dict):
    camera_id = payload.get("camera_id", "")
    image_id = payload.get("image_id", "")
    points = payload.get("points")
    if camera_id not in {"camera1", "camera2"}:
        raise CalibrationError("카메라 번호가 올바르지 않습니다.")
    if not is_hex_id(str(image_id)):
        raise CalibrationError("업로드한 이미지를 다시 선택해 주세요.")
    length_m, width_m = validate_pool_dimensions(payload.get("length_m"), payload.get("width_m"))
    try:
        rotation_degrees = int(payload.get("rotation_degrees", 0))
    except (TypeError, ValueError):
        raise CalibrationError("회전값은 0°, 90°, 180° 또는 270°여야 합니다.") from None
    if rotation_degrees not in {0, 90, 180, 270}:
        raise CalibrationError("회전값은 0°, 90°, 180° 또는 270°여야 합니다.")
    flip_horizontal = payload.get("flip_horizontal", False)
    if not isinstance(flip_horizontal, bool):
        raise CalibrationError("좌우 반전값이 올바르지 않습니다.")
    return camera_id, image_id, points, length_m, width_m, rotation_degrees, flip_horizontal


def camera_label(camera_id: str) -> str:
    return "카메라 1" if camera_id == "camera1" else "카메라 2"


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")), debug=False)
