from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path

import cv2


class CameraCaptureError(RuntimeError):
    pass


def find_capture_command() -> str:
    command = shutil.which("rpicam-still") or shutil.which("libcamera-still")
    if not command:
        raise CameraCaptureError(
            "카메라 촬영 프로그램을 찾지 못했습니다. Raspberry Pi에서 rpicam-apps를 설치해 주세요."
        )
    return command


def detected_camera_indices(command: str) -> list[int]:
    try:
        result = subprocess.run(
            [command, "--list-cameras"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise CameraCaptureError("연결된 카메라 목록을 확인하지 못했습니다.") from exc

    output = f"{result.stdout}\n{result.stderr}"
    indices = [int(match.group(1)) for match in re.finditer(r"(?m)^\s*(\d+)\s*:", output)]
    if len(indices) < 2:
        raise CameraCaptureError(
            f"카메라가 {len(indices)}대만 감지되었습니다. CSI 케이블과 카메라 1·2 연결을 확인해 주세요."
        )
    return indices


def capture_all_cameras(data_dir: Path, timeout_ms: int = 2500) -> dict:
    data_dir = Path(data_dir)
    upload_dir = data_dir / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    command = find_capture_command()
    indices = detected_camera_indices(command)[:2]
    captured = []

    for position, camera_index in enumerate(indices, start=1):
        camera_id = f"camera{position}"
        image_id = uuid.uuid4().hex
        temporary = upload_dir / f".{camera_id}-{image_id}.jpg.tmp"
        final = upload_dir / f"{camera_id}-{image_id}.jpg"
        capture_command = [
            command,
            "--camera",
            str(camera_index),
            "--width",
            "1920",
            "--height",
            "1080",
            "--timeout",
            str(timeout_ms),
            "--nopreview",
            "--encoding",
            "jpg",
            "--quality",
            "92",
            "--output",
            str(temporary),
        ]
        try:
            result = subprocess.run(
                capture_command,
                capture_output=True,
                text=True,
                timeout=max(15, timeout_ms / 1000 + 12),
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            temporary.unlink(missing_ok=True)
            raise CameraCaptureError(f"{camera_id} 촬영 시간이 초과되었습니다.") from exc
        if result.returncode != 0 or not temporary.exists():
            temporary.unlink(missing_ok=True)
            detail = (result.stderr or result.stdout).strip().splitlines()
            message = detail[-1] if detail else "알 수 없는 카메라 오류"
            raise CameraCaptureError(f"{camera_id} 촬영 실패: {message}")

        image = cv2.imread(str(temporary))
        if image is None or image.size == 0:
            temporary.unlink(missing_ok=True)
            raise CameraCaptureError(f"{camera_id}에서 촬영된 이미지를 읽을 수 없습니다.")
        height, width = image.shape[:2]
        os.replace(temporary, final)
        captured.append(
            {
                "camera_id": camera_id,
                "camera_index": camera_index,
                "image_id": image_id,
                "width": width,
                "height": height,
                "image_file": str(final.relative_to(data_dir)).replace("\\", "/"),
            }
        )

    manifest = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "cameras": captured,
    }
    manifest_path = data_dir / "latest-captures.json"
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    temporary_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary_manifest, manifest_path)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture PoolSight camera 1 and 2")
    parser.add_argument(
        "--data-dir",
        default=os.environ.get("POOL_CALIBRATOR_DATA_DIR", "data"),
        type=Path,
    )
    args = parser.parse_args()
    try:
        manifest = capture_all_cameras(args.data_dir)
    except CameraCaptureError as exc:
        print(str(exc))
        return 1
    print(json.dumps(manifest, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
