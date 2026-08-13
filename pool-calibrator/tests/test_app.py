import io
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from app import create_app
from calibration import calculate_color_calibration


class CalibrationApiTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.app = create_app(
            {
                "TESTING": True,
                "DATA_DIR": root,
                "UPLOAD_DIR": root / "uploads",
                "PREVIEW_DIR": root / "previews",
                "CONFIG_PATH": root / "config.json",
            }
        )
        self.client = self.app.test_client()

    def tearDown(self):
        self.temp.cleanup()

    def make_image(self):
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        image[:] = (150, 95, 30)
        cv2.rectangle(image, (80, 60), (560, 420), (230, 200, 100), 6)
        ok, encoded = cv2.imencode(".jpg", image)
        self.assertTrue(ok)
        return encoded.tobytes()

    def upload(self, camera_id):
        response = self.client.post(
            "/api/images",
            data={"camera_id": camera_id, "image": (io.BytesIO(self.make_image()), "pool.jpg")},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 200)
        return response.get_json()["image_id"]

    def payload(self, camera_id, image_id):
        return {
            "camera_id": camera_id,
            "image_id": image_id,
            "length_m": 25,
            "width_m": 12.5,
            "points": [
                {"x": 80, "y": 60},
                {"x": 560, "y": 60},
                {"x": 560, "y": 420},
                {"x": 80, "y": 420},
            ],
        }

    def test_health_and_home(self):
        self.assertEqual(self.client.get("/health").get_json(), {"status": "ok"})
        self.assertIn("PoolSight".encode(), self.client.get("/").data)

    def test_custom_site_name(self):
        self.app.config["SITE_NAME"] = "학교 수영장"
        self.assertIn("학교 수영장".encode(), self.client.get("/").data)

    def test_loads_latest_boot_camera_images(self):
        camera1_id = self.upload("camera1")
        camera2_id = self.upload("camera2")
        manifest = {
            "captured_at": "2026-08-03T00:00:00+00:00",
            "cameras": [
                {"camera_id": "camera1", "camera_index": 0, "image_id": camera1_id, "width": 640, "height": 480},
                {"camera_id": "camera2", "camera_index": 1, "image_id": camera2_id, "width": 640, "height": 480},
            ],
        }
        (Path(self.app.config["DATA_DIR"]) / "latest-captures.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )

        response = self.client.get("/api/cameras/latest")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual([item["camera_id"] for item in data["cameras"]], ["camera1", "camera2"])
        self.assertIn("image_url", data["cameras"][0])

    def test_manual_camera_capture_endpoint(self):
        camera1_id = self.upload("camera1")
        camera2_id = self.upload("camera2")

        def fake_capture(_data_dir):
            return {
                "captured_at": "2026-08-03T00:00:00+00:00",
                "cameras": [
                    {"camera_id": "camera1", "camera_index": 0, "image_id": camera1_id, "width": 640, "height": 480},
                    {"camera_id": "camera2", "camera_index": 1, "image_id": camera2_id, "width": 640, "height": 480},
                ],
            }

        self.app.config["CAMERA_CAPTURE_FUNCTION"] = fake_capture
        response = self.client.post("/api/cameras/capture")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.get_json()["cameras"]), 2)

    def test_preview_calculates_homography(self):
        image_id = self.upload("camera1")
        response = self.client.post("/api/preview", json=self.payload("camera1", image_id))
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        matrix = data["calibration"]["homography_pixel_to_meter"]
        self.assertEqual(len(matrix), 3)
        self.assertIn("preview_url", data)

    def test_auto_orders_points_regardless_of_click_order(self):
        image_id = self.upload("camera1")
        payload = self.payload("camera1", image_id)
        payload["points"][1], payload["points"][2] = payload["points"][2], payload["points"][1]
        response = self.client.post("/api/preview", json=payload)
        self.assertEqual(response.status_code, 200)
        points = response.get_json()["calibration"]["source_points_px"]
        self.assertEqual(
            points,
            [
                {"x": 80.0, "y": 60.0},
                {"x": 560.0, "y": 60.0},
                {"x": 560.0, "y": 420.0},
                {"x": 80.0, "y": 420.0},
            ],
        )

    def test_preview_rotation_is_applied_to_homography(self):
        image_id = self.upload("camera1")
        payload = self.payload("camera1", image_id)
        payload["rotation_degrees"] = 90
        response = self.client.post("/api/preview", json=payload)
        self.assertEqual(response.status_code, 200)
        calibration = response.get_json()["calibration"]
        self.assertEqual(calibration["rotation_degrees_clockwise"], 90)
        self.assertEqual(
            calibration["oriented_coordinate_size_m"],
            {"width": 12.5, "height": 25.0},
        )
        matrix = np.array(calibration["homography_pixel_to_meter"])
        source_point = np.array([80, 60, 1], dtype=np.float64)
        mapped = matrix @ source_point
        mapped /= mapped[2]
        np.testing.assert_allclose(mapped[:2], [12.5, 0], atol=1e-4)

    def test_horizontal_flip_is_applied_to_homography(self):
        image_id = self.upload("camera1")
        payload = self.payload("camera1", image_id)
        payload["flip_horizontal"] = True
        response = self.client.post("/api/preview", json=payload)
        self.assertEqual(response.status_code, 200)
        calibration = response.get_json()["calibration"]
        self.assertTrue(calibration["flip_horizontal"])
        matrix = np.array(calibration["homography_pixel_to_meter"])
        source_point = np.array([80, 60, 1], dtype=np.float64)
        mapped = matrix @ source_point
        mapped /= mapped[2]
        np.testing.assert_allclose(mapped[:2], [25, 0], atol=1e-4)

    def test_builds_combined_xy_alignment_preview(self):
        camera1 = self.payload("camera1", self.upload("camera1"))
        camera2 = self.payload("camera2", self.upload("camera2"))
        response = self.client.post(
            "/api/alignment-preview",
            json={"pool": {"length_m": 25, "width_m": 12.5}, "cameras": [camera1, camera2]},
        )
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["coordinate_size_m"], {"width": 25.0, "height": 12.5})
        self.assertEqual(data["grid_step_m"], 5)
        self.assertIn("color_calibration", data)
        self.assertEqual(set(data["color_calibration"]["cameras"]), {"camera1", "camera2"})
        preview = self.client.get(data["preview_url"])
        self.assertEqual(preview.status_code, 200)
        self.assertEqual(preview.mimetype, "image/jpeg")
        preview.close()

    def test_saves_complete_config(self):
        camera1 = self.payload("camera1", self.upload("camera1"))
        camera2 = self.payload("camera2", self.upload("camera2"))
        response = self.client.post(
            "/api/config",
            json={"pool": {"length_m": 25, "width_m": 12.5}, "cameras": [camera1, camera2]},
        )
        self.assertEqual(response.status_code, 200)
        config = json.loads(Path(self.app.config["CONFIG_PATH"]).read_text(encoding="utf-8"))
        self.assertEqual(config["schema_version"], 4)
        self.assertEqual(len(config["cameras"]), 2)
        self.assertEqual(config["coordinate_system"]["unit"], "meter")
        self.assertEqual(config["color_calibration"]["method"], "relative_robust_pool_roi_v1")
        self.assertIn("image_adjustments", config["cameras"][0])
        self.assertIn("white_balance", config["cameras"][0]["image_adjustments"])
        self.assertIn("exposure", config["cameras"][0]["image_adjustments"])

    def test_color_calibration_balances_two_different_images(self):
        brighter_cyan = np.full((180, 320, 3), (150, 120, 70), dtype=np.uint8)
        darker_red = np.full((180, 320, 3), (55, 65, 95), dtype=np.uint8)
        result = calculate_color_calibration([brighter_cyan, darker_red])
        camera1 = result["cameras"]["camera1"]
        camera2 = result["cameras"]["camera2"]

        self.assertLess(camera1["exposure"]["compensation_ev"], 0)
        self.assertGreater(camera2["exposure"]["compensation_ev"], 0)
        self.assertGreater(camera1["white_balance"]["red_gain_multiplier"], 1)
        self.assertLess(camera2["white_balance"]["red_gain_multiplier"], 1)

    def test_rejects_mismatched_preview_orientation(self):
        camera1 = self.payload("camera1", self.upload("camera1"))
        camera2 = self.payload("camera2", self.upload("camera2"))
        camera2["rotation_degrees"] = 90
        response = self.client.post(
            "/api/config",
            json={"pool": {"length_m": 25, "width_m": 12.5}, "cameras": [camera1, camera2]},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("가로·세로 방향", response.get_json()["error"])


if __name__ == "__main__":
    unittest.main()
