#!/usr/bin/env python3
import argparse
import gc
import math
import sys
import time
from collections import deque
from concurrent.futures import Future
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import yaml
from hailo_platform import HEF, FormatType, HailoSchedulingAlgorithm, VDevice
from picamera2 import Picamera2


CLASSES = ("SWIMMING", "FLOATING", "ACTIVE")
WINDOW = 50


class HailoModel:
    def __init__(self, device, path):
        self.hef = HEF(str(path))
        self.input_shape = tuple(self.hef.get_input_vstream_infos()[0].shape)
        self.model = device.create_infer_model(str(path))
        self.model.set_batch_size(1)
        self.model.input().set_format_type(FormatType.FLOAT32)
        for output in self.model.outputs:
            output.set_format_type(FormatType.FLOAT32)
        self.names = list(self.model.output_names)
        self.configured = self.model.configure()

    def run(self, data):
        buffers = {
            name: np.empty(self.model.output(name).shape, np.float32)
            for name in self.names
        }
        bindings = self.configured.create_bindings(output_buffers=buffers)
        bindings.input().set_buffer(np.ascontiguousarray(data, np.float32))
        future = Future()

        def done(info):
            try:
                if info.exception:
                    raise info.exception
                if len(self.names) == 1:
                    future.set_result(bindings.output().get_buffer().copy())
                else:
                    future.set_result({n: bindings.output(n).get_buffer().copy() for n in self.names})
            except Exception as exc:
                future.set_exception(exc)

        self.configured.wait_for_async_ready(timeout_ms=10000)
        job = self.configured.run_async([bindings], done)
        result = future.result(timeout=15)
        _ = job
        return result

    def close(self):
        if hasattr(self, "configured"):
            del self.configured
        if hasattr(self, "model"):
            del self.model


def letterbox(image, out_h, out_w):
    h, w = image.shape[:2]
    gain = min(out_w / w, out_h / h)
    nw, nh = int(round(w * gain)), int(round(h * gain))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    left, top = (out_w - nw) // 2, (out_h - nh) // 2
    canvas = np.full((out_h, out_w, 3), 114, np.uint8)
    canvas[top:top + nh, left:left + nw] = resized
    return canvas.astype(np.float32) / 255.0, gain, left, top


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def output_matrix(value):
    a = np.squeeze(np.asarray(value, dtype=np.float32))
    if a.ndim == 1:
        return a.reshape(1, -1)
    if a.ndim == 2:
        if a.shape[0] <= 256:
            return a
        if a.shape[1] <= 256:
            return a.T
    if a.ndim == 3:
        if a.shape[-1] <= 256:
            return a.reshape(-1, a.shape[-1]).T
        if a.shape[0] <= 256:
            return a.reshape(a.shape[0], -1)
    return None


def anchors_for(count, size):
    grids = [(size // s, s) for s in (8, 16, 32)]
    if sum(g * g for g, _ in grids) != count:
        g = int(round(math.sqrt(count)))
        if g * g != count:
            raise ValueError(f"unsupported YOLO anchor count: {count}")
        grids = [(g, size / g)]
    points, strides = [], []
    for g, stride in grids:
        yy, xx = np.meshgrid(np.arange(g), np.arange(g), indexing="ij")
        points.append(np.stack((xx.ravel() + 0.5, yy.ravel() + 0.5), axis=1))
        strides.append(np.full((g * g, 1), stride, np.float32))
    return np.concatenate(points).astype(np.float32), np.concatenate(strides)


def decode_dfl(raw, size):
    bins = raw.shape[0] // 4
    if bins < 2 or raw.shape[0] != bins * 4:
        raise ValueError(f"invalid DFL shape: {raw.shape}")
    x = raw.T.reshape(-1, 4, bins)
    x -= x.max(axis=2, keepdims=True)
    x = np.exp(x)
    x /= x.sum(axis=2, keepdims=True)
    dist = (x * np.arange(bins, dtype=np.float32)).sum(axis=2)
    anchor, stride = anchors_for(raw.shape[1], size)
    xy1 = (anchor - dist[:, :2]) * stride
    xy2 = (anchor + dist[:, 2:]) * stride
    return np.concatenate((xy1, xy2), axis=1)


def decode_xywh(raw, size):
    box = raw.T.copy()
    if np.nanmax(np.abs(box)) <= 2.0:
        box *= size
    out = np.empty_like(box)
    out[:, 0] = box[:, 0] - box[:, 2] / 2
    out[:, 1] = box[:, 1] - box[:, 3] / 2
    out[:, 2] = box[:, 0] + box[:, 2] / 2
    out[:, 3] = box[:, 1] + box[:, 3] / 2
    return out


def nms(boxes, scores, threshold):
    if not len(boxes):
        return np.empty(0, np.int32)
    x1, y1, x2, y2 = boxes.T
    areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    order, keep = scores.argsort()[::-1], []
    while order.size:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1, yy1 = np.maximum(x1[i], x1[rest]), np.maximum(y1[i], y1[rest])
        xx2, yy2 = np.minimum(x2[i], x2[rest]), np.minimum(y2[i], y2[rest])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        union = areas[i] + areas[rest] - inter + 1e-7
        order = rest[inter / union <= threshold]
    return np.asarray(keep, np.int32)


def decode_yolo(outputs, input_size, gain, pad_x, pad_y, frame_shape, conf, iou):
    values = list(outputs.values()) if isinstance(outputs, dict) else outputs
    if not isinstance(values, (list, tuple)):
        values = [values]
    matrices = [m for m in (output_matrix(v) for v in values) if m is not None]
    groups = {}
    for m in matrices:
        groups.setdefault(m.shape[1], []).append(m)
    all_boxes, all_scores = [], []

    for count, mats in groups.items():
        score = next((m[0] for m in mats if m.shape[0] == 1), None)
        box = next((m for m in mats if m.shape[0] == 4), None)
        combined = next((m for m in mats if m.shape[0] in (5, 65)), None)
        dfl = next((m for m in mats if m.shape[0] >= 64 and m.shape[0] % 4 == 0), None)

        if combined is not None:
            if combined.shape[0] == 5:
                box = combined[:4]
                score = combined[4]
            elif dfl is None:
                dfl = combined[:64]
                if score is None:
                    score = combined[64]
        if score is None or (box is None and dfl is None):
            continue
        if score.min(initial=0) < 0 or score.max(initial=0) > 1:
            score = sigmoid(score)
        boxes = decode_xywh(box, input_size) if box is not None else decode_dfl(dfl, input_size)
        mask = score >= conf
        all_boxes.append(boxes[mask])
        all_scores.append(score[mask])

    if not all_boxes:
        shapes = [tuple(np.asarray(v).shape) for v in values]
        raise ValueError(f"unsupported detector outputs: {shapes}")
    boxes = np.concatenate(all_boxes)
    scores = np.concatenate(all_scores)
    keep = nms(boxes, scores, iou)
    boxes, scores = boxes[keep], scores[keep]
    h, w = frame_shape[:2]
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / gain
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / gain
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w - 1)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h - 1)
    valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    return np.column_stack((boxes[valid], scores[valid])).astype(np.float32)


def load_tracker(path, frame_rate):
    hailo_apps = Path.home() / "hailo-apps"
    if hailo_apps.exists():
        sys.path.insert(0, str(hailo_apps))
    from hailo_apps.python.core.tracker.byte_tracker import BYTETracker

    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    tracker_args = SimpleNamespace(
        track_thresh=float(cfg.get("track_thresh", cfg.get("track_high_thresh", 0.25))),
        track_buffer=int(cfg.get("track_buffer", 30)),
        match_thresh=float(cfg.get("match_thresh", 0.8)),
        mot20=bool(cfg.get("mot20", False)),
    )
    tracker = BYTETracker(tracker_args, frame_rate=frame_rate)
    tracker.det_thresh = float(cfg.get("new_track_thresh", tracker_args.track_thresh + 0.1))
    return tracker


@dataclass
class TrackState:
    samples: deque = field(default_factory=lambda: deque(maxlen=WINDOW))
    label: str = "WARMUP"
    probability: float = 0.0
    last_seen: float = 0.0
    last_box: np.ndarray | None = None


def fill_missing(values, default):
    x = np.asarray(values, np.float32)
    good = np.flatnonzero(np.isfinite(x))
    if not len(good):
        return np.full(len(x), default, np.float32)
    return np.interp(np.arange(len(x)), good, x[good]).astype(np.float32)


def make_features(samples):
    a = np.asarray(samples, np.float32)
    detected, confidence = a[:, 0], a[:, 1]
    cx = fill_missing(a[:, 2], 0.5)
    cy = fill_missing(a[:, 3], 0.5)
    scale = fill_missing(a[:, 4], 0.0)
    dx = np.diff(cx, prepend=cx[0])
    dy = np.diff(cy, prepend=cy[0])
    speed = np.sqrt(dx * dx + dy * dy)
    scale_change = np.diff(scale, prepend=scale[0])
    change = np.diff(detected.astype(np.int32), prepend=int(detected[0]))
    lost = (change < 0).astype(np.float32)
    redetect = (change > 0).astype(np.float32)
    missing, count = [], 0
    for value in detected:
        count = 0 if value else count + 1
        missing.append(count / WINDOW)
    visible_ratio = float(detected.mean())
    return np.column_stack((
        detected, confidence, cx, cy, scale, dx, dy, speed, scale_change,
        lost, redetect, missing,
        np.full(WINDOW, visible_ratio), np.full(WINDOW, 1.0 - visible_ratio),
    )).astype(np.float32)


def classifier_input(features, shape):
    candidates = (features[..., None], features[None, ...], features)
    for value in candidates:
        if tuple(value.shape) == tuple(shape):
            return value
    raise ValueError(f"classifier input mismatch: HEF={shape}, data={features.shape}")


def softmax(x):
    x = np.asarray(x, np.float32).reshape(-1)
    x -= x.max()
    x = np.exp(x)
    return x / x.sum()


def classify(state, model, passive_steps):
    detected = [int(s[0]) for s in state.samples]
    longest = current = 0
    for value in detected:
        current = 0 if value else current + 1
        longest = max(longest, current)
    if longest >= passive_steps:
        return "PASSIVE_DROWNING", 1.0
    if len(state.samples) < WINDOW:
        return "WARMUP", len(state.samples) / WINDOW
    features = make_features(state.samples)
    output = model.run(classifier_input(features, model.input_shape))
    if isinstance(output, dict):
        output = next(iter(output.values()))
    probabilities = softmax(output)
    index = int(probabilities.argmax())
    return CLASSES[index], float(probabilities[index])


def color(label):
    return {
        "SWIMMING": (0, 220, 0),
        "FLOATING": (0, 220, 255),
        "ACTIVE": (0, 80, 255),
        "PASSIVE_DROWNING": (255, 0, 255),
    }.get(label, (200, 200, 200))


def parse_args():
    base = Path(__file__).resolve().parent
    p = argparse.ArgumentParser()
    p.add_argument("--detector", default=str(base / "pool_head_best_512.hef"))
    p.add_argument("--classifier", default=str(base / "drowning_classifier_50x14_hailo.hef"))
    p.add_argument("--tracker", default=str(base / "bytetrack_pool.yaml"))
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--camera-fps", type=int, default=30)
    p.add_argument("--sample-hz", type=float, default=10.0)
    p.add_argument("--conf", type=float, default=0.05)
    p.add_argument("--iou", type=float, default=0.70)
    p.add_argument("--passive-seconds", type=float, default=3.0)
    p.add_argument("--state-ttl", type=float, default=8.0)
    p.add_argument("--no-display", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    for path in (args.detector, args.classifier, args.tracker):
        if not Path(path).expanduser().exists():
            raise FileNotFoundError(path)

    params = VDevice.create_params()
    params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
    device = VDevice(params)
    detector = classifier = camera = None

    try:
        detector = HailoModel(device, Path(args.detector).expanduser())
        classifier = HailoModel(device, Path(args.classifier).expanduser())
        if len(detector.input_shape) != 3 or detector.input_shape[2] != 3:
            raise ValueError(f"detector input mismatch: {detector.input_shape}")
        det_h, det_w, _ = detector.input_shape
        tracker = load_tracker(args.tracker, args.camera_fps)
        states = {}
        interval = 1.0 / args.sample_hz
        passive_steps = max(1, int(round(args.passive_seconds * args.sample_hz)))
        next_sample = time.monotonic()

        camera = Picamera2()
        config = camera.create_video_configuration(
            main={"size": (args.width, args.height), "format": "RGB888"},
            controls={"FrameRate": args.camera_fps},
            buffer_count=4,
        )
        camera.configure(config)
        camera.start()

        while True:
            frame = camera.capture_array("main")
            inp, gain, px, py = letterbox(frame, det_h, det_w)
            raw = detector.run(inp)
            detections = decode_yolo(raw, det_w, gain, px, py, frame.shape, args.conf, args.iou)
            tracks = tracker.update(detections)
            active = {
                int(t.track_id): (np.asarray(t.tlbr, np.float32), float(t.score))
                for t in tracks
            }
            now = time.monotonic()

            if now >= next_sample:
                next_sample = max(next_sample + interval, now)
                for track_id, (box, score) in active.items():
                    state = states.setdefault(track_id, TrackState())
                    state.last_seen, state.last_box = now, box.copy()
                for track_id, state in list(states.items()):
                    if track_id in active:
                        box, score = active[track_id]
                        x1, y1, x2, y2 = box
                        cx = (x1 + x2) / (2 * args.width)
                        cy = (y1 + y2) / (2 * args.height)
                        scale = math.sqrt(max(0.0, (x2 - x1) / args.width * (y2 - y1) / args.height))
                        state.samples.append((1.0, score, cx, cy, scale))
                    else:
                        state.samples.append((0.0, 0.0, np.nan, np.nan, np.nan))
                    old = state.label
                    state.label, state.probability = classify(state, classifier, passive_steps)
                    if state.label != old:
                        print(f"ID={track_id} {state.label} {state.probability:.3f}", flush=True)
                    if now - state.last_seen > args.state_ttl:
                        del states[track_id]

            if not args.no_display:
                view = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                for track_id, (box, _) in active.items():
                    state = states.get(track_id, TrackState())
                    x1, y1, x2, y2 = box.astype(int)
                    c = color(state.label)
                    cv2.rectangle(view, (x1, y1), (x2, y2), c, 2)
                    cv2.putText(view, f"{track_id} {state.label} {state.probability:.2f}",
                                (x1, max(20, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 2)
                y = 24
                for track_id, state in sorted(states.items()):
                    cv2.putText(view, f"ID {track_id}: {state.label}", (12, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color(state.label), 2)
                    y += 24
                cv2.imshow("Drowning Monitor", view)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
    finally:
        if camera is not None:
            camera.stop()
            camera.close()
        cv2.destroyAllWindows()
        if classifier is not None:
            classifier.close()
        if detector is not None:
            detector.close()
        gc.collect()
        device.release()


if __name__ == "__main__":
    main()
