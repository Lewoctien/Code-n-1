from pathlib import Path
import argparse
import os
import time
import threading
from collections import deque

os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts/truetype/dejavu")
os.environ.setdefault("GPIOZERO_PIN_FACTORY", "lgpio")

import cv2
import numpy as np
from ultralytics import YOLO

from gpiozero import OutputDevice

#sẵn sàng chạy NCNN_model
MODEL_PATH = Path("lancuoi_ncnn_model")
RTSP_URL = "rtsp://admin:L223092F@192.168.100.177:554/cam/realmonitor?channel=1&subtype=0"

RTSP_TRANSPORT = "tcp"
FFMPEG_LOW_LATENCY_OPTIONS = (
    f"rtsp_transport;{RTSP_TRANSPORT}"
    "|threads;1"
)

#Cập nhật class ID
PERSON_CLASS = 1
HELMET_CLASS = 0
VEST_CLASS   = 2

# Chỉnh độ tự tin
CONF_PERSON = 0.45
CONF_HELMET = 0.55  
CONF_VEST   = 0.55 

#Các cấu hình
IOU_THRESHOLD = 0.45
INFER_SIZE = 640
INFER_EVERY_N = 2
DISPLAY_W, DISPLAY_H = 1280, 720
FRAME_TIME = 0.0
MIN_PERSON_AREA = 5000
MIN_PPE_AREA = 300
MAX_PERSON_AREA_RATIO = 0.90
MAX_PPE_AREA_RATIO = 0.35
PERSON_NMS_IOU = 0.30
PPE_NMS_IOU = 0.45
SHOW_PART_BOXES = False
DROP_STALE_GRABS = 0
CAMERA_STARTUP_TIMEOUT = 75.0

CAMERA_RECONNECT_DELAY = 2.0
GST_RTSP_LATENCY_MS = 150
USE_HW_ACCELERATION = False


USE_GSTREAMER_LOW_LATENCY = True

WARNING_LED_PIN = 17
WARNING_LED_ACTIVE_HIGH = True
WARNING_LED_STABLE_SECONDS = 0.5

_warning_led_device = None
_warning_led_output_state = False
_warning_led_candidate_state = False
_warning_led_candidate_since = time.monotonic()


def _get_warning_led():
    global _warning_led_device
    if _warning_led_device is None:
        _warning_led_device = OutputDevice(
            WARNING_LED_PIN,
            active_high=WARNING_LED_ACTIVE_HIGH,
            initial_value=False,
        )

    return _warning_led_device


def _write_warning_led(is_on):
    device = _get_warning_led()
    if is_on:
        device.on()
    else:
        device.off()


def reset_warning_led():
    global _warning_led_output_state, _warning_led_candidate_state, _warning_led_candidate_since
    _warning_led_output_state = False
    _warning_led_candidate_state = False
    _warning_led_candidate_since = time.monotonic()
    _write_warning_led(False)


def update_warning_led(has_helmet, has_vest):
    global _warning_led_output_state, _warning_led_candidate_state, _warning_led_candidate_since
    should_turn_on = not (has_helmet and has_vest)
    now = time.monotonic()

    if should_turn_on != _warning_led_candidate_state:
        _warning_led_candidate_state = should_turn_on
        _warning_led_candidate_since = now
        return _warning_led_output_state

    stable_for = now - _warning_led_candidate_since
    if should_turn_on != _warning_led_output_state and stable_for >= WARNING_LED_STABLE_SECONDS:
        _write_warning_led(should_turn_on)
        _warning_led_output_state = should_turn_on

    return _warning_led_output_state


def close_warning_led():
    global _warning_led_device
    _write_warning_led(False)
    if _warning_led_device is not None:
        _warning_led_device.close()
        _warning_led_device = None


def test_warning_led(duration=3.0):
    reset_warning_led()
    _write_warning_led(True)
    time.sleep(duration)
    close_warning_led()

# ===== CAMERA THREAD =====
class CameraReader:
    def __init__(self, url):
        self.url = url
        self.cap = None
        self._frame = None
        self._frame_time = 0.0
        self._lock = threading.Lock()
        self._running = True
        self._connected = False
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _build_gstreamer_pipeline(self, codec):
        if codec == "h265":
            depay, parser, decoder = "rtph265depay", "h265parse", "avdec_h265"
        else:
            depay, parser, decoder = "rtph264depay", "h264parse", "avdec_h264"

        return (
            f'rtspsrc location="{self.url}" latency={GST_RTSP_LATENCY_MS} protocols={RTSP_TRANSPORT} '
            f"! {depay} ! {parser} ! {decoder} ! "
            "videoconvert ! video/x-raw,format=BGR ! "
            "appsink sync=false max-buffers=1 drop=true"
        )

    def _configure_capture(self, cap):
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 2000)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if USE_HW_ACCELERATION:
            cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)

    def _capture_open_params(self):
        return [
            cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000,
            cv2.CAP_PROP_READ_TIMEOUT_MSEC, 2000,
        ]

    def _create_capture(self, source, backend):
        params = self._capture_open_params()
        return cv2.VideoCapture(source, backend, params)

    def _open_capture(self):
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = FFMPEG_LOW_LATENCY_OPTIONS

        backend_candidates = []
        if USE_GSTREAMER_LOW_LATENCY:
            backend_candidates.extend([
                ("GSTREAMER-H264", self._build_gstreamer_pipeline("h264"), cv2.CAP_GSTREAMER),
                ("GSTREAMER-H265", self._build_gstreamer_pipeline("h265"), cv2.CAP_GSTREAMER),
            ])
        backend_candidates.extend([
            ("FFMPEG", self.url, cv2.CAP_FFMPEG),
            ("DEFAULT", self.url, cv2.CAP_ANY),
        ])

        for backend_name, source, backend in backend_candidates:
            cap = self._create_capture(source, backend)
            self._configure_capture(cap)
            if cap.isOpened():
                return cap
            cap.release()
        return None

    def _init_cap(self):
        if self.cap:
            self.cap.release()
        self.cap = self._open_capture()
        self._connected = self.cap is not None and self.cap.isOpened()

    def _loop(self):
        while self._running:
            if not self._connected:
                self._init_cap()
                if not self._connected:
                    time.sleep(CAMERA_RECONNECT_DELAY)
                continue
            ret, frame = self.cap.read()
            if not ret or frame is None:
                self._connected = False
                continue
            if DROP_STALE_GRABS > 0:
                grabbed = False
                for _ in range(DROP_STALE_GRABS):
                    if not self.cap.grab():
                        break
                    grabbed = True
                if grabbed:
                    ret, fresh_frame = self.cap.retrieve()
                    if ret and fresh_frame is not None:
                        frame = fresh_frame
            with self._lock:
                self._frame = frame
                self._frame_time = time.time()

    def get(self):
        with self._lock:
            return self._frame if self._frame is not None else None

    def age_ms(self):
        with self._lock:
            if self._frame_time == 0:
                return 0.0
            return (time.time() - self._frame_time) * 1000.0



    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()
            self.cap = None

# ===== UTILS =====
def draw_label(frame, text, x1, y1, color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thick = 0.55, 1
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    pad = 4
    by1 = max(y1 - th - pad * 2, 0)
    by2 = y1 if y1 > th + pad * 2 else th + pad * 2
    cv2.rectangle(frame, (x1, by1), (x1 + tw + pad * 2, by2), color, -1)
    cv2.putText(frame, text, (x1 + pad, by2 - pad), font, scale, (255, 255, 255), thick, cv2.LINE_AA)


def box_area(box):
    x1, y1, x2, y2 = box[:4]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def box_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a[:4]
    bx1, by1, bx2, by2 = box_b[:4]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = box_area(box_a) + box_area(box_b) - inter
    return inter / union if union > 0 else 0.0


def nms_boxes(boxes, iou_threshold):
    boxes = sorted(boxes, key=lambda item: item[4], reverse=True)
    kept = []
    for box in boxes:
        if all(box_iou(box, kept_box) < iou_threshold for kept_box in kept):
            kept.append(box)
    return kept


def valid_box(box, frame_area, min_area, max_area_ratio):
    area = box_area(box)
    if area < min_area:
        return False
    if area > frame_area * max_area_ratio:
        return False
    x1, y1, x2, y2 = box[:4]
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return False
    return True


def parse_detections(result, frame_shape):
    frame_h, frame_w = frame_shape[:2]
    frame_area = frame_h * frame_w
    person_boxes, helmet_boxes, vest_boxes = [], [], []
    boxes = result.boxes

    if boxes is None or len(boxes) == 0:
        return person_boxes, helmet_boxes, vest_boxes

    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1 = max(0.0, min(float(x1), frame_w - 1.0))
        y1 = max(0.0, min(float(y1), frame_h - 1.0))
        x2 = max(0.0, min(float(x2), frame_w - 1.0))
        y2 = max(0.0, min(float(y2), frame_h - 1.0))
        det = (x1, y1, x2, y2, conf)

        if cls_id == PERSON_CLASS and conf >= CONF_PERSON:
            if valid_box(det, frame_area, MIN_PERSON_AREA, MAX_PERSON_AREA_RATIO):
                person_boxes.append(det)
        elif cls_id == HELMET_CLASS and conf >= CONF_HELMET:
            if valid_box(det, frame_area, MIN_PPE_AREA, MAX_PPE_AREA_RATIO):
                helmet_boxes.append(det)
        elif cls_id == VEST_CLASS and conf >= CONF_VEST:
            if valid_box(det, frame_area, MIN_PPE_AREA, MAX_PPE_AREA_RATIO):
                vest_boxes.append(det)

    return (
        nms_boxes(person_boxes, PERSON_NMS_IOU),
        nms_boxes(helmet_boxes, PPE_NMS_IOU),
        nms_boxes(vest_boxes, PPE_NMS_IOU),
    )


def build_display_results(person_boxes, helmet_boxes, vest_boxes, show_parts=False):
    last_results = []
    has_person = bool(person_boxes)
    all_have_helmet = has_person
    all_have_vest = has_person

    for px1, py1, px2, py2, p_conf in person_boxes:
        p_height = py2 - py1
        p_width = px2 - px1
        margin_x = p_width * 0.1

        has_helmet = False
        for hx1, hy1, hx2, hy2, h_conf in helmet_boxes:
            helmet_center_x = (hx1 + hx2) / 2
            valid_x = (px1 - margin_x) <= helmet_center_x <= (px2 + margin_x)
            valid_top_y = (py1 - p_height * 0.1) <= hy1 <= (py1 + p_height * 0.25)
            if valid_x and valid_top_y:
                has_helmet = True
                break

        has_vest = False
        for vx1, vy1, vx2, vy2, v_conf in vest_boxes:
            vest_center_x = (vx1 + vx2) / 2
            vest_width = vx2 - vx1
            valid_x = (px1 - margin_x) <= vest_center_x <= (px2 + margin_x)
            valid_top_y = (py1 + p_height * 0.08) <= vy1 <= (py1 + p_height * 0.55)
            valid_width = vest_width >= (p_width * 0.30)
            if valid_x and valid_top_y and valid_width:
                has_vest = True
                break

        if not has_helmet:
            all_have_helmet = False
        if not has_vest:
            all_have_vest = False

        if has_helmet and has_vest:
            color, label = (0, 200, 0), "SAFE"
        elif has_helmet and not has_vest:
            color, label = (0, 140, 255), "WARN: No Vest"
        elif not has_helmet and has_vest:
            color, label = (0, 140, 255), "WARN: No Helmet"
        else:
            color, label = (0, 0, 255), "DANGER: No Helmet/Vest"

        last_results.append((int(px1), int(py1), int(px2), int(py2), label, color))

    if show_parts:
        for hx1, hy1, hx2, hy2, h_conf in helmet_boxes:
            last_results.append((int(hx1), int(hy1), int(hx2), int(hy2), f"Helmet {h_conf:.2f}", (255, 255, 0)))
        for vx1, vy1, vx2, vy2, v_conf in vest_boxes:
            last_results.append((int(vx1), int(vy1), int(vx2), int(vy2), f"Vest {v_conf:.2f}", (255, 0, 255)))

    return last_results, all_have_helmet, all_have_vest


def detect_frame(model, frame, show_parts=False):
    result = model(
        frame,
        imgsz=INFER_SIZE,
        conf=min(CONF_PERSON, CONF_HELMET, CONF_VEST),
        iou=IOU_THRESHOLD,
        verbose=False,
    )[0]
    person_boxes, helmet_boxes, vest_boxes = parse_detections(result, frame.shape)
    display_results, all_have_helmet, all_have_vest = build_display_results(
        person_boxes,
        helmet_boxes,
        vest_boxes,
        show_parts=show_parts,
    )
    stats = {
        "person": len(person_boxes),
        "helmet": len(helmet_boxes),
        "vest": len(vest_boxes),
        "has_person": bool(person_boxes),
        "has_helmet": all_have_helmet,
        "has_vest": all_have_vest,
    }
    return display_results, stats


def draw_results(display, results):
    for x1, y1, x2, y2, label, color in results:
        thickness = 3 if "WARN" in label or "DANGER" in label else 2
        cv2.rectangle(display, (x1, y1), (x2, y2), color, thickness)
        draw_label(display, label, x1, y1, color)


def load_model():
    model = YOLO(str(MODEL_PATH), task="detect")
    model(np.zeros((INFER_SIZE, INFER_SIZE, 3), dtype=np.uint8), verbose=False)
    return model


def wait_for_camera_frame(cam, timeout=CAMERA_STARTUP_TIMEOUT):
    deadline = time.time() + timeout
    while time.time() < deadline:
        frame = cam.get()
        if frame is not None:
            return frame
        time.sleep(0.05)
    return None


def run_camera(model, camera_url=RTSP_URL, startup_timeout=CAMERA_STARTUP_TIMEOUT, show_parts=False):
    reset_warning_led()
    cam = CameraReader(camera_url)
    first_frame = wait_for_camera_frame(cam, startup_timeout)
    if first_frame is None:
        cam.stop()
        close_warning_led()
        return

    frame_count = 0
    smooth_results = []
    fps_buf = deque(maxlen=30)
    prev_time = time.time()
    last_stats = {
        "person": 0,
        "helmet": 0,
        "vest": 0,
        "has_person": False,
        "has_helmet": True,
        "has_vest": True,
    }



    try:
        while True:
            loop_start = time.time()
            frame = cam.get()
            if frame is None:
                time.sleep(0.01)
                continue

            frame_count += 1
            display = cv2.resize(frame, (DISPLAY_W, DISPLAY_H))

            if frame_count % INFER_EVERY_N == 0:
                smooth_results, last_stats = detect_frame(
                    model,
                    display,
                    show_parts=show_parts,
                )
                if last_stats["has_person"]:
                    update_warning_led(last_stats["has_helmet"], last_stats["has_vest"])
                else:
                    update_warning_led(True, True)

            draw_results(display, smooth_results)



            now = time.time()
            fps_buf.append(1.0 / max(now - prev_time, 1e-6))
            prev_time = now
            avg_fps = sum(fps_buf) / len(fps_buf)

            cv2.putText(display, f"FPS: {avg_fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(
                display,
                f"P:{last_stats['person']} H:{last_stats['helmet']} V:{last_stats['vest']}",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
            cv2.imshow("Safety Monitoring", display)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("d"):
                show_parts = not show_parts

            if FRAME_TIME > 0:
                elapsed = time.time() - loop_start
                if elapsed < FRAME_TIME:
                    time.sleep(FRAME_TIME - elapsed)
    except KeyboardInterrupt:
        pass
    finally:
        cam.stop()
        close_warning_led()
        cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera-url", default=RTSP_URL, help="Camera source, for example RTSP URL or /dev/video0.")
    parser.add_argument(
        "--camera-timeout",
        type=float,
        default=CAMERA_STARTUP_TIMEOUT,
        help="Seconds to wait for the first camera frame before exiting.",
    )
    parser.add_argument("--show-parts", action="store_true", help="Draw helmet/vest boxes for debug.")
    parser.add_argument("--test-led", action="store_true", help="Turn GPIO warning LED on for a quick hardware test.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.test_led:
        test_warning_led()
        return
    model = load_model()
    run_camera(
        model,
        camera_url=args.camera_url,
        startup_timeout=args.camera_timeout,
        show_parts=args.show_parts or SHOW_PART_BOXES,
    )


if __name__ == "__main__":
    main()
