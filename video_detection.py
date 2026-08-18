import threading
import time

import cv2
import numpy as np
import requests

import detection
from detection import process_frame

STREAM_FRAME_STRIDE = 2

# Latest detection counts per live stream source, updated by the stream generator
# and read by the /cctv/stream/status endpoint so the page can show live status.
STREAM_STATUS = {}


def update_stream_status(source_key, workers, safe, violations):
    workers = int(workers)
    safe = int(safe)
    violations = int(violations)
    STREAM_STATUS[source_key] = {
        "workers": workers,
        "safe": safe,
        "violations": violations,
        "status": "SAFE" if violations == 0 else "UNSAFE",
        "updated": time.time(),
    }


def build_stream_candidates(source_url):
    source_url = (source_url or "").strip()
    if not source_url:
        return []

    candidates = []
    normalized = source_url.replace("\\", "/")
    if normalized.startswith(("http://", "https://")):
        candidates.append(normalized)
        if not normalized.rstrip("/").endswith(("/video", "/mjpegfeed")):
            candidates.append(normalized.rstrip("/") + "/video")
            candidates.append(normalized.rstrip("/") + "/mjpegfeed")
    else:
        candidates.append(normalized)
        if normalized.startswith(("localhost", "127.0.0.1")):
            candidates.append(f"http://{normalized}/video")
            candidates.append(f"http://{normalized}/mjpegfeed")
        elif ":" in normalized and not normalized.startswith(("rtsp://", "http://", "https://")):
            candidates.append(f"http://{normalized}/video")
            candidates.append(f"http://{normalized}/mjpegfeed")

    seen = []
    for candidate in candidates:
        if candidate not in seen:
            seen.append(candidate)
    return seen


def _encode_frame(frame):
    ret, buffer = cv2.imencode(".jpg", frame)
    if not ret:
        return None
    return (
        b'--frame\r\n'
        b'Content-Type: image/jpeg\r\n\r\n' +
        buffer.tobytes() +
        b'\r\n'
    )


def _build_status_frame(message, width=960, height=540):
    frame = 255 * cv2.UMat(height, width, cv2.CV_8UC3).get()
    cv2.putText(
        frame,
        "SafeSite AI Webcam",
        (40, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (37, 99, 235),
        3
    )
    cv2.putText(
        frame,
        message,
        (40, 170),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (20, 36, 61),
        2
    )
    cv2.putText(
        frame,
        "Check camera permission or try another camera index.",
        (40, 220),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (92, 107, 128),
        2
    )
    return frame


def generate_frames(video_path):

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        fallback = np.full((540, 960, 3), 255, dtype=np.uint8)
        cv2.putText(fallback, "SafeSite AI Video", (40, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (37, 99, 235), 3)
        cv2.putText(fallback, "The uploaded clip could not be opened yet.", (40, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 36, 61), 2)
        cv2.putText(fallback, "Please try another file or refresh the page.", (40, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (92, 107, 128), 2)
        payload = _encode_frame(fallback)
        if payload:
            yield payload
        return

    frame_index = 0

    while True:
        success, frame = cap.read()

        if not success:
            break

        if frame_index % STREAM_FRAME_STRIDE != 0:
            frame_index += 1
            continue

        frame, _, _, _ = process_frame(frame)
        frame_index += 1

        payload = _encode_frame(frame)
        if payload:
            yield payload

    cap.release()


import cv2
from detection import process_frame

def generate_camera_frames(camera_index=0):

    # Try different camera indexes (DroidCam, webcam, etc.)
    cap = None

    for idx in [camera_index, 1, 2, 0]:
        cap = cv2.VideoCapture(idx)

        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"Using Camera Index: {idx}")
                break
            cap.release()
            cap = None

    if cap is None:
        raise Exception("No camera found")

    while True:

        success, frame = cap.read()

        if not success:
            break

        frame, workers, safe, violations = process_frame(frame)

        ret, buffer = cv2.imencode('.jpg', frame)

        frame = buffer.tobytes()

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' +
            frame +
            b'\r\n'
        )

    cap.release()

def _probe_stream(url, timeout=4):
    """Try to reach a source URL and classify what it actually returns.

    Returns (status_code, content_type, error_string).
    """
    try:
        r = requests.get(url, stream=True, timeout=timeout)
        ctype = (r.headers.get("Content-Type", "") or "").lower()
        status = r.status_code
        r.close()
        return status, ctype, None
    except Exception as exc:
        return None, "", str(exc)


def _open_cap_bounded(url, timeout=8):
    """Try to open a video capture source on url, bounded by a timeout."""
    result = {}

    def _open():
        try:
            cap = cv2.VideoCapture(url)
            result["cap"] = cap
            result["ok"] = cap.isOpened()
        except Exception as exc:
            result["err"] = str(exc)

    thread = threading.Thread(target=_open, daemon=True)
    thread.start()
    thread.join(timeout)
    if not result.get("ok"):
        if thread.is_alive():
            # Give up on a hung connect attempt.
            return None, "Timed out connecting to: " + url
        return None, result.get("err") or "Unable to open stream: " + url
    return result.get("cap"), None


def generate_device_frames(source_key, camera_index=0):
    """Stream and annotate frames from a local camera/DroidCam device index."""
    if not detection.is_model_ready():
        threading.Thread(target=detection._get_model, daemon=True).start()

    indices = []
    for idx in [camera_index, 1, 2, 0]:
        if idx not in indices and 0 <= idx <= 9:
            indices.append(idx)

    cap = None
    opened_index = None
    for idx in indices:
        try:
            probe = cv2.VideoCapture(idx)
            if probe.isOpened():
                ok, frame = probe.read()
                if ok and frame is not None and frame.size > 0:
                    cap = probe
                    opened_index = idx
                    break
                probe.release()
        except Exception:
            continue

    if cap is None:
        fallback = np.full((540, 960, 3), 255, dtype=np.uint8)
        cv2.putText(fallback, "SafeSite AI CCTV", (40, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (37, 99, 235), 3)
        cv2.putText(fallback, "No PC / DroidCam camera device found.", (40, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 38, 38), 2)
        cv2.putText(fallback, "Open the DroidCam client, connect the phone, then retry.", (40, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (92, 107, 128), 2)
        STREAM_STATUS[source_key] = {"status": "ERROR", "error": "No camera device found", "updated": time.time()}
        payload = _encode_frame(fallback)
        if payload:
            yield payload
        return

    STREAM_STATUS[source_key] = {
        "status": "CONNECTING",
        "camera_index": opened_index,
        "updated": time.time(),
    }

    frame_index = 0
    while True:
        success, frame = cap.read()
        if not success:
            break

        if frame_index % STREAM_FRAME_STRIDE != 0:
            frame_index += 1
            continue
        frame_index += 1

        if detection.is_model_ready():
            try:
                frame, workers, safe, violations = process_frame(frame)
                update_stream_status(source_key, workers, safe, violations)
            except Exception:
                pass

        payload = _encode_frame(frame)
        if payload:
            yield payload

    cap.release()


def generate_stream_frames(source_url):
    source_url = (source_url or "").strip()
    if not source_url:
        raise ValueError("No CCTV stream URL provided")

    stream_candidates = build_stream_candidates(source_url)
    last_error = None
    last_reason = None

    # Kick off model loading in the background so the raw stream appears
    # immediately instead of blocking on the first detection.
    if not detection.is_model_ready():
        threading.Thread(target=detection._get_model, daemon=True).start()

    for candidate in stream_candidates:
        status, ctype, err = _probe_stream(candidate)
        if err:
            last_error = f"Could not reach {candidate} — {err}"
            continue
        if status != 200:
            last_error = f"{candidate} replied with HTTP {status}"
            continue
        if ctype.startswith(("text/html", "application/xhtml")) or "droidcam" in ctype:
            last_error = (
                f"{candidate} is a web page, not a video stream. "
                "New DroidCam no longer exposes a raw video URL — connect the "
                "DroidCam client on the PC and use Live Monitoring's webcam "
                "instead, or point the box at an IP camera with an MJPEG URL."
            )
            continue
        if not (ctype.startswith("image/") or "multipart" in ctype or "mjpeg" in ctype):
            last_error = f"{candidate} returned {ctype or 'an unknown type'} — not a video stream."
            continue

        cap, err = _open_cap_bounded(candidate)
        if cap is not None:
            frame_index = 0
            while True:
                success, frame = cap.read()
                if not success:
                    break

                if frame_index % STREAM_FRAME_STRIDE != 0:
                    frame_index += 1
                    continue
                frame_index += 1

                if detection.is_model_ready():
                    try:
                        frame, workers, safe, violations = process_frame(frame)
                        update_stream_status(source_url, workers, safe, violations)
                    except Exception:
                        pass

                payload = _encode_frame(frame)
                if payload:
                    yield payload

            cap.release()
            return

        last_error = err

    if stream_candidates and not last_reason:
        try:
            response = requests.get(stream_candidates[0], stream=True, timeout=5)
            response.raise_for_status()
            content_type = response.headers.get("Content-Type", "").lower()
            if content_type.startswith("image/") or "multipart" in content_type or "mjpeg" in content_type:
                for chunk in response.iter_content(chunk_size=4096):
                    if chunk:
                        yield chunk
                return
        except Exception as exc:
            last_error = str(exc)

    fallback = np.full((540, 960, 3), 255, dtype=np.uint8)
    cv2.putText(fallback, "SafeSite AI CCTV", (40, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (37, 99, 235), 3)
    reason = last_error or "Unable to connect to the CCTV stream URL."
    if last_error:
        STREAM_STATUS[source_url] = {
            "status": "ERROR",
            "error": last_error,
            "updated": time.time(),
        }
    if len(reason) > 90:
        for i, line in enumerate(reason.split("— ")):
            if i >= 3:
                break
            cv2.putText(fallback, line, (40, 170 + i * 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 38, 38), 2)
    else:
        cv2.putText(fallback, reason, (40, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 38, 38), 2)
    payload = _encode_frame(fallback)
    if payload:
        yield payload
