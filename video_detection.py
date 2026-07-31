import cv2
from detection import process_frame

STREAM_FRAME_STRIDE = 2


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


def generate_camera_frames(camera_index=0):
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        payload = _encode_frame(_build_status_frame("Unable to open webcam."))
        if payload:
            while True:
                yield payload
        return

    frame_index = 0

    while True:
        success, frame = cap.read()

        if not success:
            payload = _encode_frame(_build_status_frame("Webcam disconnected."))
            if payload:
                yield payload
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
