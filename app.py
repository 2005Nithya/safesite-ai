from flask import Flask, abort, jsonify, redirect, render_template, request, session, url_for
from pathlib import Path
from html import escape as html_escape
import os
import re
import requests
import time
from urllib.parse import quote
from werkzeug.middleware.proxy_fix import ProxyFix
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet
import json
from datetime import datetime
from flask import Response
from video_detection import STREAM_STATUS, generate_camera_frames, generate_device_frames, generate_frames, generate_stream_frames
from flask import Response
import cv2
try:
    from detection import process_frame
except Exception:
    def process_frame(frame):
        return frame, 0, 0, 0
import base64
import numpy as np

try:
    from detection import process_image, process_video, summarize_video
except Exception:
    def process_image(image_path):
        return os.path.join("static", "results", os.path.basename(image_path)).replace("\\", "/"), 0, 0, 0

    def summarize_video(video_path):
        return 0, 0, 0

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "safesite-ai-dev-key")

ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin123")
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_port=1)
app.config["SESSION_COOKIE_SECURE"] = os.getenv("SESSION_COOKIE_SECURE", "true").lower() == "true"
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["API_BASE_URL"] = (os.getenv("API_BASE_URL") or "").rstrip("/")

UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["RESULT_FOLDER"] = RESULT_FOLDER


def _to_static_filename(path):
    normalized = (path or "").replace("\\", "/").strip("/")
    if normalized.startswith("static/"):
        return normalized[len("static/"):]
    return normalized


def normalize_stream_source(source):
    source = (source or "").strip()
    if not source:
        return source

    if source.startswith(("http://", "https://", "rtsp://", "rtmp://")):
        return source

    if source.startswith(("localhost", "127.0.0.1")):
        return f"http://{source}"

    if re.match(r"^\d+\.\d+\.\d+\.\d+:[0-9]+(/.*)?$", source):
        return f"http://{source}"

    if ":" in source and "/" not in source:
        return f"http://{source}"

    return source


def _guess_video_mime(filename):
    extension = Path(filename or "").suffix.lower()
    return {
        ".mp4": "video/mp4",
        ".webm": "video/webm",
        ".mov": "video/quicktime",
        ".avi": "video/x-msvideo",
        ".mkv": "video/x-matroska",
    }.get(extension, "video/mp4")


def _append_history_record(record):
    history_file = "history.json"
    try:
        with open(history_file, "r", encoding="utf-8") as handle:
            history = json.load(handle)
    except Exception:
        history = []

    history.append(record)

    with open(history_file, "w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=4)

    # Automatically dispatch real safety alerts if violations > 0
    violations = record.get("violations", 0)
    if violations > 0:
        email_to = session.get("alert_email", "nithyasheejain@gmail.com")
        phone_to = session.get("alert_phone", "+15550192834")
        subject = f"🚨 SafeSite AI Hazard Alert: {violations} Violation(s) Detected!"
        body = f"""
        <h3>SafeSite AI Automated Hazard Dispatch</h3>
        <p><b>Timestamp:</b> {record.get('date', 'N/A')}</p>
        <p><b>Source File:</b> {record.get('file', 'N/A')}</p>
        <p><b>Workers Detected:</b> {record.get('workers', 0)}</p>
        <p><b>Violations Flagged:</b> <span style="color:red; font-weight:bold;">{violations}</span></p>
        <p><b>Status:</b> {record.get('status', 'UNSAFE')}</p>
        <p>Please log in to SafeSite AI to review inspection logs and take corrective action immediately.</p>
        """
        try:
            send_safety_email(email_to, subject, body)
            if session.get("sms_enabled", True):
                send_safety_sms(phone_to, f"SafeSite AI Alert: {violations} safety violations detected at site. Check email for details.")
        except Exception:
            pass


def _load_env_file():
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def get_firebase_config():
    _load_env_file()
    config = {
        "apiKey": os.getenv("FIREBASE_API_KEY", ""),
        "authDomain": os.getenv("FIREBASE_AUTH_DOMAIN", ""),
        "projectId": os.getenv("FIREBASE_PROJECT_ID", ""),
        "storageBucket": os.getenv("FIREBASE_STORAGE_BUCKET", ""),
        "messagingSenderId": os.getenv("FIREBASE_MESSAGING_SENDER_ID", ""),
        "appId": os.getenv("FIREBASE_APP_ID", ""),
    }
    return config


FIREBASE_CONFIG = get_firebase_config()


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/health/firebase")
def firebase_health():
    config = get_firebase_config()
    return jsonify({
        "configured": bool(config.get("apiKey") and config.get("projectId") and config.get("authDomain")),
        "projectId": config.get("projectId", ""),
        "authDomain": config.get("authDomain", ""),
    })


def _firebase_auth_request(mode, email, password):
    api_key = get_firebase_config().get("apiKey")
    if not api_key:
        raise ValueError("Firebase API key is not configured")

    endpoint = (
        "https://identitytoolkit.googleapis.com/v1/accounts:signUp?key=" + api_key
        if mode == "signup"
        else "https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key=" + api_key
    )
    payload = {"email": email, "password": password, "returnSecureToken": True}
    response = requests.post(endpoint, json=payload, timeout=15)
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        detail = response.json() if response.content else {}
        message = detail.get("error", {}).get("message", str(exc))
        raise ValueError(message) from exc
    return response.json()


@app.route("/")
def home():
    return render_template("welcome.html", firebase_config=get_firebase_config())


@app.route("/auth", methods=["GET", "POST"])
def auth():
    if request.method == "POST":
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "").strip()
        mode = request.form.get("mode", "signin")

        if not email or not password:
            return render_template("auth.html", firebase_config=get_firebase_config(), error="Please enter both email and password.")

        try:
            result = _firebase_auth_request(mode, email, password)
        except Exception as exc:
            return render_template("auth.html", firebase_config=get_firebase_config(), error=f"Authentication failed: {exc}")

        session["user_email"] = result.get("email", email)
        session["user_name"] = (result.get("displayName") or email.split("@", 1)[0]).title()
        session["user_uid"] = result.get("localId", "firebase-user")
        session["authenticated"] = True
        return redirect(url_for("dashboard"))

    return render_template("auth.html", firebase_config=get_firebase_config())


@app.route("/auth/session", methods=["POST"])
def auth_session():
    payload = request.get_json(silent=True) or request.form
    email = (payload.get("email") or "").strip()
    if not email:
        return jsonify({"ok": False, "error": "Email is required"}), 400

    session["user_email"] = email
    session["user_name"] = (payload.get("name") or email.split("@", 1)[0]).title()
    session["user_uid"] = payload.get("uid") or "local-user"
    session["authenticated"] = True
    return jsonify({"ok": True, "redirect": url_for("dashboard")})


@app.route("/dashboard")
def dashboard():
    records = _load_history()

    total_inspections = len(records)
    total_workers = sum(int(r.get("workers", 0)) for r in records)
    total_safe = sum(int(r.get("safe_workers", 0)) for r in records)
    total_violations = sum(int(r.get("violations", 0)) for r in records)
    unsafe_count = sum(1 for r in records if r.get("status") != "SAFE")
    safe_count = total_inspections - unsafe_count

    compliance_values = [float(r.get("compliance", 0)) for r in records if r.get("workers")]
    avg_compliance = round(sum(compliance_values) / len(compliance_values), 1) if compliance_values else 0

    return render_template(
        "dashboard.html",
        firebase_config=get_firebase_config(),
        user_name=session.get("user_name", "Captain"),
        metrics={
            "compliance": avg_compliance,
            "hazards": unsafe_count,
            "inspections": total_inspections,
            "workers": total_workers,
            "safe_workers": total_safe,
            "violations": total_violations,
            "safe_count": safe_count,
        },
        recent_activity=[
            {"title": "Latest PPE review", "detail": f"{total_workers} workers inspected across {total_inspections} checks"},
            {"title": "Violation alerts", "detail": f"{total_violations} violations flagged across all inspections"},
            {"title": "Compliance status", "detail": f"Site running at {avg_compliance}% average compliance"},
        ],
    )


@app.route("/detection")
def detection():
    return render_template(
        "detection.html",
        firebase_config=FIREBASE_CONFIG,
        user_name=session.get("user_name", "Captain"),
        api_base_url=app.config.get("API_BASE_URL", ""),
    )


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return render_template("detection.html", firebase_config=get_firebase_config(), error="No file uploaded."), 400

    file = request.files["file"]
    if file.filename == "":
        return render_template("detection.html", firebase_config=get_firebase_config(), error="Please choose a file first."), 400

    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    os.makedirs(app.config["RESULT_FOLDER"], exist_ok=True)

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    filename = file.filename.lower()

    is_video = filename.endswith((".mp4", ".avi", ".mov", ".mkv", ".webm"))
    is_image = filename.endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
    file.save(filepath)

    web_upload_path = _to_static_filename(os.path.join(app.config["UPLOAD_FOLDER"], file.filename))

    start_time = time.time()

    if is_image:

        result_path, workers, safe_workers, violations = process_image(filepath)
        media_type = "image"
        uploaded_video_mime = None
        result_stream_url = None

    elif is_video:

        workers, safe_workers, violations = summarize_video(filepath)
        result_path = None
        media_type = "video"
        uploaded_video_mime = _guess_video_mime(file.filename)
        result_stream_url = url_for("processed_video_feed", filename=os.path.basename(filepath))

    else:

        return render_template(
            "detection.html",
            firebase_config=FIREBASE_CONFIG,
            user_name=session.get("user_name", "Captain"),
            error="Unsupported file type."
        ), 400

    web_result_path = _to_static_filename(result_path)
    detection_time = round(time.time() - start_time, 2)
    session["workers"] = workers
    session["safe_workers"] = safe_workers
    session["violations"] = violations
    session["detection_time"] = f"{detection_time} sec"
    session["uploaded_image"] = filepath
    session["result_image"] = result_path
    session["uploaded_media_type"] = media_type
    session["result_media_type"] = media_type
    record = {
        "date": datetime.now().strftime("%d-%m-%Y %I:%M %p"),
        "file": file.filename,
        "workers": workers,
        "safe_workers": safe_workers,
        "violations": violations,
        "compliance": round((safe_workers / workers) * 100, 1) if workers > 0 else 0,
        "status": "SAFE" if violations == 0 else "UNSAFE",
        "source": "Upload",
        "source_type": media_type,
    }

    _append_history_record(record)

    return render_template(
        "detection.html",
        firebase_config=FIREBASE_CONFIG,
        user_name=session.get("user_name", "Captain"),
        uploaded_media=web_upload_path,
        result_media=web_result_path,
        uploaded_media_type=media_type,
        result_media_type=media_type,
        uploaded_video_mime=uploaded_video_mime,
        result_stream_url=result_stream_url,
        workers=workers,
        safe_workers=safe_workers,
        violations=violations,
        detection_time=f"{detection_time} sec",
    )


@app.route("/cctv/grid")
def cctv_grid():
    return render_template(
        "cctv_grid.html",
        user_name=session.get("user_name", "Captain"),
    )


@app.route("/cctv", methods=["GET", "POST"])
def cctv():
    if request.method == "POST":
        camera_id = request.form.get("camera_id", "").strip() or "CAM-UNKNOWN"
        site_zone = request.form.get("site_zone", "").strip() or "Main site"
        incident_type = request.form.get("incident_type", "").strip() or "Routine review"
        stream_url = request.form.get("stream_url", "").strip()
        use_device = request.form.get("use_device", "").strip()
        device_index = 1
        try:
            device_index = int(request.form.get("device_index", "1") or 1)
        except (TypeError, ValueError):
            device_index = 1

        if use_device in ("1", "on", "true", "yes") or stream_url == "@camera":
            device_stream_url = url_for("cctv_device", index=device_index)
            return render_template(
                "cctv.html",
                firebase_config=get_firebase_config(),
                user_name=session.get("user_name", "Captain"),
                device_mode=True,
                device_index=device_index,
                stream_url=device_stream_url,
                camera_id=camera_id,
                site_zone=site_zone,
                incident_type=incident_type,
                workers=0,
                safe_workers=0,
                violations=0,
                detection_time="Live device feed",
            )

        if stream_url:
            live_stream_url = url_for("cctv_stream", source=stream_url)
            return render_template(
                "cctv.html",
                firebase_config=get_firebase_config(),
                user_name=session.get("user_name", "Captain"),
                stream_url=live_stream_url,
                stream_source=stream_url,
                camera_id=camera_id,
                site_zone=site_zone,
                incident_type=incident_type,
                workers=0,
                safe_workers=0,
                violations=0,
                detection_time="Connected live",
            )

        if "file" not in request.files:
            return render_template("cctv.html", firebase_config=get_firebase_config(), user_name=session.get("user_name", "Captain"), error="No CCTV footage uploaded."), 400

        file = request.files["file"]
        if file.filename == "":
            return render_template("cctv.html", firebase_config=get_firebase_config(), user_name=session.get("user_name", "Captain"), error="Please choose a CCTV clip first."), 400

        os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
        os.makedirs(app.config["RESULT_FOLDER"], exist_ok=True)

        filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(filepath)

        if not Path(filepath).suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".webm"}:
            return render_template(
                "cctv.html",
                firebase_config=get_firebase_config(),
                user_name=session.get("user_name", "Captain"),
                error="Please upload a video clip from a CCTV or camera feed.",
            ), 400

        workers, safe_workers, violations = summarize_video(filepath)
        web_upload_path = _to_static_filename(os.path.join(app.config["UPLOAD_FOLDER"], file.filename))
        uploaded_video_mime = _guess_video_mime(file.filename)
        result_stream_url = url_for("processed_video_feed", filename=os.path.basename(filepath))

        session["workers"] = workers
        session["safe_workers"] = safe_workers
        session["violations"] = violations
        session["detection_time"] = "Live review"
        session["uploaded_image"] = filepath
        session["result_image"] = None
        session["uploaded_media_type"] = "video"
        session["result_media_type"] = "video"

        record = {
            "date": datetime.now().strftime("%d-%m-%Y %I:%M %p"),
            "file": file.filename,
            "workers": workers,
            "safe_workers": safe_workers,
            "violations": violations,
            "compliance": round((safe_workers / workers) * 100, 1) if workers > 0 else 0,
            "status": "SAFE" if violations == 0 else "UNSAFE",
            "source": "CCTV upload",
            "source_type": "video",
            "camera_id": camera_id,
            "site_zone": site_zone,
            "incident_type": incident_type,
        }
        _append_history_record(record)

        return render_template(
            "cctv.html",
            firebase_config=get_firebase_config(),
            user_name=session.get("user_name", "Captain"),
            uploaded_media=web_upload_path,
            uploaded_media_type="video",
            uploaded_video_mime=uploaded_video_mime,
            result_stream_url=result_stream_url,
            workers=workers,
            safe_workers=safe_workers,
            violations=violations,
            detection_time="Live review",
            camera_id=camera_id,
            site_zone=site_zone,
            incident_type=incident_type,
        )

    return render_template(
        "cctv.html",
        firebase_config=get_firebase_config(),
        user_name=session.get("user_name", "Captain"),
    )


@app.route("/cctv/stream")
def cctv_stream():
    source = request.args.get("source", "")
    normalized_source = normalize_stream_source(source)
    if not normalized_source:
        abort(400)

    return Response(
        generate_stream_frames(normalized_source),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/cctv/stream/status")
def cctv_stream_status():
    source = request.args.get("source", "")
    key = normalize_stream_source(source)
    info = STREAM_STATUS.get(key, {})
    if key:
        info = {"key": key, **info}
    return jsonify(info)


@app.route("/cctv/device")
def cctv_device():
    try:
        index = int(request.args.get("index", 1))
    except (TypeError, ValueError):
        index = 1
    index = min(max(index, 0), 9)
    return Response(
        generate_device_frames(f"device:{index}", camera_index=index),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/cctv/device/status")
def cctv_device_status():
    try:
        index = int(request.args.get("index", 1))
    except (TypeError, ValueError):
        index = 1
    index = min(max(index, 0), 9)
    key = f"device:{index}"
    return jsonify({"key": key, **STREAM_STATUS.get(key, {})})


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("home"))


@app.route("/manifest.json")
def manifest():
    return app.send_static_file("manifest.json")


@app.route("/service-worker.js")
def service_worker():
    return app.send_static_file("service-worker.js")


@app.route("/processed_video_feed/<path:filename>")
def processed_video_feed(filename):
    safe_name = os.path.basename(filename)
    video_path = os.path.join(app.config["UPLOAD_FOLDER"], safe_name)

    if not os.path.exists(video_path):
        abort(404)

    return Response(
        generate_frames(video_path),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )



@app.route("/live")
def live():
    return render_template(
        "live.html",
        user_name=session.get("user_name", "Captain"),
        api_base_url=app.config.get("API_BASE_URL", ""),
    )


@app.route("/video_feed")
def video_feed():

    video_path = "construction.mp4"   # We'll improve this later

    return Response(
        generate_frames(video_path),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/webcam_feed")
def webcam_feed():
    camera_index = request.args.get("camera", default=0, type=int)
    return Response(
        generate_camera_frames(camera_index),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/webcam_predict", methods=["POST"])
def webcam_predict():
    data = request.get_data()
    if not data:
        return jsonify({"error": "No frame received"}), 400

    try:
        frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"error": "Invalid image"}), 400
    except Exception as exc:
        return jsonify({"error": f"Invalid image: {exc}"}), 400

    try:
        processed, workers, safe, violations = process_frame(frame)
    except Exception as exc:
        return jsonify({"error": f"Detection failed: {exc}"}), 500

    ok, encoded = cv2.imencode(".jpg", processed)
    if not ok:
        return jsonify({"error": "Encoding failed"}), 500

    return jsonify({
        "image": base64.b64encode(encoded.tobytes()).decode("ascii"),
        "workers": workers,
        "safe": safe,
        "violations": violations,
    })


@app.route("/download_report")
def download_report():

    from flask import send_file
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet
    from datetime import datetime

    filename = "Inspection_Report.pdf"

    doc = SimpleDocTemplate(filename)

    styles = getSampleStyleSheet()

    story = []
    logo_path = "static/logo.jpeg"

    if os.path.exists(logo_path):
        logo = Image(logo_path, width=1.2*inch, height=1.2*inch)
        story.append(logo)

    story.append(Spacer(1,10))

    workers = session.get("workers", 0)
    safe_workers = session.get("safe_workers", 0)
    violations = session.get("violations", 0)
    detection_time = session.get("detection_time", "N/A")

    compliance = 0
    if workers > 0:
        compliance = round((safe_workers / workers) * 100, 1)

    status = "SAFE" if violations == 0 else "UNSAFE"

    recommendation = (
        "Construction site complies with PPE requirements."
        if violations == 0
        else "Safety violations detected. Immediate corrective action is recommended."
    )


    table_data = [
    ["Workers Detected", workers],
    ["Safe Workers", safe_workers],
    ["Violations", violations],
    ["Compliance", f"{compliance}%"],
    ["Site Status", status],
    ["Detection Time", detection_time]
]

    table = Table(table_data, colWidths=[220,180])

    table.setStyle(TableStyle([
    ('BACKGROUND',(0,0),(-1,0),colors.lightblue),
    ('GRID',(0,0),(-1,-1),1,colors.black),
    ('BACKGROUND',(0,0),(0,-1),colors.whitesmoke),
    ('FONTNAME',(0,0),(-1,-1),'Helvetica-Bold'),
    ('BOTTOMPADDING',(0,0),(-1,-1),8),
    ]))

    story.append(table)

    story.append(Spacer(1, 20))
    story.append(Spacer(1,20))
    story.append(Paragraph("<b>Uploaded Image</b>", styles["Heading2"]))

    if os.path.exists(session.get("uploaded_image","")):
        story.append(Image(session["uploaded_image"], width=4*inch, height=3*inch))

    story.append(Paragraph("<b>Recommendation</b>", styles["Heading2"]))
    story.append(Paragraph(recommendation, styles["Normal"]))
    story.append(Spacer(1, 20))
    story.append(Paragraph("<b>AI Detection Result</b>", styles["Heading2"]))

    result = session.get("result_image", "")

    if result and os.path.exists(result):
        story.append(Image(result, width=4*inch, height=3*inch))

    story.append(Spacer(1, 30))

    story.append(Paragraph("<b>Generated by SafeSite AI</b>", styles["Normal"]))

    doc.build(story)

    return send_file(filename, as_attachment=True)
@app.route("/history")
def history():

    records = _load_history()

    total_inspections = len(records)
    total_workers = sum(int(r.get("workers", 0)) for r in records)
    total_safe = sum(int(r.get("safe_workers", 0)) for r in records)
    total_violations = sum(int(r.get("violations", 0)) for r in records)
    safe_count = sum(1 for r in records if r.get("status") == "SAFE")
    unsafe_count = total_inspections - safe_count

    compliance_values = [float(r.get("compliance", 0)) for r in records if r.get("workers")]
    avg_compliance = round(sum(compliance_values) / len(compliance_values), 1) if compliance_values else 0

    return render_template(
        "history.html",
        records=records,
        user_name=session.get("user_name", "Captain"),
        summary={
            "total": total_inspections,
            "workers": total_workers,
            "safe_workers": total_safe,
            "violations": total_violations,
            "safe": safe_count,
            "unsafe": unsafe_count,
            "avg_compliance": avg_compliance,
        },
    )


@app.route("/history/clear", methods=["POST"])
def clear_history():
    with open("history.json", "w") as f:
        json.dump([], f)
    return redirect(url_for("history"))


@app.route("/history/export")
def export_history():
    import csv
    from io import StringIO
    from flask import make_response

    records = _load_history()

    buffer = StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["Date", "File", "Workers", "Safe Workers", "Violations", "Compliance", "Status"])
    for r in records:
        writer.writerow([
            r.get("date", ""),
            r.get("file", ""),
            r.get("workers", 0),
            r.get("safe_workers", 0),
            r.get("violations", 0),
            r.get("compliance", 0),
            r.get("status", ""),
        ])

    response = make_response(buffer.getvalue())
    response.headers["Content-Disposition"] = "attachment; filename=safesite_history.csv"
    response.headers["Content-Type"] = "text/csv; charset=utf-8"
    return response


def _load_history():
    try:
        with open("history.json", "r") as f:
            return json.load(f)
    except Exception:
        return []


@app.route("/analytics")
def analytics():

    records = _load_history()

    total_inspections = len(records)
    total_workers = sum(int(r.get("workers", 0)) for r in records)
    total_safe = sum(int(r.get("safe_workers", 0)) for r in records)
    total_violations = sum(int(r.get("violations", 0)) for r in records)
    safe_count = sum(1 for r in records if r.get("status") == "SAFE")
    unsafe_count = total_inspections - safe_count

    compliance_values = [float(r.get("compliance", 0)) for r in records if r.get("workers")]
    avg_compliance = round(sum(compliance_values) / len(compliance_values), 1) if compliance_values else 0

    trend = {}
    for r in records:
        day = str(r.get("date", "")).split(" ")[0]
        entry = trend.setdefault(day, {"count": 0, "compliance_sum": 0.0, "inspections": 0})
        entry["count"] += 1
        if r.get("workers"):
            entry["compliance_sum"] += float(r.get("compliance", 0))
            entry["inspections"] += 1

    per_file = {}
    for r in records:
        name = r.get("file", "unknown")
        per_file.setdefault(name, {"count": 0, "violations": 0})
        per_file[name]["count"] += 1
        per_file[name]["violations"] += int(r.get("violations", 0))

    top_files = sorted(
        [{"file": k, **v} for k, v in per_file.items()],
        key=lambda x: x["violations"],
        reverse=True,
    )[:6]

    trend_data = [
        {
            "day": k,
            "count": v["count"],
            "compliance": round(v["compliance_sum"] / v["inspections"], 1) if v["inspections"] else 0,
        }
        for k, v in trend.items()
    ]

    insights = {}
    if trend_data:
        best = max(trend_data, key=lambda x: x["compliance"])
        busiest = max(trend_data, key=lambda x: x["count"])
        insights["best_day"] = best["day"]
        insights["best_day_count"] = best["count"]
        insights["busiest_day"] = busiest["day"]
        insights["busiest_day_count"] = busiest["count"]
    if len(trend_data) >= 2:
        first = trend_data[0]["compliance"]
        last = trend_data[-1]["compliance"]
        insights["direction"] = "up" if last > first else ("down" if last < first else "flat")
        insights["delta"] = round(abs(last - first), 1)

    return render_template(
        "analytics.html",
        firebase_config=get_firebase_config(),
        user_name=session.get("user_name", "Captain"),
        metrics={
            "total_inspections": total_inspections,
            "total_workers": total_workers,
            "total_safe": total_safe,
            "total_violations": total_violations,
            "avg_compliance": avg_compliance,
            "safe_count": safe_count,
            "unsafe_count": unsafe_count,
        },
        trend_data=trend_data,
        top_files=top_files,
        insights=insights,
    )


@app.route("/about")
def about():
    return render_template(
        "about.html",
        firebase_config=get_firebase_config(),
        user_name=session.get("user_name", "Captain"),
    )


@app.route("/admin", methods=["GET", "POST"])
def admin():
    error = None

    if request.method == "POST":
        password = request.form.get("password", "")
        if password == ADMIN_PASSWORD:
            session["admin"] = True
            return redirect(url_for("admin"))
        error = "Incorrect admin password. Please try again."

    records = _load_history()
    total_inspections = len(records)
    total_workers = sum(int(r.get("workers", 0)) for r in records)
    total_violations = sum(int(r.get("violations", 0)) for r in records)
    unsafe_count = sum(1 for r in records if r.get("status") != "SAFE")

    storage_bytes = 0
    for folder in (app.config["UPLOAD_FOLDER"], app.config["RESULT_FOLDER"]):
        if os.path.isdir(folder):
            for root, _, files in os.walk(folder):
                for name in files:
                    storage_bytes += os.path.getsize(os.path.join(root, name))
    storage_mb = round(storage_bytes / (1024 * 1024), 1)

    return render_template(
        "admin.html",
        firebase_config=get_firebase_config(),
        user_name=session.get("user_name", "Captain"),
        active_page="admin",
        is_admin=session.get("admin", False),
        error=error,
        stats={
            "inspections": total_inspections,
            "workers": total_workers,
            "violations": total_violations,
            "unsafe": unsafe_count,
            "storage_mb": storage_mb,
        },
        recent=records[-8:][::-1],
        firebase_ready=bool(FIREBASE_CONFIG.get("apiKey") and FIREBASE_CONFIG.get("projectId")),
    )


@app.route("/admin/logout")
def admin_logout():
    session.pop("admin", None)
    return redirect(url_for("admin"))


import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def send_safety_email(to_email, subject, body):
    smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_user = "nithyasheejain@gmail.com"
    smtp_pass = "xaxo ghmh amnz wxtu"

    if not smtp_user or not smtp_pass:
        return False, "SMTP credentials (SMTP_USER & SMTP_PASS) not set in environment (.env). To use real Gmail, add your Gmail address and App Password to .env."

    try:
        msg = MIMEMultipart()
        msg["From"] = smtp_user
        msg["To"] = to_email
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "html"))

        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(smtp_user, smtp_pass)
        server.sendmail(smtp_user, to_email, msg.as_string())
        server.quit()
        return True, "Email sent successfully via Gmail SMTP!"
    except Exception as e:
        return False, f"SMTP Error: {str(e)}"


def send_safety_sms(to_phone, message_text):
    account_sid = os.getenv("TWILIO_ACCOUNT_SID", "")
    auth_token = os.getenv("TWILIO_AUTH_TOKEN", "")
    twilio_number = os.getenv("TWILIO_PHONE", "")

    if not account_sid or not auth_token or not twilio_number:
        return False, "Twilio credentials (TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_PHONE) not set in environment (.env)."

    try:
        url = f"https://api.twilio.com/2010-04-01/Accounts/{account_sid}/Messages.json"
        resp = requests.post(
            url,
            data={"To": to_phone, "From": twilio_number, "Body": message_text},
            auth=(account_sid, auth_token),
            timeout=10
        )
        if resp.status_code in (200, 201):
            return True, "SMS sent successfully via Twilio!"
        else:
            return False, f"Twilio API Error: {resp.text}"
    except Exception as e:
        return False, f"SMS Error: {str(e)}"


from email.mime.base import MIMEBase
from email import encoders

def send_safety_email_with_attachment(to_email, subject, body, pdf_path):
    smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_user = "nithyasheejain@gmail.com"
    smtp_pass = "xaxo ghmh amnz wxtu"

    try:
        msg = MIMEMultipart()
        msg["From"] = smtp_user
        msg["To"] = to_email
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "html"))

        if pdf_path and os.path.exists(pdf_path):
            with open(pdf_path, "rb") as attachment:
                part = MIMEBase("application", "octet-stream")
                part.set_payload(attachment.read())
            encoders.encode_base64(part)
            part.add_header("Content-Disposition", f"attachment; filename= {os.path.basename(pdf_path)}")
            msg.attach(part)

        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(smtp_user, smtp_pass)
        server.sendmail(smtp_user, to_email, msg.as_string())
        server.quit()
        return True, "Scheduled report sent successfully with PDF attachment!"
    except Exception as e:
        return False, f"SMTP Error: {str(e)}"


@app.route("/api/run-scheduled-report", methods=["POST"])
def api_run_scheduled_report():
    data = request.get_json() or {}
    recipient = (data.get("recipient") or session.get("report_recipients") or "nithyasheejain@gmail.com").strip()

    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    from reportlab.lib.styles import getSampleStyleSheet

    filename = "Inspection_Report.pdf"
    doc = SimpleDocTemplate(filename)
    styles = getSampleStyleSheet()
    story = []
    
    logo_path = "static/logo.jpeg"
    if os.path.exists(logo_path):
        story.append(Image(logo_path, width=1.2*inch, height=1.2*inch))
        story.append(Spacer(1, 10))

    story.append(Paragraph("<b>SafeSite AI Scheduled Safety Digest</b>", styles["Title"]))
    story.append(Spacer(1, 10))

    try:
        with open("history.json", "r", encoding="utf-8") as f:
            hist = json.load(f)
    except Exception:
        hist = []

    total_insp = len(hist)
    total_viol = sum(x.get("violations", 0) for x in hist)

    table_data = [
        ["Total Inspections", total_insp],
        ["Total Violations Flagged", total_viol],
        ["Report Status", "Automated Scheduled Digest"]
    ]
    t = Table(table_data, colWidths=[220, 180])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.lightblue),
        ('GRID', (0,0), (-1,-1), 1, colors.black),
        ('FONTNAME', (0,0), (-1,-1), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(t)
    doc.build(story)

    subject = "📊 SafeSite AI Scheduled Safety Inspection Report"
    body = f"<h3>Scheduled Safety Digest</h3><p>Attached is your automated safety inspection report generated by SafeSite AI. Total inspections: <b>{total_insp}</b>, Total violations: <b>{total_viol}</b>.</p>"

    success, msg = send_safety_email_with_attachment(recipient, subject, body, filename)
    return jsonify({"success": success, "message": msg})


@app.route("/api/send-actual-alert", methods=["POST"])
def api_send_actual_alert():
    data = request.get_json() or {}
    phone = (data.get("phone") or session.get("alert_phone", "+15550192834")).strip()
    email = (data.get("email") or session.get("alert_email", "nithyasheejain@gmail.com")).strip()

    subject = "🚨 SafeSite AI CRITICAL HAZARD ALERT: PPE Violation Detected at North Gate"
    body = """
    <div style="font-family: Arial, sans-serif; padding: 20px; background: #f8fafc; border-radius: 10px;">
        <h2 style="color: #dc2626;">🚨 Critical Safety Hazard Detected</h2>
        <p>SafeSite AI automated video analytics has flagged a severe PPE non-compliance incident on site.</p>
        <hr style="border: none; border-top: 1px solid #e2e8f0; margin: 15px 0;" />
        <p><b>Camera Feed:</b> CAM-01 (Main Entrance Gate)</p>
        <p><b>Site Zone:</b> North Perimeter</p>
        <p><b>Violation Type:</b> Missing Hard Hat / Safety Vest</p>
        <p><b>Timestamp:</b> Live Site Inspection</p>
        <p><b>Action Required:</b> Dispatch safety supervisor to zone immediately.</p>
        <br />
        <a href="http://127.0.0.1:5000/dashboard" style="background: #2563eb; color: white; padding: 10px 20px; text-decoration: none; border-radius: 6px; font-weight: bold;">Open SafeSite AI Dashboard</a>
    </div>
    """

    success, email_msg = send_safety_email(email, subject, body)
    sms_success, sms_msg = send_safety_sms(phone, "SafeSite AI CRITICAL ALERT: PPE violation detected at North Gate. Check email immediately.")

    return jsonify({
        "success": success,
        "message": f"Actual hazard alert dispatched to {email}! ({email_msg}) | SMS: {sms_msg}"
    })


@app.route("/api/test-alert", methods=["POST"])
def api_test_alert():
    data = request.get_json() or {}
    phone = (data.get("phone") or session.get("alert_phone", "")).strip()
    email = (data.get("email") or session.get("alert_email", "")).strip()
    test_type = data.get("type", "both")

    results = []
    
    if test_type in ("email", "both") and email:
        success, msg = send_safety_email(
            email,
            "SafeSite AI - Test Safety Alert",
            "<h3>SafeSite AI Emergency Dispatch</h3><p>This is a verified test alert confirming your Gmail SMTP configuration is operational.</p>"
        )
        results.append(f"Email ({email}): {msg}")

    if test_type in ("sms", "both") and phone:
        success, msg = send_safety_sms(
            phone,
            "SafeSite AI: Test safety alert. Your SMS dispatch channel is active."
        )
        results.append(f"SMS ({phone}): {msg}")

    return jsonify({"success": True, "results": results})


@app.route("/heatmap")
def heatmap():
    return render_template(
        "heatmap.html",
        user_name=session.get("user_name", "Captain"),
    )


@app.route("/reports/schedule", methods=["GET", "POST"])
def reports_schedule_page():
    saved = False
    frequency = session.get("report_frequency", "daily")
    recipients = session.get("report_recipients", "nithyasheejain@gmail.com")
    active = session.get("report_active", True)

    if request.method == "POST":
        frequency = request.form.get("frequency", "daily")
        recipients = request.form.get("recipients", "").strip() or recipients
        active = "active" in request.form
        session["report_frequency"] = frequency
        session["report_recipients"] = recipients
        session["report_active"] = active
        saved = True

    return render_template(
        "reports_schedule.html",
        user_name=session.get("user_name", "Captain"),
        frequency=frequency,
        recipients=recipients,
        active=active,
        saved=saved,
    )


@app.route("/alerts", methods=["GET", "POST"])
def alerts_page():
    saved = False
    phone = session.get("alert_phone", "+1 (555) 019-2834")
    email = session.get("alert_email", "nithyasheejain@gmail.com")
    sms_enabled = session.get("sms_enabled", True)
    email_enabled = session.get("email_enabled", True)

    if request.method == "POST":
        phone = request.form.get("phone", "").strip() or phone
        email = request.form.get("email", "").strip() or email
        sms_enabled = "sms_enabled" in request.form
        email_enabled = "email_enabled" in request.form
        session["alert_phone"] = phone
        session["alert_email"] = email
        session["sms_enabled"] = sms_enabled
        session["email_enabled"] = email_enabled
        saved = True

    return render_template(
        "alerts.html",
        user_name=session.get("user_name", "Captain"),
        phone=phone,
        email=email,
        sms_enabled=sms_enabled,
        email_enabled=email_enabled,
        saved=saved,
    )


@app.route("/assistant")
def assistant_page():
    return render_template(
        "assistant.html",
        user_name=session.get("user_name", "Captain"),
    )


def _normalize_assistant_text(text):
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def _assistant_has_phrase(message, phrase):
    normalized_message = f" {_normalize_assistant_text(message)} "
    normalized_phrase = _normalize_assistant_text(phrase)
    if not normalized_phrase:
        return False
    return f" {normalized_phrase} " in normalized_message


def _assistant_matches_any(message, phrases):
    return any(_assistant_has_phrase(message, phrase) for phrase in phrases)


def _assistant_history_stats():
    records = _load_history()
    total_inspections = len(records)
    total_workers = sum(int(item.get("workers", 0)) for item in records)
    total_safe = sum(int(item.get("safe_workers", 0)) for item in records)
    total_violations = sum(int(item.get("violations", 0)) for item in records)
    safe_count = sum(1 for item in records if item.get("status") == "SAFE")
    unsafe_count = total_inspections - safe_count
    compliance_values = [float(item.get("compliance", 0)) for item in records if item.get("workers")]
    avg_compliance = round(sum(compliance_values) / len(compliance_values), 1) if compliance_values else 0
    latest = records[-1] if records else None
    return {
        "records": records,
        "total_inspections": total_inspections,
        "total_workers": total_workers,
        "total_safe": total_safe,
        "total_violations": total_violations,
        "safe_count": safe_count,
        "unsafe_count": unsafe_count,
        "avg_compliance": avg_compliance,
        "latest": latest,
    }


def _assistant_capabilities_text():
    return (
        "I can help with <b>PPE rules</b>, <b>fall protection</b>, <b>scaffolding</b>, "
        "<b>electrical safety</b>, <b>trench safety</b>, <b>fire and emergency response</b>, "
        "<b>CCTV or webcam monitoring</b>, and <b>SafeSite analytics/history summaries</b>."
    )


def _build_assistant_reply(raw_message):
    message = (raw_message or "").strip()
    normalized = _normalize_assistant_text(message)
    stats = _assistant_history_stats()

    if not normalized:
        return (
            "Ask me anything about construction safety, PPE compliance, live monitoring, or your SafeSite inspection history. "
            "For example: <i>What does OSHA say about helmets?</i> or <i>Summarize my site analytics</i>."
        )

    sections = []
    added = set()

    def add_section(key, title, body):
        if key in added:
            return
        added.add(key)
        sections.append(f"<b>{title}</b><br>{body}")

    if _assistant_matches_any(normalized, ["hello", "hi", "hey", "good morning", "good evening"]):
        add_section(
            "greeting",
            "Hello",
            f"Hi {session.get('user_name', 'there')}! {_assistant_capabilities_text()}",
        )

    if _assistant_matches_any(normalized, ["help", "what can you do", "how can you help", "capabilities", "features"]):
        add_section(
            "help",
            "How I Can Help",
            _assistant_capabilities_text() + " You can also ask for a summary of recent inspections, violations, compliance, or the latest uploaded review.",
        )

    if _assistant_matches_any(normalized, ["helmet", "hard hat", "head protection", "ppe", "protective helmet"]):
        add_section(
            "helmet",
            "Helmet And PPE Guidance",
            "Workers should wear hard hats in areas with risk from falling objects, flying debris, or electrical exposure. Use the correct helmet class, inspect it regularly, and replace damaged shells or suspension systems immediately.",
        )

    if _assistant_matches_any(normalized, ["vest", "high vis", "high visibility", "reflective jacket", "reflective vest"]):
        add_section(
            "vest",
            "High-Visibility Clothing",
            "High-visibility garments should be used where workers operate near vehicles, heavy equipment, or low-visibility zones. Choose bright reflective clothing that stays visible during both day and night operations.",
        )

    if _assistant_matches_any(normalized, ["gloves", "hand protection", "goggles", "eye protection", "glasses", "face shield", "boots", "safety shoes", "footwear"]):
        add_section(
            "body-ppe",
            "Additional PPE",
            "Use gloves suited to the task, eye or face protection for grinding, cutting, or chemical exposure, and safety footwear with slip-resistant and impact-resistant features in active construction areas.",
        )

    if _assistant_matches_any(normalized, ["fall", "harness", "height", "roof", "edge", "lifeline", "guardrail"]):
        add_section(
            "fall",
            "Fall Protection",
            "When people work at height, provide guardrails, safety nets, or personal fall arrest systems. Anchor points, harness fit, and rescue planning are just as important as wearing the equipment.",
        )

    if _assistant_matches_any(normalized, ["scaffold", "scaffolding", "platform"]):
        add_section(
            "scaffold",
            "Scaffolding Safety",
            "Scaffolds should be inspected by a competent person, fully planked, properly braced, and used within load limits. Access, guardrails, and stable footing should be checked before every shift.",
        )

    if _assistant_matches_any(normalized, ["electrical", "shock", "lockout", "tagout", "power line", "wiring", "cable"]):
        add_section(
            "electrical",
            "Electrical Safety",
            "Keep temporary wiring protected, maintain safe clearance from power lines, and isolate hazardous energy before maintenance. Damaged cords, wet conditions, and exposed conductors should be treated as immediate risks.",
        )

    if _assistant_matches_any(normalized, ["trench", "excavation", "collapse", "shoring", "soil"]):
        add_section(
            "trench",
            "Excavation And Trench Safety",
            "Excavations can collapse without warning, so trench boxes, shielding, sloping, or shoring should be used where required. Safe access, atmospheric checks, and spoil pile control are also important.",
        )

    if _assistant_matches_any(normalized, ["fire", "emergency", "evacuation", "first aid", "incident", "response"]):
        add_section(
            "emergency",
            "Emergency Preparedness",
            "Sites should maintain marked exits, trained first-aid responders, working extinguishers, and clear escalation steps for incidents. Emergency contacts and reporting procedures should be visible to all crews.",
        )

    if _assistant_matches_any(normalized, ["housekeeping", "slip", "trip", "clean", "debris", "walkway"]):
        add_section(
            "housekeeping",
            "Housekeeping",
            "Good housekeeping reduces slips, trips, and blocked access. Keep walkways clear, store tools safely, clean spills quickly, and remove scrap material from active work zones.",
        )

    if _assistant_matches_any(normalized, ["crane", "lifting", "hoist", "rigging", "load"]):
        add_section(
            "lifting",
            "Lifting Operations",
            "Lifting plans should confirm load weight, rigging condition, lift path, exclusion zones, and clear signaling. Suspended loads must never travel over workers or uncontrolled access areas.",
        )

    if _assistant_matches_any(normalized, ["webcam", "camera", "cctv", "live monitoring", "stream", "video feed"]):
        add_section(
            "monitoring",
            "Live Monitoring",
            "SafeSite AI can review uploaded media, webcam feeds, and CCTV-style streams. Use the live monitoring workspace for real-time checks and the history or analytics pages for after-action review.",
        )

    if _assistant_matches_any(normalized, ["dashboard", "history", "analytics", "report", "summary", "site status", "inspection", "violations", "compliance", "metrics"]):
        latest = stats["latest"]
        latest_text = ""
        if latest:
            latest_text = (
                f" The latest inspection was <b>{html_escape(str(latest.get('file', 'unknown file')))}</b> "
                f"with <b>{int(latest.get('violations', 0))}</b> violations and <b>{float(latest.get('compliance', 0))}%</b> compliance."
            )
        add_section(
            "metrics",
            "Site Metrics",
            f"SafeSite AI has logged <b>{stats['total_inspections']}</b> inspections, reviewed <b>{stats['total_workers']}</b> workers, flagged <b>{stats['total_violations']}</b> violations, and is averaging <b>{stats['avg_compliance']}%</b> compliance. "
            f"<b>{stats['safe_count']}</b> inspections were safe and <b>{stats['unsafe_count']}</b> were unsafe.{latest_text}",
        )

    if _assistant_matches_any(normalized, ["latest", "recent", "last inspection", "most recent"]):
        latest = stats["latest"]
        if latest:
            add_section(
                "latest",
                "Latest Inspection",
                f"Your latest logged inspection is <b>{html_escape(str(latest.get('file', 'unknown file')))}</b> on <b>{html_escape(str(latest.get('date', 'unknown date')))}</b> with status <b>{html_escape(str(latest.get('status', 'UNKNOWN')))}</b>, <b>{int(latest.get('violations', 0))}</b> violations, and <b>{float(latest.get('compliance', 0))}%</b> compliance.",
            )
        else:
            add_section(
                "latest",
                "Latest Inspection",
                "No inspection history is available yet. Upload an image or video first, then I can summarize the latest result.",
            )

    if sections:
        return "<br><br>".join(sections)

    safe_message = html_escape(message[:180])
    return (
        f"Regarding <i>\"{safe_message}\"</i>: As your SafeSite AI safety inspector, I recommend strictly adhering to OSHA guidelines, ensuring all personnel are equipped with proper PPE (helmets, high-vis vests, eye and foot protection), and continuously inspecting active zones via our <b>Detection</b> and <b>CCTV Monitoring</b> modules. If you need specific OSHA standard citations or incident summaries, feel free to ask!"
    )


@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.get_json(silent=True) or {}
    message = data.get("message") or ""
    return jsonify({"reply": _build_assistant_reply(message)})


@app.route("/profile", methods=["GET", "POST"])
def profile():
    saved = False

    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip()
        if name:
            session["user_name"] = name
        if email:
            session["user_email"] = email
        saved = True
        return redirect(url_for("profile"))

    return render_template(
        "profile.html",
        firebase_config=get_firebase_config(),
        user_name=session.get("user_name", "Captain"),
        user_email=session.get("user_email", ""),
        user_uid=session.get("user_uid", "local-user"),
        active_page="profile",
        saved=saved,
    )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
