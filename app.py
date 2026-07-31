from flask import Flask, abort, jsonify, redirect, render_template, request, session, url_for
from pathlib import Path
import os
import requests
import time
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet
import json
from datetime import datetime
from flask import Response
from video_detection import generate_camera_frames, generate_frames
from flask import Response
import cv2
from detection import process_frame
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

UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["RESULT_FOLDER"] = RESULT_FOLDER


def _to_static_filename(path):
    normalized = (path or "").replace("\\", "/").strip("/")
    if normalized.startswith("static/"):
        return normalized[len("static/"):]
    return normalized


def _guess_video_mime(filename):
    extension = Path(filename or "").suffix.lower()
    return {
        ".mp4": "video/mp4",
        ".webm": "video/webm",
        ".mov": "video/quicktime",
        ".avi": "video/x-msvideo",
        ".mkv": "video/x-matroska",
    }.get(extension, "video/mp4")


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
    history_file = "history.json"

    record = {
    "date": datetime.now().strftime("%d-%m-%Y %I:%M %p"),
    "file": file.filename,
    "workers": workers,
    "safe_workers": safe_workers,
    "violations": violations,
    "compliance": round((safe_workers/workers)*100,1) if workers > 0 else 0,
    "status": "SAFE" if violations == 0 else "UNSAFE"
    }

    try:
        with open(history_file, "r") as f:
            history = json.load(f)
    except:
            history = []

    history.append(record)

    with open(history_file, "w") as f:
        json.dump(history, f, indent=4)

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
    return render_template("live.html", user_name=session.get("user_name", "Captain"))


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
    except Exception:
        return jsonify({"error": "Invalid image"}), 400

    processed, workers, safe, violations = process_frame(frame)

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
    )


@app.route("/about")
def about():
    return render_template(
        "about.html",
        firebase_config=get_firebase_config(),
        user_name=session.get("user_name", "Captain"),
    )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
