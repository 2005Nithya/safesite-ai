import os
import cv2

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

VIDEO_SUMMARY_FRAME_STRIDE = 4
VIDEO_SUMMARY_MAX_FRAMES = 90

# Load model once when the dependency is available.
model = YOLO("best (1).pt") if YOLO else None


def evaluate_person_ppe(person_box, helmets, nonhelmets, vests, overlap_threshold=0.25):
    px1, py1, px2, py2 = person_box
    person_w = max(0, px2 - px1)
    person_h = max(0, py2 - py1)

    helmet_found = False
    vest_found = False

    def overlap_score(box):
        bx1, by1, bx2, by2 = box
        inter_w = max(0, min(px2, bx2) - max(px1, bx1))
        inter_h = max(0, min(py2, by2) - max(py1, by1))
        inter_area = inter_w * inter_h
        box_area = max(1, (bx2 - bx1) * (by2 - by1))
        person_area = max(1, person_w * person_h)
        return inter_area / max(1, min(box_area, person_area))

    helmet_scores = [overlap_score(box) for box in helmets]
    nonhelmet_scores = [overlap_score(box) for box in nonhelmets]
    vest_scores = [overlap_score(box) for box in vests]

    if helmet_scores and max(helmet_scores) >= overlap_threshold:
        helmet_found = True

    if nonhelmet_scores and max(nonhelmet_scores) >= overlap_threshold:
        helmet_found = False

    if vest_scores and max(vest_scores) >= overlap_threshold:
        vest_found = True

    return helmet_found, vest_found


def process_image(image_path):
    frame = cv2.imread(image_path)

    results = model(frame)[0]

    boxes = results.boxes

    persons = []
    helmets = []
    nonhelmets = []
    vests = []

    for box in boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])

        if conf < 0.30:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if cls == 3:
            persons.append((x1, y1, x2, y2))
        elif cls == 1:
            helmets.append((x1, y1, x2, y2))
        elif cls == 2:
            nonhelmets.append((x1, y1, x2, y2))
        elif cls == 5:
            vests.append((x1, y1, x2, y2))

    violations = 0
    safe_workers = 0

    for person in persons:
        px1, py1, px2, py2 = person
        helmet_found, vest_found = evaluate_person_ppe(
            (px1, py1, px2, py2),
            helmets,
            nonhelmets,
            vests,
        )

        # Decide worker status
        if helmet_found and vest_found:
            color = (0, 255, 0)
            label = "SAFE"
            safe_workers += 1
        else:
            color = (255, 0, 0)
            label = "RISK"
            violations += 1

        cv2.rectangle(frame, (px1, py1), (px2, py2), color, 2)

        # OUTLINE FIRST: Thick white background layer
        cv2.putText(
            frame,
            label,
            (px1, max(24, py1 - 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.6,
            (255, 255, 255),
            10,
            cv2.LINE_AA,
        )

        # TEXT SECOND: Main colored label layer on top
        cv2.putText(
            frame,
            label,
            (px1, max(24, py1 - 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.6,
            color,
            4,
            cv2.LINE_AA,
        )

    # Summary on image - Render outlines first, then text on top

    # Workers Count
    cv2.putText(frame, f"Workers : {len(persons)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 0), 10, cv2.LINE_AA)
    cv2.putText(frame, f"Workers : {len(persons)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 0), 4, cv2.LINE_AA)

    # Safe Count
    cv2.putText(frame, f"Safe : {safe_workers}", (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 0), 10, cv2.LINE_AA)
    cv2.putText(frame, f"Safe : {safe_workers}", (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 255, 0), 4, cv2.LINE_AA)

    # Violations Count
    cv2.putText(frame, f"Violations : {violations}", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 0), 10, cv2.LINE_AA)
    cv2.putText(frame, f"Violations : {violations}", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 255), 4, cv2.LINE_AA)

    # Ensure output directory exists so cv2.imwrite succeeds
    os.makedirs(os.path.join("static", "results"), exist_ok=True)

    filename = os.path.basename(image_path)
    result_path = os.path.join("static", "results", filename).replace("\\", "/")

    cv2.imwrite(result_path, frame)

    return result_path, len(persons), safe_workers, violations


def process_frame(frame):

    if model is None:
        raise RuntimeError("YOLO model not loaded")

    results = model(frame)[0]

    persons = []
    helmets = []
    nonhelmets = []
    vests = []

    for box in results.boxes:

        cls = int(box.cls[0])
        conf = float(box.conf[0])

        if conf < 0.30:
            continue

        x1,y1,x2,y2 = map(int, box.xyxy[0])

        if cls == 3:
            persons.append((x1,y1,x2,y2))

        elif cls == 1:
            helmets.append((x1,y1,x2,y2))

        elif cls == 2:
            nonhelmets.append((x1,y1,x2,y2))

        elif cls == 5:
            vests.append((x1,y1,x2,y2))


    violations = 0
    safe_workers = 0


    for person in persons:

        helmet_found, vest_found = evaluate_person_ppe(
            person,
            helmets,
            nonhelmets,
            vests
        )


        px1,py1,px2,py2 = person


        if helmet_found and vest_found:

            color=(0,255,0)
            label="SAFE"
            safe_workers +=1

        else:

            color=(0,0,255)
            label="RISK"
            violations +=1


        cv2.rectangle(
            frame,
            (px1,py1),
            (px2,py2),
            color,
            3
        )


        cv2.putText(
            frame,
            label,
            (px1, max(30,py1-10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            3
        )


    cv2.putText(
        frame,
        f"Workers : {len(persons)}",
        (20,40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255,255,0),
        2
    )


    cv2.putText(
        frame,
        f"Safe : {safe_workers}",
        (20,80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        2
    )


    cv2.putText(
        frame,
        f"Violations : {violations}",
        (20,120),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,0,255),
        2
    )


    return frame, len(persons), safe_workers, violations

def process_video(video_path):

    cap = cv2.VideoCapture(video_path)

    os.makedirs(
        os.path.join("static","results"),
        exist_ok=True
    )

    filename = os.path.splitext(os.path.basename(video_path))[0] + ".mp4"

    output_path=os.path.join(
        "static",
        "results",
        filename
    )


    fps=cap.get(cv2.CAP_PROP_FPS)

    if fps==0:
        fps=30


    width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


    writer=cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width,height)
    )


    max_workers=0
    max_safe=0
    max_violations=0


    while True:

        success,frame=cap.read()

        if not success:
            break


        processed,workers,safe,violations = process_frame(frame)


        writer.write(processed)


        max_workers=max(max_workers,workers)
        max_safe=max(max_safe,safe)
        max_violations=max(max_violations,violations)



    cap.release()
    writer.release()


    return (
        output_path.replace("\\","/"),
        max_workers,
        max_safe,
        max_violations
    )


def summarize_video(
    video_path,
    frame_stride=VIDEO_SUMMARY_FRAME_STRIDE,
    max_frames=VIDEO_SUMMARY_MAX_FRAMES,
):
    cap = cv2.VideoCapture(video_path)

    max_workers = 0
    max_safe = 0
    max_violations = 0
    processed_frames = 0
    frame_index = 0

    while True:
        success, frame = cap.read()

        if not success:
            break

        if frame_index % max(1, frame_stride) != 0:
            frame_index += 1
            continue

        _, workers, safe, violations = process_frame(frame)
        max_workers = max(max_workers, workers)
        max_safe = max(max_safe, safe)
        max_violations = max(max_violations, violations)

        processed_frames += 1
        frame_index += 1

        if processed_frames >= max_frames:
            break

    cap.release()

    return max_workers, max_safe, max_violations

    
