"""Webcam -> YOLOv8-pose (person+skeleton) + MediaPipe (haath, silhouette) -> DB.
Display thread (smooth video) aur AI thread alag chalte hain."""
import os, threading, time, traceback, uuid
from datetime import datetime
import cv2
import numpy as np
from ultralytics import YOLO
from .database import SessionLocal, Person, Event
from .hands import HandTracker, MP_OK, HAND_CONN
from .shape import ShapeTracker, measure

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
SKELETON = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13),
            (13, 15), (12, 14), (14, 16), (0, 1), (0, 2), (1, 3), (2, 4)]
COL = {"Standing": (80, 200, 80), "Sitting": (255, 170, 40), "Lying": (60, 60, 230), "Close-up": (200, 120, 200)}
LOG_GESTURES = {"Thumbs up", "Peace", "OK", "Rock", "Pointing", "Open palm"}


def classify_posture(kp, box):
    x1, y1, x2, y2 = box
    if (x2 - x1) / max(y2 - y1, 1) > 1.1:
        return "Lying"
    ok = lambda i: kp[i][2] > 0.4
    sh = [kp[i][1] for i in (5, 6) if ok(i)]
    hp = [kp[i][1] for i in (11, 12) if ok(i)]
    kn = [kp[i][1] for i in (13, 14) if ok(i)]
    if sh and hp and kn:
        return "Sitting" if abs(np.mean(kn) - np.mean(hp)) < 0.6 * abs(np.mean(hp) - np.mean(sh)) else "Standing"
    return "Close-up"


def classify_gesture(kp):
    sh = [kp[i][1] for i in (5, 6) if kp[i][2] > 0.4]
    if not sh:
        return "-"
    ref = float(np.mean(sh))
    up = [i for i in (9, 10) if kp[i][2] > 0.4 and kp[i][1] < ref]
    return "Both hands up" if len(up) == 2 else ("Hand raised" if up else "-")


class Analyzer:
    def __init__(self):
        self.model = None
        self.hand_tracker = None
        self.shape_tracker = None
        self.lock = threading.Lock()
        self._raw_lock = threading.Lock()
        self.latest = None
        self.running = False
        self.thread = None
        self.status = "idle"
        self.fps = 0.0
        self.live = []
        self._dets, self._hands, self._polys = [], [], []
        self.hands_now = []
        self._raw = None
        self.run_id = ""
        self.show_skeleton = True
        self.show_hands = True
        self.show_shape = True
        self.view = "normal"
        self.mirror = True

    def start(self, camera):
        self.stop()
        self.running = True
        self.run_id = uuid.uuid4().hex[:8]
        self.latest, self.live, self._dets, self._hands, self._polys, self.hands_now = None, [], [], [], [], []
        self.thread = threading.Thread(target=self._loop, args=(camera,), daemon=True)
        self.thread.start()

    @staticmethod
    def _open_camera(camera):
        if not str(camera).isdigit():
            cap = cv2.VideoCapture(camera)
            return cap if cap.isOpened() else None
        idx = int(camera)
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY] if os.name == "nt" else [cv2.CAP_ANY]
        for i in [idx] + [j for j in range(4) if j != idx]:
            for b in backends:
                cap = cv2.VideoCapture(i, b)
                if cap.isOpened():
                    ok = False
                    for _ in range(5):
                        ok, _f = cap.read()
                        if ok:
                            break
                        time.sleep(0.1)
                    if ok:
                        print(f"[camera] index {i}, backend {b} se camera khula")
                        return cap
                cap.release()
        return None

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=10)
        self.status = "idle"
        self.live, self.hands_now = [], []

    # ---------------- display thread ----------------
    def _loop(self, camera):
        ai = None
        try:
            self.status = "model load ho raha hai..."
            self.model = YOLO(os.getenv("POSE_MODEL", "yolov8n-pose.pt"))
            if MP_OK and self.hand_tracker is None:
                self.hand_tracker, self.shape_tracker = HandTracker(), ShapeTracker()
            cap = self._open_camera(camera)
            if cap is None:
                self.status = ("error: koi camera nahi khula. Windows Settings > Privacy & security > Camera me "
                               "'Let desktop apps access your camera' ON karo, aur Zoom/Teams/browser me camera band karo")
                return
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            max_w = int(os.getenv("MAX_WIDTH", "960"))
            self._raw = None
            self.status = "running"
            ai = threading.Thread(target=self._infer, daemon=True)
            ai.start()
            while self.running:
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.02)
                    continue
                if self.mirror:
                    frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]
                if w > max_w:
                    frame = cv2.resize(frame, (max_w, int(h * max_w / w)))
                with self._raw_lock:
                    self._raw = (time.time(), frame.copy())
                self._draw(frame)
                ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ok:
                    with self.lock:
                        self.latest = jpg.tobytes()
            cap.release()
            self.running = False
            if ai:
                ai.join(timeout=10)
            if not self.status.startswith("error"):
                self.status = "stopped"
        except Exception as e:  # noqa
            traceback.print_exc()
            self.status = f"error: {e}"
        finally:
            self.running = False

    def _draw(self, frame):
        polys, dets, hands = self._polys, self._dets, self._hands
        sil_view = self.view == "silhouette"
        if sil_view:
            frame[:] = (36, 28, 22)
        if self.show_shape and polys:
            if not sil_view:
                ov = frame.copy()
                cv2.fillPoly(ov, polys, (190, 170, 0))
                cv2.addWeighted(ov, 0.35, frame, 0.65, 0, frame)
            else:
                cv2.fillPoly(frame, polys, (170, 150, 40))
            cv2.polylines(frame, polys, True, (255, 220, 0), 2, cv2.LINE_AA)
        for (box, label, col, kp) in dets:
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), col, 1 if sil_view else 2)
            if self.show_skeleton:
                for a, b in SKELETON:
                    if kp[a][2] > 0.4 and kp[b][2] > 0.4:
                        cv2.line(frame, (int(kp[a][0]), int(kp[a][1])), (int(kp[b][0]), int(kp[b][1])), (0, 255, 255), 2)
                for x, y, c in kp:
                    if c > 0.4:
                        cv2.circle(frame, (int(x), int(y)), 3, (0, 0, 255), -1)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, max(y1 - th - 8, 0)), (x1 + tw + 6, max(y1, th + 8)), col, -1)
            cv2.putText(frame, label, (x1 + 3, max(y1 - 5, th + 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1, cv2.LINE_AA)
        if self.show_hands:
            for hd in hands:
                p = hd["pts"].astype(int)
                for a, b in HAND_CONN:
                    cv2.line(frame, tuple(p[a]), tuple(p[b]), (60, 255, 60), 2, cv2.LINE_AA)
                for q in p:
                    cv2.circle(frame, tuple(q), 4, (0, 0, 255), -1)
                txt = f"{hd['label']} {hd['gesture']} ({hd['fingers']})"
                x, y = int(p[0][0]), int(p[0][1])
                (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                cv2.rectangle(frame, (x - 4, y + 6), (x + tw + 4, y + th + 14), (0, 120, 0), -1)
                cv2.putText(frame, txt, (x, y + th + 9), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"People: {len(self.live)}  Hands: {len(hands)}  AI FPS: {self.fps:.1f}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

    # ---------------- AI thread ----------------
    def _event(self, db, pid, etype, detail):
        db.add(Event(person_id=pid, event_type=etype, detail=detail, ts=datetime.now()))

    def _infer(self):
        db = SessionLocal()
        conf = float(os.getenv("CONF_THRESHOLD", "0.4"))
        imgsz = int(os.getenv("IMGSZ", "480"))
        tracks, last_ts, n, t0, last_flush = {}, 0, 0, time.time(), 0
        try:
            while self.running:
                with self._raw_lock:
                    item = self._raw
                if item is None or item[0] == last_ts:
                    time.sleep(0.005)
                    continue
                last_ts, frame = item
                n += 1
                r = self.model.track(frame, persist=True, tracker="bytetrack.yaml", classes=[0],
                                     conf=conf, imgsz=imgsz, verbose=False)[0]
                persons = []
                if r.boxes is not None and r.boxes.id is not None and r.keypoints is not None:
                    for box, tid, kp in zip(r.boxes.xyxy.int().tolist(), r.boxes.id.int().tolist(),
                                            r.keypoints.data.cpu().numpy()):
                        persons.append((tid, [max(0, v) for v in box], kp))
                polys, fills, hands = [], [], []
                try:
                    if self.show_shape and self.shape_tracker:
                        polys, fills = self.shape_tracker.process(frame, [p[1] for p in persons])
                    if self.show_hands and self.hand_tracker:
                        hands = self.hand_tracker.process(frame, self.mirror)
                except Exception as ex:
                    print("[mediapipe] skip:", ex)
                now, dets, live = time.time(), [], []
                flush = now - last_flush > 1.0
                for i, (tid, (x1, y1, x2, y2), kp) in enumerate(persons):
                    t = tracks.get(tid)
                    if t is None:
                        p = Person(run_id=self.run_id, track_id=tid)
                        db.add(p); db.flush()
                        t = tracks[tid] = {"pid": p.id, "first": now, "last": now, "area": 0, "lying": None,
                                           "lying_alert": False, "hand_ts": 0, "left": False, "hlog": {}}
                        self._event(db, p.id, "ENTER", f"Person #{p.id} aaya")
                    t["last"], t["left"] = now, False
                    posture, gesture = classify_posture(kp, (x1, y1, x2, y2)), classify_gesture(kp)
                    face = bool(kp[0][2] > 0.5 and kp[1][2] > 0.5 and kp[2][2] > 0.5)
                    dwell = now - t["first"]
                    m = measure(kp, (x1, y1, x2, y2))
                    m["fill"] = round(fills[i] * 100) if i < len(fills) else None
                    mine = [h for h in hands if x1 <= h["wrist"][0] <= x2 and y1 <= h["wrist"][1] <= y2]
                    hsum = " | ".join(f"{h['label'][0]}: {h['gesture']} ({h['fingers']})" for h in mine) or "-"
                    for h in mine:                                           # haath gesture events
                        key = (h["label"], h["gesture"])
                        if h["gesture"] in LOG_GESTURES and now - t["hlog"].get(key, 0) > 4:
                            t["hlog"][key] = now
                            self._event(db, t["pid"], "HAND GESTURE", f"Person #{t['pid']} {h['label']} haath: {h['gesture']}")
                    crop, area = frame[y1:y2, x1:x2], (x2 - x1) * (y2 - y1)
                    snap = None
                    if crop.size and area > t["area"] * 1.5:
                        t["area"] = area
                        snap = f"snapshots/{self.run_id}_{tid}.jpg"
                        cv2.imwrite(os.path.join(RESULTS, snap), crop)
                    if posture == "Lying":
                        t["lying"] = t["lying"] or now
                        if now - t["lying"] > 3 and not t["lying_alert"]:
                            t["lying_alert"] = True
                            self._event(db, t["pid"], "LYING / FALL?", f"Person #{t['pid']} 3 sec se zyada leta hua")
                    else:
                        t["lying"], t["lying_alert"] = None, False
                    if gesture != "-" and now - t["hand_ts"] > 5:
                        t["hand_ts"] = now
                        self._event(db, t["pid"], "HAND RAISED", f"Person #{t['pid']}: {gesture}")
                    if flush or snap:
                        vals = dict(posture=posture, gesture=gesture, face_visible=face, dwell_seconds=dwell,
                                    last_seen=datetime.now(), present=True, hands_summary=hsum,
                                    shoulder_w=m["shoulder"], torso_len=m["torso"], silhouette=m["silhouette"],
                                    fill_pct=m["fill"])
                        if snap:
                            vals["snapshot"] = snap
                        db.query(Person).filter(Person.id == t["pid"]).update(vals)
                    short = " ".join(f"{h['label'][0]}:{h['gesture']}" for h in mine)
                    label = f"#{t['pid']} {posture}" + (f" | {gesture}" if gesture != "-" else "") + (f" | {short}" if short else "")
                    dets.append(((x1, y1, x2, y2), label, COL.get(posture, (255, 255, 255)), kp))
                    live.append({"id": t["pid"], "posture": posture, "gesture": gesture, "face": face,
                                 "dwell": round(dwell, 1), "hands": hsum, "shape": m})
                for tid, t in tracks.items():
                    if not t["left"] and now - t["last"] > 2.5:
                        t["left"] = True
                        db.query(Person).filter(Person.id == t["pid"]).update(
                            {"present": False, "dwell_seconds": t["last"] - t["first"], "last_seen": datetime.now()})
                        self._event(db, t["pid"], "EXIT", f"Person #{t['pid']} gaya")
                if flush:
                    last_flush = now
                db.commit()
                self._dets, self._hands, self._polys, self.live, self.hands_now = dets, hands, polys, live, hands
                self.fps = n / max(time.time() - t0, 1e-6)
            db.query(Person).filter(Person.run_id == self.run_id).update({"present": False})
            db.commit()
        except Exception as e:  # noqa
            traceback.print_exc()
            self.status = f"error: {e}"
            self.running = False
        finally:
            db.close()


analyzer = Analyzer()
