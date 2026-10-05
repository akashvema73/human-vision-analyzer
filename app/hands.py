"""Haath detection: MediaPipe Hands (21 landmarks) + ungli gintii + gesture."""
import cv2
import numpy as np

try:
    import mediapipe as mp
    MP_OK, MP_ERR = True, ""
    HAND_CONN = [(a, b) for a, b in mp.solutions.hands.HAND_CONNECTIONS]
except Exception as e:  # noqa
    mp, MP_OK, MP_ERR, HAND_CONN = None, False, str(e), []


def finger_states(p):
    d = lambda a, b: float(np.linalg.norm(p[a] - p[b]))
    st = {"thumb": d(4, 17) > d(3, 17) * 1.05}               # thumb: tip pinky-base se door
    for name, tip, pip in (("index", 8, 6), ("middle", 12, 10), ("ring", 16, 14), ("pinky", 20, 18)):
        st[name] = d(tip, 0) > d(pip, 0) * 1.2               # tip wrist se pip ke mukable door
    return st


def classify(p, st):
    f = [st["index"], st["middle"], st["ring"], st["pinky"]]
    th, n = st["thumb"], sum(f) + st["thumb"]
    palm = float(np.linalg.norm(p[0] - p[9])) + 1e-6
    if float(np.linalg.norm(p[4] - p[8])) < 0.3 * palm and f[1] and f[2] and f[3]:
        return "OK", n
    if n == 0:
        return "Fist", 0
    if n == 5:
        return "Open palm", 5
    if f == [True, False, False, False] and not th:
        return "Pointing", 1
    if f == [True, True, False, False] and not th:
        return "Peace", 2
    if th and not any(f):
        return ("Thumbs up" if p[4][1] < p[0][1] else "Thumb out"), 1
    if f == [True, False, False, True]:
        return "Rock", n
    return f"{n} fingers", n


class HandTracker:
    def __init__(self):
        self.h = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2, model_complexity=0,
                                          min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def process(self, frame, mirrored):
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        res = self.h.process(rgb)
        out = []
        if not res.multi_hand_landmarks:
            return out
        for lm, hd in zip(res.multi_hand_landmarks, res.multi_handedness):
            pts = np.array([(q.x * w, q.y * h) for q in lm.landmark], dtype=np.float32)
            label = hd.classification[0].label                # mirrored (selfie) image ke hisaab se
            if not mirrored:
                label = "Left" if label == "Right" else "Right"
            g, n = classify(pts, finger_states(pts))
            out.append({"label": label, "gesture": g, "fingers": n, "pts": pts,
                        "wrist": (float(pts[0][0]), float(pts[0][1]))})
        return out
