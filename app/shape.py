"""Body shape: silhouette (segmentation) + keypoint measurements (pixel / ratio based)."""
import cv2
import numpy as np

try:
    import mediapipe as mp
    MP_OK = True
except Exception:  # noqa
    mp, MP_OK = None, False


class ShapeTracker:
    def __init__(self):
        self.seg = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)

    def process(self, frame, boxes):
        """returns (silhouette polygons [full-frame coords], fill ratio per box)"""
        h, w = frame.shape[:2]
        sw = 320
        sh = max(int(h * sw / w), 1)
        small = cv2.cvtColor(cv2.resize(frame, (sw, sh)), cv2.COLOR_BGR2RGB)
        small.flags.writeable = False
        m = self.seg.process(small).segmentation_mask
        b = cv2.morphologyEx((m > 0.55).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        cnts, _ = cv2.findContours(b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        sx, sy = w / sw, h / sh
        polys = []
        for c in cnts:
            if cv2.contourArea(c) < 0.004 * sw * sh:
                continue
            c = cv2.approxPolyDP(c, 1.5, True).reshape(-1, 2)
            polys.append((c * np.array([sx, sy])).astype(np.int32))
        fills = []
        for (x1, y1, x2, y2) in boxes:
            a = b[int(y1 / sy):max(int(y2 / sy), int(y1 / sy) + 1), int(x1 / sx):max(int(x2 / sx), int(x1 / sx) + 1)]
            fills.append(float(a.mean()) if a.size else 0.0)
        return polys, fills


def measure(kp, box):
    """Keypoints se body ke measurements (pixels me; camera angle/doori par depend karte hain)."""
    pt = lambda i: np.array(kp[i][:2], dtype=float) if kp[i][2] > 0.4 else None
    dist = lambda a, b: float(np.linalg.norm(a - b)) if a is not None and b is not None else None
    ls, rs, lh, rh, le, re, lw, rw = [pt(i) for i in (5, 6, 11, 12, 7, 8, 9, 10)]
    shoulder, hip = dist(ls, rs), dist(lh, rh)
    torso = dist((ls + rs) / 2, (lh + rh) / 2) if all(v is not None for v in (ls, rs, lh, rh)) else None

    def arm(s, e, w):
        a, b = dist(s, e), dist(e, w)
        return a + b if a is not None and b is not None else None

    ratio = shoulder / hip if shoulder and hip else None
    sil = "N/A (hips dikh nahi rahe)"
    if ratio:
        sil = "V-shape (kandhe > hips)" if ratio > 1.25 else ("Straight (H-shape)" if ratio >= 0.9 else "Pear (hips > kandhe)")
    r = lambda v: round(v) if v is not None else None
    return {"height": round(box[3] - box[1]), "shoulder": r(shoulder), "hip": r(hip), "torso": r(torso),
            "left_arm": r(arm(ls, le, lw)), "right_arm": r(arm(rs, re, rw)), "hands_gap": r(dist(lw, rw)),
            "ratio": round(ratio, 2) if ratio else None, "silhouette": sil, "fill": None}
