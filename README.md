# Human Vision Analyzer v2 — Insaan, Haath & Body Shape Detection

Pipeline: **Laptop webcam → YOLOv8-pose (person + 17 body keypoints) → ByteTrack ID → posture/gesture → SQLite → live dashboard**

## Naye features (v2)
- **Haath tracking**: dono haath, 21 landmarks har haath me, Left/Right, ungliyan gintii (0-5)
- **Hand gestures**: Fist, Open palm, Pointing, Peace, Thumbs up, OK, Rock
- **Body shape**: silhouette outline (segmentation) + "Silhouette only" view
- **Body measurements** (pixels): shoulder width, torso, arm length L/R, hands gap, shoulder/hip ratio, silhouette type (V / Straight / Pear)
- Har person ke saath uske haath automatically jude hote hain; sab DB, table, CSV me save

## Features
- Live person detection + unique ID per person (tracking)
- Body skeleton (17 keypoints) overlay (on/off)
- Posture: Standing / Sitting / Lying / Close-up (upper body only)
- Gesture: Hand raised / Both hands up
- Face visible (haan/nahi), time in frame (dwell time)
- Events: ENTER, EXIT, HAND RAISED, LYING / FALL? (3 sec se zyada leta ho)
- Dashboard: live feed, counts, people table (photo ke saath), event log, photo save, CSV export
- 100% local: koi cloud/upload nahi

## v2 upgrade (purane project par)
```
.\.venv\Scripts\python.exe -m pip uninstall -y opencv-python opencv-contrib-python
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```
Purani `vision.db` apne aap naye columns ke saath upgrade ho jati hai.

## Setup (Windows, Python 3.12 recommended)
```
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
python run.py
```
Browser: http://localhost:8000 -> **Start camera**. Pehli baar `yolov8n-pose.pt` auto-download hota hai (internet chahiye).
Dobara chalane ke liye `start.bat` par double-click.

## Tips
- Camera kisi aur app (Zoom/Teams/browser) me chalu na ho.
- Camera index 0 = laptop camera. Webcam alag ho to 1 try karo.
- Roshni achhi rakho, poora upper body dikhe. Laptop camera me aksar sirf upper body dikhti hai, isliye "Close-up" posture normal hai; Standing/Sitting/Lying ke liye door baith/khade hokar poora body frame me lao.
- Tez/halka: `.env` me `IMGSZ=320`, ya zyada accurate: `IMGSZ=640`.

## Limitations (honest)
- Measurements **pixels** me hain (asli cm nahi); camera ki doori/angle par badalte hain. Hips tabhi dikhte hain jab poora body frame me ho.
- Silhouette type sirf shoulder/hip keypoint ratio se nikalta hai, ek rough geometric label hai, medical/fitness measure nahi.
- Haath ka gesture haath camera ki taraf, achhi roshni me hone par best chalta hai.
- Silhouette sab logon ka milkar ek mask hai; overlap karte logon ke outline jud sakte hain.
- Ye kisi ko *pehchanta* (face recognition) nahi, sirf "insaan" detect karta hai. Agar wo frame se bahar jaake wapas aaye to naya ID milta hai.
- Posture/gesture simple keypoint rules par based hain, 100% accurate nahi.
- Age/gender ya identity nikalna is project me jaanbujhkar shamil nahi hai (privacy).
- Sirf apne laptop par, apne ya sahmati wale logon ke saath use karo.

## Structure
```
app/ main.py (FastAPI) | analyzer.py (camera+AI) | database.py
templates/dashboard.html | results/ (snapshots) | run.py | start.bat
```
