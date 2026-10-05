import csv, io, os, time
from fastapi import FastAPI, Form, Request
from fastapi.responses import StreamingResponse, HTMLResponse, Response, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import func
from .database import init_db, SessionLocal, Person, Event, person_dict, fmt
from .analyzer import analyzer
from .hands import MP_OK, MP_ERR

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(f"{BASE}/results/snapshots", exist_ok=True)
app = FastAPI(title="Human Vision Analyzer")
app.mount("/media", StaticFiles(directory=f"{BASE}/results"), name="media")
templates = Jinja2Templates(directory=f"{BASE}/templates")
init_db()


@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    return templates.TemplateResponse(request, "dashboard.html")


@app.post("/api/start")
def start(camera: str = Form("0")):
    analyzer.start(camera.strip() or "0")
    return {"ok": True}


@app.post("/api/stop")
def stop():
    analyzer.stop()
    return {"ok": True}


@app.post("/api/settings")
def settings(skeleton: bool = Form(True), mirror: bool = Form(True), hands: bool = Form(True),
             shape: bool = Form(True), view: str = Form("normal")):
    analyzer.show_skeleton, analyzer.mirror = skeleton, mirror
    analyzer.show_hands, analyzer.show_shape = hands, shape
    analyzer.view = view if view in ("normal", "silhouette") else "normal"
    return {"ok": True}


def mjpeg():
    while True:
        with analyzer.lock:
            f = analyzer.latest
        if f:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + f + b"\r\n"
        time.sleep(0.03)


@app.get("/video_feed")
def video_feed():
    return StreamingResponse(mjpeg(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/snapshot")
def snapshot():
    with analyzer.lock:
        f = analyzer.latest
    if not f:
        return JSONResponse({"detail": "Camera chalu nahi hai"}, status_code=400)
    return Response(f, media_type="image/jpeg",
                    headers={"Content-Disposition": f"attachment; filename=snapshot_{int(time.time())}.jpg"})


@app.get("/api/stats")
def stats():
    db = SessionLocal()
    try:
        total = db.query(func.count(Person.id)).scalar()
        avg = db.query(func.avg(Person.dwell_seconds)).scalar() or 0
        alerts = db.query(func.count(Event.id)).filter(Event.event_type.notin_(["ENTER", "EXIT"])).scalar()
        postures = {}
        for p in analyzer.live:
            postures[p["posture"]] = postures.get(p["posture"], 0) + 1
        return {"status": analyzer.status, "running": analyzer.running, "fps": round(analyzer.fps, 1),
                "present": len(analyzer.live), "total": total, "avg_dwell": round(avg, 1),
                "alerts": alerts, "postures": postures, "live": analyzer.live,
                "hands_now": len(analyzer.hands_now),
                "features": {"mediapipe": MP_OK, "note": "" if MP_OK else "mediapipe install nahi: hands/shape band (requirements install karo)"}}
    finally:
        db.close()


@app.get("/api/people")
def people(limit: int = 50):
    db = SessionLocal()
    try:
        return [person_dict(p) for p in db.query(Person).order_by(Person.id.desc()).limit(limit).all()]
    finally:
        db.close()


@app.get("/api/events")
def events(limit: int = 40):
    db = SessionLocal()
    try:
        return [{"ts": fmt(e.ts), "type": e.event_type, "detail": e.detail}
                for e in db.query(Event).order_by(Event.id.desc()).limit(limit).all()]
    finally:
        db.close()


@app.get("/api/export.csv")
def export_csv():
    db = SessionLocal()
    try:
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["person_id", "posture", "gesture", "hands", "silhouette", "shoulder_px", "torso_px", "face_visible", "dwell_sec", "first_seen", "last_seen"])
        for p in db.query(Person).order_by(Person.id).all():
            d = person_dict(p)
            w.writerow([d["id"], d["posture"], d["gesture"], d["hands"], d["silhouette"], d["shoulder"], d["torso"], d["face"], d["dwell"], d["first_seen"], d["last_seen"]])
        return Response(buf.getvalue(), media_type="text/csv",
                        headers={"Content-Disposition": "attachment; filename=people_report.csv"})
    finally:
        db.close()


@app.post("/api/reset")
def reset():
    analyzer.stop()
    db = SessionLocal()
    db.query(Event).delete(); db.query(Person).delete(); db.commit(); db.close()
    return {"ok": True}
