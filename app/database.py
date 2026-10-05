import os
from datetime import datetime
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, inspect, text
from sqlalchemy.orm import declarative_base, sessionmaker

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./vision.db")
engine = create_engine(DATABASE_URL, pool_pre_ping=True,
                       connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {})
SessionLocal = sessionmaker(bind=engine, autoflush=False)
Base = declarative_base()


class Person(Base):
    __tablename__ = "persons"
    id = Column(Integer, primary_key=True)
    run_id = Column(String(16), index=True)
    track_id = Column(Integer)
    posture = Column(String(20), default="-")
    gesture = Column(String(20), default="-")
    face_visible = Column(Boolean, default=False)
    dwell_seconds = Column(Float, default=0)
    snapshot = Column(String(200), nullable=True)
    first_seen = Column(DateTime, default=datetime.now)
    last_seen = Column(DateTime, default=datetime.now)
    present = Column(Boolean, default=True)
    hands_summary = Column(String(80), default="-")
    shoulder_w = Column(Float, nullable=True)
    torso_len = Column(Float, nullable=True)
    silhouette = Column(String(40), nullable=True)
    fill_pct = Column(Float, nullable=True)


class Event(Base):
    __tablename__ = "events"
    id = Column(Integer, primary_key=True)
    ts = Column(DateTime, default=datetime.now, index=True)
    person_id = Column(Integer)
    event_type = Column(String(30), index=True)
    detail = Column(String(120))


def init_db():
    Base.metadata.create_all(engine)
    insp = inspect(engine)                       # purani DB me naye columns auto add
    for tbl in Base.metadata.sorted_tables:
        have = {c["name"] for c in insp.get_columns(tbl.name)}
        for col in tbl.columns:
            if col.name not in have:
                with engine.begin() as c:
                    c.execute(text(f"ALTER TABLE {tbl.name} ADD COLUMN {col.name} {col.type.compile(engine.dialect)}"))


def fmt(d):
    return d.strftime("%d-%m-%Y %H:%M:%S") if d else None


def person_dict(p):
    return {"id": p.id, "track_id": p.track_id, "posture": p.posture, "gesture": p.gesture,
            "face": bool(p.face_visible), "dwell": round(p.dwell_seconds or 0, 1), "snapshot": p.snapshot,
            "first_seen": fmt(p.first_seen), "last_seen": fmt(p.last_seen), "present": bool(p.present),
            "hands": p.hands_summary or "-", "shoulder": p.shoulder_w, "torso": p.torso_len,
            "silhouette": p.silhouette or "-", "fill": p.fill_pct}
