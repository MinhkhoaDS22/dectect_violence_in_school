"""
main.py — SafeWatch FastAPI Backend
Chạy: uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

import os
import uuid
import json
import shutil
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

load_dotenv()

from violence_detector import load_models, analyze_video
from video_clipper import clip_all_segments
from notifier import send_gmail, send_telegram

# ==========================================
# Config
# ==========================================
MODEL_PATH = os.getenv("MODEL_PATH", r"D:\DATN\results\best_model.pth")
YOLO_PATH  = os.getenv("YOLO_PATH",  "../yolo11s.pt")
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "./uploads"))
CLIPS_DIR  = Path(os.getenv("CLIPS_DIR",  "./clips"))
ALERTS_DB  = Path(os.getenv("ALERTS_DB",  "./alerts.json"))

UPLOAD_DIR.mkdir(exist_ok=True)
CLIPS_DIR.mkdir(exist_ok=True)

# ==========================================
# Lifespan: load models once on startup
# ==========================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- startup ---
    model_abs = str(Path(MODEL_PATH).resolve())
    yolo_abs  = str(Path(YOLO_PATH).resolve())
    print("\U0001f504 Loading models...")
    print(f"   Model : {model_abs}")
    print(f"   YOLO  : {yolo_abs}")
    load_models(model_abs, yolo_abs)
    print("\U0001f680 SafeWatch API ready!")
    yield
    # --- shutdown (nếu cần cleanup) ---


# ==========================================
# FastAPI App
# ==========================================
app = FastAPI(title="SafeWatch API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
# Alerts DB helpers (JSON file)
# ==========================================
def _load_alerts() -> list:
    if ALERTS_DB.exists():
        with open(ALERTS_DB, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def _save_alert(alert: dict):
    alerts = _load_alerts()
    alerts.insert(0, alert)          # newest first
    with open(ALERTS_DB, "w", encoding="utf-8") as f:
        json.dump(alerts, f, ensure_ascii=False, indent=2)





# ==========================================
# Routes
# ==========================================
@app.get("/api/health")
def health():
    return {"status": "ok", "version": "1.0.0"}


@app.post("/api/analyze")
async def analyze(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    email: Optional[str] = Form(None),
    phone: Optional[str] = Form(None),
    telegram_chat_id: Optional[str] = Form(None),
):
    """
    Nhận video upload, chạy detection, cắt clip, gửi thông báo.

    Form fields:
      - video: file video
      - email: địa chỉ Gmail nhận cảnh báo (tuỳ chọn)
      - phone: số điện thoại (hiển thị trong thông báo, tuỳ chọn)
      - telegram_chat_id: Telegram chat_id (tuỳ chọn)

    Ít nhất 1 trong [email, telegram_chat_id] phải được điền.
    """
    if not email and not telegram_chat_id:
        raise HTTPException(422, "Phải có ít nhất email hoặc Telegram chat_id.")

    # Lưu video upload
    job_id = str(uuid.uuid4())[:8]
    suffix = Path(video.filename).suffix or ".mp4"
    vid_path = UPLOAD_DIR / f"{job_id}{suffix}"

    with open(vid_path, "wb") as f:
        shutil.copyfileobj(video.file, f)

    # Chạy detection
    try:
        result = analyze_video(str(vid_path))
    except Exception as e:
        raise HTTPException(500, f"Lỗi phân tích video: {e}")

    # Cắt clip nếu phát hiện bạo lực
    clip_paths = []
    notification_status = {"email": None, "telegram": None}

    if result['is_violence'] and result['segments']:
        clip_dir = CLIPS_DIR / job_id
        clip_paths = clip_all_segments(str(vid_path), result['segments'], str(clip_dir))

        # Gửi thông báo
        if email:
            ok = send_gmail(email, result, video.filename, clip_paths)
            notification_status["email"] = "sent" if ok else "failed"

        if telegram_chat_id:
            ok = send_telegram(telegram_chat_id, result, video.filename, clip_paths)
            notification_status["telegram"] = "sent" if ok else "failed"

    # Lưu alert vào DB
    clip_names = [Path(p).name for p in clip_paths]
    alert = {
        "id": job_id,
        "timestamp": datetime.now().isoformat(),
        "video_filename": video.filename,
        "email": email,
        "phone": phone,
        "telegram_chat_id": telegram_chat_id,
        "is_violence": result['is_violence'],
        "segments": result['segments'],
        "video_duration": result['video_duration'],
        "violence_ratio": result['violence_ratio'],
        "max_violent_persons": result.get('max_violent_persons', 0),
        "summary": result['summary'],
        "clips": clip_names,
        "notification_status": notification_status,
    }
    _save_alert(alert)

    return {
        "job_id": job_id,
        "result": result,
        "clips": clip_names,
        "notification_status": notification_status,
    }


@app.get("/api/alerts")
def get_alerts(limit: int = 50, offset: int = 0):
    """Lấy danh sách cảnh báo, sắp xếp mới nhất trước."""
    alerts = _load_alerts()
    return {
        "total": len(alerts),
        "alerts": alerts[offset: offset + limit],
    }


@app.get("/api/alerts/{alert_id}")
def get_alert(alert_id: str):
    alerts = _load_alerts()
    for a in alerts:
        if a["id"] == alert_id:
            return a
    raise HTTPException(404, "Alert không tồn tại.")


@app.delete("/api/alerts/{alert_id}")
def delete_alert(alert_id: str):
    alerts = _load_alerts()
    new = [a for a in alerts if a["id"] != alert_id]
    if len(new) == len(alerts):
        raise HTTPException(404, "Alert không tồn tại.")
    with open(ALERTS_DB, "w", encoding="utf-8") as f:
        json.dump(new, f, ensure_ascii=False, indent=2)
    return {"deleted": alert_id}


@app.get("/api/clips/{job_id}/{filename}")
def download_clip(job_id: str, filename: str):
    """Download clip video."""
    path = CLIPS_DIR / job_id / filename
    if not path.exists():
        raise HTTPException(404, "Clip không tồn tại.")
    return FileResponse(str(path), media_type="video/mp4",
                        filename=filename)


# ==========================================
# Entrypoint
# ==========================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
