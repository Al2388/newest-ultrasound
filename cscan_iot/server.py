from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
import uvicorn
import os

from scanner_service import CScanService


class NoCacheMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)

        # Disable caching everywhere (UI + status + images)
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response


app = FastAPI()
app.add_middleware(NoCacheMiddleware)

# Optional: if you ever open UI from another machine
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

scanner = CScanService()
templates = Jinja2Templates(directory="templates")

# IMPORTANT: use absolute path so mount is always correct even if you run from a different folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "cscan_out")
os.makedirs(OUT_DIR, exist_ok=True)

app.mount("/local", StaticFiles(directory=OUT_DIR), name="local")


@app.get("/")
async def dashboard(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/start")
async def start_scan(payload: dict):
    success, msg = scanner.start_scan(payload)
    return {"success": success, "msg": msg}


@app.post("/api/stop")
async def stop_scan():
    ok = scanner.stop_scan()
    return {"msg": "Stopping..." if ok else "Not running"}


@app.post("/api/return")
async def return_home():
    success, msg = scanner.return_to_start()
    return {"success": success, "msg": msg}


@app.post("/api/jog_z")
async def jog_z(payload: dict):
    dist = payload.get("z", 0.0)
    success, msg = scanner.jog_z_axis(dist)
    return {"success": success, "msg": msg}


@app.get("/api/status")
async def status():
    return {
        "status": scanner.status,
        "progress": scanner.progress,
        "images": scanner.images
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)