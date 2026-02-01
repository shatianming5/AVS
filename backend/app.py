from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles

from backend.routes.api import api_router
from backend.routes.pages import pages_router
from backend.services.init_db import init_db
from shared.config import ensure_data_dirs
from shared.env import load_env


load_env()
ensure_data_dirs()
init_db()

app = FastAPI(title="paper_skill")

@app.middleware("http")
async def _disable_static_cache(request: Request, call_next):
    response = await call_next(request)
    if request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-store, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response

app.include_router(api_router, prefix="/api")
app.include_router(pages_router)

app.mount(
    "/static",
    StaticFiles(directory="frontend", html=False),
    name="static",
)
