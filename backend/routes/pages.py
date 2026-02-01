from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import FileResponse


pages_router = APIRouter()


@pages_router.get("/")
async def index() -> FileResponse:
    return FileResponse("frontend/index.html", media_type="text/html")


@pages_router.get("/job/{job_id}")
async def job_page(job_id: str) -> FileResponse:
    return FileResponse("frontend/job.html", media_type="text/html")


@pages_router.get("/job/{job_id}/events")
async def job_events_page(job_id: str) -> FileResponse:
    return FileResponse("frontend/job_events.html", media_type="text/html")


@pages_router.get("/jobs")
async def jobs_page() -> FileResponse:
    return FileResponse("frontend/jobs.html", media_type="text/html")


@pages_router.get("/pack/{pack_id}")
async def pack_page(pack_id: str) -> FileResponse:
    return FileResponse("frontend/pack.html", media_type="text/html")


@pages_router.get("/viewer/{pdf_id}")
async def viewer_page(pdf_id: str) -> FileResponse:
    return FileResponse("frontend/viewer.html", media_type="text/html")
