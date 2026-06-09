"""
Async job tracking routes.

GET /jobs          - list recent tenant-scoped training jobs
GET /jobs/{job_id} - poll the status and result of a background training job
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from backend.api.gateway.auth import require_auth
from backend.utils.logger import get_logger
from .shared import _delete_job_for_tenant, _get_job_for_tenant, _list_jobs_for_tenant

log = get_logger(__name__)

router = APIRouter(tags=["Jobs"])


@router.get("/jobs")
def list_jobs(
    request: Request,
    limit: int = Query(25, ge=1, le=100),
    _auth: dict = Depends(require_auth(scopes=["jobs:read"])),
):
    """List recent training jobs visible to the authenticated tenant."""
    caller_tenant = getattr(request.state, "tenant_id", None)
    return JSONResponse(content={"jobs": _list_jobs_for_tenant(caller_tenant, limit=limit)})


@router.get("/jobs/{job_id}")
def get_job(
    job_id: str,
    request: Request,
    _auth: dict = Depends(require_auth(scopes=["jobs:read"])),
):
    """
    Return the current status and result (or error) for an async training job.

    Possible status values:
    - **pending**   - queued, not yet started
    - **running**   - actively training
    - **completed** - finished successfully; `result` contains the training report
    - **failed**    - failed; `error` contains the exception message
    """
    caller_tenant = getattr(request.state, "tenant_id", None)
    job = _get_job_for_tenant(caller_tenant, job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    return JSONResponse(content=job)


@router.delete("/jobs/{job_id}")
def delete_job(
    job_id: str,
    request: Request,
    _auth: dict = Depends(require_auth(scope="training:write")),
):
    """Delete a completed or failed training job record for the current tenant."""
    caller_tenant = getattr(request.state, "tenant_id", None)
    job = _get_job_for_tenant(caller_tenant, job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    if job.get("status") in {"pending", "running"}:
        raise HTTPException(status_code=409, detail="Running or pending jobs cannot be deleted.")

    deleted = _delete_job_for_tenant(caller_tenant, job_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    return JSONResponse(content={"deleted": True, "job_id": job_id})
