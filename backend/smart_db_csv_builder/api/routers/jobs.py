"""
smart_db_csv_builder/api/routers/jobs.py

Endpoints
---------
GET  /smart-db-csv/api/jobs/{job_id}   — poll a single build job
GET  /smart-db-csv/api/jobs            — list all jobs for this tenant
DELETE /smart-db-csv/api/jobs/{job_id} — cancel / purge a job record
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from backend.smart_db_csv_builder.api.routers.build import get_job_store
from backend.utils.logger import get_logger

log = get_logger(__name__)
router = APIRouter()


def _tenant_id(request: Request) -> str:
    return getattr(request.state, "tenant_id", "") or ""


def _safe_job(job: dict) -> dict:
    """Return a copy of the job dict suitable for the API response."""
    return {k: v for k, v in job.items() if k != "tenant_id"}


@router.get(
    "/{job_id}",
    summary="Poll build job status",
)
def get_job(job_id: str, request: Request):
    """
    Retrieve the current status and result of a build job.

    **status** values:
    - `pending`  — queued, not yet started
    - `running`  — actively fetching / writing
    - `done`     — finished; check `output_path` for the CSV location
    - `failed`   — check `error` for the reason
    """
    tenant_id = _tenant_id(request)
    jobs      = get_job_store()
    job       = jobs.get(job_id)

    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    if tenant_id and job["tenant_id"] != tenant_id:
        raise HTTPException(status_code=403, detail="Access denied to this job.")

    return _safe_job(job)


@router.get(
    "",
    summary="List build jobs for this tenant",
)
def list_jobs(request: Request, status: str = ""):
    """
    List all build jobs belonging to the authenticated tenant.

    Optionally filter by **status**: `pending`, `running`, `done`, or `failed`.
    """
    tenant_id = _tenant_id(request)
    jobs      = get_job_store()

    results = [
        _safe_job(job)
        for job in jobs.values()
        if (not tenant_id or job["tenant_id"] == tenant_id)
        and (not status or job["status"] == status)
    ]

    return {"jobs": results, "total": len(results)}


@router.delete(
    "/{job_id}",
    summary="Remove a job record",
)
def delete_job(job_id: str, request: Request):
    """
    Remove a completed (or failed) job record from the registry.

    Only the tenant that created the job may delete it.
    Running jobs are *not* cancelled — stop the process externally first.
    """
    tenant_id = _tenant_id(request)
    jobs      = get_job_store()
    job       = jobs.get(job_id)

    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    if tenant_id and job["tenant_id"] != tenant_id:
        raise HTTPException(status_code=403, detail="Access denied to this job.")

    if job["status"] == "running":
        raise HTTPException(
            status_code=409,
            detail="Cannot delete a running job. Wait for it to complete first.",
        )

    del jobs[job_id]
    log.info("Job record deleted | tenant=%s job_id=%s", tenant_id, job_id)
    return {"deleted": job_id}
