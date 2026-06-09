from __future__ import annotations

from fastapi.testclient import TestClient

from backend.app_factory import create_app
from backend.api.routes.shared import _jobs, _persist_job_record


def test_jobs_history_keeps_completed_runs_visible_while_new_run_is_active(write_tenant_config):
    write_tenant_config(raw_api_key="secret-key", scopes=["jobs:read", "training:write"])
    _jobs.clear()

    completed_job = {
        "created_at": 100.0,
        "error": None,
        "finished_at": 130.0,
        "job_id": "job-complete",
        "result": {
            "best_algorithm": "als",
            "elapsed_s": 30.0,
            "resolved_mode": "implicit",
        },
        "source_label": "historical.csv",
        "source_type": "file",
        "started_at": 102.0,
        "status": "completed",
        "tenant_id": "tenant-a",
    }
    _persist_job_record(completed_job)

    _jobs["job-running"] = {
        "created_at": 200.0,
        "error": None,
        "finished_at": None,
        "job_id": "job-running",
        "result": None,
        "source_label": "postgresql / analytics",
        "source_type": "sql",
        "started_at": 201.0,
        "status": "running",
        "tenant_id": "tenant-a",
    }

    client = TestClient(create_app())
    headers = {"X-API-Key": "secret-key"}

    list_response = client.get("/jobs", headers=headers)
    detail_response = client.get("/jobs/job-complete", headers=headers)

    assert list_response.status_code == 200
    assert detail_response.status_code == 200

    jobs = list_response.json()["jobs"]
    assert [job["job_id"] for job in jobs[:2]] == ["job-running", "job-complete"]
    assert jobs[0]["status"] == "running"
    assert jobs[1]["status"] == "completed"
    assert jobs[1]["result"]["best_algorithm"] == "als"

    completed_payload = detail_response.json()
    assert completed_payload["job_id"] == "job-complete"
    assert completed_payload["status"] == "completed"
    assert completed_payload["result"]["resolved_mode"] == "implicit"

    _jobs.clear()


def test_completed_job_history_can_be_deleted_but_running_job_cannot(write_tenant_config):
    write_tenant_config(raw_api_key="secret-key", scopes=["jobs:read", "training:write"])
    _jobs.clear()

    _persist_job_record({
        "created_at": 100.0,
        "error": None,
        "finished_at": 130.0,
        "job_id": "job-complete",
        "result": {"best_algorithm": "als"},
        "source_label": "historical.csv",
        "source_type": "file",
        "started_at": 102.0,
        "status": "completed",
        "tenant_id": "tenant-a",
    })
    _jobs["job-running"] = {
        "created_at": 200.0,
        "error": None,
        "finished_at": None,
        "job_id": "job-running",
        "result": None,
        "source_label": "postgresql / analytics",
        "source_type": "sql",
        "started_at": 201.0,
        "status": "running",
        "tenant_id": "tenant-a",
    }

    client = TestClient(create_app())
    headers = {"X-API-Key": "secret-key"}

    delete_completed = client.delete("/jobs/job-complete", headers=headers)
    delete_running = client.delete("/jobs/job-running", headers=headers)
    list_response = client.get("/jobs", headers=headers)

    assert delete_completed.status_code == 200
    assert delete_completed.json() == {"deleted": True, "job_id": "job-complete"}
    assert delete_running.status_code == 409
    assert "cannot be deleted" in delete_running.json()["detail"].lower()
    assert [job["job_id"] for job in list_response.json()["jobs"]] == ["job-running"]

    _jobs.clear()
