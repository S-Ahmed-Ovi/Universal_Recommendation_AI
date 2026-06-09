"""
smart_db_csv_builder/api/routers/connections.py

Endpoints
---------
POST /smart-db-csv/api/connections         — register a new DB connection
GET  /smart-db-csv/api/connections         — list registered connections
DELETE /smart-db-csv/api/connections/{id}  — remove a connection

Connection configs are stored in-memory per process (keyed by a UUID).
For production, swap _store for a Redis / DB-backed registry.
"""
from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, Request, status

from backend.smart_db_csv_builder.models.schemas import (
    NoSQLConnectionRequest,
    SQLConnectionRequest,
)
from backend.utils.logger import get_logger

log = get_logger(__name__)
router = APIRouter()

# ── In-process connection registry ────────────────────────────────────────────
# Maps conn_id → {"tenant_id", "type": "sql"|"nosql", "config": <request dict>}
_store: dict[str, dict[str, Any]] = {}


def _tenant_id(request: Request) -> str:
    """Extract tenant_id stamped by TenantMiddleware / auth."""
    return getattr(request.state, "tenant_id", "") or ""


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_config(cfg: dict) -> dict:
    """Return a copy of the config dict with the password masked."""
    out = dict(cfg)
    if "password" in out:
        out["password"] = "***"
    return out


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post(
    "",
    status_code=status.HTTP_201_CREATED,
    summary="Register a SQL database connection",
    response_description="Connection ID and masked config",
)
def create_sql_connection(body: SQLConnectionRequest, request: Request):
    """
    Register a SQL connection (postgres, mysql, sqlite …) for later use
    by the /schema and /build endpoints.

    Returns a **conn_id** UUID that you pass to subsequent calls.
    """
    tenant_id = _tenant_id(request) or body.tenant_id
    conn_id   = str(uuid.uuid4())

    _store[conn_id] = {
        "tenant_id": tenant_id,
        "type":      "sql",
        "config":    body.model_dump(by_alias=False),
    }

    log.info(
        "Connection registered | tenant=%s conn_id=%s dialect=%s host=%s db=%s",
        tenant_id, conn_id, body.dialect, body.host, body.database,
    )
    return {
        "conn_id":  conn_id,
        "type":     "sql",
        "dialect":  body.dialect,
        "host":     body.host,
        "database": body.database,
        "config":   _safe_config(_store[conn_id]["config"]),
    }


@router.post(
    "/nosql",
    status_code=status.HTTP_201_CREATED,
    summary="Register a NoSQL database connection",
)
def create_nosql_connection(body: NoSQLConnectionRequest, request: Request):
    """Register a NoSQL connection (MongoDB, Cassandra, DynamoDB)."""
    tenant_id = _tenant_id(request) or body.tenant_id
    conn_id   = str(uuid.uuid4())

    _store[conn_id] = {
        "tenant_id": tenant_id,
        "type":      "nosql",
        "config":    body.model_dump(by_alias=False),
    }

    log.info(
        "NoSQL connection registered | tenant=%s conn_id=%s engine=%s",
        tenant_id, conn_id, body.engine,
    )
    return {
        "conn_id": conn_id,
        "type":    "nosql",
        "engine":  body.engine,
        "host":    body.host,
        "config":  _safe_config(_store[conn_id]["config"]),
    }


@router.get(
    "",
    summary="List registered connections for this tenant",
)
def list_connections(request: Request):
    """Return all connections registered by the current tenant (passwords masked)."""
    tenant_id = _tenant_id(request)

    results = [
        {
            "conn_id": cid,
            "type":    entry["type"],
            "config":  _safe_config(entry["config"]),
        }
        for cid, entry in _store.items()
        if not tenant_id or entry["tenant_id"] == tenant_id
    ]

    return {"connections": results, "total": len(results)}


@router.delete(
    "/{conn_id}",
    status_code=status.HTTP_200_OK,
    summary="Remove a registered connection",
)
def delete_connection(conn_id: str, request: Request):
    """Delete a connection by its UUID. The underlying database is unaffected."""
    tenant_id = _tenant_id(request)

    entry = _store.get(conn_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Connection '{conn_id}' not found.")

    if tenant_id and entry["tenant_id"] != tenant_id:
        raise HTTPException(status_code=403, detail="Access denied to this connection.")

    del _store[conn_id]
    log.info("Connection removed | tenant=%s conn_id=%s", tenant_id, conn_id)
    return {"deleted": conn_id}


# ── Internal helper used by schema.py and build.py ────────────────────────────

def get_connection_entry(conn_id: str, tenant_id: str = "") -> dict:
    """
    Retrieve a stored connection entry, raising HTTP 404/403 on failure.
    Called by schema.py and build.py to resolve a conn_id.
    """
    entry = _store.get(conn_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Connection '{conn_id}' not found.")
    if tenant_id and entry["tenant_id"] != tenant_id:
        raise HTTPException(status_code=403, detail="Access denied to this connection.")
    return entry
