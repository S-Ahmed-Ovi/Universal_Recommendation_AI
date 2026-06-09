"""
Canonical deployment control plane for tenant-scoped model serving.

The model registry remains the artifact inventory. This module owns the
authoritative production deployment pointer, deployment history, and rollback
state for a tenant.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from backend.config.settings import BASE_TENANTS_DIR
from backend.core.models.model_loader import ModelLoader
from backend.core.models.model_registry import ModelRecord, ModelRegistry
from backend.core.tenants.manager import TenantManager
from backend.utils.logger import get_logger

log = get_logger(__name__)

_STATE_FILE = "deployment_state.json"
_EVENTS_FILE = "deployment_events.json"


@dataclass
class DeploymentState:
    tenant_id: str
    status: str = "inactive"
    active_model_id: Optional[str] = None
    active_algorithm: Optional[str] = None
    active_version: Optional[int] = None
    previous_active_model_id: Optional[str] = None
    previous_active_algorithm: Optional[str] = None
    previous_active_version: Optional[int] = None
    rollback_candidate_model_id: Optional[str] = None
    rollback_candidate_algorithm: Optional[str] = None
    rollback_candidate_version: Optional[int] = None
    promoted_at: Optional[float] = None
    last_ready_at: Optional[float] = None
    last_error: Optional[str] = None
    activated_by: Optional[str] = None
    activation_source: str = ""
    last_transition_at: float = 0.0

    def to_dict(self) -> dict:
        """Serialize the persisted deployment state into a JSON-safe dictionary."""
        return asdict(self)


class DeploymentControlPlane:
    def __init__(
        self,
        tenant_id: str,
        registry: Optional[ModelRegistry] = None,
        loader: Optional[ModelLoader] = None,
        base_dir: Path | str | None = None,
    ):
        """Initialise tenant-scoped deployment state and event storage paths."""
        if not tenant_id or not isinstance(tenant_id, str):
            raise ValueError("tenant_id must be a non-empty string.")

        self.tenant_id = tenant_id
        self.registry = registry or ModelRegistry(tenant_id=tenant_id, base_dir=base_dir)
        self.loader = loader
        resolved_base_dir = base_dir or getattr(TenantManager, "_base", None) or BASE_TENANTS_DIR
        root = Path(resolved_base_dir).expanduser().resolve()
        self.tenant_dir = (root / tenant_id).resolve()
        self.tenant_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.tenant_dir / _STATE_FILE
        self.events_path = self.tenant_dir / _EVENTS_FILE

    def get_state(self) -> DeploymentState:
        """Load the current deployment state or bootstrap it from registry flags."""
        state = self._load_state()
        if state is not None:
            return state
        return self._bootstrap_state()

    def get_active_record(self) -> Optional[ModelRecord]:
        """Return the active deployment record and repair broken state if needed."""
        state = self.get_state()
        if not state.active_model_id:
            return None
        record = self.registry.get(state.active_model_id)
        if record is not None:
            return record

        log.warning(
            "Deployment state references missing model | tenant=%s model_id=%s",
            self.tenant_id,
            state.active_model_id,
        )
        repaired = DeploymentState(
            tenant_id=self.tenant_id,
            status="failed",
            previous_active_model_id=state.previous_active_model_id,
            previous_active_algorithm=state.previous_active_algorithm,
            previous_active_version=state.previous_active_version,
            rollback_candidate_model_id=state.rollback_candidate_model_id,
            rollback_candidate_algorithm=state.rollback_candidate_algorithm,
            rollback_candidate_version=state.rollback_candidate_version,
            promoted_at=state.promoted_at,
            last_ready_at=state.last_ready_at,
            last_error=(
                f"Active deployment model '{state.active_model_id}' no longer exists in the registry."
            ),
            activated_by=state.activated_by,
            activation_source=state.activation_source,
            last_transition_at=time.time(),
        )
        self._save_state(repaired)
        return None

    def promote(
        self,
        model_id: str,
        *,
        actor: Optional[str] = None,
        reason: str = "",
        source: str = "manual",
    ) -> dict:
        """Promote a model to the canonical active deployment for this tenant."""
        record = self.registry.get(model_id)
        if record is None:
            raise KeyError(
                f"Model ID '{model_id}' not found in registry for tenant='{self.tenant_id}'."
            )

        current_state = self.get_state()
        previous = self.get_active_record()
        self._prewarm(record)
        self.registry.promote(model_id)

        now = time.time()
        state = DeploymentState(
            tenant_id=self.tenant_id,
            status="active",
            active_model_id=record.model_id,
            active_algorithm=record.algorithm,
            active_version=record.version,
            previous_active_model_id=(
                previous.model_id if previous and previous.model_id != record.model_id else current_state.previous_active_model_id
            ),
            previous_active_algorithm=(
                previous.algorithm if previous and previous.model_id != record.model_id else current_state.previous_active_algorithm
            ),
            previous_active_version=(
                previous.version if previous and previous.model_id != record.model_id else current_state.previous_active_version
            ),
            rollback_candidate_model_id=(
                previous.model_id if previous and previous.model_id != record.model_id else current_state.rollback_candidate_model_id
            ),
            rollback_candidate_algorithm=(
                previous.algorithm if previous and previous.model_id != record.model_id else current_state.rollback_candidate_algorithm
            ),
            rollback_candidate_version=(
                previous.version if previous and previous.model_id != record.model_id else current_state.rollback_candidate_version
            ),
            promoted_at=now,
            last_ready_at=now,
            last_error=None,
            activated_by=actor or self.tenant_id,
            activation_source=source,
            last_transition_at=now,
        )
        self._save_state(state)
        self._append_event(
            event_type="deployment.rolled_back" if source == "rollback" else "deployment.promoted",
            status="active",
            record=record,
            previous=previous,
            actor=actor,
            reason=reason,
            source=source,
            message=f"Deployment switched to model '{record.model_id}'.",
            created_at=now,
        )
        return self.describe_active()

    def rollback(self, *, actor: Optional[str] = None, reason: str = "") -> dict:
        """Rollback the active deployment to the stored rollback candidate."""
        state = self.get_state()
        if not state.rollback_candidate_model_id:
            raise RuntimeError("No rollback candidate is available for this tenant.")
        candidate = self.registry.get(state.rollback_candidate_model_id)
        if candidate is None:
            raise RuntimeError(
                f"Rollback candidate '{state.rollback_candidate_model_id}' is no longer available."
            )
        return self.promote(
            candidate.model_id,
            actor=actor,
            reason=reason or "Rollback to previous active deployment",
            source="rollback",
        )

    def describe_active(self) -> dict:
        """Return the active deployment payload exposed by deployment endpoints."""
        state = self.get_state()
        active_record = self.get_active_record()
        rollback_record = (
            self.registry.get(state.rollback_candidate_model_id)
            if state.rollback_candidate_model_id
            else None
        )

        return {
            "tenant_id": self.tenant_id,
            "deployment": state.to_dict(),
            "active_deployment": self._record_payload(active_record, promoted_at=state.promoted_at),
            "rollback_candidate": self._record_payload(rollback_record),
            "health": self.health(state=state, active_record=active_record),
        }

    def history(self, limit: int = 20) -> list[dict]:
        """Return recent deployment events in reverse chronological order."""
        events = self._load_events()
        events.sort(key=lambda row: float(row.get("created_at") or 0), reverse=True)
        return events[: max(int(limit), 1)]

    def health(
        self,
        *,
        state: Optional[DeploymentState] = None,
        active_record: Optional[ModelRecord] = None,
    ) -> dict:
        """Summarize runtime health for the active deployment and warm cache state."""
        state = state or self.get_state()
        active_record = active_record if active_record is not None else self.get_active_record()
        loaded = None
        if self.loader is not None and active_record is not None:
            loaded = next(
                (
                    item
                    for item in self.loader.list_loaded(tenant_id=self.tenant_id)
                    if item.get("model_id") == active_record.model_id
                ),
                None,
            )

        ready = (
            state.status == "active"
            and active_record is not None
            and not state.last_error
        )
        if not active_record:
            status = "inactive"
            label = "No active deployment"
        elif state.last_error:
            status = "failed"
            label = "Deployment failed"
        elif ready:
            status = "healthy"
            label = "Healthy"
        else:
            status = "warming"
            label = "Warming"

        return {
            "status": status,
            "status_label": label,
            "ready": ready,
            "deployment_status": state.status,
            "active_model_id": getattr(active_record, "model_id", None),
            "loaded_model_id": (loaded or {}).get("model_id"),
            "loaded_at": (loaded or {}).get("loaded_at"),
            "last_ready_at": state.last_ready_at,
            "last_error": state.last_error,
        }

    def _bootstrap_state(self) -> DeploymentState:
        promoted = self.registry.list_models(promoted_only=True)
        if not promoted:
            state = DeploymentState(
                tenant_id=self.tenant_id,
                status="inactive",
                last_transition_at=time.time(),
            )
            self._save_state(state)
            return state

        promoted.sort(key=lambda row: row.created_at, reverse=True)
        record = promoted[0]
        if len(promoted) > 1:
            self.registry.promote(record.model_id)
            log.warning(
                "Deployment bootstrap repaired multiple promoted models | tenant=%s kept=%s dropped=%d",
                self.tenant_id,
                record.model_id,
                len(promoted) - 1,
            )

        now = time.time()
        state = DeploymentState(
            tenant_id=self.tenant_id,
            status="active",
            active_model_id=record.model_id,
            active_algorithm=record.algorithm,
            active_version=record.version,
            promoted_at=now,
            last_ready_at=now,
            activated_by="system",
            activation_source="registry_bootstrap",
            last_transition_at=now,
        )
        self._save_state(state)
        self._append_event(
            event_type="deployment.bootstrapped",
            status="active",
            record=record,
            previous=None,
            actor="system",
            reason="Recovered canonical deployment state from registry promotion flags.",
            source="registry_bootstrap",
            message=f"Deployment bootstrapped to model '{record.model_id}'.",
            created_at=now,
        )
        return state

    def _prewarm(self, record: ModelRecord) -> None:
        if self.loader is not None:
            self.loader.load_by_id(tenant_id=self.tenant_id, model_id=record.model_id)
            return
        self.registry.load(record.model_id)

    def _record_payload(
        self,
        record: Optional[ModelRecord],
        *,
        promoted_at: Optional[float] = None,
    ) -> Optional[dict]:
        if record is None:
            return None
        payload = record.to_dict()
        payload["promoted_at"] = promoted_at
        return payload

    def _load_state(self) -> Optional[DeploymentState]:
        if not self.state_path.exists():
            return None
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        except Exception as exc:
            log.warning("Failed to read deployment state | tenant=%s error=%s", self.tenant_id, exc)
            return None
        if not isinstance(payload, dict):
            return None
        payload.setdefault("tenant_id", self.tenant_id)
        return DeploymentState(**payload)

    def _save_state(self, state: DeploymentState) -> None:
        tmp_path = self.state_path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(state.to_dict(), indent=2), encoding="utf-8")
        tmp_path.replace(self.state_path)

    def _load_events(self) -> list[dict]:
        if not self.events_path.exists():
            return []
        try:
            payload = json.loads(self.events_path.read_text(encoding="utf-8"))
        except Exception as exc:
            log.warning("Failed to read deployment history | tenant=%s error=%s", self.tenant_id, exc)
            return []
        return payload if isinstance(payload, list) else []

    def _append_event(
        self,
        *,
        event_type: str,
        status: str,
        record: Optional[ModelRecord],
        previous: Optional[ModelRecord],
        actor: Optional[str],
        reason: str,
        source: str,
        message: str,
        created_at: Optional[float] = None,
    ) -> None:
        events = self._load_events()
        events.append(
            {
                "event_id": str(uuid.uuid4())[:12],
                "event_type": event_type,
                "status": status,
                "created_at": created_at or time.time(),
                "tenant_id": self.tenant_id,
                "model_id": getattr(record, "model_id", None),
                "algorithm": getattr(record, "algorithm", None),
                "version": getattr(record, "version", None),
                "previous_model_id": getattr(previous, "model_id", None),
                "previous_algorithm": getattr(previous, "algorithm", None),
                "previous_version": getattr(previous, "version", None),
                "actor": actor or self.tenant_id,
                "reason": reason,
                "source": source,
                "message": message,
            }
        )
        events.sort(key=lambda row: float(row.get("created_at") or 0), reverse=True)
        tmp_path = self.events_path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(events[:200], indent=2), encoding="utf-8")
        tmp_path.replace(self.events_path)
