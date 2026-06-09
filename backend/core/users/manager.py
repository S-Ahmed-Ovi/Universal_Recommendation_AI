"""Filesystem-backed user account registry."""
from __future__ import annotations

import hashlib
import hmac
import json
import re
import secrets
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from backend.config.settings import AUTH_USERNAME, USERS_FILE

_USERNAME_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]{2,62}$")
_EMAIL_RE = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")
_PASSWORD_MIN_LENGTH = 8
_HASH_ITERATIONS = 210_000
USER_ROLE = "USER"
ADMIN_ROLE = "ADMIN"
DEFAULT_ROLES: dict[str, dict[str, Any]] = {
    USER_ROLE: {
        "role": USER_ROLE,
        "label": "User",
        "description": "Tenant owner with access to their own workspace.",
        "is_admin": False,
    },
    ADMIN_ROLE: {
        "role": ADMIN_ROLE,
        "label": "Admin",
        "description": "Platform administrator with tenant management access.",
        "is_admin": True,
    },
}


class UserError(RuntimeError):
    """Base error for user account operations."""


class UserAlreadyExistsError(UserError):
    """Raised when a username is already registered."""


class UserNotFoundError(UserError):
    """Raised when a user cannot be found."""


class UserTenantLimitError(UserError):
    """Raised when a user already owns a different tenant."""


class UserValidationError(UserError):
    """Raised when user account input is invalid."""


@dataclass
class UserRecord:
    username: str
    email: str
    password_hash: str
    password_salt: str
    role: str = USER_ROLE
    tenant_id: str = ""
    created_at: float = field(default_factory=lambda: time.time())
    updated_at: float = field(default_factory=lambda: time.time())

    def to_dict(self) -> dict[str, Any]:
        return {
            "username": self.username,
            "email": self.email,
            "password_hash": self.password_hash,
            "password_salt": self.password_salt,
            "role": self.role,
            "tenant_id": self.tenant_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class UserManager:
    """Small user registry used by the admin control-plane auth flow."""

    _path: Path = Path(USERS_FILE)
    _lock = threading.RLock()

    @classmethod
    def _store_path(cls) -> Path:
        path = cls._path
        if isinstance(path, Path):
            return path.expanduser().resolve()
        return Path(path).expanduser().resolve()

    @staticmethod
    def normalize_username(username: str, *, allow_reserved: bool = False) -> str:
        value = (username or "").strip().lower()
        if not value:
            raise UserValidationError("username is required")
        if not _USERNAME_RE.fullmatch(value):
            raise UserValidationError(
                "username must be 3-63 chars and contain lowercase letters, digits, dots, underscores, or hyphens"
            )
        if not allow_reserved and value == AUTH_USERNAME.strip().lower():
            raise UserValidationError("username is reserved")
        return value

    @staticmethod
    def normalize_email(email: str) -> str:
        value = (email or "").strip().lower()
        if not value:
            raise UserValidationError("email is required")
        if len(value) > 254 or not _EMAIL_RE.fullmatch(value):
            raise UserValidationError("email must be a valid email address")
        return value

    @classmethod
    def _default_roles(cls) -> dict[str, dict[str, Any]]:
        return {role: dict(payload) for role, payload in DEFAULT_ROLES.items()}

    @classmethod
    def _read_roles_unlocked(cls) -> dict[str, dict[str, Any]]:
        path = cls._store_path()
        roles = cls._default_roles()
        if not path.exists():
            return roles
        try:
            with open(path, encoding="utf-8") as handle:
                payload = json.load(handle)
        except FileNotFoundError:
            return roles
        if not isinstance(payload, dict):
            return roles
        raw_roles = payload.get("roles", {})
        if not isinstance(raw_roles, dict):
            return roles
        for role_name, raw_role in raw_roles.items():
            role = str(role_name or "").strip().upper()
            if not role or not isinstance(raw_role, dict):
                continue
            roles[role] = {
                "role": role,
                "label": str(raw_role.get("label") or role.title()),
                "description": str(raw_role.get("description") or ""),
                "is_admin": bool(raw_role.get("is_admin", False)),
            }
        return roles

    @classmethod
    def _validate_role(cls, role: str, roles: Optional[dict[str, dict[str, Any]]] = None) -> str:
        role_value = (role or USER_ROLE).strip().upper()
        available = roles if roles is not None else cls._read_roles_unlocked()
        if role_value not in available:
            raise UserValidationError(f"role must be one of {sorted(available)}")
        return role_value

    @staticmethod
    def _validate_password(password: str) -> str:
        if not password:
            raise UserValidationError("password is required")
        if len(password) < _PASSWORD_MIN_LENGTH:
            raise UserValidationError(f"password must be at least {_PASSWORD_MIN_LENGTH} characters")
        return password

    @staticmethod
    def _hash_password(password: str, salt: str) -> str:
        digest = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            bytes.fromhex(salt),
            _HASH_ITERATIONS,
        )
        return digest.hex()

    @classmethod
    def _read_all(cls) -> dict[str, UserRecord]:
        path = cls._store_path()
        if not path.exists():
            return {}
        try:
            with open(path, encoding="utf-8") as handle:
                payload = json.load(handle)
        except FileNotFoundError:
            return {}
        if not isinstance(payload, dict):
            raise UserValidationError("user store must be a JSON object")

        raw_users = payload.get("users", {})
        if not isinstance(raw_users, dict):
            raise UserValidationError("user store users field must be a JSON object")

        users: dict[str, UserRecord] = {}
        roles = cls._read_roles_unlocked()
        for username, raw in raw_users.items():
            if not isinstance(raw, dict):
                continue
            normalized = cls.normalize_username(str(raw.get("username") or username), allow_reserved=True)
            raw_email = str(raw.get("email", "") or "").strip()
            normalized_email = cls.normalize_email(raw_email) if raw_email else ""
            users[normalized] = UserRecord(
                username=normalized,
                email=normalized_email,
                password_hash=str(raw.get("password_hash", "")),
                password_salt=str(raw.get("password_salt", "")),
                role=cls._validate_role(str(raw.get("role") or USER_ROLE), roles),
                tenant_id=str(raw.get("tenant_id", "") or ""),
                created_at=float(raw.get("created_at", time.time())),
                updated_at=float(raw.get("updated_at", time.time())),
            )
        return users

    @classmethod
    def _write_all(cls, users: dict[str, UserRecord]) -> None:
        path = cls._store_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 2,
            "roles": cls._read_roles_unlocked(),
            "users": {username: record.to_dict() for username, record in sorted(users.items())},
        }
        tmp_path = path.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        tmp_path.replace(path)

    @classmethod
    def register(cls, username: str, email: str, password: str) -> UserRecord:
        username = cls.normalize_username(username)
        email = cls.normalize_email(email)
        password = cls._validate_password(password)
        with cls._lock:
            users = cls._read_all()
            if username in users:
                raise UserAlreadyExistsError(f"User '{username}' already exists")
            if any(record.email == email for record in users.values()):
                raise UserAlreadyExistsError("Email is already registered")
            salt = secrets.token_hex(16)
            record = UserRecord(
                username=username,
                email=email,
                password_hash=cls._hash_password(password, salt),
                password_salt=salt,
                role=USER_ROLE,
            )
            users[username] = record
            cls._write_all(users)
            return record

    @classmethod
    def get(cls, username: str) -> UserRecord:
        username = cls.normalize_username(username)
        with cls._lock:
            users = cls._read_all()
            if username not in users:
                raise UserNotFoundError(f"User '{username}' not found")
            return users[username]

    @classmethod
    def validate_credentials(cls, username: str, password: str) -> Optional[UserRecord]:
        try:
            lookup = cls.normalize_username(username)
        except UserValidationError:
            try:
                lookup_email = cls.normalize_email(username)
            except UserValidationError:
                return None
            lookup = ""
        with cls._lock:
            users = cls._read_all()
            record = users.get(lookup) if lookup else None
            if record is None and "lookup_email" in locals():
                record = next((user for user in users.values() if user.email == lookup_email), None)
            if record is None or not record.password_hash or not record.password_salt:
                return None
            computed = cls._hash_password(password or "", record.password_salt)
            if not hmac.compare_digest(computed, record.password_hash):
                return None
            return record

    @classmethod
    def list_roles(cls) -> list[dict[str, Any]]:
        with cls._lock:
            roles = cls._read_roles_unlocked()
        return [roles[key] for key in sorted(roles)]

    @classmethod
    def role_label(cls, role: str) -> str:
        roles = cls._read_roles_unlocked()
        role_value = cls._validate_role(role, roles)
        return str(roles[role_value].get("label") or role_value.title())

    @classmethod
    def link_tenant(cls, username: str, tenant_id: str) -> UserRecord:
        username = cls.normalize_username(username)
        tenant_id = (tenant_id or "").strip()
        if not tenant_id:
            raise UserValidationError("tenant_id is required")
        with cls._lock:
            users = cls._read_all()
            record = users.get(username)
            if record is None:
                raise UserNotFoundError(f"User '{username}' not found")
            if record.tenant_id and record.tenant_id != tenant_id:
                raise UserTenantLimitError(f"User '{username}' already owns tenant '{record.tenant_id}'")
            record.tenant_id = tenant_id
            record.updated_at = time.time()
            users[username] = record
            cls._write_all(users)
            return record

    @classmethod
    def unlink_tenant(cls, username: str, tenant_id: str) -> None:
        username = cls.normalize_username(username)
        with cls._lock:
            users = cls._read_all()
            record = users.get(username)
            if record is None or record.tenant_id != tenant_id:
                return
            record.tenant_id = ""
            record.updated_at = time.time()
            users[username] = record
            cls._write_all(users)
