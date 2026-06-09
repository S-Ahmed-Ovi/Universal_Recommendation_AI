"""
smart_db_csv_builder/connectors/csv.py — Tenant-aware file loader

Refactored from ingestion/csv_loader.py.

Changes vs v2
-------------
- Every load() call is scoped to a tenant: files are resolved relative to
  tenants/{tenant_id}/data/ when a relative path is given.
- Uploaded bytes are optionally persisted to the tenant data directory so
  they survive the request and can be referenced in subsequent jobs.
- Audit log line on every load (tenant, format, shape, source).
- All v2 format support preserved: CSV, Excel, JSON/JSONL, Parquet.

Usage
-----
    loader = TenantFileLoader(tenant_id="acme-corp")

    # From an uploaded UploadFile (FastAPI):
    df = loader.load(await file.read(), file_format="auto", filename=file.filename)

    # From a path already in the tenant's data directory:
    df = loader.load("ratings.csv")

    # Persist uploaded bytes to tenant storage and return DataFrame:
    df = loader.load(raw_bytes, filename="events.json", persist=True)
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Optional, Union

import pandas as pd

from backend.config.settings import BASE_TENANTS_DIR
from backend.utils.logger import get_logger

log = get_logger(__name__)

FileInput = Union[str, Path, bytes, io.BytesIO]

_TENANTS_ROOT = Path(BASE_TENANTS_DIR)


class TenantFileLoader:
    """
    Load tabular data for a specific tenant.

    Supports CSV, Excel (.xlsx/.xls), JSON (records or lines), Parquet.
    Paths that are not absolute are resolved relative to the tenant's
    data directory: tenants/{tenant_id}/data/
    """

    _EXT_TO_FORMAT = {
        ".csv":     "csv",
        ".xlsx":    "excel",
        ".xls":     "excel",
        ".json":    "json",
        ".jsonl":   "json",
        ".parquet": "parquet",
    }

    def __init__(self, tenant_id: str) -> None:
        """Bind the loader to one tenant and ensure its data directory exists."""
        self.tenant_id  = tenant_id
        self.data_dir   = _TENANTS_ROOT / tenant_id / "data"
        self.data_dir.mkdir(parents=True, exist_ok=True)

    # ── Public API ─────────────────────────────────────────────────────────

    def load(
        self,
        source: FileInput,
        file_format: str = "auto",
        filename:    Optional[str] = None,
        persist:     bool = False,
    ) -> pd.DataFrame:
        """
        Load a file and return a DataFrame scoped to this tenant.

        Parameters
        ----------
        source      : file path (str/Path), raw bytes, or BytesIO.
                      Relative paths are resolved against the tenant data dir.
        file_format : "auto" (infer from extension/filename), or one of
                      "csv", "excel", "json", "parquet".
        filename    : original filename — used for format inference when
                      source is bytes/BytesIO and for the persisted filename.
        persist     : if True and source is bytes/BytesIO, write the raw
                      bytes to tenants/{tenant_id}/data/{filename} before
                      loading, so the file is available for future jobs.
        """
        buf, ext = self._prepare(source, filename, persist)
        fmt = self._resolve_format(file_format, ext)

        log.info(
            "FileLoader | tenant=%s format=%s source_type=%s filename=%s",
            self.tenant_id, fmt, type(source).__name__, filename or "-",
        )

        df = self._dispatch(buf, fmt)
        log.info(
            "Loaded %d rows × %d cols | tenant=%s",
            *df.shape, self.tenant_id,
        )
        return df

    def list_files(self) -> list[str]:
        """Return filenames available in this tenant's data directory."""
        return [f.name for f in self.data_dir.iterdir() if f.is_file()]

    def tenant_path(self, filename: str) -> Path:
        """Resolve a filename to its full path in the tenant data dir."""
        return self.data_dir / filename

    # ── Internal helpers ───────────────────────────────────────────────────

    def _prepare(
        self,
        source: FileInput,
        filename: Optional[str],
        persist: bool,
    ) -> tuple[Union[Path, io.BytesIO], str]:
        """
        Normalise source to (buffer_or_path, extension).
        Optionally persist bytes to tenant storage.
        """
        if isinstance(source, (str, Path)):
            path = Path(source)
            # Relative → tenant data dir
            if not path.is_absolute():
                path = self.data_dir / path
            if not path.exists():
                raise FileNotFoundError(
                    f"File not found in tenant '{self.tenant_id}' data dir: {path}"
                )
            return path, path.suffix.lower()

        # bytes or BytesIO
        raw: bytes
        if isinstance(source, bytes):
            raw = source
        else:
            source.seek(0)
            raw = source.read()

        ext = Path(filename).suffix.lower() if filename else ""

        if persist and filename:
            dest = self.data_dir / filename
            dest.write_bytes(raw)
            log.info(
                "Persisted upload to tenant storage: tenant=%s path=%s",
                self.tenant_id, dest,
            )
            return dest, ext

        return io.BytesIO(raw), ext

    def _dispatch(self, buf, fmt: str) -> pd.DataFrame:
        loaders = {
            "csv":     self._load_csv,
            "excel":   self._load_excel,
            "json":    self._load_json,
            "parquet": self._load_parquet,
        }
        fn = loaders.get(fmt)
        if not fn:
            raise ValueError(
                f"Unsupported format '{fmt}'. Choose from: {list(loaders)}"
            )
        return fn(buf)

    # ── Format loaders (identical logic to v2) ─────────────────────────────

    @staticmethod
    def _load_csv(buf) -> pd.DataFrame:
        try:
            return pd.read_csv(buf, low_memory=False)
        except Exception as exc:
            raise ValueError(f"Failed to parse CSV: {exc}") from exc

    @staticmethod
    def _load_excel(buf) -> pd.DataFrame:
        try:
            return pd.read_excel(buf, engine="openpyxl")
        except ImportError:
            raise ImportError(
                "Install 'openpyxl' to load Excel files: pip install openpyxl"
            )
        except Exception as exc:
            raise ValueError(f"Failed to parse Excel: {exc}") from exc

    @staticmethod
    def _load_json(buf) -> pd.DataFrame:
        try:
            try:
                return pd.read_json(buf, orient="records")
            except Exception:
                if hasattr(buf, "seek"):
                    buf.seek(0)
                return pd.read_json(buf, lines=True)
        except Exception as exc:
            raise ValueError(f"Failed to parse JSON: {exc}") from exc

    @staticmethod
    def _load_parquet(buf) -> pd.DataFrame:
        try:
            return pd.read_parquet(buf)
        except ImportError:
            raise ImportError(
                "Install 'pyarrow' or 'fastparquet' to load Parquet files."
            )
        except Exception as exc:
            raise ValueError(f"Failed to parse Parquet: {exc}") from exc

    # ── Format resolution ──────────────────────────────────────────────────

    @classmethod
    def _resolve_format(cls, file_format: str, ext: str) -> str:
        if file_format != "auto":
            return file_format.lower()
        fmt = cls._EXT_TO_FORMAT.get(ext)
        if not fmt:
            raise ValueError(
                f"Cannot infer format from extension '{ext}'. "
                "Pass file_format explicitly: csv, excel, json, parquet."
            )
        return fmt
