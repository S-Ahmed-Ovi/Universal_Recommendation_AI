"""
smart_db_csv_builder/services/llm_planner.py — Smart query generation (optional)

New in v3.

Uses an LLM (Claude via Anthropic API) to inspect a database schema and
produce a validated interaction query ready for the recommendation pipeline.

The LLM is given:
  - Table + column metadata from the live database
  - Optional sample rows (from get_sample())
  - A system prompt that constrains output to JSON with specific keys
  - The tenant's algorithm_mode preference ("explicit" | "implicit")

The planner returns a PlannerResult that can be handed directly to
SQLExecutor.execute_custom() or SQLExecutor.execute_interaction_query().

Design principles
-----------------
- **Optional dependency**: if ANTHROPIC_API_KEY is not set, the planner
  raises a clear ImportError-like error rather than silently failing.
- **No side effects**: the planner only reads schema metadata and issues
  LLM calls; it never writes to the database or filesystem.
- **Tenant-scoped**: tenant_id appears in all log lines and is included
  in the LLM prompt for context.
- **Validated output**: LLM JSON is parsed and validated against required
  keys before being returned; falls back to a safe error state on bad JSON.
- **Auditable**: the raw LLM response and the final SQL are both stored
  in PlannerResult for downstream logging / review.

Usage
-----
    from backend.config.database import SQLConfig
    from backend.smart_db_csv_builder.services.builder import ConnectionBuilder
    from backend.smart_db_csv_builder.services.llm_planner import LLMPlanner

    cfg = SQLConfig(dialect="postgres", host="...", ...)
    builder = ConnectionBuilder(cfg, tenant_id="acme-corp")

    planner = LLMPlanner(builder, tenant_id="acme-corp")
    result  = planner.plan(algorithm_mode="explicit")

    if result.success:
        from backend.smart_db_csv_builder.services.executor import SQLExecutor
        executor = SQLExecutor(cfg, tenant_id="acme-corp")
        qr = executor.execute_custom(result.sql, algorithm_mode=result.algorithm_mode)
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Optional

from backend.utils.logger import get_logger
from backend.smart_db_csv_builder.services.builder import ConnectionBuilder

log = get_logger(__name__)
_ANTHROPIC_CLIENTS: dict[str, object] = {}

# ── Constants ──────────────────────────────────────────────────────────────────

_MODEL         = "claude-sonnet-4-20250514"
_MAX_TOKENS    = 1024
_SAMPLE_ROWS   = 3          # rows fetched per table for LLM context
_MAX_TABLES    = 20         # cap to avoid bloating the prompt
_MAX_COLS      = 30         # cap columns per table

_SYSTEM_PROMPT = """\
You are a database analyst helping build a recommendation-system data pipeline.

Given schema metadata (tables, columns, optional sample rows) from a live database,
identify the best table and columns to use as a user-item interaction matrix.

Respond ONLY with a valid JSON object — no markdown, no explanation:
{
  "table": "<table_name>",
  "user_col": "<column_for_user_id>",
  "item_col": "<column_for_item_id>",
  "rating_col": "<column_for_explicit_rating_or_null>",
  "timestamp_col": "<column_for_event_timestamp_or_null>",
  "algorithm_mode": "explicit" | "implicit",
  "reasoning": "<one-sentence explanation>",
  "sql": "<ready-to-execute SELECT statement>"
}

Rules:
- algorithm_mode must be "explicit" if a reliable numeric rating/score column exists,
  otherwise "implicit".
- sql must be a plain SELECT — no CTEs, no subqueries, no LIMIT clause.
- Use only columns and tables that appear in the schema metadata.
- If no suitable interaction table is found, return {"error": "<reason>"}.
"""

_REQUIRED_KEYS = {"table", "user_col", "item_col", "algorithm_mode", "sql"}


# ── Result dataclass ───────────────────────────────────────────────────────────

@dataclass
class PlannerResult:
    """Outcome of an LLM planning run."""

    success: bool
    tenant_id: str = ""
    table: str = ""
    user_col: str = ""
    item_col: str = ""
    rating_col: Optional[str] = None
    timestamp_col: Optional[str] = None
    algorithm_mode: str = ""       # "explicit" | "implicit"
    sql: str = ""
    reasoning: str = ""
    raw_response: str = ""         # full LLM text for audit log
    error: str = ""
    schema_snapshot: dict = field(default_factory=dict)


# ── Main planner ───────────────────────────────────────────────────────────────

class LLMPlanner:
    """
    Inspects a live database schema via ConnectionBuilder and asks Claude
    to produce a validated interaction query.

    Parameters
    ----------
    builder       : ConnectionBuilder configured for the target database
    tenant_id     : owning tenant (for logging and LLM context)
    api_key       : Anthropic API key; defaults to ANTHROPIC_API_KEY env var
    include_samples : whether to fetch sample rows for each table (richer
                      context but slower and uses more tokens)
    """

    def __init__(
        self,
        builder: ConnectionBuilder,
        tenant_id: str = "",
        api_key: Optional[str] = None,
        include_samples: bool = True,
    ) -> None:
        """Initialise the planner and validate that an Anthropic API key is available."""
        self._builder         = builder
        self.tenant_id        = tenant_id
        self._api_key         = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        self._include_samples = include_samples

        if not self._api_key:
            raise EnvironmentError(
                "LLMPlanner requires ANTHROPIC_API_KEY to be set. "
                "Set it as an environment variable or pass api_key= explicitly."
            )

    # ── Public API ─────────────────────────────────────────────────────────────

    def plan(self, algorithm_mode_hint: str = "") -> PlannerResult:
        """
        Introspect the database schema and call the LLM to produce a query plan.

        Parameters
        ----------
        algorithm_mode_hint : "explicit" | "implicit" | "" (let LLM decide)

        Returns
        -------
        PlannerResult
            .success=True  → .sql and column fields are ready to use
            .success=False → .error describes what went wrong
        """
        log.info("LLMPlanner.plan() | tenant=%s", self.tenant_id)

        try:
            schema = self._gather_schema()
        except Exception as exc:
            log.error("Schema gathering failed | tenant=%s error=%s", self.tenant_id, exc)
            return PlannerResult(
                success=False,
                tenant_id=self.tenant_id,
                error=f"Schema gathering failed: {exc}",
            )

        user_prompt = self._build_user_prompt(schema, algorithm_mode_hint)

        try:
            raw = self._call_llm(user_prompt)
        except Exception as exc:
            log.error("LLM call failed | tenant=%s error=%s", self.tenant_id, exc)
            return PlannerResult(
                success=False,
                tenant_id=self.tenant_id,
                error=f"LLM call failed: {exc}",
                schema_snapshot=schema,
            )

        return self._parse_response(raw, schema)

    # ── Schema gathering ───────────────────────────────────────────────────────

    def _gather_schema(self) -> dict:
        """
        Return a dict mapping table names to their column metadata (and
        optionally sample rows).
        """
        schema: dict = {}

        with self._builder.get_connector() as conn:
            try:
                tables = conn.get_table_names()
            except AttributeError:
                raise RuntimeError(
                    "The connected database does not support schema introspection "
                    "(get_table_names). Only SQL connectors are supported by LLMPlanner."
                )

            tables = tables[:_MAX_TABLES]
            get_sample = getattr(conn, "get_sample", None)
            log.info(
                "Schema discovery: %d tables | tenant=%s",
                len(tables), self.tenant_id,
            )

            for table in tables:
                cols = conn.get_columns(table)[:_MAX_COLS]
                entry: dict = {"columns": cols}

                if self._include_samples and callable(get_sample):
                    try:
                        sample_df = get_sample(table, _SAMPLE_ROWS)
                        entry["sample"] = sample_df.to_dict(orient="records")
                    except Exception as exc:
                        log.warning(
                            "Could not fetch sample for '%s' | tenant=%s: %s",
                            table, self.tenant_id, exc,
                        )

                schema[table] = entry

        return schema

    # ── Prompt construction ────────────────────────────────────────────────────

    def _build_user_prompt(self, schema: dict, algorithm_mode_hint: str) -> str:
        """Render a compact schema-and-samples prompt for the planner LLM."""
        lines: list[str] = [
            f"Tenant: {self.tenant_id or 'unknown'}",
            f"Database dialect: {getattr(self._builder.config, 'dialect', 'unknown')}",
        ]
        if algorithm_mode_hint:
            lines.append(
                f"Algorithm mode hint (prefer this if the schema supports it): "
                f"{algorithm_mode_hint}"
            )

        lines.append("\n### Schema\n")
        for table, meta in schema.items():
            lines.append(f"Table: {table}")
            for col in meta["columns"]:
                pk_tag = " [PK]" if col.get("primary_key") else ""
                null_tag = " NULL" if col.get("nullable") else " NOT NULL"
                lines.append(f"  - {col['name']} ({col['type']}{null_tag}{pk_tag})")
            if "sample" in meta and meta["sample"]:
                lines.append(f"  Sample rows ({len(meta['sample'])}):")
                for row in meta["sample"]:
                    lines.append(f"    {row}")
            lines.append("")

        return "\n".join(lines)

    # ── LLM call ──────────────────────────────────────────────────────────────

    def _call_llm(self, user_prompt: str) -> str:
        """Send the schema prompt to Claude and return the raw text response."""
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "Install the Anthropic SDK: pip install anthropic"
            )

        client = _ANTHROPIC_CLIENTS.get(self._api_key)
        if client is None:
            client = anthropic.Anthropic(api_key=self._api_key)
            _ANTHROPIC_CLIENTS[self._api_key] = client

        log.debug("Sending schema to LLM | tenant=%s model=%s", self.tenant_id, _MODEL)

        message = client.messages.create(
            model=_MODEL,
            max_tokens=_MAX_TOKENS,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )

        raw = message.content[0].text if message.content else ""
        log.debug(
            "LLM response received | tenant=%s stop_reason=%s length=%d",
            self.tenant_id, message.stop_reason, len(raw),
        )
        return raw

    # ── Response parsing ───────────────────────────────────────────────────────

    def _parse_response(self, raw: str, schema: dict) -> PlannerResult:
        """Parse and validate the LLM JSON response into a PlannerResult."""
        base = PlannerResult(
            tenant_id=self.tenant_id,
            raw_response=raw,
            schema_snapshot=schema,
            success=False,
        )

        # Strip accidental markdown fences
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = "\n".join(
                line for line in cleaned.splitlines()
                if not line.strip().startswith("```")
            ).strip()

        try:
            data: dict = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            log.error(
                "LLM returned non-JSON | tenant=%s error=%s raw_preview=%.200s",
                self.tenant_id, exc, raw,
            )
            base.error = f"LLM returned non-JSON: {exc}"
            return base

        # LLM signalled it couldn't find a suitable table
        if "error" in data:
            base.error = data["error"]
            log.warning("LLMPlanner: no suitable table | tenant=%s reason=%s", self.tenant_id, base.error)
            return base

        # Validate required keys
        missing = _REQUIRED_KEYS - data.keys()
        if missing:
            base.error = f"LLM response missing required keys: {sorted(missing)}"
            log.error("LLMPlanner validation failed | tenant=%s error=%s", self.tenant_id, base.error)
            return base

        log.info(
            "LLMPlanner success | tenant=%s table=%s algorithm_mode=%s",
            self.tenant_id, data["table"], data.get("algorithm_mode", ""),
        )

        return PlannerResult(
            success=True,
            tenant_id=self.tenant_id,
            table=data["table"],
            user_col=data["user_col"],
            item_col=data["item_col"],
            rating_col=data.get("rating_col") or None,
            timestamp_col=data.get("timestamp_col") or None,
            algorithm_mode=data.get("algorithm_mode", ""),
            sql=data["sql"],
            reasoning=data.get("reasoning", ""),
            raw_response=raw,
            schema_snapshot=schema,
        )
