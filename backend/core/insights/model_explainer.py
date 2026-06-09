"""
model_explainer.py — Automated explanation for benchmarked models.

Generates factual and LLM-enriched explanations for why a specific model
was selected as the winner of a benchmark run.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any, Optional

import pandas as pd
from backend.core.benchmark.benchmark_engine import BenchmarkReport, BenchmarkResult
from backend.core.algorithms import ALGORITHM_REGISTRY
from backend.config.settings import (
    BASE_TENANTS_DIR,
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_EXPLANATIONS_ENABLED,
    LLM_MODEL,
    LLM_PROVIDER,
)
from backend.utils.logger import get_logger

log = get_logger(__name__)
_OPENAI_CLIENTS: dict[tuple[str, str], Any] = {}

# ── JSON Schema for LLM Output ────────────────────────────────────────────────

EXPECTED_SCHEMA = {
    "type": "object",
    "properties": {
        "why_recommended": {"type": "string"},
        "data_fit": {"type": "string"},
        "expected_accuracy": {"type": "string"},
        "scalability": {"type": "string"},
        "training_speed": {"type": "string"},
        "robustness": {"type": "string"},
        "interpretability": {"type": "string"},
        "production_ready": {"type": "string"},
        "trade_offs": {"type": "string"},
        "user_friendly_summary": {"type": "string"},
        "tags": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "why_recommended",
        "data_fit",
        "expected_accuracy",
        "scalability",
        "training_speed",
        "robustness",
        "interpretability",
        "production_ready",
        "trade_offs",
        "user_friendly_summary",
        "tags",
    ],
}

_SYSTEM_PROMPT = """\
You are an expert data scientist and machine learning engineer.
Your task is to enrich a deterministic model explanation with more nuance and natural language.
You will be provided with factual data about a benchmark run and an algorithm.

RULES:
- Return ONLY strict JSON.
- Do NOT include any markdown formatting (no ```json).
- Do NOT change the results; only enrich the explanation.
- Use the provided factual data as the ground truth.
- If you don't have enough data for a field, provide a reasonable professional estimate based on the algorithm's known characteristics.
- Ensure the tone is professional, insightful, and helpful.
"""

# ── ModelExplainer ────────────────────────────────────────────────────────────

class ModelExplainer:
    """
    Orchestrates the generation of model-level explanations.

    Always instantiate with a tenant-scoped cache path to prevent cross-tenant
    explanation leakage.  Use the for_tenant factory rather than the
    constructor directly:

        explainer = ModelExplainer.for_tenant("acme-corp")
        result = explainer.explain_winner(report)
    """

    def __init__(self, cache_path: str):
        """Initialise a tenant-scoped explanation cache backed by a JSON file."""
        self.cache_path = cache_path
        self._cache = self._load_cache()

    @classmethod
    def for_tenant(cls, tenant_id: str) -> "ModelExplainer":
        """
        Return a ModelExplainer whose cache is scoped to tenant_id.

        Cache files live at tenants/<tenant_id>/explanation_cache.json
        so no two tenants can ever read each other's cached narratives.
        """
        cache_path = os.path.join(BASE_TENANTS_DIR, tenant_id, "explanation_cache.json")
        return cls(cache_path=cache_path)

    def explain_winner(self, report: BenchmarkReport) -> dict:
        """
        Generates a full explanation for the winning model in a benchmark report.
        """
        winner = report.winner()
        if not winner:
            return {"error": "No winning model found in benchmark report."}

        # 1. Build factual input
        facts = self._build_facts(report, winner)

        # 2. Check cache first — avoid wasted computation on cache hits
        fingerprint = self._get_fingerprint(facts)
        if fingerprint in self._cache:
            log.info("Found cached explanation for %s", winner.algorithm)
            return self._cache[fingerprint]

        # 3. Generate deterministic explanation (only on cache miss)
        deterministic = self._generate_deterministic(facts)

        # 4. LLM enrichment if enabled
        final_result = deterministic
        if LLM_EXPLANATIONS_ENABLED and LLM_API_KEY:
            try:
                enriched = self._call_llm(facts, deterministic)
                if enriched:
                    # Validate and merge
                    if self._validate_schema(enriched):
                        final_result = enriched
                    else:
                        log.warning("LLM output failed schema validation. Falling back.")
            except Exception as e:
                log.error("LLM explanation failed: %s. Falling back.", e)

        self._save_to_cache(fingerprint, final_result)
        return final_result

    # ── Internal Helpers ───────────────────────────────────────────────────────

    def _build_facts(self, report: BenchmarkReport, winner: BenchmarkResult) -> dict:
        """Extract the deterministic facts that both caching and LLM enrichment depend on."""
        lb = report.leaderboard()
        winner_row = lb[lb["Algorithm"] == winner.algorithm].iloc[0]

        algo_meta = ALGORITHM_REGISTRY.get(winner.algorithm, {})

        facts = {
            "algorithm_name": winner.algorithm,
            "rank": int(winner_row["Rank"]),
            "composite_score": float(winner_row["Composite Score"]),
            "primary_metric": report.primary_metric,
            "primary_metric_value": float(winner_row[report.primary_metric]),
            "primary_metric_direction": report.primary_metric_direction,
            "training_time_s": float(winner_row["Time (s)"]),
            "scalability_score": float(winner_row["Scalability Score"]),
            "n_users": report.train["userID"].nunique(),
            "n_items": report.train["itemID"].nunique(),
            "n_interactions": len(report.train),
            "algorithm_mode": report.resolved_mode,
            "feedback_type": "implicit" if report.is_implicit else "explicit",
            "supports_hybrid": algo_meta.get("feedback") == "both",
            "description": algo_meta.get("description", ""),
            "metadata": {
                "training_speed": algo_meta.get("training_speed"),
                "scalability": algo_meta.get("scalability"),
                "robustness": algo_meta.get("robustness"),
                "interpretability": algo_meta.get("interpretability"),
                "production_readiness": algo_meta.get("production_readiness"),
            },
            "skip_reasons": [r.skip_reason for r in report.results if r.skipped],
        }
        return facts

    def _generate_deterministic(self, facts: dict) -> dict:
        """Rule-based fallback explanation."""
        algo = facts["algorithm_name"]
        metric = facts["primary_metric"]
        val = facts["primary_metric_value"]

        why = (
            f"{algo} was selected because it achieved the highest composite score "
            f"in the benchmark. It delivered a {metric} of {val:.4f}."
        )

        data_fit = (
            f"The model successfully processed {facts['n_interactions']:,} interactions "
            f"across {facts['n_users']:,} users and {facts['n_items']:,} items."
        )

        return {
            "why_recommended": why,
            "data_fit": data_fit,
            "expected_accuracy": f"High performance on {metric} relative to other candidates.",
            "scalability": f"Scalability is rated as {facts['metadata']['scalability']}.",
            "training_speed": f"Training completed in {facts['training_time_s']:.2f}s.",
            "robustness": f"Robustness level: {facts['metadata']['robustness']}.",
            "interpretability": f"Interpretability level: {facts['metadata']['interpretability']}.",
            "production_ready": f"Production readiness: {facts['metadata']['production_readiness']}.",
            "trade_offs": "Optimized for a balance of accuracy, speed, and scalability.",
            "user_friendly_summary": f"{algo} provides the best overall recommendation quality for this dataset.",
            "tags": [facts["algorithm_mode"], facts["feedback_type"], "benchmark-winner"],
        }

    def _call_llm(self, facts: dict, deterministic: dict) -> Optional[dict]:
        """Calls the configured LLM provider."""
        if LLM_PROVIDER == "openai":
            return self._call_openai(facts, deterministic)
        elif LLM_PROVIDER == "groq":
            return self._call_openai(facts, deterministic, base_url="https://api.groq.com/openai/v1")
        elif LLM_PROVIDER == "local":
            return self._call_openai(facts, deterministic, is_local=True)
        else:
            log.warning("Unsupported LLM provider: %s", LLM_PROVIDER)
            return None

    def _call_openai(
        self,
        facts: dict,
        deterministic: dict,
        is_local: bool = False,
        base_url: Optional[str] = None
    ) -> Optional[dict]:
        """Call an OpenAI-compatible chat endpoint and return parsed JSON output."""
        try:
            from openai import OpenAI
        except ImportError:
            log.error("openai package not installed. Run 'pip install openai'.")
            return None

        resolved_base_url = LLM_BASE_URL if is_local and LLM_BASE_URL else base_url
        cache_key = (resolved_base_url or "", LLM_API_KEY)
        client = _OPENAI_CLIENTS.get(cache_key)
        if client is None:
            client_kwargs = {"api_key": LLM_API_KEY}
            if resolved_base_url:
                client_kwargs["base_url"] = resolved_base_url
            client = OpenAI(**client_kwargs)
            _OPENAI_CLIENTS[cache_key] = client

        user_content = (
            f"FACTUAL DATA:\n{json.dumps(facts, indent=2)}\n\n"
            f"DETERMINISTIC EXPLANATION:\n{json.dumps(deterministic, indent=2)}\n\n"
            "Please enrich this explanation following the rules."
        )

        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            response_format={"type": "json_object"},
            timeout=30,
        )

        raw_json = response.choices[0].message.content
        if not raw_json:
            return None

        return json.loads(raw_json)

    def _validate_schema(self, data: dict) -> bool:
        """Validates that the LLM output matches the expected schema."""
        # Simple manual validation for speed and to avoid external deps if possible
        required = EXPECTED_SCHEMA["required"]
        for key in required:
            if key not in data:
                log.error("Missing key in LLM output: %s", key)
                return False
        if not isinstance(data.get("tags"), list):
            log.error("'tags' is not a list in LLM output")
            return False
        return True

    def _get_fingerprint(self, facts: dict) -> str:
        """Generates a unique fingerprint for caching."""
        # Focus on algorithm, dataset size, and benchmark results
        key_parts = [
            facts["algorithm_name"],
            str(facts["n_users"]),
            str(facts["n_items"]),
            str(facts["n_interactions"]),
            facts["feedback_type"],
            facts["primary_metric"],
            f"{facts['primary_metric_value']:.6f}",
        ]
        key_str = "|".join(key_parts)
        return hashlib.sha256(key_str.encode()).hexdigest()

    def _load_cache(self) -> dict:
        """Load the tenant's explanation cache from disk if it exists."""
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                log.warning("Failed to load explanation cache: %s", e)
        return {}

    def _save_to_cache(self, fingerprint: str, result: dict):
        """Persist a generated explanation into the tenant-scoped JSON cache."""
        self._cache[fingerprint] = result
        try:
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            with open(self.cache_path, "w") as f:
                json.dump(self._cache, f, indent=2)
        except Exception as e:
            log.warning("Failed to save explanation cache: %s", e)
