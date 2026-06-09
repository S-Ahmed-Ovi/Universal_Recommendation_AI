"""
Algorithm benchmarking orchestrator.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
from colorama import Fore, Style

from backend.core.algorithms import ALGORITHM_REGISTRY
from backend.config.settings import TIER_A_ITEMS, TIER_A_USERS, TIER_B_ITEMS, TIER_B_USERS, TOP_K
from backend.core.data_processing.feedback_detector import FeedbackType
from backend.utils.logger import get_logger
from backend.utils.metrics import evaluate_all

log = get_logger(__name__)

MODE_TO_ALLOWED_FEEDBACK = {
    "explicit": {"explicit", "both"},
    "implicit": {"implicit", "both"},
    "hybrid": {"explicit", "implicit", "both"},
}


@dataclass
class BenchmarkResult:
    algorithm: str
    skipped: bool = False
    skip_reason: str = ""
    failed: bool = False
    error: str = ""
    elapsed_s: float = 0.0
    metrics: dict = field(default_factory=dict)
    sampled: bool = False
    scalability_score: float = 0.0

    def to_dict(self) -> dict:
        """Serialize one benchmark outcome into the leaderboard row format."""
        return {
            "Algorithm": self.algorithm,
            "Skipped": self.skipped,
            "Skip Reason": self.skip_reason,
            "Failed": self.failed,
            "Error": self.error,
            "Time (s)": round(self.elapsed_s, 2),
            "Sampled": self.sampled,
            "Scalability Score": round(self.scalability_score, 4),
            **self.metrics,
        }


@dataclass
class BenchmarkReport:
    results: list[BenchmarkResult]
    train: pd.DataFrame
    test: pd.DataFrame
    sampled: bool
    is_implicit: bool
    top_k: int
    resolved_mode: str
    primary_metric: str
    primary_metric_direction: str  # maximize | minimize
    ranking_logic: dict

    def leaderboard(self) -> pd.DataFrame:
        """Build the ranked benchmark leaderboard with composite scoring columns."""
        rows = [r.to_dict() for r in self.results if not r.skipped and not r.failed]
        df = pd.DataFrame(rows)
        if df.empty:
            return df

        if self.primary_metric not in df.columns:
            df[self.primary_metric] = None

        metric_values = pd.to_numeric(df[self.primary_metric], errors="coerce")
        speed_values = pd.to_numeric(df["Time (s)"], errors="coerce")
        scale_values = pd.to_numeric(df["Scalability Score"], errors="coerce").fillna(0.5)

        perf = self._normalize_metric(metric_values, self.primary_metric_direction)
        speed = self._normalize_metric(speed_values, "minimize")
        composite = (
            0.7 * perf.fillna(0.0)
            + 0.2 * speed.fillna(0.0)
            + 0.1 * scale_values.fillna(0.5).clip(0, 1)
        )
        df["Performance Score"] = perf.round(4)
        df["Speed Score"] = speed.round(4)
        df["Composite Score"] = composite.round(4)

        df = df.sort_values(
            by=["Composite Score", "Performance Score", "Speed Score"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
        df["Rank"] = df.index + 1
        return df

    def top_n(self, n: int = 10) -> list[BenchmarkResult]:
        """Return the top-N successful benchmark results in leaderboard order."""
        lb = self.leaderboard()
        if lb.empty:
            return []
        wanted = lb["Algorithm"].head(n).tolist()
        by_name = {r.algorithm: r for r in self.results}
        return [by_name[name] for name in wanted if name in by_name]

    def winner(self) -> Optional[BenchmarkResult]:
        """Return the highest-ranked benchmark result, if one exists."""
        top = self.top_n(1)
        return top[0] if top else None

    @staticmethod
    def _normalize_metric(values: pd.Series, direction: str) -> pd.Series:
        valid = values.dropna()
        if valid.empty:
            return pd.Series([0.0] * len(values), index=values.index, dtype=float)
        vmin = float(valid.min())
        vmax = float(valid.max())
        if vmax == vmin:
            return pd.Series([1.0] * len(values), index=values.index, dtype=float)
        if direction == "maximize":
            return (values - vmin) / (vmax - vmin)
        return (vmax - values) / (vmax - vmin)


class BenchmarkEngine:
    """
    Orchestrates benchmark execution with algorithm-mode filtering and
    transparent composite ranking.
    """

    def run(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
        top_k: int = TOP_K,
        force_all: bool = False,
        algorithm_mode: str = "auto",
        feedback_profile: Optional[FeedbackType] = None,
    ) -> BenchmarkReport:
        """Execute all compatible algorithms and return a fully ranked report."""
        resolved_mode = self._resolve_mode(train, algorithm_mode, feedback_profile)
        primary_metric, direction = self._primary_metric(resolved_mode)

        train, sampled = self._maybe_downsample(train, force_all, resolved_mode)
        if sampled:
            allowed_users = set(train["userID"].unique())
            allowed_items = set(train["itemID"].unique())
            test = test[test["userID"].isin(allowed_users) & test["itemID"].isin(allowed_items)].copy()
            test.attrs.update(train.attrs)
        n_users = train["userID"].nunique()
        n_items = train["itemID"].nunique()

        print(f"\n{Fore.CYAN}{'=' * 66}")
        print("  BENCHMARK ENGINE")
        print(f"  Users: {n_users:,}  Items: {n_items:,}  TopK: {top_k}")
        print(f"  Mode: {resolved_mode.upper()}  Primary metric: {primary_metric} ({direction})")
        print(f"{'=' * 66}{Style.RESET_ALL}\n")

        results: list[BenchmarkResult] = []
        for name, meta in ALGORITHM_REGISTRY.items():
            result = self._run_one(
                name=name,
                meta=meta,
                train=train,
                test=test,
                top_k=top_k,
                resolved_mode=resolved_mode,
                n_users=n_users,
                n_items=n_items,
                sampled=sampled,
            )
            results.append(result)

        self._print_summary(results)
        ranking_logic = {
            "primary_metric": primary_metric,
            "primary_metric_direction": direction,
            "weights": {
                "performance": 0.7,
                "training_speed": 0.2,
                "scalability": 0.1,
            },
            "notes": "Composite score ranks algorithms by normalized metric, runtime, and scale fit.",
        }

        return BenchmarkReport(
            results=results,
            train=train,
            test=test,
            sampled=sampled,
            is_implicit=(resolved_mode == "implicit"),
            top_k=top_k,
            resolved_mode=resolved_mode,
            primary_metric=primary_metric,
            primary_metric_direction=direction,
            ranking_logic=ranking_logic,
        )

    def _run_one(
        self,
        name: str,
        meta: dict,
        train: pd.DataFrame,
        test: pd.DataFrame,
        top_k: int,
        resolved_mode: str,
        n_users: int,
        n_items: int,
        sampled: bool,
    ) -> BenchmarkResult:
        skip_reason = self._check_compat(meta, resolved_mode, n_users, n_items)
        if skip_reason:
            print(f"  {Fore.YELLOW}skip {name:<24} {skip_reason}{Style.RESET_ALL}")
            return BenchmarkResult(algorithm=name, skipped=True, skip_reason=skip_reason)

        print(f"  run  {name:<24}", end="", flush=True)
        t0 = time.time()
        try:
            rating_preds, ranking_preds = meta["fn"](train, test, top_k)
            elapsed = time.time() - t0
            metrics = evaluate_all(test, rating_preds, ranking_preds, name, top_k)
            metrics.pop("Algorithm", None)
            scale_score = self._scalability_score(meta, n_users, n_items)
            print(f"{Fore.GREEN}ok {elapsed:.2f}s{Style.RESET_ALL}")
            return BenchmarkResult(
                algorithm=name,
                elapsed_s=elapsed,
                metrics=metrics,
                sampled=sampled,
                scalability_score=scale_score,
            )
        except Exception as e:
            elapsed = time.time() - t0
            print(f"{Fore.RED}fail {elapsed:.2f}s: {str(e)[:70]}{Style.RESET_ALL}")
            log.error("Benchmark failed | algo=%s error=%s", name, e, exc_info=True)
            return BenchmarkResult(
                algorithm=name,
                failed=True,
                error=str(e),
                elapsed_s=elapsed,
                scalability_score=self._scalability_score(meta, n_users, n_items),
            )

    @staticmethod
    def _resolve_mode(
        train: pd.DataFrame,
        requested_mode: str,
        feedback_profile: Optional[FeedbackType],
    ) -> str:
        req = (requested_mode or "auto").lower().strip()
        if req not in ("auto", "explicit", "implicit", "hybrid"):
            raise ValueError("algorithm_mode must be one of: explicit, implicit, hybrid, auto")
        if req != "auto":
            return req
        if feedback_profile:
            return feedback_profile.detected_mode
        return "implicit" if train.attrs.get("is_implicit", False) else "explicit"

    @staticmethod
    def _primary_metric(mode: str) -> tuple[str, str]:
        if mode == "explicit":
            return "RMSE", "minimize"
        return "NDCG@K", "maximize"

    @staticmethod
    def _check_compat(meta: dict, mode: str, n_users: int, n_items: int) -> str:
        feedback = meta.get("feedback", "both")
        allowed = MODE_TO_ALLOWED_FEEDBACK.get(mode, {"explicit", "implicit", "both"})
        if feedback not in allowed:
            return f"incompatible with mode={mode} (algo feedback={feedback})"

        max_u = meta.get("max_users")
        max_i = meta.get("max_items")
        if max_u and n_users > max_u:
            return f"dataset too large: users {n_users:,} > {max_u:,}"
        if max_i and n_items > max_i:
            return f"dataset too large: items {n_items:,} > {max_i:,}"
        return ""

    def _maybe_downsample(
        self,
        df: pd.DataFrame,
        force_all: bool,
        mode: str,
    ) -> tuple[pd.DataFrame, bool]:
        n_users = df["userID"].nunique()
        n_items = df["itemID"].nunique()

        blocked = 0
        for meta in ALGORITHM_REGISTRY.values():
            feedback = meta.get("feedback", "both")
            if feedback not in MODE_TO_ALLOWED_FEEDBACK.get(mode, set()):
                continue
            if (meta.get("max_users") and n_users > meta["max_users"]) or (
                meta.get("max_items") and n_items > meta["max_items"]
            ):
                blocked += 1

        if blocked == 0:
            return df, False

        target_users = TIER_B_USERS if force_all else TIER_A_USERS
        target_items = TIER_B_ITEMS if force_all else TIER_A_ITEMS
        target_users = min(target_users, n_users)
        target_items = min(target_items, n_items)

        if target_users >= n_users and target_items >= n_items:
            return df, False

        print(
            f"\n{Fore.YELLOW}Auto-downsampling {n_users:,}u x {n_items:,}i "
            f"-> <= {target_users:,}u x <= {target_items:,}i ({blocked} blocked algos){Style.RESET_ALL}"
        )
        top_users = set(df["userID"].value_counts().head(target_users).index)
        sub = df[df["userID"].isin(top_users)]
        top_items = set(sub["itemID"].value_counts().head(target_items).index)
        sub = sub[sub["itemID"].isin(top_items)].copy()
        sub.attrs.update(df.attrs)
        return sub, True

    @staticmethod
    def _scalability_score(meta: dict, n_users: int, n_items: int) -> float:
        # Metadata priors
        prior_map = {"low": 0.35, "medium": 0.65, "high": 0.9}
        prior = prior_map.get(str(meta.get("scalability", "medium")).lower(), 0.65)

        max_u = meta.get("max_users")
        max_i = meta.get("max_items")
        user_fit = 1.0 if not max_u else max(0.0, 1.0 - (n_users / max_u))
        item_fit = 1.0 if not max_i else max(0.0, 1.0 - (n_items / max_i))
        fit = (user_fit + item_fit) / 2

        # Blend static prior and current dataset fit.
        return max(0.0, min(1.0, 0.6 * prior + 0.4 * fit))

    @staticmethod
    def _print_summary(results: list[BenchmarkResult]) -> None:
        ran = [r for r in results if not r.skipped and not r.failed]
        skipped = [r for r in results if r.skipped]
        failed = [r for r in results if r.failed]
        print(
            f"\n{Fore.CYAN}Ran: {len(ran)} | Skipped: {len(skipped)} | Failed: {len(failed)}{Style.RESET_ALL}"
        )
