"""
Optuna hyperparameter optimization with resilient fallbacks.
"""
from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from typing import Optional

try:
    import optuna  # type: ignore
except ImportError:  # pragma: no cover
    optuna = None
import pandas as pd

from backend.config.settings import (
    BASE_TENANTS_DIR,
    OPTUNA_TIMEOUT_S,
    OPTUNA_TRIALS_AUTO,
    RANDOM_SEED,
    TOP_K,
)
from backend.utils.logger import get_logger
from backend.utils.metrics import evaluate_explicit, evaluate_ranking

if optuna is not None:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
log = get_logger(__name__)

_ALGO_TRIAL_BUDGETS = {
    "SVD": 20,
    "SVD++": 20,
    "NMF": 25,
    "ALS": 20,
    "BPR": 15,
    "Ecommerce-Popularity": 10,
    "Ecommerce-Purchase-ALS": 20,
    "Movie-Item-KNN": 15,
    "Temporal-SVD": 20,
    "LightFM": 20,
    "Autoencoder-CF": 12,
    "Content-Based TF-IDF": 12,
    "Factorization Machines": 15,
}
_DEFAULT_TRIALS = 20


def get_optuna_trial_budgets() -> dict[str, int]:
    """Return the adaptive per-algorithm trial budgets used for auto tuning."""
    return dict(_ALGO_TRIAL_BUDGETS)


def _suggest_params(trial, algorithm: str) -> dict:
    spaces = {
        "SVD": lambda t: {"n_components": t.suggest_int("n_components", 10, 200, step=10)},
        "SVD++": lambda t: {
            "n_components": t.suggest_int("n_components", 10, 100, step=10),
            "n_epochs": t.suggest_int("n_epochs", 10, 50, step=5),
            "lr": t.suggest_float("lr", 1e-4, 0.1, log=True),
            "reg": t.suggest_float("reg", 1e-4, 0.5, log=True),
        },
        "NMF": lambda t: {
            "n_components": t.suggest_int("n_components", 5, 100, step=5),
            "max_iter": t.suggest_int("max_iter", 100, 1000, step=100),
            "l1_ratio": t.suggest_float("l1_ratio", 0.0, 1.0),
        },
        "ALS": lambda t: {
            "n_factors": t.suggest_int("n_factors", 10, 200, step=10),
            "n_iterations": t.suggest_int("n_iterations", 5, 30),
            "regularization": t.suggest_float("regularization", 1e-4, 1.0, log=True),
            "alpha": t.suggest_float("alpha", 1.0, 100.0),
        },
        "BPR": lambda t: {
            "n_factors": t.suggest_int("n_factors", 10, 200, step=10),
            "n_epochs": t.suggest_int("n_epochs", 20, 200, step=20),
            "lr": t.suggest_float("lr", 1e-4, 0.1, log=True),
            "reg": t.suggest_float("reg", 1e-5, 0.1, log=True),
        },
        "Ecommerce-Popularity": lambda t: {
            "recency_decay": t.suggest_float("recency_decay", 0.8, 1.0),
            "min_interactions": t.suggest_int("min_interactions", 1, 20),
        },
        "Ecommerce-Purchase-ALS": lambda t: {
            "n_factors": t.suggest_int("n_factors", 10, 200, step=10),
            "n_iterations": t.suggest_int("n_iterations", 5, 30),
            "regularization": t.suggest_float("regularization", 1e-4, 1.0, log=True),
            "purchase_weight": t.suggest_float("purchase_weight", 2.0, 10.0),
        },
        "Movie-Item-KNN": lambda t: {
            "k_neighbours": t.suggest_int("k_neighbours", 5, 100, step=5),
            "genre_boost": t.suggest_float("genre_boost", 0.0, 0.5),
        },
        "Temporal-SVD": lambda t: {
            "n_components": t.suggest_int("n_components", 10, 100, step=10),
            "n_time_bins": t.suggest_int("n_time_bins", 3, 20),
        },
        "LightFM": lambda t: {
            "n_components": t.suggest_int("n_components", 16, 96, step=16),
            "content_weight": t.suggest_float("content_weight", 0.1, 0.7),
            "user_feature_weight": t.suggest_float("user_feature_weight", 0.05, 0.4),
            "max_features": t.suggest_int("max_features", 500, 3000, step=500),
        },
        "Autoencoder-CF": lambda t: {
            "hidden_dim": t.suggest_int("hidden_dim", 16, 128, step=16),
            "alpha": t.suggest_float("alpha", 1e-5, 1e-2, log=True),
            "max_iter": t.suggest_int("max_iter", 75, 200, step=25),
        },
        "Content-Based TF-IDF": lambda t: {
            "max_features": t.suggest_int("max_features", 500, 5000, step=500),
            "embedding_dim": t.suggest_int("embedding_dim", 16, 128, step=16),
        },
        "Factorization Machines": lambda t: {
            "n_factors": t.suggest_int("n_factors", 8, 48, step=8),
            "n_epochs": t.suggest_int("n_epochs", 10, 30, step=5),
            "learning_rate": t.suggest_float("learning_rate", 1e-3, 5e-2, log=True),
            "regularization": t.suggest_float("regularization", 1e-4, 1e-1, log=True),
            "max_features": t.suggest_int("max_features", 500, 2500, step=500),
        },
    }
    fn = spaces.get(algorithm)
    return fn(trial) if fn else {}


def _make_study(algorithm: str, direction: str, tenant_id: str = ""):
    """
    Create (or load) an Optuna study scoped to *tenant_id*.

    Each tenant gets its own SQLite database at
    ``tenants/{tenant_id}/optuna_studies.db`` so that:

      * Study names cannot collide across tenants.
      * One tenant's trial history is never visible to another.
      * Simultaneous training runs for the same algorithm by two tenants do
        not raise ``DuplicatedStudyError``.

    Falls back to an in-memory (``storage=None``) study when *tenant_id* is
    empty, preserving behaviour for non-tenant call-sites and unit tests.
    """
    if optuna is None:
        raise RuntimeError("optuna is not installed")
    safe_name  = algorithm.lower().replace(" ", "_").replace("/", "_").replace("+", "p")
    study_name = f"{safe_name}_study"
    storage    = None
    if tenant_id:
        db_path = os.path.join(BASE_TENANTS_DIR, tenant_id, "optuna_studies.db")
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        storage = f"sqlite:///{db_path}"

    return optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        direction=direction,
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3, interval_steps=1),
    )


@dataclass
class TuningResult:
    algorithm: str
    best_params: dict
    best_value: Optional[float]
    metric_name: str
    n_trials: int
    elapsed_s: float
    status: str = "ok"  # ok | skipped | fallback | failed
    fallback_reason: str = ""
    trial_budget: str = ""
    study: object | None = None

    def summary(self) -> dict:
        """Serialize the tuning outcome into a compact API-friendly summary."""
        return {
            "algorithm": self.algorithm,
            "best_params": self.best_params,
            "best_value": round(self.best_value, 4) if isinstance(self.best_value, (int, float)) else None,
            "metric_name": self.metric_name,
            "n_trials": self.n_trials,
            "elapsed_s": round(self.elapsed_s, 1),
            "status": self.status,
            "fallback_reason": self.fallback_reason,
            "trial_budget": self.trial_budget,
        }


class OptunaTuner:
    def tune(
        self,
        algorithm: str,
        train: pd.DataFrame,
        test: pd.DataFrame,
        n_trials: int = OPTUNA_TRIALS_AUTO,
        top_k: int = TOP_K,
        timeout: int = OPTUNA_TIMEOUT_S,
        tenant_id: str = "",
    ) -> TuningResult:
        """Tune one algorithm with Optuna and return the best valid trial result."""
        from backend.core.algorithms import ALGORITHM_REGISTRY

        meta = ALGORITHM_REGISTRY.get(algorithm)
        if not meta:
            return TuningResult(
                algorithm=algorithm,
                best_params={},
                best_value=None,
                metric_name="N/A",
                n_trials=0,
                elapsed_s=0.0,
                status="failed",
                fallback_reason=f"unknown algorithm: {algorithm}",
                trial_budget="n/a",
            )

        requested_trials = n_trials
        if requested_trials == OPTUNA_TRIALS_AUTO:
            requested_trials = _ALGO_TRIAL_BUDGETS.get(algorithm, _DEFAULT_TRIALS)
            trial_budget = f"auto({requested_trials})"
        else:
            trial_budget = f"fixed({requested_trials})"

        if requested_trials < 0:
            return TuningResult(
                algorithm=algorithm,
                best_params={},
                best_value=None,
                metric_name="N/A",
                n_trials=0,
                elapsed_s=0.0,
                status="failed",
                fallback_reason="n_trials must be -1 (auto) or >= 0",
                trial_budget=trial_budget,
            )
        if requested_trials == 0:
            return TuningResult(
                algorithm=algorithm,
                best_params={},
                best_value=None,
                metric_name="NDCG@K" if train.attrs.get("is_implicit", False) else "RMSE",
                n_trials=0,
                elapsed_s=0.0,
                status="skipped",
                fallback_reason="optuna skipped because n_trials=0",
                trial_budget=trial_budget,
            )

        if optuna is None:
            return TuningResult(
                algorithm=algorithm,
                best_params={},
                best_value=None,
                metric_name="NDCG@K" if train.attrs.get("is_implicit", False) else "RMSE",
                n_trials=0,
                elapsed_s=0.0,
                status="skipped",
                fallback_reason="optuna is not installed; tuning skipped",
                trial_budget=trial_budget,
            )

        is_implicit = train.attrs.get("is_implicit", False)
        metric_name = "NDCG@K" if is_implicit else "RMSE"
        direction = "maximize" if is_implicit else "minimize"
        fn = meta["fn"]

        study = _make_study(algorithm, direction, tenant_id=tenant_id)
        prior_complete = len(
            [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        )
        remaining = max(0, requested_trials - prior_complete)
        t0 = time.time()

        def objective(trial: optuna.Trial) -> float:
            """Evaluate one Optuna trial against the active ranking or RMSE metric."""
            params = _suggest_params(trial, algorithm)
            rating_preds, ranking_preds = fn(train, test, top_k, **params)
            if is_implicit:
                m = evaluate_ranking(test, ranking_preds, top_k)
                value = m.get("NDCG@K")
                if value is None or not math.isfinite(float(value)):
                    raise optuna.exceptions.TrialPruned()
                return float(value)
            m = evaluate_explicit(test, rating_preds)
            value = m.get("RMSE")
            if value is None or not math.isfinite(float(value)):
                raise optuna.exceptions.TrialPruned()
            return float(value)

        try:
            if remaining > 0:
                study.optimize(
                    objective,
                    n_trials=remaining,
                    timeout=timeout,
                    show_progress_bar=False,
                )
        except Exception as e:
            elapsed = time.time() - t0
            log.error("Optuna run failed | algo=%s error=%s", algorithm, e, exc_info=True)
            return TuningResult(
                algorithm=algorithm,
                best_params={},
                best_value=None,
                metric_name=metric_name,
                n_trials=len(study.trials),
                elapsed_s=elapsed,
                status="fallback",
                fallback_reason=f"optimization failed: {e}",
                trial_budget=trial_budget,
                study=study,
            )

        elapsed = time.time() - t0
        complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if not complete:
            return TuningResult(
                algorithm=algorithm,
                best_params={},
                best_value=None,
                metric_name=metric_name,
                n_trials=len(study.trials),
                elapsed_s=elapsed,
                status="fallback",
                fallback_reason="all trials pruned or invalid",
                trial_budget=trial_budget,
                study=study,
            )

        best = study.best_trial
        if best.value is None or not math.isfinite(float(best.value)):
            return TuningResult(
                algorithm=algorithm,
                best_params={},
                best_value=None,
                metric_name=metric_name,
                n_trials=len(study.trials),
                elapsed_s=elapsed,
                status="fallback",
                fallback_reason="best trial returned non-finite value",
                trial_budget=trial_budget,
                study=study,
            )

        return TuningResult(
            algorithm=algorithm,
            best_params=dict(best.params),
            best_value=float(best.value),
            metric_name=metric_name,
            n_trials=len(study.trials),
            elapsed_s=elapsed,
            status="ok",
            trial_budget=trial_budget,
            study=study,
        )

    def tune_top_n(
        self,
        algorithms: list[str],
        train: pd.DataFrame,
        test: pd.DataFrame,
        n_trials: int = OPTUNA_TRIALS_AUTO,
        top_k: int = TOP_K,
        timeout: int = OPTUNA_TIMEOUT_S,
    ) -> list[TuningResult]:
        """Tune a shortlist of algorithms and return them sorted by final quality."""
        is_implicit = train.attrs.get("is_implicit", False)
        results: list[TuningResult] = []

        for algorithm in algorithms:
            result = self.tune(
                algorithm=algorithm,
                train=train,
                test=test,
                n_trials=n_trials,
                top_k=top_k,
                timeout=timeout,
            )
            results.append(result)

        def sort_key(r: TuningResult) -> tuple[int, float]:
            """Sort tuned results by status first, then by metric direction."""
            # Prefer successful tunings over fallbacks/skips/failed.
            status_rank = {"ok": 0, "fallback": 1, "skipped": 2, "failed": 3}.get(r.status, 4)
            if r.best_value is None:
                metric_rank = float("-inf") if is_implicit else float("inf")
            else:
                metric_rank = float(r.best_value)
            return status_rank, (-metric_rank if is_implicit else metric_rank)

        results.sort(key=sort_key)
        return results
