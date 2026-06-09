"""
Automatic explicit / implicit / hybrid feedback profiling.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
from colorama import Fore, Style

from backend.config.settings import ALGORITHM_MODES
from backend.utils.logger import get_logger

log = get_logger(__name__)

_IMPLICIT_HINTS = (
    "click",
    "view",
    "purchase",
    "buy",
    "order",
    "watch",
    "play",
    "impression",
    "dwell",
    "count",
    "event",
    "interaction",
)


@dataclass
class FeedbackType:
    # Backward-compatible fields
    is_implicit: bool
    reason: str
    label: str  # IMPLICIT | EXPLICIT | HYBRID
    rating_min: float
    rating_max: float
    n_unique: int

    # New capability fields
    supports_explicit: bool = False
    supports_implicit: bool = True
    detected_mode: str = "implicit"  # explicit | implicit | hybrid
    explicit_signals: list[str] = field(default_factory=list)
    implicit_signals: list[str] = field(default_factory=list)

    @property
    def supports_hybrid(self) -> bool:
        """Return whether the dataset can serve both explicit and implicit modes."""
        return self.supports_explicit and self.supports_implicit

    def to_dict(self) -> dict:
        """Serialize the detected feedback profile for APIs and logs."""
        return {
            "detected_mode": self.detected_mode,
            "supports_explicit": self.supports_explicit,
            "supports_implicit": self.supports_implicit,
            "supports_hybrid": self.supports_hybrid,
            "reason": self.reason,
            "rating_min": self.rating_min,
            "rating_max": self.rating_max,
            "n_unique": self.n_unique,
            "explicit_signals": self.explicit_signals,
            "implicit_signals": self.implicit_signals,
        }


class FeedbackDetector:
    """
    Determines feedback capabilities:
    - explicit-only
    - implicit-only
    - hybrid-capable (both)
    """

    def detect(self, series: pd.Series) -> FeedbackType:
        """Infer the feedback profile from a rating-like pandas Series."""
        vals = series.dropna()
        if vals.empty:
            return FeedbackType(
                is_implicit=True,
                reason="empty rating column; treating as implicit",
                label="IMPLICIT",
                rating_min=0.0,
                rating_max=0.0,
                n_unique=0,
                supports_explicit=False,
                supports_implicit=True,
                detected_mode="implicit",
                explicit_signals=[],
                implicit_signals=["rating"],
            )

        unique = sorted(vals.unique())
        n_unique = len(unique)
        vmin = float(vals.min())
        vmax = float(vals.max())

        inferred_implicit, reason = self._classify(vals, unique, n_unique, vmin, vmax)
        supports_explicit = self._supports_explicit(vals, n_unique, vmin, vmax)
        supports_implicit = True  # every interaction table can be interpreted implicitly

        detected_mode = self._mode_from_support(supports_explicit, supports_implicit)
        label = "HYBRID" if detected_mode == "hybrid" else detected_mode.upper()

        result = FeedbackType(
            is_implicit=inferred_implicit,
            reason=reason,
            label=label,
            rating_min=vmin,
            rating_max=vmax,
            n_unique=n_unique,
            supports_explicit=supports_explicit,
            supports_implicit=supports_implicit,
            detected_mode=detected_mode,
            explicit_signals=["rating"] if supports_explicit else [],
            implicit_signals=["rating"],
        )
        self._print_result(result)
        return result

    def detect_from_df(self, df: pd.DataFrame) -> FeedbackType:
        """Infer the feedback profile from a full interaction DataFrame."""
        implicit_cols = self._find_implicit_signal_columns(df)

        if "rating" not in df.columns:
            result = FeedbackType(
                is_implicit=True,
                reason="no rating column; treating as implicit",
                label="IMPLICIT",
                rating_min=0.0,
                rating_max=1.0,
                n_unique=1,
                supports_explicit=False,
                supports_implicit=True,
                detected_mode="implicit",
                explicit_signals=[],
                implicit_signals=implicit_cols or ["interaction"],
            )
            self._print_result(result)
            return result

        base = self.detect(df["rating"])
        if implicit_cols:
            # If the dataset has extra implicit behavior columns and explicit ratings,
            # classify as hybrid-capable.
            base.supports_implicit = True
            if base.supports_explicit:
                base.detected_mode = "hybrid"
                base.label = "HYBRID"
            base.implicit_signals = sorted(set(base.implicit_signals + implicit_cols))
        return base

    @staticmethod
    def resolve_mode(profile: FeedbackType, requested_mode: str) -> str:
        """Resolve the requested algorithm mode against the dataset capabilities."""
        mode = (requested_mode or "auto").lower().strip()
        if mode not in ALGORITHM_MODES:
            raise ValueError(
                f"Invalid algorithm_mode='{requested_mode}'. "
                f"Allowed: {', '.join(ALGORITHM_MODES)}."
            )

        if mode == "auto":
            return profile.detected_mode

        if mode == "explicit" and not profile.supports_explicit:
            raise ValueError("Explicit mode requested but dataset does not support explicit feedback.")
        if mode == "implicit" and not profile.supports_implicit:
            raise ValueError("Implicit mode requested but dataset does not support implicit feedback.")
        return mode

    @staticmethod
    def _find_implicit_signal_columns(df: pd.DataFrame) -> list[str]:
        out: list[str] = []
        for c in df.columns:
            lc = c.lower()
            if c == "rating":
                continue
            if any(h in lc for h in _IMPLICIT_HINTS):
                out.append(c)
        return out

    @staticmethod
    def _supports_explicit(vals: pd.Series, n_unique: int, vmin: float, vmax: float) -> bool:
        sample = vals.iloc[: min(300, len(vals))]
        has_decimals = any((float(v) % 1) != 0 for v in sample)
        if has_decimals and n_unique >= 5:
            return True
        if vmax > 1 and n_unique >= 3:
            return True
        if vmin >= 1 and vmax in (5, 10) and n_unique >= 3:
            return True
        return False

    @staticmethod
    def _classify(vals, unique, n_unique, vmin, vmax):
        if set(unique).issubset({0, 1}):
            return True, "binary values {0,1}; interpreted as implicit"

        if vmin >= 1 and vmax in (5, 10) and n_unique >= 3:
            return False, f"classic {int(vmin)}-{int(vmax)} rating scale"

        if (
            n_unique <= 5
            and vmax <= 10
            and vmax not in (5, 10)
            and all(float(v).is_integer() for v in unique)
        ):
            return True, f"low-cardinality integer counts {unique}; likely implicit"

        sample = vals.iloc[: min(200, len(vals))]
        if n_unique > 20 and not all(float(v).is_integer() for v in sample):
            return False, f"{n_unique} decimal-like values; likely explicit"

        if vmax > 1:
            return False, f"max value {vmax} > 1; assuming explicit"
        return True, f"max value {vmax} <= 1; assuming implicit"

    @staticmethod
    def _mode_from_support(supports_explicit: bool, supports_implicit: bool) -> str:
        if supports_explicit and supports_implicit:
            return "hybrid"
        if supports_explicit:
            return "explicit"
        return "implicit"

    @staticmethod
    def _print_result(result: FeedbackType) -> None:
        mode_color = {
            "explicit": Fore.MAGENTA,
            "implicit": Fore.BLUE,
            "hybrid": Fore.CYAN,
        }.get(result.detected_mode, Fore.WHITE)
        print(
            f"\n{mode_color}Feedback mode: {result.detected_mode.upper()}{Style.RESET_ALL}"
            f"\n   Reason           : {result.reason}"
            f"\n   Rating range     : [{result.rating_min}, {result.rating_max}]"
            f"\n   Unique ratings   : {result.n_unique}"
            f"\n   Supports explicit: {result.supports_explicit}"
            f"\n   Supports implicit: {result.supports_implicit}"
        )

