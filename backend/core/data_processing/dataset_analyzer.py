"""
data_processing/dataset_analyzer.py – Dataset structure analysis and column detection

Migrated and upgraded from demo app/data.py with:
  - Fuzzy column matching with edit-distance fallback
  - Confidence scoring (HIGH / MEDIUM / LOW / FALLBACK)
  - Cardinality / dtype validation for ID columns
  - Non-interactive (API) and interactive (CLI) confirmation modes
  - Schema metadata extraction for relational datasets
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from colorama import Fore, Style

from backend.utils.logger import get_logger

log = get_logger(__name__)


# ── Hint dictionaries ─────────────────────────────────────────────────────────

_HINTS: dict[str, list[str]] = {
    "userID": [
        "user", "userid", "user_id", "customerid", "customer_id",
        "customer", "member", "memberid", "member_id", "uid",
        "client", "clientid", "client_id", "buyer", "account",
        "accountid", "account_id", "visitor", "visitorid",
    ],
    "itemID": [
        "item", "itemid", "item_id", "product", "productid", "product_id",
        "movie", "movieid", "movie_id", "song", "songid", "track", "trackid",
        "sku", "asin", "iid", "article", "articleid", "book", "bookid",
        "content", "contentid", "service", "serviceid", "listing",
    ],
    "rating": [
        "rating", "score", "stars", "review", "grade", "value",
        "feedback", "points", "rank", "preference", "like",
        "reaction", "vote", "weight",
    ],
    "timestamp": [
        "time", "timestamp", "date", "datetime", "created", "ts",
        "createdat", "created_at", "updatedat", "updated_at",
        "event_time", "interaction_time", "viewed_at", "purchased_at",
    ],
}
_ID_NEGATIVE_HINTS = ("type", "name", "title", "date", "time", "status", "action", "log", "desc", "description")


@dataclass
class ColumnMapping:
    """Detected mapping of dataset columns to standard roles."""
    userID:    Optional[str]
    itemID:    Optional[str]
    rating:    Optional[str]
    timestamp: Optional[str]
    confidence: dict  # role → (label, dist)

    def to_dict(self) -> dict:
        """Serialize the detected role mapping into the standard API shape."""
        return {
            "userID":    self.userID,
            "itemID":    self.itemID,
            "rating":    self.rating,
            "timestamp": self.timestamp,
        }


@dataclass
class DatasetStats:
    """Summary statistics of the detected interaction dataset."""
    n_users: int
    n_items: int
    n_interactions: int
    sparsity: float
    is_implicit: bool
    feedback_reason: str


# ── Fuzzy matching ────────────────────────────────────────────────────────────

def _normalise(s: str) -> str:
    return s.lower().replace("_", "").replace(" ", "").replace("-", "")


def _levenshtein(a: str, b: str) -> int:
    if a == b: return 0
    if not a:  return len(b)
    if not b:  return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            curr.append(min(prev[j] + 1, curr[j-1] + 1, prev[j-1] + (ca != cb)))
        prev = curr
    return prev[-1]


def _fuzzy_match(col: str, hints: list[str], max_dist: int = 2):
    norm_col   = _normalise(col)
    norm_hints = [_normalise(h) for h in hints]
    if norm_col in norm_hints:             return True, 0
    for nh in norm_hints:
        if nh in norm_col:                 return True, 1
    for nh in norm_hints:
        if norm_col in nh and len(norm_col) >= 2: return True, 2
    best = min(_levenshtein(norm_col, nh) for nh in norm_hints)
    if best <= max_dist:                   return True, best + 2
    return False, 999


def _confidence_label(dist: int) -> tuple[str, str]:
    if dist <= 1: return "HIGH",     Fore.GREEN
    if dist == 2: return "MEDIUM",   Fore.YELLOW
    if dist <= 3: return "LOW",      Fore.RED
    return             "FALLBACK",   Fore.RED


def _role_penalty(df: pd.DataFrame, col: str, role: str) -> int:
    norm_col = _normalise(col)
    series = df[col]
    n_total = max(len(series), 1)
    n_unique = max(int(series.nunique(dropna=True)), 0)
    ratio = n_unique / n_total
    penalty = 0

    if role in ("userID", "itemID"):
        if any(bad in norm_col for bad in _ID_NEGATIVE_HINTS):
            penalty += 4
        if norm_col == "id":
            penalty += 4
        if pd.api.types.is_float_dtype(series):
            penalty += 3
        if n_unique <= 1:
            penalty += 6
        elif n_unique <= 2:
            penalty += 5
        elif n_unique <= 5:
            penalty += 4
        elif n_unique <= 10:
            penalty += 3
        if n_total > 50:
            if ratio < 0.01:
                penalty += 5
            elif ratio < 0.05:
                penalty += 3
            elif ratio < 0.10:
                penalty += 2
        if norm_col.endswith("id") and norm_col != "id":
            penalty -= 1

    if role == "rating":
        if not pd.api.types.is_numeric_dtype(series):
            penalty += 2
        if n_unique > 500:
            penalty += 4

    if role == "timestamp":
        if not (
            pd.api.types.is_datetime64_any_dtype(series)
            or any(token in norm_col for token in ("time", "date", "created", "updated", "timestamp", "ts"))
        ):
            penalty += 2

    return max(penalty, 0)


def _is_safe_id_fallback(df: pd.DataFrame, col: str) -> bool:
    norm_col = _normalise(col)
    if norm_col == "id" or any(bad in norm_col for bad in _ID_NEGATIVE_HINTS):
        return False

    series = df[col]
    n_total = max(len(series), 1)
    n_unique = int(series.nunique(dropna=True))
    ratio = n_unique / n_total
    if n_unique < 10:
        return False
    if n_total > 50 and ratio < 0.10:
        return False
    if pd.api.types.is_float_dtype(series):
        return False
    return norm_col.endswith("id")


# ── Core detection ────────────────────────────────────────────────────────────

class DatasetAnalyzer:
    """
    Analyses a raw DataFrame to:
      1. Auto-detect column roles (userID, itemID, rating, timestamp)
      2. Validate cardinality / dtype of detected columns
      3. Produce a ColumnMapping and DatasetStats
    """

    def detect_columns(self, df: pd.DataFrame) -> ColumnMapping:
        """
        Run fuzzy matching + de-duplication to map columns to standard roles.
        Returns a ColumnMapping with confidence metadata.
        """
        # Phase 1: collect candidates
        candidates: dict[str, list] = {role: [] for role in _HINTS}
        for col in df.columns:
            for role, hints in _HINTS.items():
                matched, dist = _fuzzy_match(col, hints)
                if matched:
                    candidates[role].append((dist + _role_penalty(df, col, role), col))
        for role in candidates:
            candidates[role].sort(key=lambda x: x[0])

        # Phase 2: pick best per role
        col_map: dict[str, tuple] = {}
        for role in _HINTS:
            col_map[role] = (candidates[role][0][1], candidates[role][0][0]) \
                if candidates[role] else (None, 999)

        # Phase 2.5: block "interaction" from timestamp — it is an implicit signal, not a time value
        ts_col, ts_dist = col_map["timestamp"]
        if ts_col and _normalise(ts_col) == "interaction":
            col_map["timestamp"] = (None, 999)


        # Phase 3: de-duplicate (priority: userID > itemID > rating > timestamp)
        claimed: dict[str, str] = {}
        for role in ("userID", "itemID", "rating", "timestamp"):
            col, dist = col_map[role]
            if col is None: continue
            if col not in claimed:
                claimed[col] = role
            else:
                next_best = next(
                    ((d, c) for d, c in candidates[role][1:] if c not in claimed),
                    (999, None),
                )
                col_map[role] = (next_best[1], next_best[0])
                if next_best[1]: claimed[next_best[1]] = role

        # Phase 4: numeric fallbacks for still-undetected roles
        if col_map["rating"][0] is None:
            assigned = {c for c, _ in col_map.values() if c}
            for c in df.select_dtypes(include=[np.number]).columns:
                if c not in assigned and df[c].nunique() < 100:
                    col_map["rating"] = (c, 999); break

        assigned = {c for c, _ in col_map.values() if c}
        for role in ("userID", "itemID"):
            if col_map[role][0] is None:
                for c in df.columns:
                    if c in assigned:
                        continue
                    if _is_safe_id_fallback(df, c):
                        col_map[role] = (c, 999)
                        assigned.add(c)
                        break

        confidence = {role: _confidence_label(dist) for role, (_, dist) in col_map.items()}
        return ColumnMapping(
            userID    = col_map["userID"][0],
            itemID    = col_map["itemID"][0],
            rating    = col_map["rating"][0],
            timestamp = col_map["timestamp"][0],
            confidence= confidence,
        )

    def validate_mapping(self, df: pd.DataFrame, mapping: ColumnMapping) -> list[str]:
        """Return a list of validation warnings for the detected column mapping."""
        warnings = []
        if mapping.userID is None:
            warnings.append("No userID column could be detected with acceptable confidence.")
        if mapping.itemID is None:
            warnings.append("No itemID column could be detected with acceptable confidence.")
        for role, col in [("userID", mapping.userID), ("itemID", mapping.itemID)]:
            if col is None: continue
            series   = df[col]
            n_unique = series.nunique()
            n_total  = len(series)
            if pd.api.types.is_float_dtype(series):
                warnings.append(f"'{col}' mapped as {role} has float dtype — may be a value, not an ID.")
            if n_unique < 3:
                warnings.append(f"'{col}' mapped as {role} has only {n_unique} unique values.")
            if n_unique / n_total < 0.001 and n_total > 1000:
                warnings.append(f"'{col}' mapped as {role} has very low cardinality ({n_unique}/{n_total}).")

        if mapping.rating:
            n_unique = df[mapping.rating].nunique()
            if n_unique > 500:
                warnings.append(f"'{mapping.rating}' has {n_unique} unique values — may be price/revenue, not a rating.")
        return warnings

    def print_mapping(self, mapping: ColumnMapping, all_columns: list[str]) -> None:
        """Pretty-print the detected column mapping to stdout."""
        print(f"\n{Fore.YELLOW}Auto-detected column mapping:{Style.RESET_ALL}")
        print(f"   {'Role':<12}  {'Mapped To':<28}  {'Confidence':<10}  Status")
        print(f"   {'-'*12}  {'-'*28}  {'-'*10}  {'-'*20}")
        for role in ("userID", "itemID", "rating", "timestamp"):
            col   = getattr(mapping, role)
            label, color = mapping.confidence.get(role, ("UNKNOWN", Fore.WHITE))
            if col:
                print(f"   {role:<12}  {col:<28}  {color}{label:<10}{Style.RESET_ALL}  {Fore.GREEN}found{Style.RESET_ALL}")
            else:
                print(f"   {role:<12}  {'N/A':<28}  {Fore.RED}{'MISSING':<10}{Style.RESET_ALL}  {Fore.RED}not found{Style.RESET_ALL}")
        print(f"\n   Available columns: {all_columns}")

    def confirm_or_override(
        self, mapping: ColumnMapping, all_columns: list[str], interactive: bool = False
    ) -> ColumnMapping:
        """
        In interactive (CLI) mode, show mapping and allow user to remap any column.
        In non-interactive (API) mode, just return the detected mapping.
        """
        self.print_mapping(mapping, all_columns)

        low_conf = [r for r in ("userID", "itemID", "rating", "timestamp")
                    if mapping.confidence.get(r, ("", ""))[0] in ("LOW", "FALLBACK", "MEDIUM")]
        if low_conf:
            print(f"\n{Fore.RED}Warning: low-confidence detections for: {low_conf}{Style.RESET_ALL}")

        if not interactive:
            print(f"   {Fore.YELLOW}Non-interactive mode: accepting detected mapping.{Style.RESET_ALL}")
            return mapping

        mapping_dict = mapping.to_dict()
        print(f"\n{Fore.CYAN}Accept this mapping?{Style.RESET_ALL}")
        print(f"   Press {Fore.GREEN}Enter{Style.RESET_ALL} to accept, or type a role to remap it:")

        while True:
            try:
                answer = input("   > ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                break

            if answer in ("", "yes", "y", "done", "ok", "accept"):
                break

            role_map = {"userid": "userID", "itemid": "itemID", "rating": "rating", "timestamp": "timestamp"}
            if answer in role_map:
                role_key = role_map[answer]
                print(f"\n   Available: {all_columns}")
                print(f"   Current: {role_key} -> {mapping_dict.get(role_key)}")
                try:
                    new_col = input("   New column (or 'none'/'skip'): ").strip()
                except (EOFError, KeyboardInterrupt):
                    continue
                if new_col.lower() == "skip":     continue
                elif new_col.lower() == "none":   mapping_dict[role_key] = None
                elif new_col in all_columns:      mapping_dict[role_key] = new_col
                else:                             print(f"   {Fore.RED}'{new_col}' not found.{Style.RESET_ALL}")
            else:
                print(f"   {Fore.RED}Unknown role. Type: userID, itemID, rating, timestamp{Style.RESET_ALL}")

        return ColumnMapping(
            userID    = mapping_dict.get("userID"),
            itemID    = mapping_dict.get("itemID"),
            rating    = mapping_dict.get("rating"),
            timestamp = mapping_dict.get("timestamp"),
            confidence= mapping.confidence,
        )
