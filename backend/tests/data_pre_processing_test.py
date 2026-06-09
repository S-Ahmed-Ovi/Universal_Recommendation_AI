"""
data_pre_processing_test.py – End-to-end data processing test script

Usage:
    python data_pre_processing_test.py <path_to_csv>

What it does:
    1. Loads your CSV file
    2. Auto-detects column roles  (DatasetAnalyzer)
    3. Profiles feedback type     (FeedbackDetector)
    4. Auto-tunes k-core thresholds based on data distribution
    5. Cleans the data            (DataCleaner.clean)
    6. Splits train / test        (DataCleaner.split)
    7. Builds sparse matrix       (InteractionMatrixBuilder)
    8. Prints before/after counts at every step + final cleaned DataFrame
"""
from __future__ import annotations

import sys
import os
import logging

if "pytest" in sys.modules:
    import pytest

    pytest.skip("Utility script; excluded from automated pytest collection.", allow_module_level=True)

# ── Inline config (replaces config/settings.py & utils/logger.py) ────────────

TEST_RATIO      = 0.2
RANDOM_SEED     = 42
ALGORITHM_MODES = {"auto", "explicit", "implicit", "hybrid"}

# K-core thresholds – auto-tuned at runtime; these are only the monkey-patch defaults.
MIN_USER_INTERACTIONS = 1
MIN_ITEM_INTERACTIONS = 1


def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger(name)


# ── Monkey-patch the imports the modules expect ───────────────────────────────
import types

cfg = types.ModuleType("config.settings")
cfg.TEST_RATIO             = TEST_RATIO
cfg.RANDOM_SEED            = RANDOM_SEED
cfg.MIN_USER_INTERACTIONS  = MIN_USER_INTERACTIONS
cfg.MIN_ITEM_INTERACTIONS  = MIN_ITEM_INTERACTIONS
cfg.ALGORITHM_MODES        = ALGORITHM_MODES
sys.modules["config"]          = types.ModuleType("config")
sys.modules["config.settings"] = cfg

ul = types.ModuleType("utils.logger")
ul.get_logger = get_logger
sys.modules["utils"]        = types.ModuleType("utils")
sys.modules["utils.logger"] = ul

sys.modules["core"]                 = types.ModuleType("core")
sys.modules["core.data_processing"] = types.ModuleType("core.data_processing")

# ── Load the four pipeline modules from core/data_processing/ ────────────────

import importlib.util as ilu

def _load(name: str, path: str):
    spec = ilu.spec_from_file_location(name, path)
    mod  = ilu.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

_here = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),  # tests/
    "..", "core", "data_processing"              # → core/data_processing/
)

dataset_analyzer_mod   = _load("dataset_analyzer",  os.path.join(_here, "dataset_analyzer.py"))
sys.modules["core.data_processing.dataset_analyzer"] = dataset_analyzer_mod  # must be before data_cleaning

feedback_detector_mod  = _load("feedback_detector",  os.path.join(_here, "feedback_detector.py"))
data_cleaning_mod      = _load("data_cleaning",      os.path.join(_here, "data_cleaning.py"))
interaction_matrix_mod = _load("interaction_matrix", os.path.join(_here, "interaction_matrix.py"))

from dataset_analyzer   import DatasetAnalyzer, ColumnMapping
from feedback_detector  import FeedbackDetector
from data_cleaning      import DataCleaner
from interaction_matrix import InteractionMatrixBuilder

# ── Colorama (graceful fallback if not installed) ─────────────────────────────
try:
    from colorama import Fore, Style, init as _cinit
    _cinit(autoreset=True)
except ImportError:
    class _C:
        def __getattr__(self, _): return ""
    Fore = Style = _C()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _section(title: str) -> None:
    sep = "─" * 60
    print(f"\n{Fore.YELLOW}{sep}")
    print(f"  {title}")
    print(f"{sep}{Style.RESET_ALL}")


def _auto_kcore(df, user_col: str, item_col: str):
    """
    Pick the highest k-core threshold that still retains >= 80% of interactions.
    Tries k = 1, 2, 3, 4, 5 and returns the largest k that meets the bar.
    Falls back to k=1 (keep everything) if even k=2 drops too many rows.
    """
    n_total = len(df)
    chosen = 1
    for k in range(1, 6):
        user_ok = df[user_col].value_counts() >= k
        item_ok = df[item_col].value_counts() >= k
        retained = df[
            df[user_col].isin(user_ok[user_ok].index) &
            df[item_col].isin(item_ok[item_ok].index)
        ]
        pct = len(retained) / n_total
        print(f"  k={k}  →  {len(retained):,} rows retained  ({pct*100:.1f}%)")
        if pct >= 0.80:
            chosen = k
        else:
            break  # further increases will only drop more
    return chosen, chosen


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run(csv_path: str) -> None:
    import pandas as pd

    # ── 1. Load ───────────────────────────────────────────────────────────────
    _section("STEP 1 – Load CSV")
    print(f"  File   : {csv_path}")

    df_raw = pd.read_csv(csv_path)
    n_raw  = len(df_raw)
    print(f"  Shape  : {n_raw:,} rows × {df_raw.shape[1]} columns")
    print(f"  Columns: {list(df_raw.columns)}")
    print(f"\n  First 3 rows:")
    print(df_raw.head(3).to_string(index=False))

    # ── 2. Detect columns ─────────────────────────────────────────────────────
    _section("STEP 2 – Auto-detect column roles")

    analyzer = DatasetAnalyzer()
    mapping  = analyzer.detect_columns(df_raw)

    # Fix: clear timestamp if the mapped column doesn't actually contain dates/numbers.
    # The 'interaction' column (purchase/view/etc.) is a common false-positive.
    if mapping.timestamp:
        ts_series = df_raw[mapping.timestamp]
        looks_numeric  = pd.to_numeric(ts_series, errors="coerce").notna().mean() > 0.8
        looks_datetime = False
        try:
            pd.to_datetime(ts_series, errors="raise")
            looks_datetime = True
        except Exception:
            pass
        if not looks_numeric and not looks_datetime:
            print(f"  {Fore.YELLOW}ℹ  '{mapping.timestamp}' does not look like a timestamp "
                  f"– clearing timestamp mapping.{Style.RESET_ALL}")
            mapping = ColumnMapping(
                userID=mapping.userID, itemID=mapping.itemID,
                rating=mapping.rating, timestamp=None,
                confidence=mapping.confidence,
            )

    analyzer.print_mapping(mapping, list(df_raw.columns))

    for w in analyzer.validate_mapping(df_raw, mapping):
        print(f"  {Fore.RED}⚠  {w}{Style.RESET_ALL}")

    if mapping.userID is None or mapping.itemID is None:
        print(f"\n{Fore.RED}ERROR: Could not detect userID or itemID columns. "
              f"Please rename your columns and try again.{Style.RESET_ALL}")
        sys.exit(1)

    # ── 3. Detect feedback type ───────────────────────────────────────────────
    _section("STEP 3 – Detect feedback type")

    detector = FeedbackDetector()
    feedback = detector.detect_from_df(df_raw.rename(columns={
        mapping.userID: "userID",
        mapping.itemID: "itemID",
        **({"rating": "rating"} if mapping.rating and mapping.rating != "rating" else {}),
    }))

    # ── 4. Auto-tune k-core thresholds ───────────────────────────────────────
    _section("STEP 4 – Auto-tune k-core thresholds")

    user_counts = df_raw[mapping.userID].value_counts()
    item_counts = df_raw[mapping.itemID].value_counts()
    print(f"  User interaction counts → median={user_counts.median():.0f}, "
          f"min={user_counts.min()}, max={user_counts.max()}")
    print(f"  Item interaction counts → median={item_counts.median():.0f}, "
          f"min={item_counts.min()}, max={item_counts.max()}")
    print()

    min_u, min_i = _auto_kcore(df_raw, mapping.userID, mapping.itemID)
    print(f"\n  {Fore.CYAN}Chosen MIN_USER_INTERACTIONS = {min_u}{Style.RESET_ALL}")
    print(f"  {Fore.CYAN}Chosen MIN_ITEM_INTERACTIONS = {min_i}{Style.RESET_ALL}")

    # ── 5. Clean ──────────────────────────────────────────────────────────────
    _section("STEP 5 – Clean data")

    print(f"  Before cleaning : {n_raw:,} rows\n")

    cleaner  = DataCleaner()
    clean_df, report = cleaner.clean(
        df_raw,
        mapping,
        min_user_interactions=min_u,
        min_item_interactions=min_i,
    )
    clean_df.attrs["is_implicit"] = feedback.is_implicit

    # Pretty before/after table
    print(f"\n  {'Step':<28}  {'Before':>8}    {'After':>8}    {'Removed':>8}")
    print(f"  {'─'*28}  {'─'*8}    {'─'*8}    {'─'*8}")
    steps = [
        ("Column select",     report.initial_rows,        report.after_column_select, report.initial_rows - report.after_column_select),
        ("Drop nulls",        report.after_column_select, report.after_dropna,        report.removed_null),
        ("Dedup (user+item)", report.after_dropna,        report.after_dedup,         report.removed_dup),
        ("K-core pruning",    report.after_dedup,         report.after_kcore,         report.after_dedup - report.after_kcore),
    ]
    for label, before, after, removed in steps:
        color = Fore.RED if removed > 0 else Fore.GREEN
        print(f"  {label:<28}  {before:>8,}    {after:>8,}    "
              f"{color}{removed:>8,}{Style.RESET_ALL}")

    pct_kept = len(clean_df) / n_raw * 100 if n_raw else 0
    print(f"\n  Raw rows      : {n_raw:,}")
    print(f"  Cleaned rows  : {len(clean_df):,}  ({pct_kept:.1f}% of original)")
    print(f"  Unique users  : {report.n_users:,}")
    print(f"  Unique items  : {report.n_items:,}")
    print(f"  Sparsity      : {report.sparsity:.4%}")

    if len(clean_df) == 0:
        print(f"\n{Fore.RED}ERROR: All rows removed during cleaning. "
              f"Check your data or lower the k-core thresholds.{Style.RESET_ALL}")
        sys.exit(1)

    # ── 6. Train / test split ─────────────────────────────────────────────────
    _section(f"STEP 6 – Train / test split  (test ratio = {TEST_RATIO})")

    print(f"  Before split : {len(clean_df):,} rows")
    train_df, test_df = cleaner.split(clean_df, test_ratio=TEST_RATIO, seed=RANDOM_SEED)
    print(f"  Train set    : {len(train_df):,} rows  ({len(train_df)/len(clean_df)*100:.1f}%)")
    print(f"  Test  set    : {len(test_df):,} rows  ({len(test_df)/len(clean_df)*100:.1f}%)")

    # ── 7. Build interaction matrix ───────────────────────────────────────────
    _section("STEP 7 – Build sparse interaction matrix")

    print(f"  Input (train set) : {len(train_df):,} rows")
    builder = InteractionMatrixBuilder()
    im      = builder.build(train_df)
    print(f"  Shape   : {im.n_users:,} users × {im.n_items:,} items")
    print(f"  NNZ     : {im.matrix.nnz:,}")
    print(f"  Density : {im.density:.4%}")
    print(f"  Implicit: {im.is_implicit}")

    # ── 8. Final cleaned data preview ─────────────────────────────────────────
    _section("FINAL – Cleaned DataFrame")

    print(f"  Raw rows     : {n_raw:,}")
    print(f"  Cleaned rows : {len(clean_df):,}  ({pct_kept:.1f}% retained)")
    print(f"  Columns      : {list(clean_df.columns)}")
    print()

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 120)
    print(clean_df.head(50).to_string(index=False))

    if len(clean_df) > 50:
        print(f"\n  ... and {len(clean_df) - 50:,} more rows (showing first 50).")

    print(f"\n{Fore.GREEN}✔  Pipeline complete!{Style.RESET_ALL}\n")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python data_pre_processing_test.py <path_to_csv>")
        print("Example: python data_pre_processing_test.py data.csv")
        sys.exit(1)

    run(sys.argv[1])
