"""
data_processing/data_cleaning.py - Data cleaning and validation pipeline

Steps:
  1. Select and rename standard columns (userID, itemID, rating, timestamp)
  2. Drop null / duplicate rows
  3. Coerce rating to numeric
  4. K-core pruning: remove users/items below interaction threshold
  5. Train/test split
  6. Cold-start filtering on test set
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import pandas as pd
from colorama import Fore, Style
from sklearn.model_selection import train_test_split

from backend.config.settings import TEST_RATIO, RANDOM_SEED, MIN_USER_INTERACTIONS, MIN_ITEM_INTERACTIONS
from backend.core.data_processing.dataset_analyzer import ColumnMapping
from backend.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class CleaningReport:
    """Summary of all cleaning operations applied."""

    initial_rows: int
    after_column_select: int
    after_dropna: int
    after_dedup: int
    after_kcore: int
    n_users: int
    n_items: int
    sparsity: float
    removed_null: int
    removed_dup: int
    removed_kcore_users: int
    removed_kcore_items: int


class DataCleaner:
    """
    Transforms a raw DataFrame into a clean, standardized interaction dataset
    ready for recommendation algorithm training.
    """

    def clean(
        self,
        df: pd.DataFrame,
        mapping: ColumnMapping,
        min_user_interactions: int = MIN_USER_INTERACTIONS,
        min_item_interactions: int = MIN_ITEM_INTERACTIONS,
    ) -> Tuple[pd.DataFrame, CleaningReport]:
        """
        Full cleaning pipeline.

        Returns
        -------
        clean_df : standardized DataFrame with columns [userID, itemID, rating, (timestamp)]
        report   : CleaningReport with per-step row counts
        """
        initial_rows = len(df)

        df = self._select_columns(df, mapping)
        after_select = len(df)

        df = df.dropna(subset=["userID", "itemID"])
        after_dropna = len(df)

        before_dedup = len(df)
        df = df.drop_duplicates(subset=["userID", "itemID"], keep="last")
        after_dedup = len(df)

        if "rating" in df.columns:
            df["rating"] = pd.to_numeric(df["rating"], errors="coerce").fillna(0)
        else:
            df["rating"] = 1.0

        df["userID"] = df["userID"].astype(str)
        df["itemID"] = df["itemID"].astype(str)

        df, removed_users, removed_items = self._kcore_pruning(
            df, min_user_interactions, min_item_interactions
        )
        after_kcore = len(df)

        n_users = df["userID"].nunique()
        n_items = df["itemID"].nunique()
        sparsity = 1 - len(df) / (n_users * n_items) if (n_users * n_items) > 0 else 1.0

        self._print_summary(df, n_users, n_items, sparsity)

        report = CleaningReport(
            initial_rows=initial_rows,
            after_column_select=after_select,
            after_dropna=after_dropna,
            after_dedup=after_dedup,
            after_kcore=after_kcore,
            n_users=n_users,
            n_items=n_items,
            sparsity=sparsity,
            removed_null=after_select - after_dropna,
            removed_dup=before_dedup - after_dedup,
            removed_kcore_users=removed_users,
            removed_kcore_items=removed_items,
        )
        log.info(
            "Cleaning complete | rows=%d users=%d items=%d sparsity=%.2f%%",
            after_kcore,
            n_users,
            n_items,
            sparsity * 100,
        )
        return df.reset_index(drop=True), report

    def split(
        self,
        df: pd.DataFrame,
        test_ratio: float = TEST_RATIO,
        seed: int = RANDOM_SEED,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Random train/test split preserving the implicit flag.
        Cold-start test users are removed from test.
        """
        if df.empty:
            raise ValueError(
                "Dataset validation failed: no interactions remained after cleaning, so train/test split cannot run."
            )
        if len(df) < 2:
            raise ValueError(
                f"Dataset validation failed: only {len(df)} interaction row remained after cleaning; "
                "at least 2 rows are required for train/test split."
            )

        is_implicit = df.attrs.get("is_implicit", False)

        if len(df) >= 1_000_000:
            key_cols = df[["userID", "itemID"]].astype(str)
            hashed = pd.util.hash_pandas_object(key_cols, index=False).astype("uint64")
            threshold = int(test_ratio * 10_000)
            mask = (hashed % 10_000) < threshold
            test = df[mask].copy()
            train = df[~mask].copy()
        else:
            train, test = train_test_split(df, test_size=test_ratio, random_state=seed)

        train.attrs["is_implicit"] = is_implicit
        test.attrs["is_implicit"] = is_implicit

        train_users = set(train["userID"].unique())
        test_users = set(test["userID"].unique())
        cold = test_users - train_users
        if cold:
            pct = len(cold) / len(test_users) * 100
            print(
                f"\n{Fore.YELLOW}Cold-start: {len(cold):,} test users "
                f"({pct:.1f}%) have no training history - removed.{Style.RESET_ALL}"
            )
            test = test[~test["userID"].isin(cold)].copy()
            test.attrs["is_implicit"] = is_implicit

        log.info("Split | train=%d test=%d cold_removed=%d", len(train), len(test), len(cold))
        return train.reset_index(drop=True), test.reset_index(drop=True)

    @staticmethod
    def _select_columns(df: pd.DataFrame, mapping: ColumnMapping) -> pd.DataFrame:
        rename_map = {}
        keep = list(df.columns)
        for role in ("userID", "itemID", "rating", "timestamp"):
            src = getattr(mapping, role)
            if src:
                rename_map[src] = role

        # If "interaction" was incorrectly mapped to timestamp and no rating exists,
        # promote it to rating instead
        if mapping.timestamp and "interaction" in mapping.timestamp.lower():
            if not mapping.rating:
                rename_map[mapping.timestamp] = "rating"
            else:
                # rating already exists — just drop the timestamp mapping, don't rename
                del rename_map[mapping.timestamp]

        if not rename_map.get(mapping.userID) and mapping.userID:
            rename_map[mapping.userID] = "userID"

        df = df[keep].rename(columns=rename_map)
        df = df.loc[:, ~df.columns.duplicated()]
        return df

    @staticmethod
    def _kcore_pruning(
        df: pd.DataFrame,
        min_user: int,
        min_item: int,
    ) -> Tuple[pd.DataFrame, int, int]:
        """
        Iteratively remove users and items below the interaction threshold.
        Iterates until convergence.
        """
        removed_users = removed_items = 0
        prev_len = -1

        while len(df) != prev_len:
            prev_len = len(df)

            user_counts = df["userID"].value_counts()
            valid_users = user_counts[user_counts >= min_user].index
            dropped_users = df["userID"].nunique() - len(valid_users)
            df = df[df["userID"].isin(valid_users)]
            removed_users += dropped_users

            item_counts = df["itemID"].value_counts()
            valid_items = item_counts[item_counts >= min_item].index
            dropped_items = df["itemID"].nunique() - len(valid_items)
            df = df[df["itemID"].isin(valid_items)]
            removed_items += dropped_items

        return df, removed_users, removed_items

    @staticmethod
    def _print_summary(df: pd.DataFrame, n_users: int, n_items: int, sparsity: float) -> None:
        print(f"\n   {Fore.CYAN}Users        : {n_users:,}{Style.RESET_ALL}")
        print(f"   {Fore.CYAN}Items        : {n_items:,}{Style.RESET_ALL}")
        print(f"   {Fore.CYAN}Interactions : {len(df):,}{Style.RESET_ALL}")
        print(f"   {Fore.CYAN}Sparsity     : {sparsity:.2%}{Style.RESET_ALL}")
        if sparsity > 0.99:
            print(
                f"   {Fore.YELLOW}Warning: very high sparsity - "
                f"memory-based methods (User-KNN) may perform poorly.{Style.RESET_ALL}"
            )
