"""
feature_extractor.py
--------------------
Centralised feature engineering for PatGPT ML layer.

Rules:
  - All rolling features are computed from rows STRICTLY BEFORE the
    target interaction date (no data leakage).
  - extract(doctor_id) → Dict ready for sklearn pipelines.
  - extract_batch(df) → pd.DataFrame of one row per interaction,
    used for offline training.

Usage (serving):
    extractor = FeatureExtractor(df)
    features  = extractor.extract("DOC001")

Usage (training):
    extractor = FeatureExtractor(df)
    X, y      = extractor.extract_batch(df)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List, Optional


# ── Ordinal mapping for AIDA stage ────────────────────────────────────────────
AIDA_ORDINAL: Dict[str, int] = {
    "awareness": 0,
    "interest": 1,
    "desire": 2,
    "action": 3,
}

# ── Specialty Bayesian priors (fallback for cold-start) ───────────────────────
# Approximate positive-outcome base rates per specialty; update from data.
SPECIALTY_PRIORS: Dict[str, float] = {
    "cardiologist":    0.42,
    "diabetologist":   0.38,
    "general physician": 0.35,
    "neurologist":     0.33,
    "oncologist":      0.30,
    "pulmonologist":   0.36,
    "dermatologist":   0.40,
    "pediatrician":    0.37,
    "__default__":     0.35,
}


class FeatureExtractor:
    """
    Shared feature engineering layer for training and serving.

    Parameters
    ----------
    df : pd.DataFrame
        The full interaction DataFrame (all doctors, all dates).
    rolling_window_months : int
        How many months back to look for rolling statistics.
        Default is 6 (matches the plan spec).
    """

    ROLLING_MONTHS = 6

    # Columns that must exist; others are derived.
    REQUIRED_COLS = [
        "doctor_id", "interaction_date", "outcome", "interest_level",
        "follow_up", "actual_time_seconds",
    ]

    def __init__(self, df: pd.DataFrame, rolling_window_months: int = 6):
        self.df = self._prepare(df.copy())
        self.ROLLING_MONTHS = rolling_window_months

        # Pre-compute territory-level target encodings (used for doctor territory feature)
        self._territory_enc: Dict[str, float] = self._compute_target_encoding("territory")
        self._product_enc:   Dict[str, float] = self._compute_target_encoding("product_name")

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def extract(self, doctor_id: str) -> Dict[str, Any]:
        """
        Extract a single feature dict for online serving.

        Rolling statistics are computed from ALL historical rows for this
        doctor (no future rows exist at serving time, so no leakage).
        """
        d = self.df[self.df["doctor_id"] == str(doctor_id).strip()].copy()
        if d.empty:
            return self._cold_start_features(doctor_id)

        static = d.iloc[0]  # static doctor attributes (stable across rows)
        return self._build_feature_dict(d, static, as_of_date=None)

    def extract_batch(
        self,
        df: Optional[pd.DataFrame] = None,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Build a training matrix (X, y) with one row per interaction.

        For each row i, rolling features are computed from rows
        STRICTLY BEFORE row i's interaction_date — no leakage.

        Returns
        -------
        X : pd.DataFrame  — feature matrix
        y : pd.Series     — binary label (1 = positive outcome)
        """
        src = self._prepare((df if df is not None else self.df).copy())
        rows: List[Dict[str, Any]] = []
        labels: List[int] = []

        src_sorted = src.sort_values(["doctor_id", "interaction_date"])

        for _, group in src_sorted.groupby("doctor_id"):
            group = group.reset_index(drop=True)
            for i, row in group.iterrows():
                as_of = row["interaction_date"]
                history = group[group["interaction_date"] < as_of]
                static = group.iloc[0]

                feats = self._build_feature_dict(history, static, as_of_date=as_of)

                # Inject the current interaction's product (for product-level features)
                feats["product_name_enc"] = self._product_enc.get(
                    str(row.get("product_name", "")), self._product_enc.get("__default__", 0.35)
                )

                rows.append(feats)
                labels.append(1 if str(row["outcome"]).lower() == "positive" else 0)

        X = pd.DataFrame(rows)
        y = pd.Series(labels, name="label")
        return X, y

    def get_specialty_prior(self, specialty: str) -> float:
        """Return Bayesian prior conversion rate for a specialty (cold-start)."""
        return SPECIALTY_PRIORS.get(
            str(specialty).strip().lower(),
            SPECIALTY_PRIORS["__default__"],
        )

    def get_feature_names(self) -> List[str]:
        """Return the ordered list of feature column names produced by extract()."""
        dummy = self.extract(self.df["doctor_id"].iloc[0])
        return list(dummy.keys())

    # ──────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────────

    # Full outcome normalisation map — kept here so _prepare() is self-contained
    _OUTCOME_MAP = {
        "positive": "positive", "converted": "positive", "success": "positive",
        "won": "positive", "yes": "positive", "1": "positive", "true": "positive",
        "negative": "negative", "lost": "negative", "no": "negative",
        "0": "negative", "false": "negative",
        "neutral": "neutral", "pending": "neutral",
        "nan": "neutral", "none": "neutral", "na": "neutral",
    }

    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["interaction_date"] = pd.to_datetime(
            df["interaction_date"], dayfirst=True, errors="coerce"
        )
        # Full normalisation — safe to call even if load_data() already ran
        raw = df["outcome"].astype(str).str.strip().str.lower()
        df["outcome"] = raw.map(self._OUTCOME_MAP).fillna(
            raw.apply(lambda v: "positive" if v not in ("negative", "lost", "no", "neutral", "pending", "nan", "none", "na", "0", "false") and v else "neutral")
        )
        df["follow_up"] = df["follow_up"].astype(str).str.strip().str.lower()
        df["doctor_id"] = df["doctor_id"].astype(str).str.strip()

        for col in ["interest_level", "actual_time_seconds", "sales_volume",
                    "experience_years", "patient_load", "publications_count",
                    "social_media_reach"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        if "specialty" not in df.columns:
            df["specialty"] = "unknown"
        if "territory" not in df.columns:
            df["territory"] = "unknown"
        if "product_name" not in df.columns:
            df["product_name"] = "unknown"
        if "employee_type" not in df.columns:
            df["employee_type"] = "mr"

        return df

    def _compute_target_encoding(self, col: str) -> Dict[str, float]:
        """Target-encode a categorical column using global positive-outcome rate."""
        if col not in self.df.columns:
            return {"__default__": 0.35}
        global_rate = (self.df["outcome"] == "positive").mean()
        enc: Dict[str, float] = {}
        for val, grp in self.df.groupby(col):
            enc[str(val)] = round(
                (grp["outcome"] == "positive").mean() * 0.8 + global_rate * 0.2, 4
            )
        enc["__default__"] = round(global_rate, 4)
        return enc

    def _build_feature_dict(
        self,
        history: pd.DataFrame,
        static: pd.Series,
        as_of_date: Optional[pd.Timestamp],
    ) -> Dict[str, Any]:
        """Core feature computation. `history` must already be leakage-free."""

        # ── Static doctor attributes ───────────────────────────────────────
        specialty      = str(static.get("specialty", "unknown")).strip().lower()
        experience_yrs = float(static.get("experience_years", 0) or 0)
        patient_load   = float(static.get("patient_load", 0) or 0)
        publications   = float(static.get("publications_count", 0) or 0)
        social_reach   = float(static.get("social_media_reach", 0) or 0)
        territory      = str(static.get("territory", "unknown")).strip().lower()
        employee_type  = str(static.get("employee_type", "mr")).strip().lower()

        # ── Specialty one-hot (top-8 + other) ─────────────────────────────
        top_specialties = list(SPECIALTY_PRIORS.keys())[:-1]  # exclude __default__
        specialty_ohe = {f"spec_{s.replace(' ', '_')}": int(specialty == s) for s in top_specialties}
        specialty_ohe["spec_other"] = int(specialty not in top_specialties)

        # ── Employee type ordinal ──────────────────────────────────────────
        emp_type_map = {"mr": 0, "sr": 1, "asm": 2, "rsm": 3, "nsm": 4}
        emp_type_enc = emp_type_map.get(employee_type, 0)

        # ── Territory target encoding ──────────────────────────────────────
        territory_enc = self._territory_enc.get(territory, self._territory_enc.get("__default__", 0.35))

        # ── Rolling statistics (last ROLLING_MONTHS months) ───────────────
        if history.empty or as_of_date is None:
            # Serving mode: use full history
            window_df = history
        else:
            cutoff = as_of_date - pd.DateOffset(months=self.ROLLING_MONTHS)
            window_df = history[history["interaction_date"] >= cutoff]

        if window_df.empty:
            avg_interest_6m  = self.get_specialty_prior(specialty)  # Bayesian fallback
            follow_up_rate   = 0.0
            avg_conv_6m      = self.get_specialty_prior(specialty)
            n_interactions   = 0
            interest_trend   = 0.0
            follow_up_trend  = 0.0
        else:
            avg_interest_6m = float(window_df["interest_level"].mean())
            follow_up_rate  = float(
                (window_df["follow_up"] == "yes").sum() / max(len(window_df), 1)
            )
            avg_conv_6m = float(
                (window_df["outcome"] == "positive").sum() / max(len(window_df), 1)
            )
            n_interactions = len(window_df)

            # Linear trend slopes for interest and follow-up (monthly)
            interest_trend  = self._monthly_slope(window_df, "interest_level")
            follow_up_trend = self._monthly_slope(
                window_df.assign(fu_bin=(window_df["follow_up"] == "yes").astype(float)),
                "fu_bin",
            )

        # ── Days since last visit ──────────────────────────────────────────
        if history.empty:
            days_since_last = 999
        else:
            last_date = history["interaction_date"].max()
            ref_date  = as_of_date if as_of_date is not None else pd.Timestamp.now()
            days_since_last = max(0, (ref_date - last_date).days) if pd.notna(last_date) else 999

        # ── Days since last positive interaction ───────────────────────────
        pos_history = history[history["outcome"] == "positive"] if not history.empty else pd.DataFrame()
        if pos_history.empty:
            days_since_positive = 999
        else:
            last_pos = pos_history["interaction_date"].max()
            ref_date = as_of_date if as_of_date is not None else pd.Timestamp.now()
            days_since_positive = max(0, (ref_date - last_pos).days) if pd.notna(last_pos) else 999

        # ── AIDA stage (ordinal, rule-based for training bootstrap) ───────
        aida_stage_ord = self._rule_based_aida(avg_conv_6m, avg_interest_6m, follow_up_rate)

        # ── High-interest recency ──────────────────────────────────────────
        if not history.empty and "interest_level" in history.columns:
            hi_df = history[history["interest_level"] >= 4]
            if not hi_df.empty:
                last_hi = hi_df["interaction_date"].max()
                ref_date = as_of_date if as_of_date is not None else pd.Timestamp.now()
                days_since_high_interest = max(0, (ref_date - last_hi).days)
            else:
                days_since_high_interest = 999
        else:
            days_since_high_interest = 999

        # ── Assemble final feature dict ────────────────────────────────────
        features: Dict[str, Any] = {
            # static
            "experience_years":    experience_yrs,
            "patient_load":        patient_load,
            "publications_count":  publications,
            "social_media_reach":  social_reach,
            "territory_enc":       territory_enc,
            "emp_type_enc":        emp_type_enc,
            # rolling
            "avg_interest_6m":     round(avg_interest_6m, 4),
            "avg_conv_rate_6m":    round(avg_conv_6m, 4),
            "follow_up_rate_6m":   round(follow_up_rate, 4),
            "n_interactions_6m":   n_interactions,
            "interest_trend":      round(interest_trend, 6),
            "follow_up_trend":     round(follow_up_trend, 6),
            # recency
            "days_since_last_visit":     days_since_last,
            "days_since_positive":       days_since_positive,
            "days_since_high_interest":  days_since_high_interest,
            # AIDA
            "aida_stage_ord":      aida_stage_ord,
            # specialty OHE
            **specialty_ohe,
        }

        return features

    def _monthly_slope(self, df: pd.DataFrame, value_col: str) -> float:
        """Return the linear slope (per month) of a metric over time."""
        if df.empty or value_col not in df.columns:
            return 0.0
        df = df.copy()
        df["month"] = df["interaction_date"].dt.to_period("M")
        monthly = df.groupby("month")[value_col].mean().reset_index()
        if len(monthly) < 2:
            return 0.0
        x = np.arange(len(monthly), dtype=float)
        y = monthly[value_col].astype(float).values
        mask = ~np.isnan(y)
        if mask.sum() < 2:
            return 0.0
        slope = float(np.polyfit(x[mask], y[mask], 1)[0])
        return slope

    def _rule_based_aida(
        self, conv_rate: float, avg_interest: float, follow_up_rate: float
    ) -> int:
        """Bootstrap AIDA stage ordinal for training (mirrors AIDAClassifier logic)."""
        if conv_rate >= 0.5:
            return AIDA_ORDINAL["action"]
        elif avg_interest >= 4 and follow_up_rate > 0.4:
            return AIDA_ORDINAL["desire"]
        elif avg_interest >= 3 and conv_rate < 0.3:
            return AIDA_ORDINAL["interest"]
        return AIDA_ORDINAL["awareness"]

    def _cold_start_features(self, doctor_id: str) -> Dict[str, Any]:
        """Return a safe zero-vector for doctors with no history."""
        top_specialties = list(SPECIALTY_PRIORS.keys())[:-1]
        specialty_ohe = {f"spec_{s.replace(' ', '_')}": 0 for s in top_specialties}
        specialty_ohe["spec_other"] = 1

        prior = SPECIALTY_PRIORS["__default__"]
        return {
            "experience_years":    0.0,
            "patient_load":        0.0,
            "publications_count":  0.0,
            "social_media_reach":  0.0,
            "territory_enc":       prior,
            "emp_type_enc":        0,
            "avg_interest_6m":     prior * 5,   # rough interest estimate
            "avg_conv_rate_6m":    prior,
            "follow_up_rate_6m":   0.0,
            "n_interactions_6m":   0,
            "interest_trend":      0.0,
            "follow_up_trend":     0.0,
            "days_since_last_visit":     999,
            "days_since_positive":       999,
            "days_since_high_interest":  999,
            "aida_stage_ord":      AIDA_ORDINAL["awareness"],
            **specialty_ohe,
        }


# ── FeatureStore (LRU cache wrapper for serving) ───────────────────────────────

from functools import lru_cache


class FeatureStore:
    """
    Thin caching layer around FeatureExtractor for online serving.

    Features for a given doctor are computed once per request cycle
    and reused by both ML inference and LLM prompt building.

    Parameters
    ----------
    extractor : FeatureExtractor
    max_size  : int — LRU cache size (number of doctor entries)
    """

    def __init__(self, extractor: FeatureExtractor, max_size: int = 500):
        self._extractor = extractor
        self._cache: dict = {}
        self._max_size = max_size

    def get(self, doctor_id: str) -> Dict[str, Any]:
        """Return cached features or compute fresh."""
        key = str(doctor_id).strip()
        if key not in self._cache:
            if len(self._cache) >= self._max_size:
                # Evict oldest entry (simple FIFO for now)
                oldest = next(iter(self._cache))
                del self._cache[oldest]
            self._cache[key] = self._extractor.extract(key)
        return self._cache[key]

    def invalidate(self, doctor_id: str) -> None:
        """Call this when new interaction data arrives for a doctor."""
        self._cache.pop(str(doctor_id).strip(), None)

    def invalidate_all(self) -> None:
        self._cache.clear()