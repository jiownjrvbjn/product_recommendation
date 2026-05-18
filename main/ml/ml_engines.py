"""
ml_engines.py
-----------------------------
PatGPT ML Engine Layer — Four modular engines as per upgrade plan.

Engines:
  1. ProductRecommendationEngine  — per-doctor product suggestions with time allocation
  2. ProductPerformanceEngine     — portfolio-level analytics with QoQ growth
  3. DoctorReviewEngine           — doctor KPIs, LTV, radar metrics, objection intelligence
  4. EmployeeReportEngine         — employee effectiveness score, peer comparison

All engines:
  - Use pure pandas/numpy (no LLM calls)
  - Accept the shared DataFrame from DoctorAnalyticsEnhanced.df
  - Implement formulae exactly as specified in plan §7
  - LLM is NOT called here — stub hooks are provided for llm_agents.py
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime

from main.analytics_engine import (
    DoctorAnalyticsEnhanced,
    AIDAClassifier,
    TrendAnalytics,
    ObjectionResolutionTracker,
)

_aida_classifier = AIDAClassifier()


class ProductRecommendationEngine:

    GLOBAL_AVG_TIME = 120

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self._normalise()
        self._main_engine = DoctorAnalyticsEnhanced(df)
        self._trend = TrendAnalytics(df)
        self._aida = AIDAClassifier()

    def _normalise(self):
        self.df["outcome"] = self.df["outcome"].astype(str).str.strip().str.lower()
        outcome_map = {
            "positive": "positive", "converted": "positive", "success": "positive",
            "won": "positive", "yes": "positive",
            "negative": "negative", "lost": "negative", "no": "negative",
            "neutral": "neutral", "pending": "neutral",
        }
        self.df["outcome"] = self.df["outcome"].map(outcome_map).fillna("neutral")
        self.df["follow_up"] = self.df["follow_up"].astype(str).str.strip().str.lower()
        self.df["doctor_id"] = self.df["doctor_id"].astype(str).str.strip()
        self.df["interaction_date"] = pd.to_datetime(self.df["interaction_date"], dayfirst=True, errors="coerce")
        for col in ["actual_time_seconds", "sales_volume", "interest_level"]:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce").fillna(0)

    def _doctor_df(self, doctor_id: str) -> pd.DataFrame:
        return self.df[self.df["doctor_id"] == str(doctor_id).strip()].copy()

    def get_last_meeting_recap(self, doctor_id: str) -> Optional[Dict[str, Any]]:
        d = self._doctor_df(doctor_id)
        if d.empty:
            return None
        latest = d.sort_values("interaction_date", ascending=False).iloc[0]
        obj_col = "objection" if "objection" in d.columns else (
                  "objection_type" if "objection_type" in d.columns else None)
        return {
            "date": str(latest["interaction_date"].date()) if pd.notna(latest["interaction_date"]) else None,
            "product": latest.get("product_name", ""),
            "objection": str(latest.get(obj_col, "")) if obj_col else "",
            "outcome": latest.get("outcome", ""),
            "interest_level": int(latest.get("interest_level", 0)),
            "sales_volume": int(latest.get("sales_volume", 0)),
            "notes": "Auto-generated from last interaction record.",
        }

    def predict_available_time(self, doctor_id: str) -> int:
        d = self._doctor_df(doctor_id)
        if d.empty or "actual_time_seconds" not in d.columns:
            return self.GLOBAL_AVG_TIME
        times = d["actual_time_seconds"].dropna()
        times = times[times > 0].tolist()
        if len(times) >= 5:
            return int(np.median(times[-5:]))
        global_avg = self.df["actual_time_seconds"]
        global_avg = global_avg[global_avg > 0].mean()
        return int(global_avg) if not np.isnan(global_avg) else self.GLOBAL_AVG_TIME

    def score_products_for_doctor(self, doctor_id: str) -> List[Dict[str, Any]]:
        d = self._doctor_df(doctor_id)
        if d.empty:
            return []
        results = []
        for product in d["product_name"].unique():
            p_df = d[d["product_name"] == product]
            if p_df.empty:
                continue
            conv = (p_df["outcome"] == "positive").sum() / max(len(p_df), 1)
            interest = p_df["interest_level"].mean()
            follow_up = (p_df["follow_up"] == "yes").sum() / max(len(p_df), 1)
            p_df2 = p_df.copy()
            p_df2["month"] = p_df2["interaction_date"].dt.to_period("M")
            monthly_conv = (
                p_df2.groupby("month")
                .apply(lambda x: (x["outcome"] == "positive").sum() / len(x) if len(x) else 0)
                .tolist()
            )
            recent_trend = self._trend.calc_recent_trend_score(monthly_conv)
            score = max(0.4 * conv + 0.3 * (interest / 5) + 0.2 * follow_up + 0.1 * recent_trend, 0.0)
            aida_result = self._aida.classify(p_df)
            results.append({
                "product_name": product,
                "score": round(score, 3),
                "conversion_rate": round(conv, 3),
                "avg_interest": round(interest, 2),
                "follow_up_rate": round(follow_up, 3),
                "recent_trend": recent_trend,
                "aida_stage": aida_result["aida_stage"],
                "aida_label": aida_result["aida_label"],
                "aida_color": aida_result["aida_color"],
                "aida_emoji": aida_result["aida_emoji"],
                "total_sales_volume": int(p_df["sales_volume"].sum()) if "sales_volume" in p_df.columns else 0,
            })
        results.sort(key=lambda x: x["score"], reverse=True)
        return results

    def recommend_for_time(self, doctor_id: str, time_sec: int) -> Dict[str, Any]:
        products = self.score_products_for_doctor(doctor_id)
        if not products:
            return {"products": [], "mode": "no_data", "effective_time": time_sec}
        if time_sec <= 60:
            count, mode = 1, "short"
        elif time_sec <= 120:
            count, mode = 3, "medium"
        else:
            count, mode = 5, "long"
        selected = products[:count]
        total_score = sum(p["score"] for p in selected) or 1
        for p in selected:
            p["suggested_duration_sec"] = max(10, int((p["score"] / total_score) * time_sec))
        return {
            "doctor_id": doctor_id,
            "effective_time": time_sec,
            "mode": mode,
            "products": selected,
        }

    def get_full_product_suggestions(self, doctor_id, selected_time_sec, employee_type):
        d = self._doctor_df(doctor_id)
        if d.empty:
            return None
        static = d.iloc[0]
        doc_info = {
            "patient_load": int(static.get("patient_load", 0)),
            "publications_count": int(static.get("publications_count", 0)),
            "social_media_reach": int(static.get("social_media_reach", 0)),
            "conversion_rate": round((d["outcome"] == "positive").sum() / len(d), 3),
        }
        score = self._main_engine.reco_engine.score_doctor(doc_info)
        tier = self._main_engine.reco_engine.classify_doctor_tier(score)
        recap = self.get_last_meeting_recap(doctor_id)
        predicted_time = self.predict_available_time(doctor_id)
        time_to_use = selected_time_sec if selected_time_sec else predicted_time
        reco = self.recommend_for_time(doctor_id, time_to_use)
        return {
            "doctor_id": doctor_id,
            "doctor_name": static.get("doctor_name", ""),
            "specialty": static.get("specialty", ""),
            "territory": static.get("territory", ""),
            "doctor_rating": {"score": score, "tier": tier},
            "last_meeting": recap,
            "predicted_time": predicted_time,
            "effective_time": time_to_use,
            "products": reco["products"],
            "rule_based_suggestion": "",
            "llm_suggestion": None,
        }


from main.analytics_engine import AIDAClassifier

_aida_classifier = AIDAClassifier()


class ProductPerformanceEngine:
    """
    Enterprise product-analytics engine — PatGPT v5.1
    All heavy lifting is pure pandas/numpy.  LLM is NEVER called
    from inside this class; callers pass llm_engine explicitly to
    generate_ai_analysis() only after the user expands the AI card.
    """

    # ── thresholds ─────────────────────────────────────────────────────────
    _UNDERPERF_CONV   = 0.30   # below this conv_rate → flag
    _UNDERPERF_QOQ    = -0.10  # QoQ drop worse than this → flag
    _UNDERPERF_INT    = 3.0    # avg interest below this → partial flag
    _TREND_SLOPE_UP   = 5.0    # linear-regression slope threshold
    _TREND_SLOPE_DN   = -5.0

    # ── lifecycle ──────────────────────────────────────────────────────────
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self._normalise()

    def _normalise(self):
        df = self.df

        # outcome
        df["outcome"] = df["outcome"].astype(str).str.strip().str.lower()
        _omap = {
            "positive": "positive", "converted": "positive",
            "success":  "positive", "won":       "positive", "yes": "positive",
            "negative": "negative", "lost":       "negative", "no": "negative",
            "neutral":  "neutral",  "pending":    "neutral",
        }
        df["outcome"] = df["outcome"].map(_omap).fillna("neutral")

        # dates
        df["interaction_date"] = pd.to_datetime(
            df["interaction_date"], dayfirst=True, errors="coerce"
        )

        # numerics
        for col in ("sales_volume", "interest_level", "actual_time_seconds"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        # strings
        df["follow_up"] = df["follow_up"].astype(str).str.strip().str.lower()
        if "quarter" in df.columns:
            df["quarter"] = df["quarter"].astype(str).str.strip().str.upper()
        if "region" in df.columns:
            df["region"] = df["region"].astype(str).str.strip()

        # unified objection column
        if "objection" not in df.columns:
            if "objection_type" in df.columns:
                df["objection"] = df["objection_type"].astype(str).str.strip().str.lower()
            else:
                df["objection"] = "none"
        else:
            df["objection"] = df["objection"].astype(str).str.strip().str.lower()

        self.df = df

    # ── internal helpers ───────────────────────────────────────────────────

    def _filter(self, df: pd.DataFrame, territory=None, quarter=None) -> pd.DataFrame:
        if territory:
            df = df[df["territory"].str.lower() == territory.strip().lower()]
        if quarter:
            df = df[df["quarter"].str.upper() == quarter.strip().upper()]
        return df

    def _dated(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop rows where interaction_date is NaT."""
        return df[df["interaction_date"].notna()].copy()

    def _lin_trend(self, series: List[float]) -> str:
        """Linear-regression trend over a numeric list."""
        if len(series) < 3:
            return "stable"
        y = np.array(series, dtype=float)
        mask = ~np.isnan(y)
        if mask.sum() < 2:
            return "stable"
        x = np.arange(len(y))
        slope = np.polyfit(x[mask], y[mask], 1)[0]
        if slope > self._TREND_SLOPE_UP:
            return "improving"
        if slope < self._TREND_SLOPE_DN:
            return "declining"
        return "stable"

    def _qoq_growth(self, p_df: pd.DataFrame) -> float:
        if p_df.empty or "quarter" not in p_df.columns:
            return 0.0
        grp_cols = ["year", "quarter"] if "year" in p_df.columns else ["quarter"]
        quarterly = p_df.groupby(grp_cols)["sales_volume"].sum()
        if len(quarterly) < 2:
            return 0.0
        vals = quarterly.values
        prev, curr = vals[-2], vals[-1]
        return 0.0 if prev == 0 else round((curr - prev) / prev, 4)

    def _underperf_flag(self, conv: float, qoq: float, interest: float, trend: str) -> bool:
        return (
            conv     < self._UNDERPERF_CONV
            or trend == "declining"
            or (qoq < self._UNDERPERF_QOQ and interest < self._UNDERPERF_INT)
        )

    def _objection_breakdown(self, p_df: pd.DataFrame) -> Dict[str, int]:
        obj = p_df["objection"].astype(str).str.strip().str.lower()
        return obj[~obj.isin(["nan", "none", ""])].value_counts().to_dict()

    def _top_region(self, p_df: pd.DataFrame) -> str:
        if "region" not in p_df.columns or p_df["region"].isna().all():
            return "N/A"
        rs = p_df.groupby("region")["sales_volume"].sum()
        return str(rs.idxmax()) if rs.sum() > 0 else "N/A"

    def _monthly_sales_series(self, p_df: pd.DataFrame):
        d = self._dated(p_df).copy()
        d["month"] = d["interaction_date"].dt.to_period("M")
        ms = d.groupby("month")["sales_volume"].sum().reset_index()
        ms.columns = ["month", "sales_volume"]
        ms["month"] = ms["month"].astype(str)
        return ms

    def _monthly_conv_series(self, p_df: pd.DataFrame):
        d = self._dated(p_df).copy()
        d["month"] = d["interaction_date"].dt.to_period("M")
        mc = (
            d.groupby("month")["outcome"]
            .agg(lambda x: round((x == "positive").sum() / max(len(x), 1), 3))
            .reset_index()
        )
        mc.columns = ["month", "conversion_rate"]
        mc["month"] = mc["month"].astype(str)
        return mc

    # ══════════════════════════════════════════════════════════════════════════
    # PUBLIC API
    # ══════════════════════════════════════════════════════════════════════════

    # ── 1. Overall summary ─────────────────────────────────────────────────

    def get_overall_summary(
        self,
        territory: Optional[str] = None,
        quarter:   Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Returns a DataFrame, one row per product, with:
          product, total_sales, total_interactions, unique_doctors,
          qoq_growth, conversion_rate, avg_interest, follow_up_rate,
          top_region, trend, underperformance_flag
        """
        df = self._filter(self.df.copy(), territory, quarter)
        if df.empty:
            return pd.DataFrame()

        rows = []
        for product, p_df in df.groupby("product_name"):
            if p_df.empty:
                continue

            total_sales     = int(p_df["sales_volume"].sum())
            total_int       = len(p_df)
            unique_doctors  = p_df["doctor_id"].nunique()
            conv_rate       = round((p_df["outcome"] == "positive").sum() / max(total_int, 1), 3)
            avg_interest    = round(p_df["interest_level"].mean(), 2)
            follow_up_rate  = round((p_df["follow_up"] == "yes").sum() / max(total_int, 1), 3)
            qoq_growth      = self._qoq_growth(p_df)
            top_region      = self._top_region(p_df)

            ms              = self._monthly_sales_series(p_df)
            trend           = self._lin_trend(ms["sales_volume"].tolist())
            flag            = self._underperf_flag(conv_rate, qoq_growth, avg_interest, trend)

            rows.append({
                "product":              product,
                "total_sales":          total_sales,
                "total_interactions":   total_int,
                "unique_doctors":       unique_doctors,
                "qoq_growth":           qoq_growth,
                "conversion_rate":      conv_rate,
                "avg_interest":         avg_interest,
                "follow_up_rate":       follow_up_rate,
                "top_region":           top_region,
                "trend":                trend,
                "underperformance_flag": flag,
            })

        return (
            pd.DataFrame(rows)
            .sort_values("total_sales", ascending=False)
            .reset_index(drop=True)
        )

    # ── 2. Individual product detail ───────────────────────────────────────

    def get_product_detail(
        self,
        product_name: str,
        territory:    Optional[str] = None,
        quarter:      Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Comprehensive per-product report:
          monthly_sales, monthly_conversion, region_comparison,
          objection_breakdown, top_doctors, lowest_doctors,
          engagement_depth, follow_up_analytics, aida_distribution,
          conversion_funnel, qoq_performance, competitor_objections,
          adoption_trend, quarterly_table, summary KPIs
        """
        p_df = self.df[self.df["product_name"] == product_name].copy()
        if p_df.empty:
            return {"error": f"Product '{product_name}' not found."}

        if territory:
            p_df = p_df[p_df["territory"].str.lower() == territory.strip().lower()]
            if p_df.empty:
                return {"error": f"No data for '{product_name}' in territory '{territory}'."}
        if quarter:
            p_df = p_df[p_df["quarter"].str.upper() == quarter.strip().upper()]
            if p_df.empty:
                return {"error": f"No data for '{product_name}' in quarter '{quarter}'."}

        # ── KPIs ──────────────────────────────────────────────────────────
        total_interactions = len(p_df)
        unique_doctors     = p_df["doctor_id"].nunique()
        total_sales        = int(p_df["sales_volume"].sum())
        conv_rate          = round((p_df["outcome"] == "positive").sum() / max(total_interactions, 1), 3)
        avg_interest       = round(p_df["interest_level"].mean(), 2)
        follow_up_rate     = round((p_df["follow_up"] == "yes").sum() / max(total_interactions, 1), 3)
        engagement_depth   = round(total_interactions / max(unique_doctors, 1), 2)
        qoq_growth         = self._qoq_growth(p_df)

        # ── Monthly trends ────────────────────────────────────────────────
        monthly_sales = self._monthly_sales_series(p_df)
        monthly_conv  = self._monthly_conv_series(p_df)
        trend         = self._lin_trend(monthly_sales["sales_volume"].tolist())
        is_underperforming = self._underperf_flag(conv_rate, qoq_growth, avg_interest, trend)

        # ── Adoption trend (cumulative unique doctors over time) ──────────
        d = self._dated(p_df).copy()
        if not d.empty:
            d["month"] = d["interaction_date"].dt.to_period("M")
            adoption = (
                d.groupby("month")["doctor_id"].nunique()
                .cumsum()
                .reset_index()
            )
            adoption.columns = ["month", "cumulative_doctors"]
            adoption["month"] = adoption["month"].astype(str)
            adoption_trend = adoption.to_dict(orient="records")
        else:
            adoption_trend = []

        # ── Objections ────────────────────────────────────────────────────
        objection_breakdown = self._objection_breakdown(p_df)

        # Competitor objection analysis:
        # objections that mention competitor-style language
        _COMP_KEYWORDS = {"competitor", "alternative", "cheaper", "price", "generic",
                          "brand", "cost", "efficacy", "trial", "evidence"}
        competitor_objections = {
            k: v for k, v in objection_breakdown.items()
            if any(kw in k for kw in _COMP_KEYWORDS)
        }

        # ── Follow-up analytics ───────────────────────────────────────────
        fu_outcomes = (
            p_df.groupby("follow_up")["outcome"]
            .agg(
                count="count",
                positive_count=lambda x: (x == "positive").sum(),
            )
            .reset_index()
        )
        fu_outcomes["conv_rate"] = (
            fu_outcomes["positive_count"] / fu_outcomes["count"].clip(lower=1)
        ).round(3)
        follow_up_analytics = fu_outcomes.to_dict(orient="records")

        # ── Doctor performance tables ──────────────────────────────────────
        def _doctor_perf(p_df_inner: pd.DataFrame, top: bool, n: int = 5):
            agg = (
                p_df_inner.groupby("doctor_id")
                .agg(
                    doctor_name    = ("doctor_name",  "first"),
                    total_sales    = ("sales_volume", "sum"),
                    conv_rate      = ("outcome", lambda x: round((x == "positive").sum() / max(len(x), 1), 3)),
                    interactions   = ("interaction_id", "count"),
                    avg_interest   = ("interest_level", "mean"),
                )
                .reset_index()
                .sort_values("total_sales", ascending=not top)
                .head(n)
            )
            records = agg.to_dict(orient="records")
            for r in records:
                r["doctor_id"]   = str(r["doctor_id"])
                r["doctor_name"] = str(r["doctor_name"])
                r["total_sales"] = int(r["total_sales"])
                r["conv_rate"]   = round(float(r["conv_rate"]), 3)
                r["avg_interest"]= round(float(r["avg_interest"]), 2)
            return records

        top_doctors    = _doctor_perf(p_df, top=True)
        lowest_doctors = _doctor_perf(p_df, top=False)

        # ── AIDA stage distribution ───────────────────────────────────────
        aida_dist: Dict[str, int] = {"awareness": 0, "interest": 0, "desire": 0, "action": 0}
        for did, grp in p_df.groupby("doctor_id"):
            stage = _aida_classifier.classify(grp).get("aida_stage", "awareness")
            aida_dist[stage] = aida_dist.get(stage, 0) + 1

        # ── Conversion funnel ─────────────────────────────────────────────
        total_ints    = len(p_df)
        with_followup = int((p_df["follow_up"] == "yes").sum())
        converted     = int((p_df["outcome"] == "positive").sum())
        conversion_funnel = [
            {"stage": "Total Interactions",   "count": total_ints},
            {"stage": "Requested Follow-up",  "count": with_followup},
            {"stage": "Converted",            "count": converted},
        ]

        # ── Region comparison ─────────────────────────────────────────────
        region_col = "region" if "region" in p_df.columns else "territory"
        region_perf = (
            p_df.groupby(region_col)
            .agg(
                total_sales   = ("sales_volume", "sum"),
                interactions  = ("interaction_id", "count"),
                conv_rate     = ("outcome", lambda x: round((x == "positive").sum() / max(len(x), 1), 3)),
                avg_interest  = ("interest_level", "mean"),
            )
            .reset_index()
            .sort_values("total_sales", ascending=False)
        )
        region_perf.columns = [region_col] + list(region_perf.columns[1:])
        region_comparison = region_perf.to_dict(orient="records")
        for r in region_comparison:
            r["total_sales"]  = int(r["total_sales"])
            r["conv_rate"]    = round(float(r["conv_rate"]), 3)
            r["avg_interest"] = round(float(r["avg_interest"]), 2)

        # ── QoQ performance table ─────────────────────────────────────────
        qoq_table = self.get_quarterly_table(product_name=product_name)
        qoq_records = qoq_table.to_dict(orient="records") if not qoq_table.empty else []

        return {
            # ── summary KPIs ──────────────────────────────────────────────
            "product_name":         product_name,
            "total_sales":          total_sales,
            "total_interactions":   total_interactions,
            "unique_doctors":       unique_doctors,
            "conversion_rate":      conv_rate,
            "avg_interest":         avg_interest,
            "follow_up_rate":       follow_up_rate,
            "engagement_depth":     engagement_depth,
            "qoq_growth":           qoq_growth,
            "trend":                trend,
            "is_underperforming":   is_underperforming,
            # ── chart data ────────────────────────────────────────────────
            "monthly_sales":            monthly_sales.to_dict(orient="records"),
            "monthly_conversion":       monthly_conv.to_dict(orient="records"),
            "adoption_trend":           adoption_trend,
            "conversion_funnel":        conversion_funnel,
            "aida_distribution":        aida_dist,
            # ── tables ────────────────────────────────────────────────────
            "top_doctors":              top_doctors,
            "lowest_doctors":           lowest_doctors,
            "region_comparison":        region_comparison,
            "objection_breakdown":      objection_breakdown,
            "competitor_objections":    competitor_objections,
            "follow_up_analytics":      follow_up_analytics,
            "qoq_performance":          qoq_records,
        }

    # ── 3. Region breakdown ────────────────────────────────────────────────

    def get_region_breakdown(
        self,
        product_name: Optional[str] = None,
        quarter:      Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Per-region KPIs, optionally filtered by product and quarter.
        Returns list of dicts: region, total_sales, conv_rate,
        avg_interest, interactions, unique_doctors, top_product
        """
        df = self.df.copy()
        region_col = "region" if "region" in df.columns else "territory"
        if product_name:
            df = df[df["product_name"] == product_name]
        if quarter and "quarter" in df.columns:
            df = df[df["quarter"].str.upper() == quarter.strip().upper()]
        if df.empty:
            return []

        rows = []
        for region, grp in df.groupby(region_col):
            total_sales    = int(grp["sales_volume"].sum())
            interactions   = len(grp)
            unique_doctors = grp["doctor_id"].nunique()
            conv_rate      = round((grp["outcome"] == "positive").sum() / max(interactions, 1), 3)
            avg_interest   = round(grp["interest_level"].mean(), 2)
            if "product_name" in grp.columns and grp["sales_volume"].sum() > 0:
                top_product = str(grp.groupby("product_name")["sales_volume"].sum().idxmax())
            else:
                top_product = "N/A"
            rows.append({
                "region":          str(region),
                "total_sales":     total_sales,
                "interactions":    interactions,
                "unique_doctors":  unique_doctors,
                "conversion_rate": conv_rate,
                "avg_interest":    avg_interest,
                "top_product":     top_product,
            })

        return sorted(rows, key=lambda r: r["total_sales"], reverse=True)

    # ── 4. Trend analysis (portfolio level) ────────────────────────────────

    def get_trend_analysis(
        self,
        territory: Optional[str] = None,
        quarter:   Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Portfolio-level trend signals:
          - monthly portfolio sales
          - top-3 growing products
          - top-3 declining products
          - overall portfolio trend
        """
        df = self._filter(self.df.copy(), territory, quarter)
        if df.empty:
            return {}

        d = self._dated(df).copy()
        d["month"] = d["interaction_date"].dt.to_period("M")

        # portfolio monthly sales
        port_monthly = (
            d.groupby("month")["sales_volume"].sum().reset_index()
        )
        port_monthly.columns = ["month", "sales_volume"]
        port_monthly["month"] = port_monthly["month"].astype(str)
        portfolio_trend = self._lin_trend(port_monthly["sales_volume"].tolist())

        # per-product slopes
        product_slopes = []
        for product, grp in d.groupby("product_name"):
            ms = grp.groupby("month")["sales_volume"].sum().tolist()
            if len(ms) < 2:
                continue
            y = np.array(ms, dtype=float)
            x = np.arange(len(y))
            slope = float(np.polyfit(x, y, 1)[0])
            product_slopes.append({"product": product, "slope": slope})

        product_slopes.sort(key=lambda r: r["slope"])
        declining = product_slopes[:3]
        growing   = list(reversed(product_slopes[-3:]))

        return {
            "portfolio_monthly_sales": port_monthly.to_dict(orient="records"),
            "portfolio_trend":         portfolio_trend,
            "top_growing_products":    growing,
            "top_declining_products":  declining,
        }

    # ── 5. AI Analysis (rule-based; LLM optional) ─────────────────────────

    def generate_ai_analysis(
        self,
        product_name: str,
        llm_engine=None,   # pass LLMInsightsEngineEnhanced instance for LLM path
    ) -> Dict[str, Any]:
        """
        Called ONLY when user expands the AI Analysis card.
        Rule-based root-cause + recs always run.
        LLM explanation is generated only if llm_engine is provided.
        """
        detail = self.get_product_detail(product_name)
        if "error" in detail:
            return {"error": detail["error"]}

        objections   = detail.get("objection_breakdown", {})
        conv_rate    = detail.get("conversion_rate", 0)
        qoq_growth   = detail.get("qoq_growth", 0)
        avg_interest = detail.get("avg_interest", 0)
        trend        = detail.get("trend", "stable")
        eng_depth    = detail.get("engagement_depth", 0)
        region_comp  = detail.get("region_comparison", [])

        # ── rule-based root cause ─────────────────────────────────────────
        root_causes: List[str] = []
        if conv_rate < self._UNDERPERF_CONV:
            root_causes.append(
                f"Low conversion rate ({conv_rate:.0%}) — objection handling or messaging may be weak."
            )
        if trend == "declining":
            root_causes.append("Sales trend is declining over time — market fatigue or rep de-prioritisation.")
        if qoq_growth < self._UNDERPERF_QOQ:
            root_causes.append(
                f"QoQ growth is {qoq_growth:.1%} — significant sequential sales drop."
            )
        if avg_interest < self._UNDERPERF_INT:
            root_causes.append(
                f"Average doctor interest is low ({avg_interest:.1f}/5) — product relevance or pitch quality issue."
            )
        if eng_depth > 5 and conv_rate < 0.3:
            root_causes.append(
                "High engagement depth but low conversion suggests reps are re-visiting without closing."
            )
        top_obj = sorted(objections.items(), key=lambda x: -x[1])[:3]
        if top_obj:
            obj_text = ", ".join(f"'{k}' ({v}×)" for k, v in top_obj)
            root_causes.append(f"Dominant objections: {obj_text} — targeted rebuttal training needed.")
        if not root_causes:
            root_causes = ["No critical underperformance signals detected — product is stable."]

        # ── actionable recommendations ────────────────────────────────────
        recommendations: List[str] = []
        if conv_rate < 0.25:
            recommendations.append(
                "Run objection-handling workshop focused on top 3 objections before next territory cycle."
            )
        if avg_interest < 2.5:
            recommendations.append(
                "Redesign product pitch deck — lead with clinical outcome data instead of mechanism."
            )
        if qoq_growth < -0.15:
            recommendations.append(
                "Reassign product to high-performing reps in territories showing positive conv_rate."
            )
        if eng_depth > 6:
            recommendations.append(
                "Coach reps to use closing techniques after 3rd visit — set visit-cap triggers."
            )
        if region_comp:
            worst_region = min(region_comp, key=lambda r: r.get("conversion_rate", 1))
            recommendations.append(
                f"Focus territory plan on {worst_region.get('region', 'worst region')} "
                f"(conv: {worst_region.get('conversion_rate', 0):.0%}) — "
                "peer-shadow programme with top region reps."
            )
        while len(recommendations) < 3:
            recommendations.append(
                "Increase sample distribution to doctors in 'Interest' AIDA stage to accelerate funnel."
            )

        # ── quick win ─────────────────────────────────────────────────────
        if top_obj:
            quick_win = (
                f"Address the most frequent objection ('{top_obj[0][0]}') "
                "with a one-pager rebuttal card in the next 2 weeks."
            )
        else:
            quick_win = "Share a patient success story with the top 5 interested doctors this week."

        # ── next best action ──────────────────────────────────────────────
        if conv_rate > 0.4:
            nba = "Upsell complementary products to converted doctors to increase revenue per doctor."
        elif avg_interest > 3.5:
            nba = "Push doctors in 'Desire' AIDA stage into 'Action' with a limited trial or sample campaign."
        else:
            nba = "Re-engage lapsed doctors with updated clinical evidence and request a 10-minute slot."

        # ── territory-specific suggestion ─────────────────────────────────
        if region_comp:
            best_region = max(region_comp, key=lambda r: r.get("conversion_rate", 0))
            territory_suggestion = (
                f"Best performing region: {best_region.get('region')} "
                f"(conv: {best_region.get('conversion_rate', 0):.0%}). "
                "Replicate its rep approach and sample strategy across other territories."
            )
        else:
            territory_suggestion = "Insufficient region data — enable regional tagging for territory insights."

        # ── optional LLM explanation ──────────────────────────────────────
        llm_explanation: Optional[str] = None
        if llm_engine is not None:
            try:
                llm_explanation = llm_engine.explain_product_underperformance(
                    product_name=product_name,
                    metrics={
                        "total_sales":        detail.get("total_sales", 0),
                        "total_interactions":  detail.get("total_interactions", 0),
                        "conversion_rate":     conv_rate,
                        "avg_interest":        avg_interest,
                        "qoq_growth":          qoq_growth,
                        "trend":               trend,
                        "engagement_depth":    eng_depth,
                    },
                    objections=objections,
                )
            except Exception as exc:
                llm_explanation = f"LLM unavailable: {exc}"

        return {
            "product_name":          product_name,
            "root_causes":           root_causes,
            "recommendations":       recommendations[:3],
            "quick_win":             quick_win,
            "next_best_action":      nba,
            "territory_suggestion":  territory_suggestion,
            "llm_explanation":       llm_explanation,   # None unless llm_engine passed
        }

    # ── 6. Quarterly table (unchanged contract) ────────────────────────────

    def get_quarterly_table(self, product_name: Optional[str] = None) -> pd.DataFrame:
        df = self.df if not product_name else self.df[self.df["product_name"] == product_name]
        if df.empty or "quarter" not in df.columns:
            return pd.DataFrame()
        grp_cols = ["quarter"]
        if product_name:
            grp_cols = ["product_name"] + grp_cols
        if "year" in df.columns:
            grp_cols = ["year"] + grp_cols
        table = (
            df.groupby(grp_cols)
            .agg(
                total_sales  = ("sales_volume", "sum"),
                conv_rate    = ("outcome", lambda x: round((x == "positive").sum() / max(len(x), 1), 3)),
                interactions = ("interaction_id", "count"),
            )
            .reset_index()
            .sort_values(grp_cols)
        )
        return table

    # ── 7. Region-product matrix (unchanged contract) ──────────────────────

    def get_region_product_matrix(self, quarter: Optional[str] = None) -> pd.DataFrame:
        df = self.df.copy()
        if quarter:
            df = df[df["quarter"].str.upper() == quarter.strip().upper()]
        if df.empty or "region" not in df.columns:
            return pd.DataFrame()
        pivot = df.pivot_table(
            index="region",
            columns="product_name",
            values="sales_volume",
            aggfunc="sum",
            fill_value=0,
        )
        return pivot.reset_index()


class DoctorReviewEngine:
    DEFAULT_PRICE_PER_UNIT = 500

    def __init__(self, df: pd.DataFrame, price_per_unit=DEFAULT_PRICE_PER_UNIT):
        self.df = df.copy()
        self.price_per_unit = price_per_unit
        self._normalise()
        self._objection_tracker = ObjectionResolutionTracker(df)
        self._trend_engine = TrendAnalytics(df)

    def _normalise(self):
        self.df["outcome"] = self.df["outcome"].astype(str).str.strip().str.lower()
        outcome_map = {
            "positive": "positive", "converted": "positive", "success": "positive",
            "won": "positive", "yes": "positive",
            "negative": "negative", "lost": "negative", "no": "negative",
            "neutral": "neutral", "pending": "neutral",
        }
        self.df["outcome"] = self.df["outcome"].map(outcome_map).fillna("neutral")
        self.df["follow_up"] = self.df["follow_up"].astype(str).str.strip().str.lower()
        self.df["doctor_id"] = self.df["doctor_id"].astype(str).str.strip()
        self.df["territory"] = self.df["territory"].astype(str).str.strip().str.lower()
        self.df["interaction_date"] = pd.to_datetime(self.df["interaction_date"], dayfirst=True, errors="coerce")
        for col in ["sales_volume", "interest_level"]:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce").fillna(0)

    def _lifetime_value(self, doctor_df):
        if "sales_volume" not in doctor_df.columns:
            return 0
        return int(doctor_df["sales_volume"].sum() * self.price_per_unit)

    def _doctor_score(self, row):
        def norm(x, mn, mx): return (x - mn) / (mx - mn + 1e-5)
        return round(
            0.4 * norm(row.get("patient_load", 0), 0, 200)
            + 0.2 * norm(row.get("publications_count", 0), 0, 50)
            + 0.2 * norm(row.get("social_media_reach", 0), 0, 10000)
            + 0.2 * row.get("conv_rate", 0),
            3,
        )

    def get_all_doctors_summary(self, territory=None):
        df = self.df.copy()
        if territory:
            df = df[df["territory"] == territory.strip().lower()]
        if df.empty:
            return pd.DataFrame()
        rows = []
        for did in df["doctor_id"].unique():
            d = df[df["doctor_id"] == did]
            if d.empty:
                continue
            conv_rate = round((d["outcome"] == "positive").sum() / max(len(d), 1), 3)
            avg_interest = round(d["interest_level"].mean(), 2)
            follow_up = round((d["follow_up"] == "yes").sum() / max(len(d), 1), 3)
            ltv = self._lifetime_value(d)
            static = d.iloc[0]
            score = self._doctor_score({
                "patient_load": int(static.get("patient_load", 0)),
                "publications_count": int(static.get("publications_count", 0)),
                "social_media_reach": int(static.get("social_media_reach", 0)),
                "conv_rate": conv_rate,
            })
            tier = "high" if score >= 0.75 else ("medium" if score >= 0.45 else "low")
            rows.append({
                "doctor_id": did,
                "doctor_name": static.get("doctor_name", ""),
                "specialty": static.get("specialty", ""),
                "territory": static.get("territory", ""),
                "total_interactions": len(d),
                "conv_rate": conv_rate,
                "avg_interest": avg_interest,
                "follow_up_rate": follow_up,
                "ltv": ltv,
                "doctor_score": score,
                "tier": tier,
            })
        return pd.DataFrame(rows).sort_values("ltv", ascending=False).reset_index(drop=True)

    def get_doctor_analysis(self, doctor_id):
        d = self.df[self.df["doctor_id"] == str(doctor_id).strip()].copy()
        if d.empty:
            return {"error": f"Doctor '{doctor_id}' not found."}
        conv_rate = round((d["outcome"] == "positive").sum() / max(len(d), 1), 3)
        avg_interest = round(d["interest_level"].mean(), 2)
        follow_up = round((d["follow_up"] == "yes").sum() / max(len(d), 1), 3)
        ltv = self._lifetime_value(d)
        static = d.iloc[0]
        territory = static.get("territory", "")
        specialty = static.get("specialty", "")

        score = self._doctor_score({
            "patient_load": int(static.get("patient_load", 0)),
            "publications_count": int(static.get("publications_count", 0)),
            "social_media_reach": int(static.get("social_media_reach", 0)),
            "conv_rate": conv_rate,
        })

        terr_df = self.df[self.df["territory"] == territory]
        peer_conv = round((terr_df["outcome"] == "positive").sum() / max(len(terr_df), 1), 3)
        peer_interest = round(terr_df["interest_level"].mean(), 2) if not terr_df.empty else 0

        terr_convs = [
            (self.df[self.df["doctor_id"] == did]["outcome"] == "positive").sum() / max(len(self.df[self.df["doctor_id"] == did]), 1)
            for did in terr_df["doctor_id"].unique()
        ]
        rank_in_territory = next(
            (i + 1 for i, v in enumerate(sorted(terr_convs, reverse=True)) if abs(v - conv_rate) < 1e-6), 0
        )

        spec_df = self.df[self.df["specialty"] == specialty]
        spec_convs = [
            (self.df[self.df["doctor_id"] == did]["outcome"] == "positive").sum() / max(len(self.df[self.df["doctor_id"] == did]), 1)
            for did in spec_df["doctor_id"].unique()
        ]
        rank_in_specialty = next(
            (i + 1 for i, v in enumerate(sorted(spec_convs, reverse=True)) if abs(v - conv_rate) < 1e-6), 0
        )

        product_affinity = []
        for product in d["product_name"].unique():
            p_df = d[d["product_name"] == product]
            if p_df.empty:
                continue
            product_affinity.append({
                "product_name": product,
                "conv_rate": round((p_df["outcome"] == "positive").sum() / max(len(p_df), 1), 3),
                "avg_interest": round(p_df["interest_level"].mean(), 2),
                "total_sales": int(p_df["sales_volume"].sum()) if "sales_volume" in p_df.columns else 0,
            })
        product_affinity.sort(key=lambda x: x["conv_rate"], reverse=True)

        aida_result = _aida_classifier.classify(d)

        return {
            "doctor_id": doctor_id,
            "doctor_name": static.get("doctor_name", ""),
            "specialty": specialty,
            "territory": territory,
            "conv_rate": conv_rate,
            "avg_interest": avg_interest,
            "follow_up_rate": follow_up,
            "ltv": ltv,
            "doctor_score": score,
            "tier": "high" if score >= 0.75 else ("medium" if score >= 0.45 else "low"),
            "rank_in_territory": rank_in_territory,
            "rank_in_specialty": rank_in_specialty,
            "peer_metrics": {
                "territory_avg_conv": peer_conv,
                "territory_avg_interest": peer_interest,
            },
            "product_affinity": product_affinity,
            "trend_analytics": self._trend_engine.get_doctor_trends(doctor_id),
            "objection_intelligence": self._objection_tracker.get_objection_analysis(doctor_id),
            "aida_stage": aida_result["aida_stage"],
            "aida_label": aida_result["aida_label"],
            "aida_color": aida_result["aida_color"],
            "aida_emoji": aida_result["aida_emoji"],
            "aida_confidence": aida_result["aida_confidence"],
        }

    def get_doctor_overview_stats(self):
        all_docs = self.get_all_doctors_summary()
        if all_docs.empty:
            return {"specialty_avg_conversion": [], "top5_doctors": [], "territory_comparison": []}
        specialty_avg = (
            all_docs.groupby("specialty")["conv_rate"].mean().reset_index().to_dict(orient="records")
        )
        top5 = all_docs.nlargest(5, "conv_rate")[["doctor_id", "doctor_name", "specialty", "conv_rate"]].to_dict(orient="records")
        territory_comp = (
            all_docs.groupby("territory")["conv_rate"].mean().reset_index().to_dict(orient="records")
        )
        return {
            "specialty_avg_conversion": specialty_avg,
            "top5_doctors": top5,
            "territory_comparison": territory_comp,
        }


class EmployeeReportEngine:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self._normalise()

    def _normalise(self):
        self.df["outcome"] = self.df["outcome"].astype(str).str.strip().str.lower()
        outcome_map = {
            "positive": "positive", "converted": "positive", "success": "positive",
            "won": "positive", "yes": "positive",
            "negative": "negative", "lost": "negative", "no": "negative",
            "neutral": "neutral", "pending": "neutral",
        }
        self.df["outcome"] = self.df["outcome"].map(outcome_map).fillna("neutral")
        if "employee_id" in self.df.columns:
            self.df["employee_id"] = self.df["employee_id"].astype(str).str.strip()
        else:
            self.df["employee_id"] = "unknown"
        self.df["territory"] = self.df["territory"].astype(str).str.strip().str.lower()
        for col in ["actual_time_seconds", "sales_volume", "interest_level"]:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce").fillna(0)

    def _emp_score(self, conv_rate, avg_duration, products_pitched,
                   territory_avg_conv, territory_avg_duration, max_products):
        conv_ratio = conv_rate / max(territory_avg_conv, 1e-5)
        duration_ratio = avg_duration / max(territory_avg_duration, 1e-5)
        product_ratio = products_pitched / max(max_products, 1)
        return round(min(0.5 * conv_ratio + 0.3 * duration_ratio + 0.2 * product_ratio, 2.0), 3)

    def get_team_summary(self, territory=None):
        df = self.df.copy()
        if territory:
            df = df[df["territory"] == territory.strip().lower()]
        if df.empty:
            return pd.DataFrame()
        terr_conv = (df["outcome"] == "positive").sum() / max(len(df), 1)
        terr_dur = df["actual_time_seconds"].mean() if "actual_time_seconds" in df.columns else 120
        max_prods = df["product_name"].nunique()

        rows = []
        for eid in df["employee_id"].unique():
            e = df[df["employee_id"] == eid]
            if e.empty:
                continue
            static = e.iloc[0]
            conv_rate = round((e["outcome"] == "positive").sum() / max(len(e), 1), 3)
            avg_duration = round(e["actual_time_seconds"].mean(), 1) if "actual_time_seconds" in e.columns else 0
            total_sales = int(e["sales_volume"].sum()) if "sales_volume" in e.columns else 0
            products = e["product_name"].nunique()
            visits = len(e)
            score = self._emp_score(conv_rate, avg_duration, products, terr_conv, terr_dur, max_prods)
            rows.append({
                "employee_id": eid,
                "employee_name": static.get("employee_name", ""),
                "employee_type": static.get("employee_type", ""),
                "territory": static.get("territory", ""),
                "total_visits": visits,
                "conv_rate": conv_rate,
                "avg_duration_sec": avg_duration,
                "total_sales": total_sales,
                "products_pitched": products,
                "emp_score": score,
            })
        return pd.DataFrame(rows).sort_values("emp_score", ascending=False).reset_index(drop=True)

    def get_employee_report(self, employee_id):
        e = self.df[self.df["employee_id"] == str(employee_id).strip()].copy()
        if e.empty:
            return {"error": f"Employee '{employee_id}' not found."}
        static = e.iloc[0]
        territory = static.get("territory", "")
        emp_type = static.get("employee_type", "")

        conv_rate = round((e["outcome"] == "positive").sum() / max(len(e), 1), 3)
        avg_duration = round(e["actual_time_seconds"].mean(), 1) if "actual_time_seconds" in e.columns else 0
        total_sales = int(e["sales_volume"].sum()) if "sales_volume" in e.columns else 0
        products = e["product_name"].nunique()
        visits = len(e)

        terr_df = self.df[self.df["territory"] == territory]
        terr_conv = round((terr_df["outcome"] == "positive").sum() / max(len(terr_df), 1), 3)
        terr_dur = round(terr_df["actual_time_seconds"].mean(), 1) if "actual_time_seconds" in terr_df.columns else 120
        max_prods = self.df["product_name"].nunique()

        score = self._emp_score(conv_rate, avg_duration, products, terr_conv, terr_dur, max_prods)
        if "sales_volume" in e.columns and e["sales_volume"].sum() > 0:
            best_product = str(e.groupby("product_name")["sales_volume"].sum().idxmax())
        else:
            best_product = "N/A"

        doctor_rows = []
        for did in e["doctor_id"].unique():
            d_df = e[e["doctor_id"] == did]
            d_conv = round((d_df["outcome"] == "positive").sum() / max(len(d_df), 1), 3)
            doctor_rows.append({
                "doctor_id": str(did),                                      # cast numpy.str_ → str
                "doctor_name": str(d_df["doctor_name"].iloc[0]),
                "visits": int(len(d_df)),
                "conv_rate": float(d_conv),
                "total_sales": int(d_df["sales_volume"].sum()) if "sales_volume" in d_df.columns else 0,
            })
        doctor_rows.sort(key=lambda x: x["conv_rate"], reverse=True)

        suggestions = []
        if conv_rate < terr_conv:
            suggestions.append("Conversion below territory average. Improve objection handling.")
        if avg_duration < terr_dur * 0.8:
            suggestions.append("Meeting duration low. Spend more time building rapport.")
        if products < max_prods * 0.5:
            suggestions.append("Only pitching few products. Expand portfolio coverage.")

        return {
            "employee_id": str(employee_id),
            "employee_name": str(static.get("employee_name", "")),
            "employee_type": str(emp_type),
            "territory": str(territory),
            "total_visits": int(visits),
            "conv_rate": float(conv_rate),
            "avg_duration_sec": float(avg_duration),
            "total_sales": int(total_sales),
            "products_pitched": int(products),
            "emp_score": float(score),
            "most_successful_product": best_product,
            "territory_avg": {"conv_rate": float(terr_conv), "avg_duration_sec": float(terr_dur)},
            "comparison": {
                "conv_vs_avg": float(round(conv_rate - terr_conv, 3)),
                "duration_vs_avg": float(round(avg_duration - terr_dur, 1)),
                "outperforming": bool(conv_rate >= terr_conv),
            },
            "doctors_handled": doctor_rows,
            "improvement_suggestions": suggestions,
        }