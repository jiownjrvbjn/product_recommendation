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


# ═══════════════════════════════════════════════════════════
# 1. PRODUCT RECOMMENDATION ENGINE  (plan §2.1)
# ═══════════════════════════════════════════════════════════
class ProductRecommendationEngine:
    """
    Input : doctor_id, selected_time_sec, employee_type
    Output: last_meeting_recap, predicted_time, scored_products,
            time-slotted recommendations, AIDA stage per product
    """

    GLOBAL_AVG_TIME = 120  # fallback when < 5 meetings exist

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self._normalise()
        self._main_engine = DoctorAnalyticsEnhanced(df)
        self._trend = TrendAnalytics(df)
        self._aida = AIDAClassifier()

    # ── Internal normalisation ──────────────────────────────
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
        self.df["interaction_date"] = pd.to_datetime(
            self.df["interaction_date"], dayfirst=True, errors="coerce"
        )
        for col in ["actual_time_seconds", "sales_volume", "interest_level"]:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce").fillna(0)

    def _doctor_df(self, doctor_id: str) -> pd.DataFrame:
        return self.df[self.df["doctor_id"] == str(doctor_id).strip()].copy()

    # ── Public methods ──────────────────────────────────────

    def get_last_meeting_recap(self, doctor_id: str) -> Optional[Dict[str, Any]]:
        """Fetch the most recent interaction for a doctor (plan §2.1)."""
        d = self._doctor_df(doctor_id)
        if d.empty:
            return None
        latest = d.sort_values("interaction_date", ascending=False).iloc[0]
        obj_col = "objection" if "objection" in d.columns else \
                  ("objection_type" if "objection_type" in d.columns else None)
        return {
            "date":     str(latest["interaction_date"].date()) if pd.notna(latest["interaction_date"]) else None,
            "product":  latest.get("product_name", "—"),
            "objection": str(latest.get(obj_col, "—")) if obj_col else "—",
            "outcome":  latest.get("outcome", "—"),
            "interest_level": int(latest.get("interest_level", 0)),
            "sales_volume": int(latest.get("sales_volume", 0)),
            "notes":    "Auto-generated from last interaction record.",
        }

    def predict_available_time(self, doctor_id: str) -> int:
        """
        Plan §7.3: predicted_time = median(last_5_actual_times).
        If fewer than 5, use global average (120 sec).
        """
        d = self._doctor_df(doctor_id)
        if "actual_time_seconds" not in d.columns or d.empty:
            return self.GLOBAL_AVG_TIME
        times = d["actual_time_seconds"].dropna()
        times = times[times > 0].tolist()
        if len(times) >= 5:
            return int(np.median(times[-5:]))
        global_avg = self.df["actual_time_seconds"]
        global_avg = global_avg[global_avg > 0].mean()
        return int(global_avg) if not np.isnan(global_avg) else self.GLOBAL_AVG_TIME

    def score_products_for_doctor(self, doctor_id: str) -> List[Dict[str, Any]]:
        """
        Plan §7.2: product scoring formula.
        score = 0.4*conv_rate + 0.3*(avg_interest/5) + 0.2*follow_up_rate + 0.1*recent_trend
        """
        d = self._doctor_df(doctor_id)
        if d.empty:
            return []

        results = []
        for product in d["product_name"].unique():
            p_df = d[d["product_name"] == product]
            conv = (p_df["outcome"] == "positive").sum() / max(len(p_df), 1)
            interest = p_df["interest_level"].mean()
            follow_up = (p_df["follow_up"] == "yes").sum() / max(len(p_df), 1)

            # Recent trend component
            p_df2 = p_df.copy()
            p_df2["month"] = p_df2["interaction_date"].dt.to_period("M")
            monthly_conv = (
                p_df2.groupby("month")
                .apply(lambda x: (x["outcome"] == "positive").sum() / len(x))
                .tolist()
            )
            recent_trend = self._trend.calc_recent_trend_score(monthly_conv)

            score = max(
                0.4 * conv
                + 0.3 * (interest / 5)
                + 0.2 * follow_up
                + 0.1 * recent_trend,
                0.0,
            )

            # AIDA stage per product
            aida_result = self._aida.classify(p_df)

            results.append({
                "product_name":       product,
                "score":              round(score, 3),
                "conversion_rate":    round(conv, 3),
                "avg_interest":       round(interest, 2),
                "follow_up_rate":     round(follow_up, 3),
                "recent_trend":       recent_trend,
                "aida_stage":         aida_result["aida_stage"],
                "aida_label":         aida_result["aida_label"],
                "aida_color":         aida_result["aida_color"],
                "aida_emoji":         aida_result["aida_emoji"],
                "total_sales_volume": int(p_df["sales_volume"].sum()) if "sales_volume" in p_df.columns else 0,
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results

    def recommend_for_time(self, doctor_id: str, time_sec: int) -> Dict[str, Any]:
        """
        Plan §2.1: Time rules → product count, proportional duration allocation.
        ≤60s → 1 product | ≤120s → 3 products | else → 5 products
        """
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
            "doctor_id":      doctor_id,
            "effective_time": time_sec,
            "mode":           mode,
            "products":       selected,
        }

    def generate_custom_suggestion(
        self,
        doctor_id: str,
        time_sec: int,
        employee_type: str,
    ) -> Dict[str, Any]:
        """
        Rule-based meeting suggestion. LLM fallback is hooked here.
        Call llm_agents.custom_meeting_suggestion() if you need narrative text.
        """
        recap = self.get_last_meeting_recap(doctor_id)
        reco  = self.recommend_for_time(doctor_id, time_sec)
        top   = reco["products"][0] if reco["products"] else {}

        rule_text = (
            f"[{employee_type}] Focus on {top.get('product_name', '—')} "
            f"(AIDA: {top.get('aida_stage', '—')}). "
            f"Suggested duration: {top.get('suggested_duration_sec', time_sec)} sec. "
            f"Last objection: {recap.get('objection', 'none') if recap else 'none'}. "
            "Use evidence-based pitch."
        )

        return {
            "doctor_id":        doctor_id,
            "employee_type":    employee_type,
            "top_product":      top.get("product_name"),
            "rule_suggestion":  rule_text,
            "llm_hook":         "call llm_agents.custom_meeting_suggestion(doctor_id, top_product, persona, objections)",
            "last_recap":       recap,
            "recommendations":  reco,
        }

    def get_full_product_suggestions(
        self,
        doctor_id: str,
        selected_time_sec: int,
        employee_type: str = "MR",
    ) -> Optional[Dict[str, Any]]:
        """
        Composite method — matches /recommendations/product_suggestions endpoint.
        Aggregates: doctor rating, last meeting, top products, AIDA per product,
        time allocation, rule-based suggestion.
        """
        d = self._doctor_df(doctor_id)
        if d.empty:
            return None

        # Doctor rating (plan §7.1)
        doc_info = {
            "patient_load":      int(d["patient_load"].iloc[0]) if "patient_load" in d.columns else 0,
            "publications_count": int(d["publications_count"].iloc[0]) if "publications_count" in d.columns else 0,
            "social_media_reach": int(d["social_media_reach"].iloc[0]) if "social_media_reach" in d.columns else 0,
            "conversion_rate":   round((d["outcome"] == "positive").sum() / max(len(d), 1), 3),
        }
        doctor_score = self._doctor_rating(doc_info)
        doctor_tier  = "high" if doctor_score >= 0.75 else ("medium" if doctor_score >= 0.45 else "low")

        predicted_time = self.predict_available_time(doctor_id)
        effective_time = selected_time_sec if selected_time_sec > 0 else predicted_time
        recap          = self.get_last_meeting_recap(doctor_id)
        reco           = self.recommend_for_time(doctor_id, effective_time)
        suggestion     = self.generate_custom_suggestion(doctor_id, effective_time, employee_type)

        return {
            "doctor_id":      doctor_id,
            "doctor_name":    str(d["doctor_name"].iloc[0]),
            "doctor_rating":  doctor_score,
            "doctor_tier":    doctor_tier,
            "predicted_time": predicted_time,
            "selected_time":  effective_time,
            "last_meeting":   recap,
            "recommendations": reco["products"],
            "custom_suggestion": suggestion["rule_suggestion"],
            "llm_hook":       suggestion["llm_hook"],
        }

    def _doctor_rating(self, doc_info: Dict) -> float:
        """Plan §7.1 formula."""
        def norm(x, mn, mx): return (x - mn) / (mx - mn + 1e-5)
        return round(
            0.4 * norm(doc_info.get("patient_load", 0), 0, 200)
            + 0.2 * norm(doc_info.get("publications_count", 0), 0, 50)
            + 0.2 * norm(doc_info.get("social_media_reach", 0), 0, 10000)
            + 0.2 * doc_info.get("conversion_rate", 0),
            3,
        )


# ═══════════════════════════════════════════════════════════
# 2. PRODUCT PERFORMANCE ENGINE  (plan §2.2)
# ═══════════════════════════════════════════════════════════
class ProductPerformanceEngine:
    """
    Input : product_name (optional), region, quarter
    Output: overall summary DataFrame, per-product deep dive
    Metrics: sales, QoQ growth, conversion, interest, follow-up, region breakdown, trend
    """

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
        self.df["interaction_date"] = pd.to_datetime(
            self.df["interaction_date"], dayfirst=True, errors="coerce"
        )
        for col in ["sales_volume", "interest_level"]:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce").fillna(0)
        self.df["follow_up"] = self.df["follow_up"].astype(str).str.strip().str.lower()
        if "quarter" in self.df.columns:
            self.df["quarter"] = self.df["quarter"].astype(str).str.strip().str.upper()
        if "region" in self.df.columns:
            self.df["region"] = self.df["region"].astype(str).str.strip()

    def _filter(self, region: Optional[str] = None, quarter: Optional[str] = None) -> pd.DataFrame:
        df = self.df.copy()
        if region:
            df = df[df["region"].str.lower() == region.strip().lower()]
        if quarter:
            df = df[df["quarter"].str.upper() == quarter.strip().upper()]
        return df

    def _qoq_growth(self, product_name: str) -> float:
        """
        Plan §7.4: qoq_growth = (current_q_sales - prev_q_sales) / prev_q_sales
        Uses the two most recent quarters in the data.
        """
        p_df = self.df[self.df["product_name"] == product_name]
        if "quarter" not in p_df.columns or p_df.empty:
            return 0.0

        quarterly = (
            p_df.groupby(["year", "quarter"])["sales_volume"].sum()
            if "year" in p_df.columns
            else p_df.groupby("quarter")["sales_volume"].sum()
        )
        if len(quarterly) < 2:
            return 0.0
        vals = quarterly.values
        prev, curr = vals[-2], vals[-1]
        if prev == 0:
            return 0.0
        return round((curr - prev) / prev, 4)

    def _sales_trend(self, monthly_sales: List[float]) -> str:
        if len(monthly_sales) < 2:
            return "stable"
        x = np.arange(len(monthly_sales))
        y = np.array(monthly_sales, dtype=float)
        mask = ~np.isnan(y)
        if mask.sum() < 2:
            return "stable"
        slope = np.polyfit(x[mask], y[mask], 1)[0]
        if slope > 5:
            return "improving"
        elif slope < -5:
            return "declining"
        return "stable"

    def get_overall_summary(
        self,
        region: Optional[str] = None,
        quarter: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Returns DataFrame: product, total_sales, qoq_growth, conv_rate,
                           avg_interest, follow_up_rate, top_region, trend
        Matches plan §2.2 overall summary report.
        """
        df = self._filter(region, quarter)
        if df.empty:
            return pd.DataFrame()

        rows = []
        for product in df["product_name"].unique():
            p_df = df[df["product_name"] == product]
            total_sales   = int(p_df["sales_volume"].sum())
            conv_rate     = round((p_df["outcome"] == "positive").sum() / max(len(p_df), 1), 3)
            avg_interest  = round(p_df["interest_level"].mean(), 2)
            follow_up_rate = round((p_df["follow_up"] == "yes").sum() / max(len(p_df), 1), 3)
            qoq_growth    = self._qoq_growth(product)

            # Monthly sales for trend
            p_all = self.df[self.df["product_name"] == product].copy()
            p_all["month"] = p_all["interaction_date"].dt.to_period("M")
            monthly_sales = p_all.groupby("month")["sales_volume"].sum().tolist()
            trend = self._sales_trend(monthly_sales)

            # Top region
            if "region" in p_df.columns:
                top_region = p_df.groupby("region")["sales_volume"].sum().idxmax()
            else:
                top_region = "N/A"

            rows.append({
                "product":            product,
                "total_sales":        total_sales,
                "qoq_growth":         qoq_growth,
                "conv_rate":          conv_rate,
                "avg_interest":       avg_interest,
                "follow_up_rate":     follow_up_rate,
                "top_region":         top_region,
                "trend":              trend,
                "is_underperforming": (
                    conv_rate < 0.3
                    or trend == "declining"
                    or (qoq_growth < -0.1 and avg_interest < 3.0)
                ),
            })

        summary_df = pd.DataFrame(rows).sort_values("total_sales", ascending=False).reset_index(drop=True)
        return summary_df

    def get_product_detail(self, product_name: str) -> Dict[str, Any]:
        """
        Per-product deep dive (plan §2.2):
        - Time series of sales & conversion
        - Objection breakdown
        - Top performing doctors
        - Region-wise breakdown
        - QoQ growth
        """
        p_df = self.df[self.df["product_name"] == product_name].copy()
        if p_df.empty:
            return {"error": f"Product '{product_name}' not found."}

        # Monthly time-series
        p_df["month"] = p_df["interaction_date"].dt.to_period("M")
        monthly_sales = p_df.groupby("month")["sales_volume"].sum().reset_index()
        monthly_sales.columns = ["month", "sales_volume"]
        monthly_sales["month"] = monthly_sales["month"].astype(str)

        monthly_conv = (
            p_df.groupby("month")
            .apply(lambda x: round((x["outcome"] == "positive").sum() / len(x), 3))
            .reset_index()
        )
        monthly_conv.columns = ["month", "conversion_rate"]
        monthly_conv["month"] = monthly_conv["month"].astype(str)

        # Objection breakdown — handle both 'objection' and 'objection_type' column names
        obj_col = "objection" if "objection" in p_df.columns else \
                  ("objection_type" if "objection_type" in p_df.columns else None)
        if obj_col:
            obj_series = p_df[obj_col].astype(str).str.strip().str.lower()
            obj_df = p_df[~obj_series.isin(["nan", "none", ""])]
            objection_breakdown = obj_series[~obj_series.isin(["nan", "none", ""])].value_counts().to_dict()
        else:
            objection_breakdown = {}

        # Top doctors for this product
        doctor_perf = (
            p_df.groupby("doctor_id")
            .agg(
                doctor_name=("doctor_name", "first"),
                total_sales=("sales_volume", "sum"),
                conv_rate=("outcome", lambda x: (x == "positive").sum() / max(len(x), 1)),
                interactions=("interaction_id", "count"),
            )
            .reset_index()
            .sort_values("total_sales", ascending=False)
            .head(5)
        )
        top_doctors = doctor_perf.to_dict(orient="records")
        for doc in top_doctors:
            doc["total_sales"] = int(doc["total_sales"])
            doc["conv_rate"]   = round(doc["conv_rate"], 3)

        # Region breakdown
        region_breakdown = {}
        if "region" in p_df.columns:
            region_group = p_df.groupby("region").agg(
                total_sales=("sales_volume", "sum"),
                conv_rate=("outcome", lambda x: round((x == "positive").sum() / max(len(x), 1), 3)),
                interactions=("interaction_id", "count"),
            ).reset_index()
            region_breakdown = region_group.to_dict(orient="records")

        conv_rate = round((p_df["outcome"] == "positive").sum() / max(len(p_df), 1), 3)
        avg_interest = round(p_df["interest_level"].mean(), 2)
        trend = self._sales_trend(monthly_sales["sales_volume"].tolist())
        qoq_growth = self._qoq_growth(product_name)

        # Engagement depth: avg interactions per unique doctor (mirrors sessions/call ratio)
        unique_doctors = p_df["doctor_id"].nunique()
        engagement_depth = round(len(p_df) / max(unique_doctors, 1), 2)

        # Underperformance flag: conv < 0.3 OR declining trend OR negative QoQ AND low interest
        is_underperforming = (
            conv_rate < 0.3
            or trend == "declining"
            or (qoq_growth < -0.1 and avg_interest < 3.0)
        )

        return {
            "product_name":        product_name,
            "total_sales":         int(p_df["sales_volume"].sum()),
            "total_interactions":  len(p_df),
            "conversion_rate":     conv_rate,
            "avg_interest":        avg_interest,
            "qoq_growth":          qoq_growth,
            "trend":               trend,
            "engagement_depth":    engagement_depth,
            "is_underperforming":  is_underperforming,
            "monthly_sales":       monthly_sales.to_dict(orient="records"),
            "monthly_conversion":  monthly_conv.to_dict(orient="records"),
            "objection_breakdown": objection_breakdown,
            "top_doctors":         top_doctors,
            "region_breakdown":    region_breakdown,
            "llm_hook":            "call llm_agents.explain_product_underperformance(product_name, metrics, objections)",
        }

    def get_quarterly_table(self, product_name: Optional[str] = None) -> pd.DataFrame:
        """Sales by quarter (and optionally by product) for dashboard table."""
        df = self.df if not product_name else self.df[self.df["product_name"] == product_name]
        if "quarter" not in df.columns:
            return pd.DataFrame()
        grp_cols = ["quarter"] if not product_name else ["product_name", "quarter"]
        if "year" in df.columns:
            grp_cols = (["year"] + grp_cols) if "year" not in grp_cols else grp_cols
        table = (
            df.groupby(grp_cols)
            .agg(
                total_sales=("sales_volume", "sum"),
                conv_rate=("outcome", lambda x: round((x == "positive").sum() / max(len(x), 1), 3)),
                interactions=("interaction_id", "count"),
            )
            .reset_index()
            .sort_values(grp_cols)
        )
        return table


# ═══════════════════════════════════════════════════════════
# 3. DOCTOR REVIEW ENGINE  (plan §2.3)
# ═══════════════════════════════════════════════════════════
class DoctorReviewEngine:
    """
    Input : doctor_id (optional)
    Output: all-doctors summary DataFrame or individual doctor analysis
    KPIs  : interactions, conv_rate, avg_interest, follow_up, LTV, trend, rank
    """

    # Assumed price per unit for LTV (can be overridden)
    DEFAULT_PRICE_PER_UNIT = 500  # ₹500 per unit

    def __init__(self, df: pd.DataFrame, price_per_unit: int = DEFAULT_PRICE_PER_UNIT):
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
        self.df["interaction_date"] = pd.to_datetime(
            self.df["interaction_date"], dayfirst=True, errors="coerce"
        )
        for col in ["sales_volume", "interest_level"]:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce").fillna(0)

    def _lifetime_value(self, doctor_df: pd.DataFrame) -> int:
        """LTV = sum(sales_volume) * price_per_unit"""
        if "sales_volume" not in doctor_df.columns:
            return 0
        return int(doctor_df["sales_volume"].sum() * self.price_per_unit)

    def _doctor_score(self, row: pd.Series) -> float:
        """Plan §7.1: doctor score for summary table."""
        def norm(x, mn, mx): return (x - mn) / (mx - mn + 1e-5)
        return round(
            0.4 * norm(row.get("patient_load", 0), 0, 200)
            + 0.2 * norm(row.get("publications_count", 0), 0, 50)
            + 0.2 * norm(row.get("social_media_reach", 0), 0, 10000)
            + 0.2 * row.get("conv_rate", 0),
            3,
        )

    def get_all_doctors_summary(self, territory: Optional[str] = None) -> pd.DataFrame:
        """
        Returns DataFrame: doctor_id, doctor_name, specialty, territory,
                           conv_rate, ltv, trend, score, tier
        Sortable, filterable (plan §2.3).
        """
        df = self.df.copy()
        if territory:
            df = df[df["territory"] == territory.strip().lower()]

        rows = []
        for did in df["doctor_id"].unique():
            d = df[df["doctor_id"] == did]
            conv_rate   = round((d["outcome"] == "positive").sum() / max(len(d), 1), 3)
            avg_interest = round(d["interest_level"].mean(), 2)
            follow_up   = round((d["follow_up"] == "yes").sum() / max(len(d), 1), 3)
            ltv         = self._lifetime_value(d)

            # Trend using monthly conversion
            d2 = d.copy()
            d2["month"] = d2["interaction_date"].dt.to_period("M")
            monthly_conv = (
                d2.groupby("month")
                .apply(lambda x: (x["outcome"] == "positive").sum() / len(x))
                .tolist()
            )
            trend = self._trend_engine._calc_trend(monthly_conv)

            static = d.iloc[0]
            score = self._doctor_score({
                "patient_load":      int(static.get("patient_load", 0)),
                "publications_count": int(static.get("publications_count", 0)),
                "social_media_reach": int(static.get("social_media_reach", 0)),
                "conv_rate":         conv_rate,
            })
            tier = "high" if score >= 0.75 else ("medium" if score >= 0.45 else "low")

            rows.append({
                "doctor_id":       did,
                "doctor_name":     str(static.get("doctor_name", "")),
                "specialty":       str(static.get("specialty", "")),
                "territory":       str(static.get("territory", "")),
                "total_interactions": len(d),
                "conv_rate":       conv_rate,
                "avg_interest":    avg_interest,
                "follow_up_rate":  follow_up,
                "ltv":             ltv,
                "trend":           trend,
                "doctor_score":    score,
                "tier":            tier,
            })

        return pd.DataFrame(rows).sort_values("ltv", ascending=False).reset_index(drop=True)

    def get_doctor_analysis(self, doctor_id: str) -> Dict[str, Any]:
        """
        Individual doctor deep dive (plan §2.3):
        - Radar chart metrics vs peers
        - Objection intelligence
        - Product affinity
        - Next best action hook
        - Rank within territory/specialty
        """
        d = self.df[self.df["doctor_id"] == str(doctor_id).strip()].copy()
        if d.empty:
            return {"error": f"Doctor '{doctor_id}' not found."}

        conv_rate    = round((d["outcome"] == "positive").sum() / max(len(d), 1), 3)
        avg_interest = round(d["interest_level"].mean(), 2)
        follow_up    = round((d["follow_up"] == "yes").sum() / max(len(d), 1), 3)
        ltv          = self._lifetime_value(d)
        static       = d.iloc[0]
        territory    = str(static.get("territory", "")).lower()
        specialty    = str(static.get("specialty", ""))

        # Radar metrics: conv_rate, avg_interest, follow_up, ltv_norm, doctor_score
        score = self._doctor_score({
            "patient_load":      int(static.get("patient_load", 0)),
            "publications_count": int(static.get("publications_count", 0)),
            "social_media_reach": int(static.get("social_media_reach", 0)),
            "conv_rate":         conv_rate,
        })

        # Peer stats (territory)
        terr_df = self.df[self.df["territory"] == territory]
        peer_conv   = round((terr_df["outcome"] == "positive").sum() / max(len(terr_df), 1), 3)
        peer_interest = round(terr_df["interest_level"].mean(), 2)

        # Rank in territory by conv_rate
        terr_convs = [
            (self.df[self.df["doctor_id"] == did]["outcome"] == "positive").sum()
            / max(len(self.df[self.df["doctor_id"] == did]), 1)
            for did in terr_df["doctor_id"].unique()
        ]
        terr_rank = sorted(terr_convs, reverse=True)
        rank_in_territory = next((i + 1 for i, v in enumerate(terr_rank) if abs(v - conv_rate) < 1e-6), 0)

        # Rank in specialty by conv_rate
        spec_df = self.df[self.df["specialty"] == specialty]
        spec_convs = [
            (self.df[self.df["doctor_id"] == did]["outcome"] == "positive").sum()
            / max(len(self.df[self.df["doctor_id"] == did]), 1)
            for did in spec_df["doctor_id"].unique()
        ]
        spec_rank_list = sorted(spec_convs, reverse=True)
        rank_in_specialty = next((i + 1 for i, v in enumerate(spec_rank_list) if abs(v - conv_rate) < 1e-6), 0)

        # Product affinity
        product_affinity = []
        for product in d["product_name"].unique():
            p_df = d[d["product_name"] == product]
            product_affinity.append({
                "product_name":    product,
                "conv_rate":       round((p_df["outcome"] == "positive").sum() / max(len(p_df), 1), 3),
                "avg_interest":    round(p_df["interest_level"].mean(), 2),
                "total_sales":     int(p_df["sales_volume"].sum()) if "sales_volume" in p_df.columns else 0,
            })
        product_affinity.sort(key=lambda x: x["conv_rate"], reverse=True)

        # Monthly trend
        trends = self._trend_engine.get_doctor_trends(doctor_id)

        # Objections
        objections = self._objection_tracker.get_objection_analysis(doctor_id)

        return {
            "doctor_id":      doctor_id,
            "doctor_name":    str(static.get("doctor_name", "")),
            "specialty":      specialty,
            "territory":      territory,
            "conv_rate":      conv_rate,
            "avg_interest":   avg_interest,
            "follow_up_rate": follow_up,
            "ltv":            ltv,
            "doctor_score":   score,
            "tier":           "high" if score >= 0.75 else ("medium" if score >= 0.45 else "low"),
            "rank_in_territory":  rank_in_territory,
            "rank_in_specialty":  rank_in_specialty,
            "peer_metrics": {
                "territory_avg_conv":     peer_conv,
                "territory_avg_interest": peer_interest,
            },
            "radar_metrics": {
                "conv_rate":      conv_rate,
                "avg_interest":   avg_interest / 5,
                "follow_up_rate": follow_up,
                "doctor_score":   score,
                "ltv_norm":       min(ltv / 500000, 1.0),  # normalise for radar
            },
            "product_affinity": product_affinity,
            "trend_analytics":  trends,
            "objection_intelligence": objections,
            "nba_hook": "Use DoctorAnalyticsEnhanced.get_doctor_summary() for NBA engine output.",
        }


# ═══════════════════════════════════════════════════════════
# 4. EMPLOYEE REPORT ENGINE  (plan §2.4)
# ═══════════════════════════════════════════════════════════
class EmployeeReportEngine:
    """
    Input : employee_id (optional)
    Output: team summary DataFrame or individual employee report
    KPIs  : visits, conv_rate, avg_duration, product_mix, win_rate vs peers
    Formula §7.5:
      emp_score = 0.5*(conv_rate/territory_avg_conv)
                + 0.3*(avg_duration/territory_avg_duration)
                + 0.2*(products_pitched/max_products)
    """

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
        self.df["employee_id"] = self.df["employee_id"].astype(str).str.strip()
        self.df["territory"]   = self.df["territory"].astype(str).str.strip().str.lower()
        for col in ["actual_time_seconds", "sales_volume", "interest_level"]:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce").fillna(0)

    def _emp_score(
        self,
        conv_rate: float,
        avg_duration: float,
        products_pitched: int,
        territory_avg_conv: float,
        territory_avg_duration: float,
        max_products: int,
    ) -> float:
        """Plan §7.5 formula."""
        conv_ratio     = conv_rate / max(territory_avg_conv, 1e-5)
        duration_ratio = avg_duration / max(territory_avg_duration, 1e-5)
        product_ratio  = products_pitched / max(max_products, 1)
        score = 0.5 * conv_ratio + 0.3 * duration_ratio + 0.2 * product_ratio
        return round(min(score, 2.0), 3)  # cap at 2× to avoid outlier explosion

    def get_team_summary(self, territory: Optional[str] = None) -> pd.DataFrame:
        """
        Group by employee_type → average conversion, total sales (plan §2.4).
        Also returns individual rows for granular table.
        """
        df = self.df.copy()
        if territory:
            df = df[df["territory"] == territory.strip().lower()]

        # Territory-level averages for score calculation
        terr_conv = (df["outcome"] == "positive").sum() / max(len(df), 1)
        terr_dur  = df["actual_time_seconds"].mean() if "actual_time_seconds" in df.columns else 120
        max_prods = df["product_name"].nunique()

        rows = []
        for eid in df["employee_id"].unique():
            e = df[df["employee_id"] == eid]
            static = e.iloc[0]
            conv_rate    = round((e["outcome"] == "positive").sum() / max(len(e), 1), 3)
            avg_duration = round(e["actual_time_seconds"].mean(), 1) if "actual_time_seconds" in e.columns else 0
            total_sales  = int(e["sales_volume"].sum()) if "sales_volume" in e.columns else 0
            products     = e["product_name"].nunique()
            visits       = len(e)

            score = self._emp_score(conv_rate, avg_duration, products, terr_conv, terr_dur, max_prods)

            rows.append({
                "employee_id":     eid,
                "employee_name":   str(static.get("employee_name", "")),
                "employee_type":   str(static.get("employee_type", "")),
                "territory":       str(static.get("territory", "")),
                "total_visits":    visits,
                "conv_rate":       conv_rate,
                "avg_duration_sec": avg_duration,
                "total_sales":     total_sales,
                "products_pitched": products,
                "emp_score":       score,
            })

        df_out = pd.DataFrame(rows).sort_values("emp_score", ascending=False).reset_index(drop=True)
        return df_out

    def get_employee_report(self, employee_id: str) -> Dict[str, Any]:
        """
        Individual employee deep dive (plan §2.4):
        - Compare to territory average
        - Doctor list with per-doctor conversion
        - Product portfolio mix
        - Improvement suggestions (rule-based)
        - emp_score (plan §7.5)
        """
        e = self.df[self.df["employee_id"] == str(employee_id).strip()].copy()
        if e.empty:
            return {"error": f"Employee '{employee_id}' not found."}

        static   = e.iloc[0]
        territory = str(static.get("territory", "")).lower()
        emp_type  = str(static.get("employee_type", ""))

        conv_rate    = round((e["outcome"] == "positive").sum() / max(len(e), 1), 3)
        avg_duration = round(e["actual_time_seconds"].mean(), 1) if "actual_time_seconds" in e.columns else 0
        total_sales  = int(e["sales_volume"].sum()) if "sales_volume" in e.columns else 0
        products     = e["product_name"].nunique()
        visits       = len(e)

        # Territory averages
        terr_df   = self.df[self.df["territory"] == territory]
        terr_conv = round((terr_df["outcome"] == "positive").sum() / max(len(terr_df), 1), 3)
        terr_dur  = round(terr_df["actual_time_seconds"].mean(), 1) if "actual_time_seconds" in terr_df.columns else 120
        max_prods = self.df["product_name"].nunique()

        # emp_score (plan §7.5)
        score = self._emp_score(conv_rate, avg_duration, products, terr_conv, terr_dur, max_prods)

        # Doctors handled
        doctor_rows = []
        for did in e["doctor_id"].unique():
            d_df = e[e["doctor_id"] == did]
            d_conv = round((d_df["outcome"] == "positive").sum() / max(len(d_df), 1), 3)
            doctor_rows.append({
                "doctor_id":    did,
                "doctor_name":  str(d_df["doctor_name"].iloc[0]),
                "visits":       len(d_df),
                "conv_rate":    d_conv,
                "total_sales":  int(d_df["sales_volume"].sum()) if "sales_volume" in d_df.columns else 0,
            })
        doctor_rows.sort(key=lambda x: x["conv_rate"], reverse=True)

        # Product portfolio mix
        product_mix = []
        for prod in e["product_name"].unique():
            p_df = e[e["product_name"] == prod]
            product_mix.append({
                "product_name":  prod,
                "pitch_count":   len(p_df),
                "conv_rate":     round((p_df["outcome"] == "positive").sum() / max(len(p_df), 1), 3),
                "total_sales":   int(p_df["sales_volume"].sum()) if "sales_volume" in p_df.columns else 0,
            })
        product_mix.sort(key=lambda x: x["total_sales"], reverse=True)

        # Rule-based improvement suggestions
        suggestions = []
        if conv_rate < terr_conv:
            suggestions.append(f"Conversion rate ({conv_rate}) is below territory average ({terr_conv}). Focus on objection handling and follow-up.")
        if avg_duration < terr_dur * 0.8:
            suggestions.append(f"Average meeting duration ({avg_duration:.0f}s) is significantly below territory average ({terr_dur:.0f}s). Spend more time building rapport.")
        if products < max_prods * 0.5:
            suggestions.append(f"Pitching only {products}/{max_prods} products. Expand product portfolio coverage.")
        if not suggestions:
            suggestions.append("Performance is on par with territory. Consider mentoring peers or upselling complementary products.")

        return {
            "employee_id":     employee_id,
            "employee_name":   str(static.get("employee_name", "")),
            "employee_type":   emp_type,
            "territory":       territory,
            "total_visits":    visits,
            "conv_rate":       conv_rate,
            "avg_duration_sec": avg_duration,
            "total_sales":     total_sales,
            "products_pitched": products,
            "emp_score":       score,
            "territory_avg": {
                "conv_rate":       terr_conv,
                "avg_duration_sec": terr_dur,
            },
            "comparison": {
                "conv_vs_avg":       round(conv_rate - terr_conv, 3),
                "duration_vs_avg":   round(avg_duration - terr_dur, 1),
                "outperforming":     conv_rate >= terr_conv,
            },
            "doctors_handled":   doctor_rows,
            "product_mix":       product_mix,
            "improvement_suggestions": suggestions,
            "llm_hook":          "call llm_agents.employee_coaching_tips(employee_id, report) for narrative.",
        }