"""
analytics_engine.py
-------------------
Pure Python data analysis engine for doctor sales metrics.
Handles all numerical computations, aggregations, and statistical analysis.
"""

import pandas as pd
from typing import Dict, Any, Optional


class RecommendationEngine:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def normalize(self, x, min_val, max_val):
        return (x - min_val) / (max_val - min_val + 1e-5)

    def score_products(self, doctor_df: pd.DataFrame):
        results = []

        for product in doctor_df["product_name"].unique():
            p_df = doctor_df[doctor_df["product_name"] == product]

            conversion = (p_df["outcome"] == "Positive").sum() / len(p_df)
            interest = p_df["interest_level"].mean()
            follow_up = (p_df["follow_up"] == "Yes").sum() / len(p_df)

            score = (0.5 * conversion) + (0.3 * (interest / 5)) + (0.2 * follow_up)

            if interest >= 3.5 and conversion >= 0.5:
                category = "high_performer"
            elif interest >= 3.5:
                category = "high_interest_low_conversion"
            elif conversion < 0.3:
                category = "low_performer"
            else:
                category = "potential"

            results.append({
                "product_name": product,
                "score": round(score, 3),
                "conversion_rate": round(conversion, 2),
                "avg_interest": round(interest, 2),
                "category": category
            })

        return sorted(results, key=lambda x: x["score"], reverse=True)

    def score_doctor(self, doctor_info):
        return (
            0.3 * self.normalize(doctor_info["patient_load"], 0, 100) +
            0.2 * self.normalize(doctor_info["experience_years"], 0, 40) +
            0.2 * self.normalize(doctor_info["publications_count"], 0, 50) +
            0.3 * self.normalize(doctor_info["social_media_reach"], 0, 10000)
        )

    def generate_strategy(self, doctor_info, top_products):
        strategy = []

        if doctor_info["experience_years"] > 10:
            strategy.append("clinical_data")
        else:
            strategy.append("benefit_selling")

        if doctor_info["social_media_reach"] > 5000:
            strategy.append("brand_products")

        if doctor_info["patient_load"] > 50:
            strategy.append("fast_moving_products")

        strategy.append(f"pitch_{min(2, len(top_products))}_products")

        return strategy


class DoctorAnalytics:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.df.columns = self.df.columns.str.strip().str.lower()
        self.df["territory"] = self.df["territory"].str.strip().str.lower()
        self.df["doctor_name"] = self.df["doctor_name"].str.strip()
        self.reco_engine = RecommendationEngine(self.df)

    def get_doctor_summary(self, doctor_id: int) -> Optional[Dict[str, Any]]:
        doctor_df = self.df[self.df["doctor_id"] == doctor_id]

        if doctor_df.empty:
            return None

        doctor_info = {
            "doctor_id": int(doctor_id),
            "doctor_name": doctor_df["doctor_name"].iloc[0],
            "territory": doctor_df["territory"].iloc[0],
            "specialty": doctor_df["specialty"].iloc[0],
            "patient_load": int(doctor_df["patient_load"].iloc[0]),
            "experience_years": int(doctor_df["experience_years"].iloc[0]),
            "publications_count": int(doctor_df["publications_count"].iloc[0]),
            "social_media_reach": int(doctor_df["social_media_reach"].iloc[0]),
        }
        doctor_df["outcome"] = doctor_df["outcome"].str.strip().str.lower()
        doctor_df["follow_up"] = doctor_df["follow_up"].str.strip().str.lower()

        engagement = {
            "total_interactions": doctor_df["interaction_id"].nunique(),
            "avg_interest_level": round(doctor_df["interest_level"].mean(), 2),
            "conversion_rate": round((doctor_df["outcome"] == "positive").sum() / len(doctor_df), 2),
            "follow_up_rate": round((doctor_df["follow_up"] == "yes").sum() / len(doctor_df), 2)
        }

        product_list = []
        for p in doctor_df["product_name"].unique():
            p_df = doctor_df[doctor_df["product_name"] == p]
            p_df["outcome"] = p_df["outcome"].str.strip().str.lower()
            product_list.append({
                "product_name": p,
                "avg_interest": round(p_df["interest_level"].mean(), 2),
                "conversion_rate": round((p_df["outcome"] == "positive").sum() / len(p_df), 2),
                "times_presented": len(p_df)
            })

        doctor_df["objection"] = doctor_df["objection"].str.strip().replace("", None)
        objection_counts = (
            doctor_df["objection"]
            .dropna()
            .dropna()
            .value_counts()
            .to_dict()
        )

        objection_analysis = {
            "total_objections": int(doctor_df["objection"].notna().sum()),
            "objection_breakdown": objection_counts,
            "has_objections": bool(objection_counts)
        }
        
        scored_products = self.reco_engine.score_products(doctor_df)
        doctor_score = self.reco_engine.score_doctor(doctor_info)
        strategy = self.reco_engine.generate_strategy(doctor_info, scored_products)

        return {
            "doctor_info": doctor_info,
            "engagement_metrics": engagement,
            "product_performance": {"product_breakdown": product_list},
            "objection_analysis": objection_analysis,
            "recommendation_engine": {
                "top_products": scored_products[:3],
                "all_products_ranked": scored_products,
                "doctor_score": round(doctor_score, 3),
                "strategy": strategy
            }
        }