"""
analytics_engine.py
-----------------------------
AI Sales Assistant Engine with:
- AIDA Stage Classification (Awareness / Interest / Desire / Action)
- Intent Score (interest + follow_up + recent_activity)
- Doctor Persona Classification
- 3-Layer Product Filtering (AIDA → Score → Time)
- Structured product output (primary / support / closing / reminder)
- Next Best Action engine
- Visit Success Predictor
- Doctor Rating formula from plan (patient_load, publications, social_reach, conv_rate)
- Product Score formula from plan (conv_rate, avg_interest, follow_up, recent_trend)
- Sales volume tracking, QoQ growth, territory/quarter filters
- No auto employee mapping — employee_type comes from UI
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta


# ─────────────────────────────────────────────
# TREND ANALYTICS
# ─────────────────────────────────────────────
class TrendAnalytics:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.df["interaction_date"] = pd.to_datetime(self.df["interaction_date"], dayfirst=True, errors="coerce")

    def get_doctor_trends(self, doctor_id: str) -> Dict[str, Any]:
        doctor_df = self.df[self.df["doctor_id"].astype(str).str.strip() == str(doctor_id).strip()]
        if doctor_df.empty:
            return None

        doctor_df = doctor_df.sort_values("interaction_date")
        doctor_df["month"] = doctor_df["interaction_date"].dt.to_period("M")

        monthly_conversion = (
            doctor_df.groupby("month")
            .apply(lambda x: (x["outcome"] == "positive").sum() / len(x))
            .reset_index()
        )
        monthly_conversion.columns = ["month", "conversion_rate"]
        monthly_conversion["month"] = monthly_conversion["month"].astype(str)

        monthly_interest = (
            doctor_df.groupby("month")["interest_level"].mean().reset_index()
        )
        monthly_interest.columns = ["month", "avg_interest"]
        monthly_interest["month"] = monthly_interest["month"].astype(str)

        interaction_freq = doctor_df.groupby("month").size().reset_index()
        interaction_freq.columns = ["month", "interaction_count"]
        interaction_freq["month"] = interaction_freq["month"].astype(str)

        # Monthly sales volume trend
        monthly_sales = doctor_df.groupby("month")["sales_volume"].sum().reset_index() if "sales_volume" in doctor_df.columns else pd.DataFrame()
        if not monthly_sales.empty:
            monthly_sales.columns = ["month", "sales_volume"]
            monthly_sales["month"] = monthly_sales["month"].astype(str)

        return {
            "monthly_conversion": monthly_conversion.to_dict(orient="records"),
            "monthly_interest": monthly_interest.to_dict(orient="records"),
            "interaction_frequency": interaction_freq.to_dict(orient="records"),
            "monthly_sales": monthly_sales.to_dict(orient="records") if not monthly_sales.empty else [],
            "trends": {
                "conversion": self._calc_trend(monthly_conversion["conversion_rate"].tolist()),
                "interest": self._calc_trend(monthly_interest["avg_interest"].tolist()),
                "sales": self._calc_trend(monthly_sales["sales_volume"].tolist()) if not monthly_sales.empty else "stable",
            },
        }

    def _calc_trend(self, values: List[float]) -> str:
        if not values or len(values) < 2:
            return "stable"
        x = np.arange(len(values))
        y = np.array(values, dtype=float)
        mask = ~np.isnan(y)
        if mask.sum() < 2:
            return "stable"
        slope = np.polyfit(x[mask], y[mask], 1)[0]
        if slope > 0.05:
            return "improving"
        elif slope < -0.05:
            return "declining"
        return "stable"

    def calc_recent_trend_score(self, monthly_conv_rates: List[float]) -> float:
        """
        Plan §7.2: recent_trend component for product score.
        slope of last 3 months conv rates > 0 → +0.1, < 0 → -0.05
        """
        last3 = monthly_conv_rates[-3:] if len(monthly_conv_rates) >= 3 else monthly_conv_rates
        if len(last3) < 2:
            return 0.0
        x = np.arange(len(last3))
        y = np.array(last3, dtype=float)
        slope = np.polyfit(x, y, 1)[0]
        return 0.1 if slope > 0 else -0.05


# ─────────────────────────────────────────────
# COMPETITIVE INTELLIGENCE
# ─────────────────────────────────────────────
class CompetitiveIntelligence:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        if "objection" not in self.df.columns:
            self.df["objection"] = self.df["objection_type"].astype(str).str.strip().str.lower() \
                if "objection_type" in self.df.columns else "none"
        else:
            self.df["objection"] = self.df["objection"].astype(str).str.strip().str.lower()
        self.df["outcome"] = self.df["outcome"].astype(str).str.strip().str.lower()

    def get_competitive_analysis(self, doctor_id: str) -> Dict[str, Any]:
        doctor_df = self.df[self.df["doctor_id"].astype(str).str.strip() == str(doctor_id).strip()]
        if doctor_df.empty:
            return None

        comp_loyalty = doctor_df[doctor_df["objection"].str.contains("competitor|alternative|already using", na=False)]
        threat_score = len(comp_loyalty) / len(doctor_df) if len(doctor_df) > 0 else 0

        products_at_risk = []
        for product in doctor_df["product_name"].unique():
            p_df = doctor_df[doctor_df["product_name"] == product]
            comp_obj = p_df[p_df["objection"].str.contains("competitor|alternative|already using", na=False)]
            if len(comp_obj) > 0:
                products_at_risk.append({
                    "product_name": product,
                    "competitor_objections": len(comp_obj),
                    "risk_level": "high" if len(comp_obj) / len(p_df) > 0.3 else "medium",
                })

        wins = (doctor_df["outcome"] == "positive").sum()
        losses = (doctor_df["outcome"] == "negative").sum()
        total = len(doctor_df)

        level = "high" if threat_score > 0.4 else ("medium" if threat_score > 0.2 else "low")
        return {
            "competitor_threat_score": round(threat_score, 3),
            "threat_level": level,
            "products_at_risk": products_at_risk,
            "win_loss_analysis": {
                "wins": int(wins),
                "losses": int(losses),
                "win_rate": round(wins / total, 2) if total > 0 else 0,
            },
        }


# ─────────────────────────────────────────────
# MANAGER COMPARISON
# ─────────────────────────────────────────────
class ManagerComparison:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.df["outcome"] = self.df["outcome"].astype(str).str.strip().str.lower()
        self.df["territory"] = self.df["territory"].astype(str).str.strip().str.lower()

    def compare_doctor_to_territory(self, doctor_id: str, territory: str) -> Dict[str, Any]:
        doctor_df = self.df[self.df["doctor_id"].astype(str).str.strip() == str(doctor_id).strip()]
        if doctor_df.empty:
            return None

        doc_conv = (doctor_df["outcome"] == "positive").sum() / len(doctor_df)
        doc_interest = doctor_df["interest_level"].mean()

        territory_df = self.df[self.df["territory"] == territory.strip().lower()]
        all_convs = []
        for did in territory_df["doctor_id"].unique():
            d = territory_df[territory_df["doctor_id"] == did]
            all_convs.append((d["outcome"] == "positive").sum() / len(d))

        terr_avg = np.mean(all_convs) if all_convs else 0
        pct_rank = (sum(1 for x in all_convs if x < doc_conv) / len(all_convs) * 100) if all_convs else 50

        tier = (
            "Top Performer" if pct_rank >= 75
            else ("Above Average" if pct_rank >= 50
            else ("Below Average" if pct_rank >= 25
            else "Needs Attention"))
        )
        return {
            "doctor_conversion": round(doc_conv, 3),
            "territory_avg_conversion": round(terr_avg, 3),
            "percentile_rank": round(pct_rank, 1),
            "performance_tier": tier,
        }

    def get_specialty_comparison(self, doctor_id: str) -> Dict[str, Any]:
        doctor_df = self.df[self.df["doctor_id"].astype(str).str.strip() == str(doctor_id).strip()]
        if doctor_df.empty:
            return None

        specialty = doctor_df["specialty"].iloc[0]
        spec_df = self.df[self.df["specialty"] == specialty]
        doc_conv = (doctor_df["outcome"] == "positive").sum() / len(doctor_df)

        spec_convs = []
        for did in spec_df["doctor_id"].unique():
            d = spec_df[spec_df["doctor_id"] == did]
            spec_convs.append((d["outcome"] == "positive").sum() / len(d))

        spec_avg = np.mean(spec_convs) if spec_convs else 0
        sorted_convs = sorted(spec_convs, reverse=True)
        rank = next((i + 1 for i, v in enumerate(sorted_convs) if abs(v - doc_conv) < 1e-6), 0)

        return {
            "specialty": specialty,
            "total_in_specialty": len(spec_convs),
            "doctor_conversion": round(doc_conv, 3),
            "specialty_avg_conversion": round(spec_avg, 3),
            "rank_in_specialty": rank,
        }

    def get_territory_benchmarks(self, territory: str) -> Dict[str, Any]:
        t = territory.strip().lower()
        territory_df = self.df[self.df["territory"] == t]
        if territory_df.empty:
            return None

        doctor_stats = []
        for did in territory_df["doctor_id"].unique():
            d = territory_df[territory_df["doctor_id"] == did]
            conv = (d["outcome"] == "positive").sum() / len(d)
            sales = int(d["sales_volume"].sum()) if "sales_volume" in d.columns else 0
            doctor_stats.append({
                "doctor_id": str(did),
                "conversion_rate": conv,
                "avg_interest": d["interest_level"].mean(),
                "total_interactions": len(d),
                "total_sales_volume": sales,
            })

        df_s = pd.DataFrame(doctor_stats)
        return {
            "territory_name": t,
            "total_doctors": len(doctor_stats),
            "avg_conversion_rate": round(df_s["conversion_rate"].mean(), 3),
            "avg_interest_level": round(df_s["avg_interest"].mean(), 2),
            "total_sales_volume": int(df_s["total_sales_volume"].sum()),
            "top_performer": {
                "doctor_id": str(df_s.loc[df_s["conversion_rate"].idxmax(), "doctor_id"]),
                "conversion_rate": round(df_s["conversion_rate"].max(), 3),
            },
            "percentiles": {
                "p25": round(df_s["conversion_rate"].quantile(0.25), 3),
                "p50": round(df_s["conversion_rate"].quantile(0.50), 3),
                "p75": round(df_s["conversion_rate"].quantile(0.75), 3),
            },
        }


# ─────────────────────────────────────────────
# OBJECTION RESOLUTION TRACKER
# ─────────────────────────────────────────────
class ObjectionResolutionTracker:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        if "objection" not in self.df.columns:
            self.df["objection"] = self.df["objection_type"].astype(str).str.strip().str.lower() \
                if "objection_type" in self.df.columns else "none"
        else:
            self.df["objection"] = self.df["objection"].astype(str).str.strip().str.lower()
        self.df["outcome"] = self.df["outcome"].astype(str).str.strip().str.lower()
        self.df["follow_up"] = self.df["follow_up"].astype(str).str.strip().str.lower()

    def get_objection_analysis(self, doctor_id: str) -> Dict[str, Any]:
        doctor_df = self.df[self.df["doctor_id"].astype(str).str.strip() == str(doctor_id).strip()]
        if doctor_df.empty:
            return None

        obj_df = doctor_df[
            doctor_df["objection"].notna()
            & (doctor_df["objection"] != "")
            & (doctor_df["objection"] != "nan")
            & (doctor_df["objection"] != "none")
        ]
        if obj_df.empty:
            return {"has_objections": False, "total_objections": 0}

        obj_types = obj_df["objection"].value_counts().to_dict()
        resolution_data = []
        for obj_type in obj_df["objection"].unique():
            o_df = obj_df[obj_df["objection"] == obj_type]
            overcome = (o_df["outcome"] == "positive").sum() / len(o_df)
            resolution_data.append({
                "objection_type": obj_type,
                "occurrence_count": len(o_df),
                "overcome_rate": round(overcome, 3),
                "resolution_difficulty": "easy" if overcome > 0.6 else ("moderate" if overcome > 0.3 else "difficult"),
            })

        resolution_data.sort(key=lambda x: x["occurrence_count"], reverse=True)
        persistent = [r for r in resolution_data if r["occurrence_count"] >= 2 and r["overcome_rate"] < 0.3]

        return {
            "has_objections": True,
            "total_objections": len(obj_df),
            "objection_breakdown": obj_types,
            "resolution_analysis": resolution_data,
            "persistent_objections": persistent,
            "overall_resolution_rate": round((obj_df["outcome"] == "positive").sum() / len(obj_df), 3),
        }


# ─────────────────────────────────────────────
# AIDA CLASSIFIER
# ─────────────────────────────────────────────
class AIDAClassifier:
    """
    Classifies doctor into AIDA stage based on behavioral signals.
    Stage logic:
      - action    : conversion >= 0.5
      - desire    : interest >= 4 AND follow_up_rate > 0.4
      - interest  : interest >= 3 AND conversion < 0.3
      - awareness : interactions < 3  OR  interest < 3
    """

    STAGES = ["awareness", "interest", "desire", "action"]
    STAGE_LABELS = {
        "awareness": "Awareness",
        "interest":  "Interest",
        "desire":    "Desire",
        "action":    "Action",
    }
    STAGE_COLORS = {
        "awareness": "#64748B",
        "interest":  "#F59E0B",
        "desire":    "#8B5CF6",
        "action":    "#10B981",
    }
    STAGE_EMOJI = {
        "awareness": "👁️",
        "interest":  "🔍",
        "desire":    "🔥",
        "action":    "✅",
    }

    def classify(self, doctor_df: pd.DataFrame) -> Dict[str, Any]:
        doctor_df = doctor_df.copy()
        doctor_df["outcome"] = doctor_df["outcome"].astype(str).str.strip().str.lower()
        doctor_df["follow_up"] = doctor_df["follow_up"].astype(str).str.strip().str.lower()

        interactions = len(doctor_df)
        conversion_rate = (doctor_df["outcome"] == "positive").sum() / max(interactions, 1)
        avg_interest = doctor_df["interest_level"].mean()
        follow_up_rate = (doctor_df["follow_up"] == "yes").sum() / max(interactions, 1)

        if conversion_rate >= 0.5:
            stage = "action"
            confidence = min(0.95, 0.5 + conversion_rate * 0.45)
        elif avg_interest >= 4 and follow_up_rate > 0.4:
            stage = "desire"
            confidence = min(0.9, 0.4 + avg_interest * 0.08 + follow_up_rate * 0.3)
        elif avg_interest >= 3 and conversion_rate < 0.3:
            stage = "interest"
            confidence = min(0.85, 0.3 + avg_interest * 0.1 + (1 - conversion_rate) * 0.2)
        else:
            stage = "awareness"
            confidence = min(0.8, 0.4 + (1 - min(avg_interest / 5, 1)) * 0.4)

        stage_idx = self.STAGES.index(stage)

        stage_guidance = {
            "awareness": {
                "what_to_say":   "Introduce your brand. Lead with a bold stat or unmet need. Keep it under 30 seconds.",
                "what_to_show":  "One-pager or product brief. Focus on awareness, not conversion.",
                "what_to_avoid": "Avoid pushing for prescriptions. Don't overwhelm with product range.",
                "next_step":     "Leave a visual aid. Book a follow-up for a deeper session.",
            },
            "interest": {
                "what_to_say":   "They're curious — feed it. Share clinical comparison data or differentiation vs. competitors.",
                "what_to_show":  "Clinical study summaries, comparison charts vs alternatives.",
                "what_to_avoid": "Avoid making it a one-way pitch. Invite questions.",
                "next_step":     "Ask if they'd like to trial a sample. Set a follow-up with your medical team.",
            },
            "desire": {
                "what_to_say":   "Reinforce conviction. Use patient success stories and peer endorsements.",
                "what_to_show":  "Patient case studies, KOL endorsements, formulary approvals.",
                "what_to_avoid": "Avoid discounting too early — they're already sold on value.",
                "next_step":     "Push for a trial prescription or limited batch. Get a verbal commitment.",
            },
            "action": {
                "what_to_say":   "Maintain momentum. Thank them, reinforce loyalty, upsell complementary products.",
                "what_to_show":  "Volume data, co-marketing opportunities, exclusive programs.",
                "what_to_avoid": "Don't re-sell what they already believe in. Avoid overloading with new products.",
                "next_step":     "Secure repeat orders. Introduce a second product from the portfolio.",
            },
        }

        return {
            "aida_stage": stage,
            "aida_stage_index": stage_idx,
            "aida_label": self.STAGE_LABELS[stage],
            "aida_color": self.STAGE_COLORS[stage],
            "aida_emoji": self.STAGE_EMOJI[stage],
            "aida_confidence": round(confidence, 2),
            "aida_signals": {
                "interactions": interactions,
                "conversion_rate": round(conversion_rate, 3),
                "avg_interest": round(avg_interest, 2),
                "follow_up_rate": round(follow_up_rate, 3),
            },
            "stage_guidance": stage_guidance[stage],
            "all_stages": self.STAGES,
            "stage_colors": self.STAGE_COLORS,
            "stage_labels": self.STAGE_LABELS,
            "stage_emojis": self.STAGE_EMOJI,
        }


# ─────────────────────────────────────────────
# INTENT SCORE ENGINE
# ─────────────────────────────────────────────
class IntentScoreEngine:
    """
    intent_score = 0.4*interest_norm + 0.3*follow_up_rate + 0.3*recent_activity
    """

    def compute(self, doctor_df: pd.DataFrame) -> Dict[str, Any]:
        doctor_df = doctor_df.copy()
        doctor_df["interaction_date"] = pd.to_datetime(doctor_df["interaction_date"], dayfirst=True, errors="coerce")
        doctor_df["follow_up"] = doctor_df["follow_up"].astype(str).str.strip().str.lower()

        avg_interest = doctor_df["interest_level"].mean()
        interest_norm = min(avg_interest / 5.0, 1.0)

        follow_up_rate = (doctor_df["follow_up"] == "yes").sum() / max(len(doctor_df), 1)

        latest = doctor_df["interaction_date"].max()
        recent = doctor_df[doctor_df["interaction_date"] >= (latest - timedelta(days=30))]
        recent_activity = len(recent) / max(len(doctor_df), 1)

        intent_score = (0.4 * interest_norm) + (0.3 * follow_up_rate) + (0.3 * recent_activity)
        intent_score = round(min(intent_score, 1.0), 3)

        if intent_score >= 0.65:
            intent_label = "High Intent"
            intent_color = "#10B981"
            pitch_aggression = "aggressive"
        elif intent_score >= 0.40:
            intent_label = "Moderate Intent"
            intent_color = "#F59E0B"
            pitch_aggression = "balanced"
        else:
            intent_label = "Low Intent"
            intent_color = "#EF4444"
            pitch_aggression = "soft"

        return {
            "intent_score": intent_score,
            "intent_label": intent_label,
            "intent_color": intent_color,
            "pitch_aggression": pitch_aggression,
            "components": {
                "interest_norm": round(interest_norm, 3),
                "follow_up_rate": round(follow_up_rate, 3),
                "recent_activity": round(recent_activity, 3),
            },
        }


# ─────────────────────────────────────────────
# DOCTOR PERSONA CLASSIFIER
# ─────────────────────────────────────────────
class PersonaClassifier:
    """
    Classifies doctors into: Analytical / Emotional / Fast Decision / Resistant
    """

    def classify(self, doctor_df: pd.DataFrame, doctor_info: Dict) -> Dict[str, Any]:
        doctor_df = doctor_df.copy()
        if "objection" not in doctor_df.columns:
            doctor_df["objection"] = doctor_df["objection_type"].astype(str).str.lower() \
                if "objection_type" in doctor_df.columns else "none"
        else:
            doctor_df["objection"] = doctor_df["objection"].astype(str).str.lower()
        doctor_df["outcome"] = doctor_df["outcome"].astype(str).str.lower()

        conversion = (doctor_df["outcome"] == "positive").sum() / max(len(doctor_df), 1)
        avg_interest = doctor_df["interest_level"].mean()
        publications = doctor_info.get("publications_count", 0)
        experience = doctor_info.get("experience_years", 0)
        has_persistent_obj = any(
            "not interested" in o or "price" in o or "already" in o
            for o in doctor_df["objection"]
        )
        follow_up = (doctor_df["follow_up"].astype(str).str.lower() == "yes").sum() / max(len(doctor_df), 1)

        scores = {
            "analytical":    publications * 0.4 + experience * 0.01 + (1 - conversion) * 0.3,
            "emotional":     follow_up * 0.5 + avg_interest / 5 * 0.3 + (1 - publications / 50) * 0.2,
            "fast_decision": conversion * 0.5 + avg_interest / 5 * 0.3 + (1 - experience / 40) * 0.2,
            "resistant":     (1 - conversion) * 0.4 + (1 - follow_up) * 0.3 + (0.3 if has_persistent_obj else 0),
        }

        persona = max(scores, key=scores.get)

        descriptions = {
            "analytical": {
                "label":       "🔬 Analytical",
                "description": "Evidence-driven. Responds to data, clinical trials, and peer publications.",
                "approach":    "Lead with clinical evidence. Bring published studies. Avoid emotional appeals.",
            },
            "emotional": {
                "label":       "❤️ Relationship-Driven",
                "description": "Values trust and relationship. Responds well to rapport and stories.",
                "approach":    "Build personal rapport first. Use patient success stories. Follow up consistently.",
            },
            "fast_decision": {
                "label":       "⚡ Fast Decision Maker",
                "description": "Decides quickly. Responds to clear value props and efficiency.",
                "approach":    "Get to the point fast. One clear CTA. No long pitches — respect their time.",
            },
            "resistant": {
                "label":       "🛡️ Resistant / Skeptical",
                "description": "Hard to convert. Has persistent objections or competitor loyalty.",
                "approach":    "Don't push hard. Plant seeds. Address objections with data, not pressure.",
            },
        }

        return {
            "persona": persona,
            **descriptions[persona],
        }


# ─────────────────────────────────────────────
# NEXT BEST ACTION ENGINE
# ─────────────────────────────────────────────
class NextBestActionEngine:
    def generate(self, aida_stage: str, intent_score: float, persona: str, top_product: str) -> Dict[str, Any]:
        stage_transitions = {
            "awareness": "Interest",
            "interest":  "Desire",
            "desire":    "Action",
            "action":    "Retention",
        }
        goal = f"Move from {aida_stage.capitalize()} → {stage_transitions[aida_stage]}"

        action_map = {
            ("awareness", "soft"):        ("Introduce brand briefly", "Leave a single visual aid or product brief"),
            ("awareness", "balanced"):    ("Share a compelling product stat", "End with 'May I share more next visit?'"),
            ("awareness", "aggressive"):  ("Pitch one key product now", "Book a detailed follow-up before leaving"),
            ("interest",  "soft"):        ("Share clinical comparison data", "Ask open-ended questions about their patients"),
            ("interest",  "balanced"):    ("Position against competitor", "Ask: 'Would you like to trial this?'"),
            ("interest",  "aggressive"):  ("Push for trial prescription", "Offer a sample and set a follow-up date"),
            ("desire",    "soft"):        ("Reinforce with patient case study", "Ask: 'When can we start?'"),
            ("desire",    "balanced"):    ("Use KOL endorsement", "Push for prescription commitment"),
            ("desire",    "aggressive"):  ("Close with urgency", "Ask for immediate prescription commitment"),
            ("action",    "soft"):        ("Thank and maintain rapport", "Check satisfaction with current product"),
            ("action",    "balanced"):    ("Upsell a complementary product", "Share volume achievement milestones"),
            ("action",    "aggressive"):  ("Introduce second portfolio product", "Secure repeat orders proactively"),
        }

        aggression = "balanced" if intent_score >= 0.4 else "soft"
        if intent_score >= 0.65:
            aggression = "aggressive"

        key = (aida_stage, aggression)
        action_text, cta_text = action_map.get(key, ("Engage naturally", "Build rapport"))

        return {
            "goal": goal,
            "action": action_text,
            "cta": cta_text,
            "product_focus": top_product,
        }


# ─────────────────────────────────────────────
# VISIT SUCCESS PREDICTOR
# ─────────────────────────────────────────────
class VisitSuccessPredictor:
    def predict(self, doctor_df: pd.DataFrame, aida_stage: str, intent_score: float) -> Dict[str, Any]:
        doctor_df = doctor_df.copy()
        doctor_df["outcome"] = doctor_df["outcome"].astype(str).str.lower()

        base_conversion = (doctor_df["outcome"] == "positive").sum() / max(len(doctor_df), 1)

        stage_weights = {"awareness": 0.15, "interest": 0.35, "desire": 0.60, "action": 0.80}
        stage_weight = stage_weights.get(aida_stage, 0.3)

        probability = (0.4 * base_conversion) + (0.35 * stage_weight) + (0.25 * intent_score)
        probability = round(min(probability, 0.99), 3)

        if probability >= 0.6:
            label, color, emoji = "High Chance", "#10B981", "🟢"
        elif probability >= 0.35:
            label, color, emoji = "Moderate", "#F59E0B", "🟡"
        else:
            label, color, emoji = "Low Chance", "#EF4444", "🔴"

        return {
            "probability": probability,
            "probability_pct": f"{probability * 100:.0f}%",
            "label": label,
            "color": color,
            "emoji": emoji,
        }


# ─────────────────────────────────────────────
# RECOMMENDATION ENGINE (3-Layer: AIDA → Score → Time)
# ─────────────────────────────────────────────
class RecommendationEngine:
    """
    3-Layer product selection:
      Layer 1 — AIDA filter (what products fit this stage)
      Layer 2 — Score ranking (plan §7.2 formula)
      Layer 3 — Time constraint (strict business rules)

    Product score formula (plan §7.2):
      score = 0.4 * conv_rate + 0.3 * (avg_interest/5) + 0.2 * follow_up_rate + 0.1 * recent_trend

    Doctor score formula (plan §7.1):
      score = 0.4*norm(patient_load,0,200) + 0.2*norm(publications,0,50)
            + 0.2*norm(social_media_reach,0,10000) + 0.2*conversion_rate
    """

    AIDA_PRODUCT_FILTER = {
        "awareness": ["high_interest_low_conversion", "potential"],
        "interest":  ["potential", "high_interest_low_conversion"],
        "desire":    ["high_performer", "potential"],
        "action":    ["high_performer"],
    }

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self._trend_engine = TrendAnalytics(df)

    def normalize(self, x, min_val, max_val):
        return (x - min_val) / (max_val - min_val + 1e-5)

    def _get_product_recent_trend(self, doctor_df: pd.DataFrame, product: str) -> float:
        """Compute recent_trend component for a product (plan §7.2)."""
        p_df = doctor_df[doctor_df["product_name"] == product].copy()
        p_df["interaction_date"] = pd.to_datetime(p_df["interaction_date"], dayfirst=True, errors="coerce")
        p_df["month"] = p_df["interaction_date"].dt.to_period("M")
        monthly = (
            p_df.groupby("month")
            .apply(lambda x: (x["outcome"] == "positive").sum() / len(x))
            .tolist()
        )
        return self._trend_engine.calc_recent_trend_score(monthly)

    # ── Layer 2: Score all products (plan §7.2 formula) ──
    def score_products(self, doctor_df: pd.DataFrame) -> Dict[str, Any]:
        results = []
        for product in doctor_df["product_name"].unique():
            p_df = doctor_df[doctor_df["product_name"] == product]
            conversion = (p_df["outcome"] == "positive").sum() / max(len(p_df), 1)
            interest = p_df["interest_level"].mean()
            follow_up = (p_df["follow_up"].astype(str).str.lower() == "yes").sum() / max(len(p_df), 1)
            recent_trend = self._get_product_recent_trend(doctor_df, product)

            # Plan §7.2 formula
            score = (
                0.4 * conversion
                + 0.3 * (interest / 5)
                + 0.2 * follow_up
                + 0.1 * recent_trend
            )
            score = round(max(score, 0.0), 3)

            # Sales volume summary for this product
            total_sales = int(p_df["sales_volume"].sum()) if "sales_volume" in p_df.columns else 0

            if interest >= 3.5 and conversion >= 0.5:
                category = "high_performer"
            elif interest >= 3.5:
                category = "high_interest_low_conversion"
            elif conversion < 0.3:
                category = "low_performer"
            else:
                category = "potential"

            results.append({
                "product_name":     product,
                "score":            score,
                "conversion_rate":  round(conversion, 3),
                "avg_interest":     round(interest, 2),
                "follow_up_rate":   round(follow_up, 3),
                "recent_trend":     recent_trend,
                "total_sales_volume": total_sales,
                "category":         category,
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        scores = np.array([r["score"] for r in results])
        mean, std = np.mean(scores), np.std(scores)

        recommended, secondary, low = [], [], []
        for r in results:
            if r["score"] >= mean:
                recommended.append(r)
            elif r["score"] >= (mean - std):
                secondary.append(r)
            else:
                low.append(r)

        return {
            "recommended":        recommended,
            "secondary":          secondary,
            "low_performers":     low,
            "all_products_ranked": results,
        }

    # ── Layer 1: AIDA filter ──
    def _aida_filter(self, products: List[Dict], aida_stage: str) -> List[Dict]:
        preferred_cats = self.AIDA_PRODUCT_FILTER.get(aida_stage, [])
        preferred = [p for p in products if p["category"] in preferred_cats]
        rest = [p for p in products if p["category"] not in preferred_cats]
        return preferred + rest

    # ── Layer 3: Time rules (strict) ──
    def _time_rules(self, time_sec: int) -> Dict[str, Any]:
        if time_sec <= 30:
            return {"primary_count": 1, "support_count": 0, "closing_count": 0, "reminder_count": 0, "mode": "ultra_short"}
        elif time_sec <= 60:
            return {"primary_count": 1, "support_count": 0, "closing_count": 0, "reminder_count": 1, "mode": "short"}
        elif time_sec <= 120:
            return {"primary_count": 3, "support_count": 1, "closing_count": 0, "reminder_count": 0, "mode": "medium"}
        else:
            return {"primary_count": 5, "support_count": 2, "closing_count": 1, "reminder_count": 0, "mode": "long"}

    # ── Event override ──
    def _apply_event_override(self, time_sec: int, doctor_df: pd.DataFrame) -> Dict[str, Any]:
        event_type = doctor_df["event_type"].iloc[0] if "event_type" in doctor_df.columns else None
        extra_time = doctor_df["event_extra_time_seconds"].iloc[0] if "event_extra_time_seconds" in doctor_df.columns else 0
        event_type = str(event_type).strip().lower() if event_type else None
        valid_events = {"birthday", "medical_day", "product_launch"}

        if event_type not in valid_events:
            return {"adjusted_time": time_sec, "event_active": False, "event_type": None}

        adjusted = time_sec + max(int(extra_time), 60)
        return {
            "adjusted_time": adjusted,
            "event_active": True,
            "event_type": event_type,
            "push_new_launches": event_type == "product_launch",
        }

    def _identify_new_products_global(self, days_threshold: int = 30) -> set:
        df = self.df.copy()
        latest = pd.to_datetime(df["interaction_date"], dayfirst=True, errors="coerce").max()
        first_seen = df.groupby("product_name")["interaction_date"].min().reset_index()
        first_seen["interaction_date"] = pd.to_datetime(first_seen["interaction_date"], dayfirst=True, errors="coerce")
        first_seen["is_new"] = (latest - first_seen["interaction_date"]).dt.days <= days_threshold
        return set(first_seen[first_seen["is_new"]]["product_name"])

    # ── Time allocation per product (plan §2.1) ──
    def _allocate_time_per_product(self, products: List[Dict], total_time: int) -> List[Dict]:
        """Proportional time allocation based on product score."""
        if not products:
            return products
        total_score = sum(p["score"] for p in products) or 1
        result = []
        for p in products:
            suggested_sec = max(10, int((p["score"] / total_score) * total_time))
            result.append({**p, "suggested_duration_sec": suggested_sec})
        return result

    # ── Main recommendation builder ──
    def build_recommendations(
        self,
        doctor_df: pd.DataFrame,
        scored_products: Dict,
        selected_time: int,
        aida_stage: str,
    ) -> Dict[str, Any]:
        event_data = self._apply_event_override(selected_time, doctor_df)
        effective_time = event_data["adjusted_time"]
        rules = self._time_rules(effective_time)

        all_ranked = scored_products["all_products_ranked"]

        # LAYER 1: AIDA reorder
        aida_ordered = self._aida_filter(all_ranked, aida_stage)

        # Handle product launch boost
        if event_data.get("push_new_launches"):
            new_products = self._identify_new_products_global()
            aida_ordered = sorted(aida_ordered, key=lambda x: 1 if x["product_name"] in new_products else 0, reverse=True)

        # LAYER 3: Time-based slicing
        primary   = aida_ordered[: rules["primary_count"]]
        support   = aida_ordered[rules["primary_count"]: rules["primary_count"] + rules["support_count"]]
        closing   = aida_ordered[rules["primary_count"] + rules["support_count"]: rules["primary_count"] + rules["support_count"] + rules["closing_count"]]
        remaining = aida_ordered[rules["primary_count"] + rules["support_count"] + rules["closing_count"]:]
        reminder  = remaining[: rules["reminder_count"]]

        # Time allocation per product group
        primary  = self._allocate_time_per_product(primary, effective_time)
        support  = self._allocate_time_per_product(support, max(effective_time // 3, 10))

        return {
            "primary_products":  primary,
            "support_products":  support,
            "closing_products":  closing,
            "reminder_items":    reminder,
            "mode":              rules["mode"],
            "effective_time":    effective_time,
            "event_active":      event_data["event_active"],
            "event_type":        event_data.get("event_type"),
            "total_pitched":     len(primary) + len(support) + len(closing),
        }

    # ── Doctor score (plan §7.1) ──
    def score_doctor(self, doctor_info: Dict) -> float:
        """
        doctor_score = 0.4*norm(patient_load,0,200) + 0.2*norm(publications,0,50)
                     + 0.2*norm(social_reach,0,10000) + 0.2*conversion_rate
        """
        patient_load = self.normalize(doctor_info.get("patient_load", 0), 0, 200)
        publications = self.normalize(doctor_info.get("publications_count", 0), 0, 50)
        social_reach = self.normalize(doctor_info.get("social_media_reach", 0), 0, 10000)
        conversion   = doctor_info.get("conversion_rate", 0)
        score = 0.4 * patient_load + 0.2 * publications + 0.2 * social_reach + 0.2 * conversion
        return round(score, 3)

    def classify_doctor_tier(self, score: float) -> str:
        if score >= 0.75:
            return "high"
        elif score >= 0.45:
            return "medium"
        return "low"


# ─────────────────────────────────────────────
# MAIN ENGINE
# ─────────────────────────────────────────────
class DoctorAnalyticsEnhanced:
    """
    Main analytics engine.
    employee_type is NOT auto-assigned — it must come from the UI.
    Supports: actual_time_seconds, sales_volume, quarter, year columns.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.df.columns = self.df.columns.str.strip().str.lower()
        self.df["territory"]   = self.df["territory"].astype(str).str.strip().str.lower()
        self.df["doctor_name"] = self.df["doctor_name"].astype(str).str.strip()
        self.df["doctor_id"]   = self.df["doctor_id"].astype(str).str.strip()
        self.df["interaction_date"] = pd.to_datetime(
            self.df["interaction_date"], dayfirst=True, errors="coerce"
        )

        # Normalise interest_level if still string
        if self.df["interest_level"].dtype == object:
            _map = {"low": 1, "medium": 3, "high": 5}
            self.df["interest_level"] = (
                self.df["interest_level"].astype(str).str.strip().str.lower()
                .map(_map).fillna(0)
            )

        # Normalise outcome
        self.df["outcome"] = self.df["outcome"].astype(str).str.strip().str.lower()
        outcome_map = {
            "positive": "positive", "converted": "positive", "success": "positive",
            "won": "positive", "yes": "positive",
            "negative": "negative", "lost": "negative", "no": "negative",
            "neutral": "neutral", "pending": "neutral",
        }
        self.df["outcome"] = self.df["outcome"].map(outcome_map).fillna("neutral")

        # Ensure numeric columns
        for col in ["actual_time_seconds", "sales_volume", "patient_load",
                    "experience_years", "publications_count", "social_media_reach"]:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce").fillna(0).astype(int)

        # Sub-engines
        self.reco_engine        = RecommendationEngine(self.df)
        self.trend_engine       = TrendAnalytics(self.df)
        self.competitive_engine = CompetitiveIntelligence(self.df)
        self.manager_engine     = ManagerComparison(self.df)
        self.objection_engine   = ObjectionResolutionTracker(self.df)
        self.aida_classifier    = AIDAClassifier()
        self.intent_engine      = IntentScoreEngine()
        self.persona_classifier = PersonaClassifier()
        self.nba_engine         = NextBestActionEngine()
        self.success_predictor  = VisitSuccessPredictor()

    # ── Time prediction (plan §7.3) ──
    def predict_available_time(self, doctor_id: str) -> int:
        """
        predicted_time = median(last_5_actual_times).
        If fewer than 5, use global average (120 sec).
        """
        if "actual_time_seconds" not in self.df.columns:
            return 120
        doc_df = self.df[self.df["doctor_id"] == str(doctor_id).strip()]
        times = doc_df["actual_time_seconds"].dropna().tolist()
        times = [t for t in times if t > 0]
        if len(times) >= 5:
            return int(np.median(times[-5:]))
        global_avg = self.df["actual_time_seconds"]
        global_avg = global_avg[global_avg > 0].mean()
        return int(global_avg) if not np.isnan(global_avg) else 120

    def get_doctor_summary(
        self,
        doctor_id: str,
        selected_time: int = 60,
        employee_type: str = "MR",
        auto_predict_time: bool = False,
    ) -> Optional[Dict[str, Any]]:
        doctor_df = self.df[self.df["doctor_id"].astype(str).str.strip() == str(doctor_id).strip()]
        if doctor_df.empty:
            return None

        doctor_df = doctor_df.copy()

        # Use time prediction if requested (plan §7.3)
        if auto_predict_time:
            selected_time = self.predict_available_time(doctor_id)

        # ── Basic info ──
        doctor_info = {
            "doctor_id":          str(doctor_id),
            "doctor_name":        doctor_df["doctor_name"].iloc[0],
            "territory":          doctor_df["territory"].iloc[0],
            "specialty":          doctor_df["specialty"].iloc[0],
            "patient_load":       int(doctor_df["patient_load"].iloc[0]),
            "experience_years":   int(doctor_df["experience_years"].iloc[0]),
            "publications_count": int(doctor_df["publications_count"].iloc[0]),
            "social_media_reach": int(doctor_df["social_media_reach"].iloc[0]),
            "conversion_rate":    round((doctor_df["outcome"] == "positive").sum() / max(len(doctor_df), 1), 3),
        }

        # ── Engagement ──
        engagement = {
            "total_interactions": int(doctor_df["interaction_id"].nunique()),
            "avg_interest_level": round(doctor_df["interest_level"].mean(), 2),
            "conversion_rate":    round((doctor_df["outcome"] == "positive").sum() / max(len(doctor_df), 1), 3),
            "follow_up_rate":     round((doctor_df["follow_up"].astype(str).str.lower() == "yes").sum() / max(len(doctor_df), 1), 3),
            "total_sales_volume": int(doctor_df["sales_volume"].sum()) if "sales_volume" in doctor_df.columns else 0,
            "avg_meeting_duration_sec": int(doctor_df["actual_time_seconds"].mean()) if "actual_time_seconds" in doctor_df.columns else None,
        }

        # ── AIDA ──
        aida = self.aida_classifier.classify(doctor_df)

        # ── Intent ──
        intent = self.intent_engine.compute(doctor_df)

        # ── Persona ──
        persona = self.persona_classifier.classify(doctor_df, doctor_info)

        # ── Doctor score / tier (plan §7.1) ──
        doctor_score = self.reco_engine.score_doctor(doctor_info)
        doctor_tier  = self.reco_engine.classify_doctor_tier(doctor_score)

        # ── Products ──
        scored_products = self.reco_engine.score_products(doctor_df)
        recommendations = self.reco_engine.build_recommendations(
            doctor_df=doctor_df,
            scored_products=scored_products,
            selected_time=selected_time,
            aida_stage=aida["aida_stage"],
        )

        # ── Top product for NBA ──
        all_ranked = scored_products["all_products_ranked"]
        top_product = all_ranked[0]["product_name"] if all_ranked else "—"

        # ── Next Best Action ──
        nba = self.nba_engine.generate(
            aida_stage=aida["aida_stage"],
            intent_score=intent["intent_score"],
            persona=persona["persona"],
            top_product=top_product,
        )

        # ── Visit Success ──
        visit_success = self.success_predictor.predict(
            doctor_df=doctor_df,
            aida_stage=aida["aida_stage"],
            intent_score=intent["intent_score"],
        )

        # ── Product performance ──
        product_list = []
        for p in doctor_df["product_name"].unique():
            p_df = doctor_df[doctor_df["product_name"] == p]
            product_list.append({
                "product_name":    p,
                "avg_interest":    round(p_df["interest_level"].mean(), 2),
                "conversion_rate": round((p_df["outcome"] == "positive").sum() / max(len(p_df), 1), 3),
                "times_presented": len(p_df),
                "total_sales_volume": int(p_df["sales_volume"].sum()) if "sales_volume" in p_df.columns else 0,
            })

        # ── Objection ──
        obj_col = "objection" if "objection" in doctor_df.columns else \
                  ("objection_type" if "objection_type" in doctor_df.columns else None)
        if obj_col:
            obj_series = doctor_df[obj_col].astype(str).str.strip().str.lower()
            valid_obj = obj_series[~obj_series.isin(["nan", "none", ""])]
            objection_analysis = {
                "total_objections":    int(valid_obj.notna().sum()),
                "objection_breakdown": valid_obj.value_counts().to_dict(),
                "has_objections":      bool(len(valid_obj) > 0),
            }
        else:
            objection_analysis = {"total_objections": 0, "objection_breakdown": {}, "has_objections": False}

        # ── Deep analytics ──
        trends               = self.trend_engine.get_doctor_trends(doctor_id)
        competitive          = self.competitive_engine.get_competitive_analysis(doctor_id)
        territory_comparison = self.manager_engine.compare_doctor_to_territory(doctor_id, doctor_info["territory"])
        specialty_comparison = self.manager_engine.get_specialty_comparison(doctor_id)
        objection_deep       = self.objection_engine.get_objection_analysis(doctor_id)

        return {
            # Core
            "doctor_info":        doctor_info,
            "engagement_metrics": engagement,
            "product_performance": {"product_breakdown": product_list},
            "objection_analysis": objection_analysis,
            "session_time":       selected_time,
            "predicted_time":     self.predict_available_time(doctor_id),

            # Behavioral intelligence
            "aida":          aida,
            "intent":        intent,
            "persona":       persona,
            "visit_success": visit_success,

            # Sales guidance
            "next_best_action": nba,
            "recommendations":  recommendations,

            # Doctor scoring (plan §7.1)
            "doctor_scoring": {
                "score": doctor_score,
                "tier":  doctor_tier,
            },
            "selected_employee_type": employee_type,

            # Deep analytics
            "trend_analytics":         trends,
            "competitive_intelligence": competitive,
            "territory_comparison":    territory_comparison,
            "specialty_comparison":    specialty_comparison,
            "objection_resolution":    objection_deep,
        }

    def get_territory_overview(self, territory: str) -> Optional[Dict[str, Any]]:
        benchmarks = self.manager_engine.get_territory_benchmarks(territory)
        if not benchmarks:
            return None
        return {
            "territory_benchmarks": benchmarks,
            "doctor_list": self._get_territory_doctor_list(territory),
        }

    def _get_territory_doctor_list(self, territory: str) -> List[Dict[str, Any]]:
        t = territory.strip().lower()
        territory_df = self.df[self.df["territory"] == t]
        doctors = []
        for did in territory_df["doctor_id"].unique():
            d = territory_df[territory_df["doctor_id"] == did]
            conv = (d["outcome"] == "positive").sum() / max(len(d), 1)
            doc_info = {
                "patient_load":      int(d["patient_load"].iloc[0]),
                "publications_count": int(d["publications_count"].iloc[0]),
                "social_media_reach": int(d["social_media_reach"].iloc[0]),
                "conversion_rate":   round(conv, 3),
            }
            doctors.append({
                "doctor_id":           str(did),
                "doctor_name":         d["doctor_name"].iloc[0],
                "specialty":           d["specialty"].iloc[0],
                "conversion_rate":     round(conv, 3),
                "avg_interest":        round(d["interest_level"].mean(), 2),
                "total_interactions":  len(d),
                "total_sales_volume":  int(d["sales_volume"].sum()) if "sales_volume" in d.columns else 0,
                "doctor_score":        self.reco_engine.score_doctor(doc_info),
            })
        return sorted(doctors, key=lambda x: x["conversion_rate"], reverse=True)