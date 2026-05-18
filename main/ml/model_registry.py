"""
main/ml/model_registry.py
--------------------------
PatGPT ML serving layer.

Loaded ONCE at server startup (api/server.py lifespan).
Exposes four clean inference wrappers consumed by analytics_engine
and llm_insights — replaces ALL rule-based scoring.

API response shapes are IDENTICAL to the old rule-based outputs
so the frontend is completely unaffected.

Usage in api/server.py:
    from main.ml.model_registry import ModelRegistry
    models = ModelRegistry.load_all("models/")
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger("patgpt.model_registry")

# ── AIDA constants ─────────────────────────────────────────────────────────────
AIDA_ORDINAL = {"awareness": 0, "interest": 1, "desire": 2, "action": 3}
AIDA_LABELS  = {v: k for k, v in AIDA_ORDINAL.items()}
AIDA_COLORS  = {
    "awareness": "#64748B", "interest": "#F59E0B",
    "desire":    "#8B5CF6", "action":   "#10B981",
}
AIDA_EMOJI = {
    "awareness": "👁️", "interest": "🔍", "desire": "🔥", "action": "✅",
}
AIDA_STAGE_LABELS = {s: s.capitalize() for s in AIDA_ORDINAL}

CHURN_TIERS = [
    (0.75, "critical"),
    (0.50, "high"),
    (0.25, "medium"),
    (0.0,  "low"),
]

PERSONA_KEYS = ["analytical", "emotional", "fast_decision", "resistant", "balanced"]
PERSONA_META = {
    "analytical":    {"label": "🔬 Analytical",          "description": "Evidence-driven. Responds to data and peer publications.",    "approach": "Lead with clinical evidence. Bring published studies."},
    "emotional":     {"label": "❤️ Relationship-Driven",  "description": "Values trust and relationship. Responds well to rapport.",    "approach": "Build personal rapport first. Use patient success stories."},
    "fast_decision": {"label": "⚡ Fast Decision Maker",  "description": "Decides quickly. Responds to clear value props.",             "approach": "Get to the point fast. One clear CTA. No long pitches."},
    "resistant":     {"label": "🛡️ Resistant / Skeptical","description": "Hard to convert. Has persistent objections.",                "approach": "Don't push hard. Plant seeds. Address objections with data."},
    "balanced":      {"label": "⚖️ Balanced",             "description": "Moderate engagement across all signals.",                    "approach": "Mix data and rapport based on the visit mood."},
}

SPECIALTY_PRIORS: Dict[str, float] = {
    "cardiologist": 0.42, "diabetologist": 0.38, "general physician": 0.35,
    "neurologist": 0.33, "oncologist": 0.30, "pulmonologist": 0.36,
    "dermatologist": 0.40, "pediatrician": 0.37, "__default__": 0.35,
}


# ─────────────────────────────────────────────────────────────────────────────
# 1. CONVERSION MODEL WRAPPER
# ─────────────────────────────────────────────────────────────────────────────

class ConversionModel:
    """P(outcome == positive) — XGBoost/LGBM pipeline."""

    def __init__(self, pipeline, feature_names: List[str]):
        self.pipeline      = pipeline
        self.feature_names = feature_names

    def predict(self, feature_dict: Dict[str, Any]) -> float:
        """Returns probability 0–1. Inference < 50ms."""
        row = self._to_df(feature_dict)
        return round(float(self.pipeline.predict_proba(row)[:, 1][0]), 4)

    def predict_with_shap(self, feature_dict: Dict[str, Any]) -> Tuple[float, List[str]]:
        """Returns (probability, top-3 driving feature names)."""
        prob = self.predict(feature_dict)
        row  = self._to_df(feature_dict)
        clf  = self.pipeline.named_steps["clf"]
        try:
            import shap
            exp = shap.TreeExplainer(clf)
            sv  = exp.shap_values(row)
            top3 = [self.feature_names[i] for i in np.argsort(np.abs(sv[0]))[::-1][:3]]
        except Exception:
            if hasattr(clf, "feature_importances_"):
                top3 = [self.feature_names[i] for i in np.argsort(clf.feature_importances_)[::-1][:3]]
            else:
                top3 = []
        return prob, top3

    def _to_df(self, fd: Dict[str, Any]) -> pd.DataFrame:
        return pd.DataFrame([{k: fd.get(k, 0) for k in self.feature_names}])[self.feature_names]


# ─────────────────────────────────────────────────────────────────────────────
# 2. AIDA MODEL WRAPPER
# ─────────────────────────────────────────────────────────────────────────────

class AIDAModel:
    """LightGBM multi-class + CalibratedClassifierCV (Platt scaling)."""

    def __init__(self, pipeline, feature_names: List[str], label_map: Dict[int, str]):
        self.pipeline      = pipeline
        self.feature_names = feature_names
        self.label_map     = label_map

    def predict(self, feature_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Returns dict identical in shape to AIDAClassifier.classify()."""
        n = int(feature_dict.get("n_interactions_6m", 0))

        # Cold-start: < 3 interactions → Bayesian prior
        if n < 3:
            return self._cold_start(feature_dict, n)

        row   = pd.DataFrame([{k: feature_dict.get(k, 0) for k in self.feature_names}])[self.feature_names]
        proba = self.pipeline.predict_proba(row)[0]
        idx   = int(np.argmax(proba))
        stage = self.label_map.get(idx, "awareness")
        return self._format(stage, float(proba[idx]), n, feature_dict)

    def _cold_start(self, fd: Dict[str, Any], n: int) -> Dict[str, Any]:
        conv     = float(fd.get("avg_conv_rate_6m", 0.35))
        interest = float(fd.get("avg_interest_6m", 1.5))
        stage = ("action" if conv >= 0.5 else "desire" if interest >= 4
                 else "interest" if interest >= 3 else "awareness")
        return self._format(stage, 0.55 if n >= 1 else 0.45, n, fd)

    def _format(self, stage: str, conf: float, n: int, fd: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "aida_stage":       stage,
            "aida_stage_index": AIDA_ORDINAL.get(stage, 0),
            "aida_label":       AIDA_STAGE_LABELS.get(stage, stage.capitalize()),
            "aida_color":       AIDA_COLORS.get(stage, "#64748B"),
            "aida_emoji":       AIDA_EMOJI.get(stage, "👁️"),
            "aida_confidence":  round(conf, 3),
            "aida_signals": {
                "interactions":    n,
                "conversion_rate": round(float(fd.get("avg_conv_rate_6m", 0)), 3),
                "avg_interest":    round(float(fd.get("avg_interest_6m", 0)), 2),
                "follow_up_rate":  round(float(fd.get("follow_up_rate_6m", 0)), 3),
            },
            "stage_guidance":  self._guidance(stage),
            "all_stages":      list(AIDA_ORDINAL.keys()),
            "stage_colors":    AIDA_COLORS,
            "stage_labels":    AIDA_STAGE_LABELS,
            "stage_emojis":    AIDA_EMOJI,
        }

    @staticmethod
    def _guidance(stage: str) -> Dict[str, str]:
        G = {
            "awareness": {"what_to_say": "Introduce your brand. Lead with a bold stat or unmet need.", "what_to_show": "One-pager or product brief.", "what_to_avoid": "Avoid pushing for prescriptions.", "next_step": "Leave a visual aid. Book a follow-up."},
            "interest":  {"what_to_say": "Share clinical comparison data.", "what_to_show": "Clinical study summaries, comparison charts.", "what_to_avoid": "Avoid making it a one-way pitch.", "next_step": "Offer a trial sample."},
            "desire":    {"what_to_say": "Reinforce conviction with patient success stories.", "what_to_show": "Patient case studies, KOL endorsements.", "what_to_avoid": "Avoid discounting too early.", "next_step": "Push for a trial prescription."},
            "action":    {"what_to_say": "Maintain momentum, upsell complementary products.", "what_to_show": "Volume data, exclusive programs.", "what_to_avoid": "Don't re-sell what they already believe in.", "next_step": "Secure repeat orders."},
        }
        return G.get(stage, G["awareness"])


# ─────────────────────────────────────────────────────────────────────────────
# 3. PERSONA MODEL WRAPPER
# ─────────────────────────────────────────────────────────────────────────────

class PersonaModel:
    """KMeans k=5 clustering on behavioral vectors."""

    def __init__(self, kmeans, scaler, feature_cols: List[str],
                 cluster_meta: List[Dict], assignments: Dict[str, int],
                 assignment_month: str):
        self.kmeans           = kmeans
        self.scaler           = scaler
        self.feature_cols     = feature_cols
        self.cluster_meta     = cluster_meta
        self.assignments      = assignments        # {doctor_id: cluster_id}
        self.assignment_month = assignment_month

    def predict(self, doctor_id: str, doctor_df: pd.DataFrame) -> Dict[str, Any]:
        """Returns dict identical in shape to PersonaClassifier.classify(), plus persona_drift."""
        doc_id = str(doctor_id).strip()
        if doctor_df.empty:
            return self._fallback()

        row = self._build_row(doctor_df)
        X_scaled   = self.scaler.transform([row])
        cluster_id = int(self.kmeans.predict(X_scaled)[0])
        meta       = self.cluster_meta[cluster_id] if cluster_id < len(self.cluster_meta) else {}
        persona_key = PERSONA_KEYS[cluster_id % len(PERSONA_KEYS)]
        info        = PERSONA_META.get(persona_key, PERSONA_META["balanced"])

        prev        = self.assignments.get(doc_id)
        drift       = (prev is not None) and (prev != cluster_id)

        return {
            "persona":       persona_key,
            "cluster_id":    cluster_id,
            "persona_drift": drift,
            "top_features":  meta.get("top_features", []),
            **info,
        }

    def _build_row(self, doc_df: pd.DataFrame) -> List[float]:
        doc_df = doc_df.copy()
        doc_df["outcome"]   = doc_df["outcome"].astype(str).str.lower()
        doc_df["follow_up"] = doc_df["follow_up"].astype(str).str.lower()
        static = doc_df.iloc[0]

        # Build the full feature map — superset of any feature_cols combination
        pubs_norm  = min(float(static.get("publications_count", 0)) / 50.0, 1.0)
        soc_norm   = min(float(static.get("social_media_reach", 0)) / 10000.0, 1.0)
        fu_rate    = float((doc_df["follow_up"] == "yes").sum() / max(len(doc_df), 1))
        conv_rate  = float((doc_df["outcome"] == "positive").sum() / max(len(doc_df), 1))
        avg_int    = float(doc_df["interest_level"].astype(float).mean()) / 5.0
        dur_norm   = min(float(doc_df["actual_time_seconds"].mean()) / 300.0, 1.0) \
                     if "actual_time_seconds" in doc_df.columns else 0.0

        obj_col    = "objection" if "objection" in doc_df.columns else (
            "objection_type" if "objection_type" in doc_df.columns else None)
        obj_rate   = 0.0
        if obj_col:
            s = doc_df[obj_col].astype(str).str.lower()
            obj_rate = len(s[~s.isin(["nan", "none", ""])]) / max(len(doc_df), 1)
        int_vol    = float(doc_df["interest_level"].astype(float).std()) / 5.0 if len(doc_df) > 1 else 0.0

        # Fully dynamic lookup — works for any feature_cols saved in pkl
        full_map = {
            "publications_norm":     pubs_norm,
            "social_reach_norm":     soc_norm,
            "follow_up_rate":        fu_rate,
            "follow_up_accept_rate": fu_rate,   # alias — old pkl compat
            "conversion_rate":       conv_rate,
            "avg_interest_norm":     avg_int,
            "avg_meeting_duration":  dur_norm,
            "objection_rate":        obj_rate,
            "interest_volatility":   int_vol,
        }
        return [full_map.get(c, 0.0) for c in self.feature_cols]

    def _fallback(self) -> Dict[str, Any]:
        return {"persona": "analytical", "cluster_id": -1, "persona_drift": False,
                "top_features": [], **PERSONA_META["analytical"]}


# ─────────────────────────────────────────────────────────────────────────────
# 4. CHURN MODEL WRAPPER
# ─────────────────────────────────────────────────────────────────────────────

class ChurnModel:
    """Logistic Regression churn/disengagement risk (60-day horizon)."""

    def __init__(self, pipeline, feature_cols: List[str]):
        self.pipeline     = pipeline
        self.feature_cols = feature_cols

    def predict(self, rfm: Dict[str, Any]) -> Dict[str, Any]:
        """Returns {risk_score, risk_tier, components}."""
        row  = pd.DataFrame([{k: rfm.get(k, 0) for k in self.feature_cols}])
        prob = float(self.pipeline.predict_proba(row)[:, 1][0])
        tier = next((t for thr, t in CHURN_TIERS if prob >= thr), "low")
        return {"risk_score": round(prob, 4), "risk_tier": tier,
                "components": {k: rfm.get(k, 0) for k in self.feature_cols}}

    def build_rfm(self, doctor_id: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Extract RFM features for a doctor from the interaction DataFrame."""
        doc_id  = str(doctor_id).strip()
        doc_df  = df[df["doctor_id"].astype(str).str.strip() == doc_id].copy()
        if doc_df.empty:
            return {k: 0 for k in self.feature_cols}

        doc_df["interaction_date"] = pd.to_datetime(doc_df["interaction_date"], dayfirst=True, errors="coerce")
        doc_df["outcome"]   = doc_df["outcome"].astype(str).str.lower()
        doc_df["follow_up"] = doc_df["follow_up"].astype(str).str.lower()

        dataset_max = df["interaction_date"].max() if not df.empty else pd.Timestamp.now()
        latest      = doc_df["interaction_date"].max()
        recency     = int((dataset_max - latest).days) if pd.notna(latest) else 999
        freq        = len(doc_df)
        monetary    = float(doc_df["sales_volume"].sum()) if "sales_volume" in doc_df.columns else 0.0
        avg_int     = float(doc_df["interest_level"].mean())
        conv        = float((doc_df["outcome"] == "positive").sum() / max(freq, 1))

        def slope(series: pd.Series) -> float:
            if len(series) < 2: return 0.0
            x = np.arange(len(series), dtype=float)
            y = series.values.astype(float)
            mask = ~np.isnan(y)
            return float(np.polyfit(x[mask], y[mask], 1)[0]) if mask.sum() >= 2 else 0.0

        doc_df["month"] = doc_df["interaction_date"].dt.to_period("M")
        int_slope = slope(doc_df.groupby("month")["interest_level"].mean())
        fu_slope  = slope(doc_df.groupby("month").apply(lambda g: (g["follow_up"] == "yes").sum() / max(len(g), 1)))

        return {
            "recency_days":      recency,
            "frequency":         freq,
            "monetary":          monetary,
            "avg_interest":      avg_int,
            "interest_trend":    int_slope,
            "follow_up_decline": -fu_slope,
            "conv_rate":         conv,
        }


# ─────────────────────────────────────────────────────────────────────────────
# 5. EMPLOYEE SKILL WRAPPER
# ─────────────────────────────────────────────────────────────────────────────

class EmployeeSkillModel:
    """
    Percentile-rank based skill decomposition — no ML model file needed.
    Computed at serve time from the full DataFrame.

    Skill dimensions (PDF §5):
      opening_score  — early interest lift vs territory peers
      handling_score — objection resolution rate vs territory peers
      closing_score  — desire-to-action conversion rate vs territory peers
    """

    def predict(self, employee_id: str, df: pd.DataFrame) -> Dict[str, Any]:
        emp_id = str(employee_id).strip()
        emp_df = df[df["employee_id"].astype(str).str.strip() == emp_id].copy()
        if emp_df.empty:
            return {"skill_vector": [0.0, 0.0, 0.0], "coaching_priority": True,
                    "opening_score": 0.0, "handling_score": 0.0, "closing_score": 0.0}

        territory = str(emp_df["territory"].iloc[0]).strip().lower()
        terr_df   = df[df["territory"].astype(str).str.strip().str.lower() == territory]

        # Compute per-employee raw scores across all employees in territory
        emp_raw   = self._raw_scores_for_all(terr_df)
        my_raw    = emp_raw.get(emp_id, {"opening": 0.0, "handling": 0.0, "closing": 0.0})

        all_open    = [v["opening"]  for v in emp_raw.values()]
        all_handle  = [v["handling"] for v in emp_raw.values()]
        all_close   = [v["closing"]  for v in emp_raw.values()]

        open_pct    = self._percentile_rank(my_raw["opening"],  all_open)
        handle_pct  = self._percentile_rank(my_raw["handling"], all_handle)
        close_pct   = self._percentile_rank(my_raw["closing"],  all_close)

        avg         = (open_pct + handle_pct + close_pct) / 3.0
        coaching    = avg < 0.4   # bottom 40% overall → flag for coaching

        return {
            "opening_score":    round(open_pct,   3),
            "handling_score":   round(handle_pct, 3),
            "closing_score":    round(close_pct,  3),
            "skill_vector":     [round(open_pct, 3), round(handle_pct, 3), round(close_pct, 3)],
            "coaching_priority": coaching,
        }

    def _raw_scores_for_all(self, terr_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        result = {}
        terr_df = terr_df.copy()
        terr_df["outcome"]   = terr_df["outcome"].astype(str).str.lower()
        terr_df["follow_up"] = terr_df["follow_up"].astype(str).str.lower()
        if "objection" not in terr_df.columns:
            obj_col = "objection_type" if "objection_type" in terr_df.columns else None
            terr_df["objection"] = terr_df[obj_col].astype(str).str.lower() if obj_col else "none"
        else:
            terr_df["objection"] = terr_df["objection"].astype(str).str.lower()

        for eid, grp in terr_df.groupby("employee_id"):
            grp = grp.copy()
            # opening_score: avg interest in FIRST visit per doctor
            first_visits = grp.sort_values("interaction_date").groupby("doctor_id").first()
            opening = float(first_visits["interest_level"].mean()) / 5.0 if not first_visits.empty else 0.0

            # handling_score: rate at which objections end in positive outcome
            obj_df = grp[~grp["objection"].isin(["nan", "none", ""])]
            handling = float((obj_df["outcome"] == "positive").sum() / max(len(obj_df), 1)) if not obj_df.empty else 0.0

            # closing_score: positive outcome rate among desire/action-stage rows (high interest)
            hi_df  = grp[grp["interest_level"] >= 4]
            closing = float((hi_df["outcome"] == "positive").sum() / max(len(hi_df), 1)) if not hi_df.empty else 0.0

            result[str(eid).strip()] = {"opening": opening, "handling": handling, "closing": closing}
        return result

    @staticmethod
    def _percentile_rank(value: float, population: List[float]) -> float:
        if not population:
            return 0.5
        below = sum(1 for v in population if v < value)
        return round(below / len(population), 4)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN REGISTRY
# ─────────────────────────────────────────────────────────────────────────────

class ModelRegistry:
    """
    Loads all models from disk and exposes typed attributes.

    Attributes
    ----------
    conversion   : ConversionModel | None
    aida         : AIDAModel       | None
    persona      : PersonaModel    | None
    churn        : ChurnModel      | None
    skill        : EmployeeSkillModel  (always available — no pkl needed)
    status       : Dict[str, str]  — for /health endpoint
    """

    EXPECTED = ["conversion_model", "aida_classifier", "persona_model", "churn_model"]

    def __init__(self):
        self.conversion : Optional[ConversionModel]    = None
        self.aida       : Optional[AIDAModel]           = None
        self.persona    : Optional[PersonaModel]        = None
        self.churn      : Optional[ChurnModel]          = None
        self.skill      : EmployeeSkillModel             = EmployeeSkillModel()
        self.status     : Dict[str, str]                 = {m: "missing" for m in self.EXPECTED}

    @classmethod
    def load_all(cls, models_dir: str = "models/") -> "ModelRegistry":
        reg  = cls()
        base = Path(models_dir)

        # Conversion
        p = base / "conversion_model.pkl"
        if p.exists():
            try:
                pipeline = joblib.load(p)
                feat_names = cls._feat_names(base, "conversion")
                reg.conversion = ConversionModel(pipeline, feat_names)
                reg.status["conversion_model"] = "ok"
                logger.info("✅ conversion_model loaded")
            except Exception as e:
                logger.error(f"❌ conversion_model: {e}")
        else:
            logger.warning("⚠  conversion_model.pkl not found — rule-based fallback active")

        # AIDA
        p = base / "aida_classifier.pkl"
        if p.exists():
            try:
                b = joblib.load(p)
                reg.aida = AIDAModel(
                    pipeline=b["pipeline"],
                    feature_names=b["feature_names"],
                    label_map={int(k): v for k, v in b["label_map"].items()},
                )
                reg.status["aida_classifier"] = "ok"
                logger.info("✅ aida_classifier loaded")
            except Exception as e:
                logger.error(f"❌ aida_classifier: {e}")
        else:
            logger.warning("⚠  aida_classifier.pkl not found — rule-based fallback active")

        # Persona
        p = base / "persona_model.pkl"
        if p.exists():
            try:
                b = joblib.load(p)
                reg.persona = PersonaModel(
                    kmeans=b["kmeans"],
                    scaler=b["scaler"],
                    feature_cols=b["feature_cols"],
                    cluster_meta=b["cluster_meta"],
                    assignments=b.get("assignments", {}),
                    assignment_month=b.get("assignment_month", ""),
                )
                reg.status["persona_model"] = "ok"
                logger.info("✅ persona_model loaded")
            except Exception as e:
                logger.error(f"❌ persona_model: {e}")
        else:
            logger.warning("⚠  persona_model.pkl not found — rule-based fallback active")

        # Churn
        p = base / "churn_model.pkl"
        if p.exists():
            try:
                b = joblib.load(p)
                reg.churn = ChurnModel(pipeline=b["pipeline"], feature_cols=b["feature_cols"])
                reg.status["churn_model"] = "ok"
                logger.info("✅ churn_model loaded")
            except Exception as e:
                logger.error(f"❌ churn_model: {e}")
        else:
            logger.warning("⚠  churn_model.pkl not found")

        return reg

    def hot_reload(self, models_dir: str = "models/") -> None:
        """Hot-reload all models without restarting the server."""
        logger.info("🔄 Hot-reloading models...")
        fresh = ModelRegistry.load_all(models_dir)
        self.conversion = fresh.conversion
        self.aida       = fresh.aida
        self.persona    = fresh.persona
        self.churn      = fresh.churn
        self.status     = fresh.status
        logger.info("✅ Hot-reload complete.")

    def health_dict(self) -> Dict[str, str]:
        return dict(self.status)

    @staticmethod
    def _feat_names(base: Path, key: str) -> List[str]:
        try:
            with open(base / "training_manifest.json") as f:
                return json.load(f).get("models", {}).get(key, {}).get("feature_names", [])
        except Exception:
            return []