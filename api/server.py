"""
api/server.py
------------------
FastAPI server — PatGPT v6.0 (ML-backed)

New vs v5:
  - ML models + FeatureStore loaded at startup
  - set_model_registry() wired into analytics_engine after load
  - /health now includes "models" field with per-model status
  - POST /feedback/{interaction_id} — closed-loop outcome logging
  - GET  /analytics/doctor/{doctor_id}/churn_risk — new ML endpoint
  - GET  /analytics/employee/{employee_id}/skill  — new ML endpoint
  - Hot-reload endpoint POST /admin/reload_models
  - All existing routes 100% preserved and unmodified
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from init_services.init import initialize_services
from main.analytics_engine import DoctorAnalyticsEnhanced, set_model_registry
from main.llm_insights import LLMInsightsEngineEnhanced
from main.ml.feature_extractor import FeatureExtractor, FeatureStore
from main.ml.model_registry import ModelRegistry
from main.ml.ml_engines import (
    DoctorReviewEngine,
    EmployeeReportEngine,
    ProductPerformanceEngine,
    ProductRecommendationEngine,
)

# ── Background jobs ───────────────────────────────────────────────────────────
try:
    from jobs.scheduler import start_scheduler, stop_scheduler
    _SCHEDULER_AVAILABLE = True
except ImportError:
    _SCHEDULER_AVAILABLE = False
    print("[WARN] jobs.scheduler not found — background jobs disabled.")

# ── Directory setup ────────────────────────────────────────────────────────────
MODELS_DIR   = Path(os.getenv("MODELS_DIR",   "models/"))
DATA_DIR     = Path(os.getenv("DATA_DIR",     "data/"))
LOGS_DIR     = Path(os.getenv("LOGS_DIR",     "logs/"))
REPORTS_DIR  = Path(os.getenv("REPORTS_DIR",  "reports/"))
FEEDBACK_LOG = DATA_DIR / "feedback_log.jsonl"

for d in [MODELS_DIR, LOGS_DIR, REPORTS_DIR, DATA_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
app = FastAPI(title="PatGPT", version="6.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

REQUIRED_COLUMNS = {
    "doctor_id", "doctor_name", "territory", "specialty",
    "patient_load", "experience_years", "publications_count",
    "social_media_reach", "outcome", "interest_level",
    "follow_up", "interaction_id", "product_name",
    "interaction_date", "employee_type", "actual_time_seconds",
}


# ── Data loading ───────────────────────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "doctor_sales_dummy_data.csv")
    df.columns = df.columns.str.strip().str.lower()
    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]) or df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip()

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing columns: {missing}")

    if "area" in df.columns:
        df["territory"] = df["area"].str.lower()
    else:
        df["territory"] = df["territory"].str.lower()

    if "objection" not in df.columns:
        df["objection"] = df["objection_type"].astype(str).str.strip().str.lower() \
            if "objection_type" in df.columns else "none"

    df["doctor_id"]    = df["doctor_id"].str.strip()
    df["follow_up"]    = df["follow_up"].str.lower()
    df["specialty"]    = df["specialty"].str.lower()
    df["employee_type"] = df["employee_type"].str.lower()
    df["region"]       = df["region"].str.strip() if "region" in df.columns else df["territory"]

    outcome_map = {
        "positive": "positive", "converted": "positive", "success": "positive",
        "won": "positive",      "yes": "positive",
        "negative": "negative", "lost": "negative", "no": "negative",
        "neutral": "neutral",   "pending": "neutral",
    }
    df["outcome"] = df["outcome"].str.lower().map(outcome_map).fillna("neutral")

    if df["interest_level"].dtype == object:
        interest_map = {"low": 1, "medium": 3, "high": 5}
        df["interest_level"] = df["interest_level"].str.lower().map(interest_map).fillna(0)
    else:
        df["interest_level"] = pd.to_numeric(df["interest_level"], errors="coerce").fillna(0)

    for col in ["actual_time_seconds", "sales_volume", "patient_load",
                "experience_years", "publications_count", "social_media_reach"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    if "quarter" in df.columns:
        df["quarter"] = df["quarter"].str.upper()
    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce").fillna(0).astype(int)

    return df


# ── Startup ────────────────────────────────────────────────────────────────────
initialize_services()
df = load_data()

# Core analytics engines
analytics_engine       = DoctorAnalyticsEnhanced(df)
product_reco_engine    = ProductRecommendationEngine(df)
product_perf_engine    = ProductPerformanceEngine(df)
doctor_review_engine   = DoctorReviewEngine(df)
employee_report_engine = EmployeeReportEngine(df)

# LLM engine
try:
    llm_engine = LLMInsightsEngineEnhanced()
except Exception as _llm_err:
    llm_engine = None
    print(f"[WARN] LLM engine not initialised: {_llm_err}")

# ── ML model loading ───────────────────────────────────────────────────────────
models        = ModelRegistry.load_all(str(MODELS_DIR))
_extractor    = FeatureExtractor(df)
feature_store = FeatureStore(_extractor, max_size=500)

# Wire ML models into analytics_engine (replaces rule-based classifiers)
set_model_registry(models, feature_store)
print(f"[INFO] ML model status: {models.health_dict()}")

# ── Start background scheduler (weekly report + monthly retrain) ──────────────
if _SCHEDULER_AVAILABLE:
    start_scheduler()


# ── JSON serialisation helper ──────────────────────────────────────────────────
def clean_json(obj):
    if isinstance(obj, dict):
        return {str(k): clean_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean_json(v) for v in obj]
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass
    return obj


# ══════════════════════════════════════════════════════════════════════════════
# EXISTING ROUTES  (all preserved, zero changes)
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/")
def root():
    return {"status": "running", "version": "6.0.0"}


@app.get("/health")
def health():
    return {
        "status":      "healthy",
        "rows":        len(analytics_engine.df),
        "doctors":     analytics_engine.df["doctor_id"].nunique(),
        "products":    analytics_engine.df["product_name"].nunique(),
        "territories": analytics_engine.df["territory"].nunique(),
        # ── New: ML model status ──────────────────────────────────────────
        "models":      models.health_dict(),
    }


@app.get("/territories")
def get_territories():
    return {"territories": sorted(analytics_engine.df["territory"].dropna().unique().tolist())}


@app.get("/doctors")
def get_doctors(territory: Optional[str] = Query(None), specialty: Optional[str] = Query(None)):
    data = analytics_engine.df.copy()
    if territory:
        data = data[data["territory"] == territory.strip().lower()]
    if specialty:
        data = data[data["specialty"] == specialty.strip().lower()]
    return data[["doctor_id", "doctor_name", "territory", "specialty"]].drop_duplicates().to_dict(orient="records")


@app.get("/products")
def get_products():
    return {"products": sorted(analytics_engine.df["product_name"].dropna().unique().tolist())}


@app.get("/employees")
def get_employees(territory: Optional[str] = Query(None)):
    data = analytics_engine.df.copy()
    if territory:
        data = data[data["territory"] == territory.strip().lower()]
    cols     = ["employee_id", "employee_name", "employee_type", "territory"]
    existing = [c for c in cols if c in data.columns]
    return data[existing].drop_duplicates().to_dict(orient="records")


@app.get("/analytics/doctor/{doctor_id}")
def get_doctor(doctor_id: str, time_sec: int = Query(60, ge=1, le=3600), employee_type: str = Query("mr")):
    result = analytics_engine.get_doctor_summary(
        doctor_id=doctor_id, selected_time=time_sec, employee_type=employee_type.strip().lower()
    )
    if result is None:
        raise HTTPException(status_code=404, detail=f"Doctor '{doctor_id}' not found")
    return JSONResponse(content=clean_json(result))


@app.get("/analytics/territory/{territory}")
def get_territory(territory: str):
    result = analytics_engine.get_territory_overview(territory)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Territory '{territory}' not found")
    return JSONResponse(content=clean_json(result))


@app.get("/recommendation/preview/{doctor_id}")
def recommendation_preview(doctor_id: str, time_sec: int = Query(...), employee_type: str = Query("mr")):
    doctor_df = analytics_engine.df[analytics_engine.df["doctor_id"] == doctor_id]
    if doctor_df.empty:
        raise HTTPException(status_code=404, detail="Doctor not found")
    scored = analytics_engine.reco_engine.score_products(doctor_df)
    aida   = analytics_engine.aida_classifier.classify(doctor_df)
    reco   = analytics_engine.reco_engine.build_recommendations(
        doctor_df=doctor_df, scored_products=scored,
        selected_time=time_sec, aida_stage=aida["aida_stage"]
    )
    return {"mode": reco["mode"], "effective_time": reco["effective_time"],
            "primary_products": reco["primary_products"], "aida_stage": aida["aida_stage"]}


@app.get("/recommendations/product_suggestions")
def product_suggestions(doctor_id: str = Query(...), time_sec: int = Query(60), employee_type: str = Query("MR")):
    result = product_reco_engine.get_full_product_suggestions(doctor_id, time_sec, employee_type)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Doctor '{doctor_id}' not found")
    return JSONResponse(content=clean_json(result))


@app.get("/suggestion/customize/{doctor_id}")
def customize_suggestion(doctor_id: str, time_sec: int = Query(60), employee_type: str = Query("MR")):
    if llm_engine is None:
        raise HTTPException(status_code=503, detail="LLM engine not available")
    suggestion = product_reco_engine.generate_custom_suggestion(
        doctor_id, time_sec, employee_type, llm_client=llm_engine.client, deployment=llm_engine.deployment
    )
    return {"suggestion": suggestion}


@app.get("/analytics/product_performance")
def product_performance(
    product: Optional[str] = Query(None),
    territory: Optional[str] = Query(None),
    quarter:   Optional[str] = Query(None),
):
    if product:
        detail = product_perf_engine.get_product_detail(product_name=product, territory=territory, quarter=quarter)
        if not detail or "error" in detail:
            raise HTTPException(status_code=404, detail=detail.get("error", "Product not found"))
        quarterly = product_perf_engine.get_quarterly_table(product_name=product)
        detail["quarterly_table"] = quarterly.to_dict(orient="records") if not quarterly.empty else []
        return JSONResponse(content=clean_json(detail))

    summary_df  = product_perf_engine.get_overall_summary(territory=territory, quarter=quarter)
    quarterly_df = product_perf_engine.get_quarterly_table()
    return JSONResponse(content=clean_json({
        "summary":         summary_df.to_dict(orient="records")  if not summary_df.empty  else [],
        "quarterly_table": quarterly_df.to_dict(orient="records") if not quarterly_df.empty else [],
    }))


@app.get("/analytics/product_performance/detail/{product_name}")
def product_performance_detail(
    product_name: str,
    territory: Optional[str] = Query(None),
    quarter:   Optional[str] = Query(None),
):
    detail = product_perf_engine.get_product_detail(product_name=product_name, territory=territory, quarter=quarter)
    if not detail or "error" in detail:
        raise HTTPException(status_code=404, detail=detail.get("error", "Product not found"))
    quarterly = product_perf_engine.get_quarterly_table(product_name=product_name)
    detail["quarterly_table"] = quarterly.to_dict(orient="records") if not quarterly.empty else []
    return JSONResponse(content=clean_json(detail))


@app.get("/analytics/product_performance/summary")
def product_performance_summary(territory: Optional[str] = Query(None), quarter: Optional[str] = Query(None)):
    summary_df = product_perf_engine.get_overall_summary(territory=territory, quarter=quarter)
    return JSONResponse(content={"summary": summary_df.to_dict(orient="records") if not summary_df.empty else []})


@app.get("/analytics/region_breakdown")
def region_breakdown(product: Optional[str] = Query(None), quarter: Optional[str] = Query(None)):
    data = product_perf_engine.get_region_breakdown(product_name=product, quarter=quarter)
    return JSONResponse(content={"regional_breakdown": data})


@app.get("/analytics/trend_analysis")
def trend_analysis(territory: Optional[str] = Query(None), quarter: Optional[str] = Query(None)):
    data = product_perf_engine.get_trend_analysis(territory=territory, quarter=quarter)
    return JSONResponse(content=data)


@app.get("/analytics/region_product_matrix")
def region_product_matrix(quarter: Optional[str] = Query(None)):
    pivot = product_perf_engine.get_region_product_matrix(quarter)
    if pivot.empty:
        return JSONResponse(content={"matrix": [], "regions": [], "products": []})
    return JSONResponse(content={
        "matrix":   pivot.to_dict(orient="records"),
        "regions":  pivot["region"].tolist() if "region" in pivot.columns else [],
        "products": [col for col in pivot.columns if col != "region"],
    })


@app.get("/analytics/doctor_review")
def doctor_review(doctor_id: Optional[str] = Query(None), territory: Optional[str] = Query(None)):
    if doctor_id:
        analysis = doctor_review_engine.get_doctor_analysis(doctor_id)
        if "error" in analysis:
            raise HTTPException(status_code=404, detail=analysis["error"])
        return JSONResponse(content=clean_json(analysis))
    summary_df = doctor_review_engine.get_all_doctors_summary(territory=territory)
    return JSONResponse(content={"doctors": summary_df.to_dict(orient="records")})


@app.get("/analytics/doctor_overview")
def doctor_overview():
    return JSONResponse(content=clean_json(doctor_review_engine.get_doctor_overview_stats()))


@app.get("/analytics/employee_report")
def employee_report(employee_id: Optional[str] = Query(None), territory: Optional[str] = Query(None)):
    if employee_id:
        report = employee_report_engine.get_employee_report(employee_id)
        if "error" in report:
            raise HTTPException(status_code=404, detail=report["error"])
        return JSONResponse(content=clean_json(report))
    team_df = employee_report_engine.get_team_summary(territory=territory)
    return JSONResponse(content={"team": team_df.to_dict(orient="records")})


@app.get("/analytics/doctor/{doctor_id}/product_aida")
def get_product_aida(doctor_id: str, time_sec: int = Query(60), employee_type: str = Query("mr")):
    doctor_df = analytics_engine.df[analytics_engine.df["doctor_id"].astype(str).str.strip() == str(doctor_id).strip()]
    if doctor_df.empty:
        raise HTTPException(status_code=404, detail=f"Doctor '{doctor_id}' not found")
    scored       = analytics_engine.reco_engine.score_products(doctor_df)
    aida_overall = analytics_engine.aida_classifier.classify(doctor_df)
    product_aida_list = []
    for p in scored["all_products_ranked"][:3]:
        p_df = doctor_df[doctor_df["product_name"] == p["product_name"]]
        if p_df.empty:
            continue
        p_aida = analytics_engine.aida_classifier.classify(p_df)
        product_aida_list.append({
            "product_name":    p["product_name"],
            "aida_stage":      p_aida["aida_stage"],
            "aida_label":      p_aida["aida_label"],
            "aida_color":      p_aida["aida_color"],
            "aida_emoji":      p_aida["aida_emoji"],
            "aida_confidence": p_aida["aida_confidence"],
            "conversion_rate": p["conversion_rate"],
            "avg_interest":    p["avg_interest"],
        })
    return JSONResponse(content={
        "doctor_id":    doctor_id,
        "overall_aida": aida_overall["aida_label"],
        "product_aida": product_aida_list,
    })


# LLM routes (unchanged)
@app.get("/llm/doctor_insights/{doctor_id}")
def llm_doctor_insights(doctor_id: str, time_sec: int = Query(60), employee_type: str = Query("mr")):
    if llm_engine is None:
        raise HTTPException(status_code=503, detail="LLM engine not available")
    analytics_data = analytics_engine.get_doctor_summary(doctor_id, selected_time=time_sec, employee_type=employee_type.strip().lower())
    if analytics_data is None:
        raise HTTPException(status_code=404, detail=f"Doctor '{doctor_id}' not found")
    reco = analytics_data.get("recommendations", {})
    primary = reco.get("primary_products", [])
    analytics_data["recommendation_engine"] = {
        "top_products": [{"product_name": p.get("product_name", ""), "conversion_rate": p.get("conversion_rate", 0), "avg_interest": p.get("avg_interest", 0)} for p in primary],
        "doctor_score": analytics_data.get("doctor_scoring", {}).get("score", 0),
    }
    insights = llm_engine.generate_doctor_insights(analytics_data)
    return JSONResponse(content=clean_json(insights))


@app.get("/llm/meeting_playbook/{doctor_id}")
def meeting_playbook(doctor_id: str, time_sec: int = Query(60), employee_type: str = Query("mr")):
    if llm_engine is None:
        raise HTTPException(status_code=503, detail="LLM engine not available")
    summary = analytics_engine.get_doctor_summary(doctor_id, selected_time=time_sec, employee_type=employee_type.strip().lower())
    if summary is None:
        raise HTTPException(status_code=404, detail=f"Doctor '{doctor_id}' not found")
    playbook = llm_engine.generate_meeting_playbook(summary)
    return {"playbook": playbook}


@app.get("/llm/product_insight/{product_name}")
def llm_product_insight(product_name: str):
    if llm_engine is None:
        raise HTTPException(status_code=503, detail="LLM engine not available")
    detail = product_perf_engine.get_product_detail(product_name)
    if "error" in detail:
        raise HTTPException(status_code=404, detail=detail["error"])
    explanation = llm_engine.explain_product_underperformance(
        product_name=product_name, metrics=detail, objections=detail.get("objection_breakdown", {}),
    )
    return JSONResponse(content={"product_name": product_name, "explanation": explanation})


@app.get("/llm/product_ai_analysis/{product_name}")
def product_ai_analysis(
    product_name: str,
    territory: Optional[str] = Query(None),
    quarter:   Optional[str] = Query(None),
):
    if llm_engine is None:
        raise HTTPException(status_code=503, detail="LLM engine not available")
    detail = product_perf_engine.get_product_detail(product_name=product_name, territory=territory, quarter=quarter)
    if not detail or "error" in detail:
        raise HTTPException(status_code=404, detail=detail.get("error", "Product not found"))
    analysis = llm_engine.explain_product_underperformance(
        product_name=product_name, metrics=detail, objections=detail.get("objection_breakdown", {})
    )
    return {"product_name": product_name, "analysis": analysis}


# ══════════════════════════════════════════════════════════════════════════════
# NEW ROUTES  (ML-backed)
# ══════════════════════════════════════════════════════════════════════════════

# ── Churn risk for a single doctor ────────────────────────────────────────────
@app.get("/analytics/doctor/{doctor_id}/churn_risk")
def doctor_churn_risk(doctor_id: str):
    if models.churn is None:
        raise HTTPException(status_code=503, detail="Churn model not loaded")
    rfm  = models.churn.build_rfm(doctor_id, df)
    risk = models.churn.predict(rfm)
    return JSONResponse(content={"doctor_id": doctor_id, **risk})


# ── Employee skill decomposition ──────────────────────────────────────────────
@app.get("/analytics/employee/{employee_id}/skill")
def employee_skill(employee_id: str):
    skill = models.skill.predict(employee_id, df)
    return JSONResponse(content={"employee_id": employee_id, **skill})


# ── FEEDBACK ENDPOINT ─────────────────────────────────────────────────────────
class FeedbackBody(BaseModel):
    playbook_used:  str
    outcome:        str   # "positive" | "negative" | "neutral"
    interest_after: int
    notes:          str = ""


@app.post("/feedback/{interaction_id}")
def log_feedback(interaction_id: str, body: FeedbackBody):
    record = {
        "interaction_id": interaction_id,
        "playbook_used":  body.playbook_used,
        "outcome":        body.outcome,
        "interest_after": body.interest_after,
        "notes":          body.notes,
        "logged_at":      datetime.utcnow().isoformat(),
        "prompt_version": getattr(llm_engine, "PROMPT_VERSION", "unknown") if llm_engine else "unknown",
    }
    with open(FEEDBACK_LOG, "a") as f:
        f.write(json.dumps(record) + "\n")

    # Invalidate FeatureStore cache so next request recomputes features
    # (interaction_id is not always the doctor_id — best-effort lookup)
    try:
        row = df[df["interaction_id"].astype(str) == str(interaction_id)]
        if not row.empty:
            feature_store.invalidate(str(row["doctor_id"].iloc[0]))
    except Exception:
        pass

    return {"logged": True, "interaction_id": interaction_id}


# ── HOT-RELOAD (admin — call after monthly retraining) ────────────────────────
@app.post("/admin/reload_models")
def reload_models():
    models.hot_reload(str(MODELS_DIR))
    set_model_registry(models, feature_store)
    return {"reloaded": True, "status": models.health_dict()}


# ──────────────────────────────────────────────────────────────────────────────
# ── Graceful scheduler shutdown on process exit ───────────────────────────────
import atexit
if _SCHEDULER_AVAILABLE:
    atexit.register(stop_scheduler)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)