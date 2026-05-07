"""
server.py
------------------
FastAPI server — AI Sales Assistant Edition (v5.0)
- Robust CSV whitespace stripping on load
- /analytics/doctor/{doctor_id}
- /recommendation/preview/{doctor_id}
- /recommendations/product_suggestions
- /analytics/product_performance
- /analytics/doctor_review
- /analytics/employee_report
- /llm/doctor_insights (new — calls LLMInsightsEngineEnhanced)
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from typing import Optional

from main.analytics_engine import DoctorAnalyticsEnhanced
from main.ml_engines import (
    ProductRecommendationEngine,
    ProductPerformanceEngine,
    DoctorReviewEngine,
    EmployeeReportEngine,
)
from init_services.init import initialize_services
from main.llm_insights import LLMInsightsEngineEnhanced


app = FastAPI(title="PatGPT — AI Sales Assistant", version="5.0.0")

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

VALID_EMPLOYEE_TYPES = {"mr", "area manager", "vp", "gm", "general manager"}


# ─────────────────────────────────────────────
# DATA LOADER — strips ALL column & value whitespace
# ─────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    df = pd.read_csv("data/doctor_sales_dummy_data.csv")

    # Strip whitespace from ALL column names
    df.columns = df.columns.str.strip().str.lower()

    # Strip whitespace from ALL string values across the entire DataFrame
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip()

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing columns after normalisation: {missing}")

    # Territory alias (keep both 'territory' and 'area' working)
    if "area" in df.columns:
        df["territory"] = df["area"].str.lower()
    else:
        df["territory"] = df["territory"].str.lower()

    # Objection column alias — CSV uses 'objection_type'; normalise to 'objection' everywhere
    if "objection" not in df.columns:
        if "objection_type" in df.columns:
            df["objection"] = df["objection_type"].astype(str).str.strip().str.lower()
        else:
            df["objection"] = "none"  # graceful fallback if column is entirely absent

    df["doctor_id"]     = df["doctor_id"].str.strip()
    df["follow_up"]     = df["follow_up"].str.lower()
    df["specialty"]     = df["specialty"].str.lower()
    df["employee_type"] = df["employee_type"].str.lower()
    df["region"]        = df["region"].str.strip() if "region" in df.columns else df["territory"]

    # Outcome normalisation
    outcome_map = {
        "positive": "positive", "converted": "positive", "success": "positive",
        "won": "positive", "yes": "positive",
        "negative": "negative", "lost": "negative", "no": "negative",
        "neutral": "neutral", "pending": "neutral",
    }
    df["outcome"] = df["outcome"].str.lower().map(outcome_map).fillna("neutral")

    # Interest level → numeric (handles both string and float input)
    if df["interest_level"].dtype == object:
        interest_map = {"low": 1, "medium": 3, "high": 5}
        df["interest_level"] = (
            df["interest_level"].str.lower().map(interest_map).fillna(0)
        )
    else:
        df["interest_level"] = pd.to_numeric(df["interest_level"], errors="coerce").fillna(0)

    # Numeric columns
    for col in ["actual_time_seconds", "sales_volume", "patient_load",
                "experience_years", "publications_count", "social_media_reach"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    # Quarter / year
    if "quarter" in df.columns:
        df["quarter"] = df["quarter"].str.upper()
    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce").fillna(0).astype(int)

    return df

# Initialize Azure OpenAI FIRST
initialize_services()

df = load_data()
analytics_engine        = DoctorAnalyticsEnhanced(df)
product_reco_engine     = ProductRecommendationEngine(df)
product_perf_engine     = ProductPerformanceEngine(df)
doctor_review_engine    = DoctorReviewEngine(df)
employee_report_engine  = EmployeeReportEngine(df)

try:
    llm_engine = LLMInsightsEngineEnhanced()
except Exception as _llm_err:
    llm_engine = None
    print(f"[WARN] LLM engine not initialised: {_llm_err}")


# ─────────────────────────────────────────────
# ROOT / HEALTH
# ─────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "running", "version": "5.0.0", "mode": "AI Sales Assistant"}


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "rows": len(analytics_engine.df),
        "doctors": analytics_engine.df["doctor_id"].nunique(),
        "products": analytics_engine.df["product_name"].nunique(),
        "territories": analytics_engine.df["territory"].nunique(),
    }


# ─────────────────────────────────────────────
# LOOKUP ENDPOINTS
# ─────────────────────────────────────────────
@app.get("/territories")
def get_territories():
    territories = analytics_engine.df["territory"].dropna().unique().tolist()
    return {"territories": sorted(territories)}


@app.get("/doctors")
def get_doctors(
    territory: Optional[str] = Query(None),
    specialty: Optional[str] = Query(None),
):
    data = analytics_engine.df.copy()
    if territory:
        data = data[data["territory"] == territory.strip().lower()]
    if specialty:
        data = data[data["specialty"] == specialty.strip().lower()]

    result = (
        data[["doctor_id", "doctor_name", "territory", "specialty"]]
        .drop_duplicates()
    )
    return result.to_dict(orient="records")


@app.get("/products")
def get_products():
    products = analytics_engine.df["product_name"].dropna().unique().tolist()
    return {"products": sorted(products)}


@app.get("/employees")
def get_employees(territory: Optional[str] = Query(None)):
    data = analytics_engine.df.copy()
    if territory:
        data = data[data["territory"] == territory.strip().lower()]
    result = (
        data[["employee_id", "employee_name", "employee_type", "territory"]]
        .drop_duplicates()
    )
    return result.to_dict(orient="records")


# ─────────────────────────────────────────────
# CORE ANALYTICS
# ─────────────────────────────────────────────
@app.get("/analytics/doctor/{doctor_id}")
def get_doctor(
    doctor_id: str,
    time_sec: int = Query(60, ge=1, le=3600),
    employee_type: str = Query("mr"),
):
    et = employee_type.strip().lower()
    result = analytics_engine.get_doctor_summary(
        doctor_id=doctor_id,
        selected_time=time_sec,
        employee_type=et,
    )
    if result is None:
        raise HTTPException(status_code=404, detail=f"Doctor '{doctor_id}' not found")
    return JSONResponse(content=result)


@app.get("/analytics/territory/{territory}")
def get_territory(territory: str):
    result = analytics_engine.get_territory_overview(territory)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Territory '{territory}' not found")
    return JSONResponse(content=result)


# ─────────────────────────────────────────────
# RECOMMENDATION PREVIEW (lightweight — UI slider)
# ─────────────────────────────────────────────
@app.get("/recommendation/preview/{doctor_id}")
def recommendation_preview(
    doctor_id: str,
    time_sec: int = Query(..., ge=1, le=3600),
    employee_type: str = Query("mr"),
):
    doctor_df = analytics_engine.df[analytics_engine.df["doctor_id"] == doctor_id]
    if doctor_df.empty:
        raise HTTPException(status_code=404, detail="Doctor not found")

    scored = analytics_engine.reco_engine.score_products(doctor_df)
    aida   = analytics_engine.aida_classifier.classify(doctor_df)
    reco   = analytics_engine.reco_engine.build_recommendations(
        doctor_df=doctor_df,
        scored_products=scored,
        selected_time=time_sec,
        aida_stage=aida["aida_stage"],
    )

    return {
        "mode":             reco["mode"],
        "effective_time":   reco["effective_time"],
        "event_active":     reco["event_active"],
        "event_type":       reco["event_type"],
        "total_pitched":    reco["total_pitched"],
        "primary_products": reco["primary_products"],
        "support_products": reco["support_products"],
        "closing_products": reco["closing_products"],
        "reminder_items":   reco["reminder_items"],
        "aida_stage":       aida["aida_stage"],
        "employee_type":    employee_type.strip().lower(),
    }


# ─────────────────────────────────────────────
# ML ENGINE ENDPOINTS (plan §3)
# ─────────────────────────────────────────────

@app.get("/recommendations/product_suggestions")
def product_suggestions(
    doctor_id: str = Query(...),
    time_sec: int  = Query(60, ge=1, le=3600),
    employee_type: str = Query("MR"),
):
    """
    Plan §4.1 — full product suggestion payload:
    doctor rating, last meeting, top products with AIDA stage
    and time allocation, rule-based suggestion.
    """
    result = product_reco_engine.get_full_product_suggestions(
        doctor_id=doctor_id,
        selected_time_sec=time_sec,
        employee_type=employee_type,
    )
    if result is None:
        raise HTTPException(status_code=404, detail=f"Doctor '{doctor_id}' not found")
    return JSONResponse(content=result)


@app.get("/analytics/product_performance")
def product_performance(
    product: Optional[str] = Query(None, description="Specific product name. Omit for overall summary."),
    region:  Optional[str] = Query(None),
    quarter: Optional[str] = Query(None),
):
    """
    Plan §4.2 — overall summary or per-product deep dive.
    """
    if product:
        detail = product_perf_engine.get_product_detail(product)
        return JSONResponse(content=detail)

    summary_df = product_perf_engine.get_overall_summary(region=region, quarter=quarter)
    if summary_df.empty:
        raise HTTPException(status_code=404, detail="No data for the given filters")

    quarterly_df = product_perf_engine.get_quarterly_table()

    return JSONResponse(content={
        "summary": summary_df.to_dict(orient="records"),
        "quarterly_table": quarterly_df.to_dict(orient="records"),
    })


@app.get("/analytics/doctor_review")
def doctor_review(
    doctor_id: Optional[str] = Query(None, description="Specific doctor ID. Omit for all doctors."),
    territory: Optional[str] = Query(None),
):
    """
    Plan §4.3 — all-doctors summary (sortable) or individual doctor deep dive.
    """
    if doctor_id:
        analysis = doctor_review_engine.get_doctor_analysis(doctor_id)
        if "error" in analysis:
            raise HTTPException(status_code=404, detail=analysis["error"])
        return JSONResponse(content=analysis)

    summary_df = doctor_review_engine.get_all_doctors_summary(territory=territory)
    if summary_df.empty:
        raise HTTPException(status_code=404, detail="No doctors found")
    return JSONResponse(content={"doctors": summary_df.to_dict(orient="records")})


@app.get("/analytics/employee_report")
def employee_report(
    employee_id: Optional[str] = Query(None, description="Specific employee ID. Omit for team summary."),
    territory:   Optional[str] = Query(None),
):
    """
    Plan §4.4 — team summary by employee_type or individual employee deep dive.
    """
    if employee_id:
        report = employee_report_engine.get_employee_report(employee_id)
        if "error" in report:
            raise HTTPException(status_code=404, detail=report["error"])
        return JSONResponse(content=report)

    team_df = employee_report_engine.get_team_summary(territory=territory)
    if team_df.empty:
        raise HTTPException(status_code=404, detail="No employees found")
    return JSONResponse(content={"team": team_df.to_dict(orient="records")})


# ─────────────────────────────────────────────
# LLM ENDPOINTS (plan §5)
# ─────────────────────────────────────────────

@app.get("/llm/doctor_insights/{doctor_id}")
def llm_doctor_insights(
    doctor_id: str,
    time_sec: int = Query(60, ge=1, le=3600),
    employee_type: str = Query("mr"),
):
    """
    Generates AI narrative insights for a doctor using LLMInsightsEngineEnhanced.
    Only called when user explicitly requests AI insights (plan §5 — reduce LLM load).
    """
    if llm_engine is None:
        raise HTTPException(status_code=503, detail="LLM engine not available (Azure OpenAI not configured)")

    analytics_data = analytics_engine.get_doctor_summary(
        doctor_id=doctor_id,
        selected_time=time_sec,
        employee_type=employee_type.strip().lower(),
    )
    if analytics_data is None:
        raise HTTPException(status_code=404, detail=f"Doctor '{doctor_id}' not found")

    # Build recommendation_engine sub-dict expected by LLM engine
    reco = analytics_data.get("recommendations", {})
    primary = reco.get("primary_products", [])
    analytics_data["recommendation_engine"] = {
        "top_products": [
            {
                "product_name":   p.get("product_name", ""),
                "conversion_rate": p.get("conversion_rate", 0),
                "avg_interest":   p.get("avg_interest", 0),
            }
            for p in primary
        ],
        "doctor_score": analytics_data.get("doctor_scoring", {}).get("score", 0),
    }

    insights = llm_engine.generate_doctor_insights(analytics_data)
    return JSONResponse(content=insights)


@app.get("/llm/product_insight/{product_name}")
def llm_product_insight(product_name: str):
    """
    Explains why a product is underperforming.
    Only called when user clicks 'Why underperforming?' (plan §5).
    """
    if llm_engine is None:
        raise HTTPException(status_code=503, detail="LLM engine not available")

    detail = product_perf_engine.get_product_detail(product_name)
    if "error" in detail:
        raise HTTPException(status_code=404, detail=detail["error"])

    explanation = llm_engine.explain_product_underperformance(
        product_name=product_name,
        metrics=detail,
        objections=detail.get("objection_breakdown", {}),
    )
    return JSONResponse(content={"product_name": product_name, "explanation": explanation})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)