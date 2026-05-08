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


app = FastAPI(title="PatGPT", version="5.0.0")

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


def load_data():
    df = pd.read_csv("data/doctor_sales_dummy_data.csv")
    df.columns = df.columns.str.strip().str.lower()
    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]) or df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip()
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing columns after normalisation: {missing}")
    if "area" in df.columns:
        df["territory"] = df["area"].str.lower()
    else:
        df["territory"] = df["territory"].str.lower()
    if "objection" not in df.columns:
        if "objection_type" in df.columns:
            df["objection"] = df["objection_type"].astype(str).str.strip().str.lower()
        else:
            df["objection"] = "none"
    df["doctor_id"] = df["doctor_id"].str.strip()
    df["follow_up"] = df["follow_up"].str.lower()
    df["specialty"] = df["specialty"].str.lower()
    df["employee_type"] = df["employee_type"].str.lower()
    df["region"] = df["region"].str.strip() if "region" in df.columns else df["territory"]
    outcome_map = {
        "positive": "positive", "converted": "positive", "success": "positive",
        "won": "positive", "yes": "positive",
        "negative": "negative", "lost": "negative", "no": "negative",
        "neutral": "neutral", "pending": "neutral",
    }
    df["outcome"] = df["outcome"].str.lower().map(outcome_map).fillna("neutral")
    if df["interest_level"].dtype == object:
        interest_map = {"low": 1, "medium": 3, "high": 5}
        df["interest_level"] = (
            df["interest_level"].str.lower().map(interest_map).fillna(0)
        )
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

initialize_services()
df = load_data()
analytics_engine = DoctorAnalyticsEnhanced(df)
product_reco_engine = ProductRecommendationEngine(df)
product_perf_engine = ProductPerformanceEngine(df)
doctor_review_engine = DoctorReviewEngine(df)
employee_report_engine = EmployeeReportEngine(df)

try:
    llm_engine = LLMInsightsEngineEnhanced()
except Exception as _llm_err:
    llm_engine = None
    print(f"[WARN] LLM engine not initialised: {_llm_err}")


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
    aida = analytics_engine.aida_classifier.classify(doctor_df)
    reco = analytics_engine.reco_engine.build_recommendations(
        doctor_df=doctor_df,
        scored_products=scored,
        selected_time=time_sec,
        aida_stage=aida["aida_stage"],
    )
    return {
        "mode": reco["mode"],
        "effective_time": reco["effective_time"],
        "event_active": reco["event_active"],
        "event_type": reco["event_type"],
        "total_pitched": reco["total_pitched"],
        "primary_products": reco["primary_products"],
        "support_products": reco["support_products"],
        "closing_products": reco["closing_products"],
        "reminder_items": reco["reminder_items"],
        "aida_stage": aida["aida_stage"],
        "employee_type": employee_type.strip().lower(),
    }


@app.get("/recommendations/product_suggestions")
def product_suggestions(
    doctor_id: str = Query(...),
    time_sec: int = Query(60, ge=1, le=3600),
    employee_type: str = Query("MR"),
):
    result = product_reco_engine.get_full_product_suggestions(
        doctor_id=doctor_id,
        selected_time_sec=time_sec,
        employee_type=employee_type,
    )
    if result is None:
        raise HTTPException(status_code=404, detail=f"Doctor '{doctor_id}' not found")
    return JSONResponse(content=result)


@app.get("/suggestion/customize/{doctor_id}")
def customize_suggestion(
    doctor_id: str,
    time_sec: int = Query(60, ge=1, le=3600),
    employee_type: str = Query("MR"),
):
    if llm_engine is None:
        raise HTTPException(status_code=503, detail="LLM engine not available")
    suggestion = product_reco_engine.generate_custom_suggestion(
        doctor_id, time_sec, employee_type,
        llm_client=llm_engine.client, deployment=llm_engine.deployment,
    )
    return {"suggestion": suggestion}


@app.get("/analytics/product_performance")
def product_performance(
    product: Optional[str] = Query(None),
    region: Optional[str] = Query(None),
    quarter: Optional[str] = Query(None),
):
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
    doctor_id: Optional[str] = Query(None),
    territory: Optional[str] = Query(None),
):
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
    employee_id: Optional[str] = Query(None),
    territory: Optional[str] = Query(None),
):
    if employee_id:
        report = employee_report_engine.get_employee_report(employee_id)
        if "error" in report:
            raise HTTPException(status_code=404, detail=report["error"])
        return JSONResponse(content=report)
    team_df = employee_report_engine.get_team_summary(territory=territory)
    if team_df.empty:
        raise HTTPException(status_code=404, detail="No employees found")
    return JSONResponse(content={"team": team_df.to_dict(orient="records")})


@app.get("/llm/doctor_insights/{doctor_id}")
def llm_doctor_insights(
    doctor_id: str,
    time_sec: int = Query(60, ge=1, le=3600),
    employee_type: str = Query("mr"),
):
    if llm_engine is None:
        raise HTTPException(status_code=503, detail="LLM engine not available")
    analytics_data = analytics_engine.get_doctor_summary(
        doctor_id=doctor_id,
        selected_time=time_sec,
        employee_type=employee_type.strip().lower(),
    )
    if analytics_data is None:
        raise HTTPException(status_code=404, detail=f"Doctor '{doctor_id}' not found")
    reco = analytics_data.get("recommendations", {})
    primary = reco.get("primary_products", [])
    analytics_data["recommendation_engine"] = {
        "top_products": [
            {
                "product_name": p.get("product_name", ""),
                "conversion_rate": p.get("conversion_rate", 0),
                "avg_interest": p.get("avg_interest", 0),
            }
            for p in primary
        ],
        "doctor_score": analytics_data.get("doctor_scoring", {}).get("score", 0),
    }
    insights = llm_engine.generate_doctor_insights(analytics_data)
    return JSONResponse(content=insights)


@app.get("/llm/product_insight/{product_name}")
def llm_product_insight(product_name: str):
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

@app.get("/llm/meeting_playbook/{doctor_id}")
def meeting_playbook(
    doctor_id: str,
    time_sec: int = Query(60, ge=1, le=3600),
    employee_type: str = Query("mr"),
):
    if llm_engine is None:
        raise HTTPException(status_code=503, detail="LLM engine not available")

    summary = analytics_engine.get_doctor_summary(
        doctor_id=doctor_id,
        selected_time=time_sec,
        employee_type=employee_type.strip().lower(),
    )
    if summary is None:
        raise HTTPException(status_code=404, detail=f"Doctor '{doctor_id}' not found")

    playbook = llm_engine.generate_meeting_playbook(summary)
    return {"playbook": playbook}

@app.get("/analytics/doctor/{doctor_id}/product_aida")
def get_product_aida(
    doctor_id: str,
    time_sec: int = Query(60, ge=1, le=3600),
    employee_type: str = Query("mr"),
):
    """
    Returns per-product AIDA stage for the doctor's recommended products.
    Classifies each product's interaction subset independently using AIDAClassifier,
    then overlays score and interest data from the recommendation engine.
    """
    doctor_df = analytics_engine.df[
        analytics_engine.df["doctor_id"].astype(str).str.strip() == str(doctor_id).strip()
    ]
    if doctor_df.empty:
        raise HTTPException(status_code=404, detail=f"Doctor '{doctor_id}' not found")
 
    # Score all products for this doctor
    scored = analytics_engine.reco_engine.score_products(doctor_df)
    aida_overall = analytics_engine.aida_classifier.classify(doctor_df)
 
    # Get the top N recommended products (primary + support, capped at 3)
    all_ranked = scored["all_products_ranked"][:5]  # limit lookup scope
 
    product_aida_list = []
    for p in all_ranked[:3]:
        product_name = p["product_name"]
        p_df = doctor_df[doctor_df["product_name"] == product_name]
 
        if p_df.empty:
            continue
 
        # Classify AIDA on the product-specific interaction slice
        p_aida = analytics_engine.aida_classifier.classify(p_df)
 
        product_aida_list.append({
            "product_name":    product_name,
            "aida_stage":      p_aida["aida_stage"],
            "aida_label":      p_aida["aida_label"],
            "aida_color":      p_aida["aida_color"],
            "aida_emoji":      p_aida["aida_emoji"],
            "aida_confidence": p_aida["aida_confidence"],
            "aida_signals":    p_aida["aida_signals"],
            "stage_guidance":  p_aida["stage_guidance"],
            # Product-level metrics for display
            "conversion_rate": p["conversion_rate"],
            "avg_interest":    p["avg_interest"],
            "score":           p["score"],
            "category":        p["category"],
            "suggested_duration_sec": p.get("suggested_duration_sec", 0),
        })
 
    return JSONResponse(content={
        "doctor_id":     doctor_id,
        "overall_aida":  aida_overall["aida_label"],
        "product_aida":  product_aida_list,
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)