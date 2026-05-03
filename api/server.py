"""
server.py
------------------
FastAPI server — AI Sales Assistant Edition
- employee_type comes from UI (query param), NOT auto-assigned
- /analytics/doctor/{doctor_id}?time_sec=60&employee_type=MR
- /recommendation/preview/{doctor_id}?time_sec=60&employee_type=MR
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from typing import Optional

from main.analytics_engine import DoctorAnalyticsEnhanced


app = FastAPI(title="PatGPT — AI Sales Assistant", version="4.0.0")

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

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing columns: {missing}")

    # String normalisation
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.strip()

    df["territory"]    = df["area"].str.lower()
    df["doctor_id"]    = df["doctor_id"].astype(str).str.strip()
    df["follow_up"]    = df["follow_up"].str.lower()
    df["specialty"]    = df["specialty"].str.lower()
    df["employee_type"] = df["employee_type"].str.lower()

    # Outcome normalisation
    outcome_map = {
        "positive": "positive", "converted": "positive", "success": "positive",
        "won": "positive", "yes": "positive",
        "negative": "negative", "lost": "negative", "no": "negative",
        "neutral": "neutral", "pending": "neutral",
    }
    df["outcome"] = df["outcome"].str.lower().map(outcome_map).fillna("neutral")

    # Interest level → numeric
    interest_map = {"low": 1, "medium": 3, "high": 5}
    df["interest_level"] = (
        df["interest_level"].astype(str).str.strip().str.lower()
        .map(interest_map).fillna(0)
    )

    return df


df = load_data()
analytics_engine = DoctorAnalyticsEnhanced(df)


@app.get("/")
def root():
    return {"status": "running", "version": "4.0.0", "mode": "AI Sales Assistant"}


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "rows": len(analytics_engine.df),
        "doctors": analytics_engine.df["doctor_id"].nunique(),
    }


@app.get("/territories")
def get_territories():
    territories = analytics_engine.df["area"].dropna().unique().tolist()
    return {"territories": sorted(territories)}


@app.get("/doctors")
def get_doctors(
    territory: Optional[str] = Query(None),
    specialty: Optional[str] = Query(None),
):
    data = analytics_engine.df

    if territory:
        data = data[data["territory"] == territory.strip().lower()]
    if specialty:
        data = data[data["specialty"] == specialty.strip().lower()]

    result = (
        data[["doctor_id", "doctor_name", "territory", "specialty"]]
        .drop_duplicates()
    )
    return result.to_dict(orient="records")


@app.get("/analytics/doctor/{doctor_id}")
def get_doctor(
    doctor_id: str,
    time_sec: int = Query(60, ge=1, le=3600, description="Visit duration in seconds"),
    employee_type: str = Query("mr", description="Employee type selected from UI: mr / area manager / vp / gm"),
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
    """
    Lightweight recommendation preview — used by the time-slider in the UI.
    Returns structured product buckets (primary / support / closing / reminder).
    """
    doctor_df = analytics_engine.df[analytics_engine.df["doctor_id"] == doctor_id]

    if doctor_df.empty:
        raise HTTPException(status_code=404, detail="Doctor not found")

    scored = analytics_engine.reco_engine.score_products(doctor_df)

    # Need AIDA stage
    aida = analytics_engine.aida_classifier.classify(doctor_df)

    reco = analytics_engine.reco_engine.build_recommendations(
        doctor_df=doctor_df,
        scored_products=scored,
        selected_time=time_sec,
        aida_stage=aida["aida_stage"],
    )

    return {
        "mode":              reco["mode"],
        "effective_time":    reco["effective_time"],
        "event_active":      reco["event_active"],
        "event_type":        reco["event_type"],
        "total_pitched":     reco["total_pitched"],
        "primary_products":  reco["primary_products"],
        "support_products":  reco["support_products"],
        "closing_products":  reco["closing_products"],
        "reminder_items":    reco["reminder_items"],
        "aida_stage":        aida["aida_stage"],
        "employee_type":     employee_type.strip().lower(),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)