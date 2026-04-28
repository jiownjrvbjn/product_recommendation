"""
server.py
------------------
Enhanced FastAPI server with:
- Original analytics and insights endpoints
- Trend analytics endpoints
- Competitive intelligence endpoints
- Manager comparison endpoints
- Objection resolution endpoints
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from typing import Optional

from main.analytics_engine import DoctorAnalyticsEnhanced
from main.llm_insights import LLMInsightsEngineEnhanced
from init_services.init import initialize_services

app = FastAPI(
    title="Pharma Analytics API",
    description="Enhanced analytics with trends, competitive intelligence, and AI insights",
    version="2.0.0"
)

# CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
initialize_services()
df = pd.read_csv("data/doctor_sales_dummy_data.csv")
analytics_engine = DoctorAnalyticsEnhanced(df)
llm_engine = LLMInsightsEngineEnhanced()


# ============================================================================
# ORIGINAL ENDPOINTS (Unchanged)
# ============================================================================

@app.get("/")
def root():
    return {
        "status": "running",
        "version": "2.0.0",
        "features": [
            "basic_analytics",
            "llm_insights",
            "trend_analytics",
            "competitive_intelligence",
            "manager_comparisons",
            "objection_resolution"
        ]
    }


@app.get("/doctors/{territory}")
def get_doctors(territory: str):
    """Get list of doctors in a territory"""
    territory = territory.strip().lower()
    data = analytics_engine.df.copy()
    data["territory"] = data["territory"].str.strip().str.lower()
    docs = data[data["territory"] == territory][["doctor_id", "doctor_name"]].drop_duplicates()
    return docs.to_dict(orient="records")


@app.get("/analytics/doctor/{doctor_id}")
def get_doctor(doctor_id: int):
    """Get comprehensive doctor analytics (enhanced with all new features)"""
    result = analytics_engine.get_doctor_summary(doctor_id)
    
    if result is None:
        raise HTTPException(status_code=404, detail="Doctor not found")
    
    return JSONResponse(content=result)


@app.get("/insights/doctor/{doctor_id}")
def get_insights(doctor_id: int):
    """Get LLM-powered insights (enhanced with sentiment, objections, competitive)"""
    analytics_data = analytics_engine.get_doctor_summary(doctor_id)
    
    if analytics_data is None:
        raise HTTPException(status_code=404, detail="Doctor not found")
    
    insights = llm_engine.generate_doctor_insights(analytics_data)
    return JSONResponse(content={"insights": insights})


# ============================================================================
# NEW ENDPOINTS - Trend Analytics
# ============================================================================

@app.get("/analytics/trends/doctor/{doctor_id}")
def get_doctor_trends(doctor_id: int, days: Optional[int] = 90):
    """Get time-series trends for a doctor"""
    trends = analytics_engine.trend_engine.get_doctor_trends(doctor_id, days)
    
    if trends is None:
        raise HTTPException(status_code=404, detail="Doctor not found or insufficient data")
    
    return JSONResponse(content=trends)


@app.get("/analytics/trends/products/{doctor_id}")
def get_product_trends(doctor_id: int):
    """Get product-specific trends for a doctor"""
    product_trends = analytics_engine.trend_engine.get_product_trends(doctor_id)
    
    if not product_trends:
        raise HTTPException(status_code=404, detail="Doctor not found or no product data")
    
    return JSONResponse(content={"products": product_trends})


# ============================================================================
# NEW ENDPOINTS - Competitive Intelligence
# ============================================================================

@app.get("/analytics/competitive/doctor/{doctor_id}")
def get_competitive_analysis(doctor_id: int):
    """Get competitive intelligence for a doctor"""
    competitive = analytics_engine.competitive_engine.get_competitive_analysis(doctor_id)
    
    if competitive is None:
        raise HTTPException(status_code=404, detail="Doctor not found")
    
    return JSONResponse(content=competitive)


# ============================================================================
# NEW ENDPOINTS - Manager Comparisons
# ============================================================================

@app.get("/analytics/territory/{territory}")
def get_territory_benchmarks(territory: str):
    """Get territory-wide benchmarks and statistics"""
    benchmarks = analytics_engine.manager_engine.get_territory_benchmarks(territory)
    
    if benchmarks is None:
        raise HTTPException(status_code=404, detail="Territory not found")
    
    return JSONResponse(content=benchmarks)


@app.get("/analytics/comparison/doctor/{doctor_id}/territory/{territory}")
def compare_doctor_to_territory(doctor_id: int, territory: str):
    """Compare a doctor against territory benchmarks"""
    comparison = analytics_engine.manager_engine.compare_doctor_to_territory(
        doctor_id, territory
    )
    
    if comparison is None:
        raise HTTPException(status_code=404, detail="Doctor or territory not found")
    
    return JSONResponse(content=comparison)


@app.get("/analytics/comparison/specialty/{doctor_id}")
def compare_doctor_to_specialty(doctor_id: int):
    """Compare a doctor against others in same specialty"""
    comparison = analytics_engine.manager_engine.get_specialty_comparison(doctor_id)
    
    if comparison is None:
        raise HTTPException(status_code=404, detail="Doctor not found")
    
    return JSONResponse(content=comparison)


@app.get("/analytics/territory/overview/{territory}")
def get_territory_overview(territory: str):
    """Get complete territory overview for managers"""
    overview = analytics_engine.get_territory_overview(territory)
    
    if overview is None:
        raise HTTPException(status_code=404, detail="Territory not found")
    
    return JSONResponse(content=overview)


# ============================================================================
# NEW ENDPOINTS - Objection Resolution
# ============================================================================

@app.get("/analytics/objections/doctor/{doctor_id}")
def get_objection_analysis(doctor_id: int):
    """Get detailed objection analysis and resolution strategies"""
    objection_analysis = analytics_engine.objection_engine.get_objection_analysis(doctor_id)
    
    if objection_analysis is None:
        raise HTTPException(status_code=404, detail="Doctor not found")
    
    return JSONResponse(content=objection_analysis)


# ============================================================================
# UTILITY ENDPOINTS
# ============================================================================

@app.get("/territories")
def get_all_territories():
    """Get list of all territories"""
    territories = analytics_engine.df['territory'].unique().tolist()
    return {"territories": sorted(territories)}


@app.get("/specialties")
def get_all_specialties():
    """Get list of all specialties"""
    specialties = analytics_engine.df['specialty'].unique().tolist()
    return {"specialties": sorted(specialties)}


@app.get("/doctors")
def get_all_doctors(
    territory: Optional[str] = None,
    specialty: Optional[str] = None
):
    """Get list of all doctors with optional filters"""
    df = analytics_engine.df.copy()
    
    if territory:
        df = df[df['territory'] == territory.strip().lower()]
    
    if specialty:
        df = df[df['specialty'] == specialty.strip()]
    
    doctors = df[['doctor_id', 'doctor_name', 'territory', 'specialty']].drop_duplicates()
    return doctors.to_dict(orient='records')


# ============================================================================
# HEALTH CHECK
# ============================================================================

@app.get("/health")
def health_check():
    """API health check"""
    return {
        "status": "healthy",
        "analytics_engine": "operational",
        "llm_engine": "operational" if llm_engine.client else "unavailable",
        "data_loaded": len(analytics_engine.df) > 0,
        "total_doctors": analytics_engine.df['doctor_id'].nunique()
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)