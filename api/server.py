"""
server_improved.py
------------------
FastAPI server with dual analytics endpoints:
1. /analytics/doctor/{doctor_id} - Raw data analytics
2. /insights/doctor/{doctor_id} - LLM-powered insights
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd

from main.analytics_engine import DoctorAnalytics
from main.llm_insights import LLMInsightsEngine
from init_services.init import initialize_services

app = FastAPI()
initialize_services() 
df = pd.read_csv("data/doctor_sales_dummy_data.csv")
analytics_engine = DoctorAnalytics(df)
llm_engine = LLMInsightsEngine()

@app.get("/")
def root():
    return {"status": "running"}


@app.get("/doctors/{territory}")
def get_doctors(territory: str):
    territory = territory.strip().lower()
    data = analytics_engine.df.copy()
    data["territory"] = data["territory"].str.strip().str.lower()
    docs = data[data["territory"] == territory][["doctor_id", "doctor_name"]].drop_duplicates()
    return docs.to_dict(orient="records")


@app.get("/analytics/doctor/{doctor_id}")
def get_doctor(doctor_id: int):
    result = analytics_engine.get_doctor_summary(doctor_id)

    if result is None:
        raise HTTPException(status_code=404, detail="Doctor not found")

    return JSONResponse(content=result)

@app.get("/insights/doctor/{doctor_id}")
def get_insights(doctor_id: int):
    analytics_data = analytics_engine.get_doctor_summary(doctor_id)

    if analytics_data is None:
        raise HTTPException(status_code=404, detail="Doctor not found")

    insights = llm_engine.generate_doctor_insights(analytics_data)
    return JSONResponse(content={"insights": insights})