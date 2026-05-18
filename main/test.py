from analytics_engine import TrendAnalytics
from main.ml.ml_engines import ProductRecommendationEngine
import pandas as pd
import random


# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
df = pd.read_csv("C:\\Users\\hardi\\Desktop\\GlobalspaceTechnology\\Work\\Doctor\\data\\doctor_sales_dummy_data.csv")
df.columns = df.columns.str.strip()
print(df.columns.tolist())
# Pick random doctors
doctor_ids = df["doctor_id"].unique().tolist()
sample_doctors = random.sample(doctor_ids, 5)


# ─────────────────────────────────────────────
# INIT ENGINES
# ─────────────────────────────────────────────
trend_engine = TrendAnalytics(df)
product_engine = ProductRecommendationEngine(df)


# ─────────────────────────────────────────────
# TEST FUNCTION WRAPPER
# ─────────────────────────────────────────────
def test_doctor(doctor_id):
    print(f"\n==============================")
    print(f"🧪 Testing Doctor: {doctor_id}")
    print(f"==============================")

    # ───────────── TREND TEST ─────────────
    trends = trend_engine.get_doctor_trends(doctor_id)

    if not trends:
        print("❌ No trend data found")
    else:
        print("\n📈 Trend Analysis:")
        print("Conversion Trend:", trends["trends"]["conversion"])
        print("Interest Trend:", trends["trends"]["interest"])
        print("Sales Trend:", trends["trends"]["sales"])

    # ───────── PRODUCT RECOMMENDATION TEST ─────────
    suggestions = product_engine.get_full_product_suggestions(
        doctor_id,
        selected_time_sec=120,
        employee_type="MR"
    )

    if not suggestions:
        print("❌ No recommendation data")
    else:
        print("\n💊 Product Suggestions:")
        print("Doctor:", suggestions["doctor_name"])
        print("Rating:", suggestions["doctor_rating"], "| Tier:", suggestions["doctor_tier"])

        print("\nTop Products:")
        for p in suggestions["recommendations"]:
            print(
                f"- {p['product_name']} | Score: {p['score']} | "
                f"Conv: {p['conversion_rate']} | Interest: {p['avg_interest']}"
            )

        print("\n🧠 Suggestion:")
        print(suggestions["custom_suggestion"])


# ─────────────────────────────────────────────
# RUN TESTS
# ─────────────────────────────────────────────
for doc_id in sample_doctors:
    test_doctor(doc_id)