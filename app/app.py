import streamlit as st
import requests
import pandas as pd
import plotly.express as px

BASE_URL = "http://localhost:8000"

st.set_page_config(page_title="Doctor Dashboard", layout="wide")

st.title("🧠 Doctor Analytics Dashboard")


# ============================
# HELPERS
# ============================

@st.cache_data
def get_territories():
    df = pd.read_csv("data/doctor_sales_dummy_data.csv")
    df.columns = df.columns.str.strip().str.lower()
    return sorted(df["territory"].unique().tolist())


@st.cache_data
def get_doctors(territory):
    res = requests.get(f"{BASE_URL}/doctors/{territory}")
    return res.json()


def get_doctor_data(doctor_id):
    analytics = requests.get(f"{BASE_URL}/analytics/doctor/{doctor_id}").json()
    return analytics


# ============================
# SIDEBAR
# ============================

st.sidebar.header("Filters")

territory = st.sidebar.selectbox("Select Territory", get_territories())

doctors = get_doctors(territory)

doctor_map = {
    f"{d['doctor_name']} (ID: {d['doctor_id']})": d["doctor_id"]
    for d in doctors
}

doctor_selected = st.sidebar.selectbox(
    "Select Doctor",
    list(doctor_map.keys()),
    index=None,  # 👈 IMPORTANT
    placeholder="Select a doctor"
)

if doctor_selected is None:
    st.warning("Please select a doctor")
    st.stop()  # ⛔ stops execution here

doctor_id = doctor_map[doctor_selected]


# ============================
# DATA FETCH
# ============================

analytics = get_doctor_data(doctor_id)

doc = analytics["doctor_info"]
eng = analytics["engagement_metrics"]
products = analytics["product_performance"]["product_breakdown"]


# ============================
# HEADER
# ============================

st.subheader(f"👨‍⚕️ {doc['doctor_name']} ({doc['specialty']})")
st.write(f"**Experience:** { doc['experience_years']} years | **Publications:** {doc['publications_count']} | **Social Reach:** {doc['social_media_reach']} followers")
st.write(f"**Patient Load:** {doc['patient_load']} patients")
col1, col2, col3, col4 = st.columns(4)

col1.metric("Conversion Rate", f"{eng['conversion_rate']:.0%}")
col2.metric("Avg Interest", f"{eng['avg_interest_level']:.1f}/5")
col3.metric("Total Interactions", eng["total_interactions"])
col4.metric("Follow-up Rate", f"{eng['follow_up_rate']:.0%}")


# ============================
# PRODUCT PERFORMANCE
# ============================

st.markdown("## 📊 Product Performance")

product_df = pd.DataFrame(products)

fig = px.bar(
    product_df,
    x="product_name",
    y="conversion_rate",
    title="Conversion Rate per Product"
)

st.plotly_chart(fig, width='stretch')


# ============================
# INTEREST VS PERFORMANCE
# ============================

fig2 = px.scatter(
    product_df,
    x="avg_interest",
    y="conversion_rate",
    size="times_presented",
    hover_name="product_name",
    title="Interest vs Conversion"
)

st.plotly_chart(fig2, width='stretch')


# ============================
# OBJECTIONS
# ============================

st.markdown("## ⚠️ Remarkable Objections")

obj_data = analytics.get("objection_analysis", {})

if not obj_data.get("has_objections", False):
    st.info("No objections recorded")
else:
    obj_df = pd.DataFrame(
        list(obj_data["objection_breakdown"].items()),
        columns=["Objection", "Count"]
    )
    fig3 = px.pie(obj_df, names="Objection", values="Count")
    st.plotly_chart(fig3)


# ============================
# AI INSIGHTS
# ============================

st.markdown("## 🤖 AI Insights")
 
if st.button("Generate AI Insights"):
    with st.spinner("Generating insights..."):
        try:
            insights_resp = requests.get(f"{BASE_URL}/insights/doctor/{doctor_id}")
            insights_resp.raise_for_status()
            insights = insights_resp.json()
 
            if "insights" not in insights:
                st.error(f"Insights API Error: {insights}")
            else:
                insights_data = insights["insights"]
 
                best_product = insights_data.get("best_product", "").strip()
                similar_products = insights_data.get("similar_products", "").strip()
                doctor_value = insights_data.get("doctor_value", "").strip()
                suggestion = insights_data.get("suggestion", "").strip()
 
                if not best_product and not similar_products and not doctor_value and not suggestion:
                    st.warning("No insights generated")
                else:
                    col1, col2 = st.columns(2)
 
                    with col1:
                        st.markdown("### 🎯 Recommended Product")
                        if best_product:
                            st.success(best_product)
                        else:
                            st.info("No data")
 
                        st.markdown("### 🔁 Similar Products")
                        if similar_products:
                            st.info(similar_products)
                        else:
                            st.info("No data")
 
                    with col2:
                        st.markdown("### 🧠 Doctor Rating")
                        if doctor_value:
                            st.warning(doctor_value)
                        else:
                            st.info("No data")
 
                        st.markdown("### 📍 Territory Suggestion")
                        if suggestion:
                            st.markdown(suggestion)
                        else:
                            st.info("No data")
 
        except requests.exceptions.RequestException as e:
            st.error(f"API request failed: {str(e)}")
        except Exception as e:
            st.error(f"Error: {str(e)}")