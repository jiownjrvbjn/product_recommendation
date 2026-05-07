"""
app.py — PatGPT AI Sales Assistant (v5.0)
Main menu (sidebar):
  1. 🏥 Sales Assistant   — doctor visit dashboard
  2. 📦 Product Performance — portfolio analytics
  3. 🩺 Doctor Review      — all doctors or individual deep dive
  4. 👤 Employee Reports   — team summary or individual report
"""
import streamlit as st
import requests
import html

BASE_URL = "http://localhost:8000"

st.set_page_config(
    page_title="PatGPT",
    layout="wide",
    initial_sidebar_state="expanded",
)

@st.cache_data(ttl=300)
def get_territories():
    try:
        r = requests.get(f"{BASE_URL}/territories")
        r.raise_for_status()
        return r.json().get("territories", [])
    except Exception as e:
        st.error(f"Could not fetch territories: {e}")
        return []

@st.cache_data(ttl=300)
def get_doctors_by_territory(territory: str):
    try:
        r = requests.get(f"{BASE_URL}/doctors", params={"territory": territory})
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"Could not fetch doctors: {e}")
        return []

@st.cache_data(ttl=300)
def get_products():
    try:
        r = requests.get(f"{BASE_URL}/products")
        r.raise_for_status()
        return r.json().get("products", [])
    except Exception as e:
        return []

@st.cache_data(ttl=300)
def get_employees(territory=None):
    try:
        params = {"territory": territory} if territory else {}
        r = requests.get(f"{BASE_URL}/employees", params=params)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return []

def get_product_suggestions(doctor_id, time_sec, employee_type):
    try:
        r = requests.get(
            f"{BASE_URL}/recommendations/product_suggestions",
            params={"doctor_id": doctor_id, "time_sec": time_sec, "employee_type": employee_type},
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return None

def get_custom_suggestion(doctor_id, time_sec, employee_type):
    try:
        r = requests.get(
            f"{BASE_URL}/suggestion/customize/{doctor_id}",
            params={"time_sec": time_sec, "employee_type": employee_type},
        )
        r.raise_for_status()
        return r.json().get("suggestion", "")
    except Exception:
        return None

def get_product_performance(product=None, region=None, quarter=None):
    try:
        params = {}
        if product: params["product"] = product
        if region: params["region"] = region
        if quarter: params["quarter"] = quarter
        r = requests.get(f"{BASE_URL}/analytics/product_performance", params=params)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return None

def get_doctor_review(doctor_id=None, territory=None):
    try:
        params = {}
        if doctor_id: params["doctor_id"] = doctor_id
        if territory: params["territory"] = territory
        r = requests.get(f"{BASE_URL}/analytics/doctor_review", params=params)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return None

def get_employee_report(employee_id=None, territory=None):
    try:
        params = {}
        if employee_id: params["employee_id"] = employee_id
        if territory: params["territory"] = territory
        r = requests.get(f"{BASE_URL}/analytics/employee_report", params=params)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return None

def get_doctor_analytics(doctor_id: str, time_sec: int, employee_type: str):
    try:
        r = requests.get(
            f"{BASE_URL}/analytics/doctor/{doctor_id}",
            params={"time_sec": time_sec, "employee_type": employee_type},
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"Could not fetch doctor analytics: {e}")
        return None

def get_meeting_playbook(doctor_id: str, time_sec: int, employee_type: str):
    try:
        r = requests.get(
            f"{BASE_URL}/llm/meeting_playbook/{doctor_id}",
            params={"time_sec": time_sec, "employee_type": employee_type},
            timeout=30
        )
        if r.status_code == 503:
            return "❌ AI service is currently unavailable. Please check server logs."
        r.raise_for_status()
        return r.json().get("playbook", "")
    except requests.exceptions.ConnectionError:
        return "❌ Cannot connect to backend server. Is it running?"
    except Exception as e:
        return f"❌ Error: {str(e)}"

def sales_assistant_section():
    st.header("Sales Assistant")
    if "ps_data" not in st.session_state:
        st.session_state.ps_data = None
    if "ps_doctor_id" not in st.session_state:
        st.session_state.ps_doctor_id = None
    if "ps_doctor_name" not in st.session_state:
        st.session_state.ps_doctor_name = None
    if "ps_employee_type" not in st.session_state:
        st.session_state.ps_employee_type = "mr"
    if "ps_selected_time" not in st.session_state:
        st.session_state.ps_selected_time = 60
    if "ps_submitted" not in st.session_state:
        st.session_state.ps_submitted = False

    territories = get_territories()
    if not territories:
        st.warning("No territories found.")
        return

    territory = st.selectbox("Select Territory", territories, index=None, placeholder="Choose a territory...")
    if territory:
        doctors = get_doctors_by_territory(territory)
        if doctors:
            doctor_options = {f"{d['doctor_name']} (ID: {d['doctor_id']})": d["doctor_id"] for d in doctors}
            selected_label = st.selectbox(
                "Select Doctor", list(doctor_options.keys()),
                index=None, placeholder="Choose a doctor...",
            )
            if selected_label:
                st.session_state.ps_doctor_id = str(doctor_options[selected_label]).strip()
                st.session_state.ps_doctor_name = selected_label.split(" (ID:")[0]
            else:
                st.session_state.ps_doctor_id = None
        else:
            st.info("No doctors in this territory.")
            return

    if not st.session_state.ps_doctor_id:
        return

    # Auto-load doctor card when doctor and role are selected
    if st.session_state.ps_doctor_id and st.session_state.ps_employee_type:
        with st.spinner("Loading doctor insights..."):
            analytics = get_doctor_analytics(
                st.session_state.ps_doctor_id,
                st.session_state.ps_selected_time,
                st.session_state.ps_employee_type,
            )
        if analytics:
            st.session_state.doctor_analytics = analytics
            st.session_state.last_meeting = analytics.get("last_meeting")
            st.session_state.top_historical_products = analytics.get("top_historical_products", [])

    # Display doctor card if analytics exist
    if "doctor_analytics" in st.session_state and st.session_state.doctor_analytics:
        da = st.session_state.doctor_analytics
        doc_info = da.get("doctor_info", {})
        engagement = da.get("engagement_metrics", {})
        aida = da.get("aida", {})
        scoring = da.get("doctor_scoring", {})

        with st.container():
            st.subheader(f"📋 Doctor Card: {doc_info.get('doctor_name')}")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Specialty", doc_info.get("specialty", "N/A"))
            col2.metric("Patient Load", doc_info.get("patient_load", 0))
            col3.metric("Total Interactions", engagement.get("total_interactions", 0))
            col4.metric("Avg Time (sec)", engagement.get("avg_meeting_duration_sec", "N/A"))

            col1.metric("Doctor Score", f"{scoring.get('score', 0):.2f}", scoring.get("tier", "").upper())
            col2.metric("AIDA Stage", aida.get("aida_label", "Unknown"))

            # Last meeting
            lm = st.session_state.last_meeting
            if lm and lm.get("date"):
                with st.expander("🕒 Last Meeting Recap"):
                    st.write(f"**Date:** {lm.get('date')}")
                    st.write(f"**Product:** {lm.get('product')}")
                    st.write(f"**Interest:** {lm.get('interest_level')}/5")
                    st.write(f"**Outcome:** {lm.get('outcome')}")
                    st.write(f"**Objection:** {lm.get('objection', 'none')}")
                    st.write(f"**Duration:** {lm.get('actual_time_seconds')} sec")
                    if lm.get("meeting_notes"):
                        st.info(f"📝 Notes: {lm.get('meeting_notes')}")
            else:
                st.caption("No recent meeting data.")

            # Top 3 historical products
            top_prods = st.session_state.top_historical_products
            if top_prods:
                with st.expander("🏆 Top 3 Historical Products (by total time)"):
                    for p in top_prods:
                        st.write(f"**{p['product_name']}** – {p['times_presented']} times, avg {p['avg_time_per_presentation']} sec")

        # Playbook button
        if st.button("📖 Want an AI-powered meeting playbook?"):
            with st.spinner("Generating AI playbook..."):
                playbook = get_meeting_playbook(
                    st.session_state.ps_doctor_id,
                    st.session_state.ps_selected_time,
                    st.session_state.ps_employee_type,
                )
            if playbook:
                with st.expander("📘 AI Helping Playbook", expanded=True):
                    st.markdown(playbook)


def product_performance_section():
    st.header("Product Performance")
    col_p, col_r, col_q = st.columns(3)
    with col_p:
        products = get_products()
        product_sel = st.selectbox("Product (optional)", ["Overall Summary"] + products)
    with col_r:
        territories = get_territories()
        region_sel = st.selectbox("Region (optional)", ["All"] + territories)
    with col_q:
        quarter_sel = st.selectbox("Quarter (optional)", ["All", "Q1", "Q2", "Q3", "Q4"])

    product_param = None if product_sel == "Overall Summary" else product_sel
    region_param = None if region_sel == "All" else region_sel
    quarter_param = None if quarter_sel == "All" else quarter_sel

    if st.button("Load Product Data", type="primary"):
        with st.spinner("Loading..."):
            data = get_product_performance(product=product_param, region=region_param, quarter=quarter_param)
        if not data:
            return

        if product_param:
            st.subheader(product_param)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Sales", data.get("total_sales", 0))
            m2.metric("Conversion", f"{data.get('conversion_rate', 0):.0%}")
            m3.metric("Avg Interest", f"{data.get('avg_interest', 0):.1f}/5")
            m4.metric("QoQ Growth", f"{data.get('qoq_growth', 0):.1%}")
            tre = data.get("trend", "stable")
            st.caption(f"Trend: {tre}")
            obj = data.get("objection_breakdown", {})
            if obj:
                with st.expander("Objection Breakdown"):
                    for k, v in sorted(obj.items(), key=lambda x: -x[1]):
                        st.write(f"- {k}: {v}")
        else:
            summary = data.get("summary", [])
            if summary:
                import pandas as pd
                df_s = pd.DataFrame(summary)
                st.dataframe(df_s, use_container_width=True)
            quarterly = data.get("quarterly_table", [])
            if quarterly:
                with st.expander("Quarterly Sales"):
                    st.dataframe(pd.DataFrame(quarterly), use_container_width=True)


def doctor_review_section():
    st.header("Doctor Review")
    col_t, col_d = st.columns(2)
    with col_t:
        territories = get_territories()
        territory_sel = st.selectbox("Territory (optional)", ["All"] + territories, key="dr_territory")
    with col_d:
        territory_filter = None if territory_sel == "All" else territory_sel
        doctors = get_doctors_by_territory(territory_filter) if territory_filter else []
        doctor_opts = {f"{d['doctor_name']} ({d['doctor_id']})": d["doctor_id"] for d in doctors}
        doctor_sel_label = st.selectbox("Doctor (optional)", ["All Doctors"] + list(doctor_opts.keys()), key="dr_doctor")
    doctor_id_param = None if doctor_sel_label == "All Doctors" else doctor_opts.get(doctor_sel_label)

    if st.button("Load Doctor Review", type="primary"):
        with st.spinner("Loading..."):
            data = get_doctor_review(doctor_id=doctor_id_param, territory=territory_filter)
        if not data:
            return
        if doctor_id_param and "doctor_name" in data:
            st.subheader(data.get("doctor_name", ""))
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Conv", f"{data.get('conv_rate', 0):.0%}")
            m2.metric("Interest", f"{data.get('avg_interest', 0):.1f}/5")
            m3.metric("Follow-up", f"{data.get('follow_up_rate', 0):.0%}")
            m4.metric("LTV", f"${data.get('ltv', 0):,}")
            st.caption(f"Score: {data.get('doctor_score',0):.2f} ({data.get('tier','').upper()})")
        else:
            doctors_list = data.get("doctors", [])
            if doctors_list:
                import pandas as pd
                st.dataframe(pd.DataFrame(doctors_list), use_container_width=True)


def employee_reports_section():
    st.header("Employee Reports")
    col_t, col_e = st.columns(2)
    with col_t:
        territories = get_territories()
        territory_sel = st.selectbox("Territory (optional)", ["All"] + territories, key="er_territory")
    with col_e:
        territory_filter = None if territory_sel == "All" else territory_sel
        employees = get_employees(territory=territory_filter)
        emp_opts = {f"{e.get('employee_name','')} ({e.get('employee_id','')})": e.get("employee_id") for e in employees}
        emp_sel_label = st.selectbox("Employee (optional)", ["Team Summary"] + list(emp_opts.keys()), key="er_emp")
    emp_id_param = None if emp_sel_label == "Team Summary" else emp_opts.get(emp_sel_label)

    if st.button("Load Employee Report", type="primary"):
        with st.spinner("Loading..."):
            data = get_employee_report(employee_id=emp_id_param, territory=territory_filter)
        if not data:
            return
        if emp_id_param and "employee_name" in data:
            st.subheader(data.get("employee_name", ""))
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Visits", data.get("total_visits", 0))
            m2.metric("Conv", f"{data.get('conv_rate', 0):.0%}")
            m3.metric("Avg Duration", f"{data.get('avg_duration_sec', 0):.0f}s")
            m4.metric("Score", f"{data.get('emp_score', 0):.2f}")
        else:
            team = data.get("team", [])
            if team:
                import pandas as pd
                st.dataframe(pd.DataFrame(team), use_container_width=True)


def main():
    st.sidebar.title("PatGPT")
    menu = st.sidebar.radio(
        "Main Menu",
        ["Sales Assistant", "Product Performance", "Doctor Review", "Employee Reports"],
        index=0,
    )
    if menu == "Sales Assistant":
        sales_assistant_section()
    elif menu == "Product Performance":
        product_performance_section()
    elif menu == "Doctor Review":
        doctor_review_section()
    elif menu == "Employee Reports":
        employee_reports_section()


if __name__ == "__main__":
    main()