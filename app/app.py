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

    emp_types = ["mr", "area manager", "vp", "general manager"]
    st.selectbox("Your Role", emp_types, index=emp_types.index(st.session_state.ps_employee_type),
                 key="ps_role_select", on_change=lambda: setattr(st.session_state, 'ps_employee_type', st.session_state.ps_role_select))

    time_map = {"20 sec": 20, "30 sec": 30, "1 min": 60, "2 min": 120, "5 min": 300}
    selected_time_label = st.select_slider("Visit Duration", options=list(time_map.keys()), value="1 min", key="ps_time_slider")
    st.session_state.ps_selected_time = time_map[selected_time_label]

    if st.button("Submit & Get Recommendations", type="primary"):
        with st.spinner("Loading..."):
            data = get_product_suggestions(
                st.session_state.ps_doctor_id,
                st.session_state.ps_selected_time,
                st.session_state.ps_employee_type,
            )
            if data:
                st.session_state.ps_data = data
                st.session_state.ps_submitted = True
                st.rerun()
            else:
                st.error("Failed to load suggestions.")

    if not st.session_state.ps_submitted:
        st.info("Select a doctor, role, and duration above, then click Submit.")
        return

    data = st.session_state.ps_data
    dr_rating = data.get("doctor_rating", {})
    doc_name = data.get("doctor_name", "Unknown")
    specialty = data.get("specialty", "").replace("_", " ").title()
    score = dr_rating.get("score", 0)
    tier = dr_rating.get("tier", "low")

    st.subheader(f"Doctor: {doc_name}")
    st.caption(f"{specialty} - Score: {score:.2f} ({tier.upper()})")

    last_meet = data.get("last_meeting")
    if last_meet:
        with st.expander("Last Meeting Recap"):
            st.write(f"Date: {last_meet.get('date','')}")
            st.write(f"Product: {last_meet.get('product','')}")
            st.write(f"Interest: {last_meet.get('interest_level','')}/5")
            st.write(f"Outcome: {last_meet.get('outcome','')}")
            st.write(f"Objection: {last_meet.get('objection','none')}")

    products = data.get("products", [])
    if products:
        st.subheader(f"Top {len(products)} Product(s) for {data.get('effective_time','')} sec")
        for idx, p in enumerate(products, 1):
            dur = p.get("suggested_duration_sec", 0)
            aida = p.get("aida_stage", "unknown")
            score_val = p.get("score", 0)
            st.write(f"{idx}. {p.get('product_name','')}  | Score: {score_val:.2%}  | Suggested: {dur}s  | AIDA: {aida}")
    else:
        st.info("No product recommendations found.")

    rule_sugg = data.get("rule_based_suggestion", "")
    if rule_sugg:
        st.markdown(f"**Rule-Based Suggestion:** {rule_sugg}")

    if st.button("Customize with AI"):
        with st.spinner("Generating custom suggestion..."):
            custom = get_custom_suggestion(
                st.session_state.ps_doctor_id,
                st.session_state.ps_selected_time,
                st.session_state.ps_employee_type,
            )
            if custom:
                st.success(custom)
            else:
                st.warning("AI suggestion not available, using rule-based.")


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