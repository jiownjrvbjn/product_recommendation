"""
app/app.py — PatGPT AI Sales Assistant v6.0
=============================================

HOW TO RUN (Windows — avoids Application Control block on streamlit.exe):
    python -m streamlit run app/app.py
    python -m streamlit run app/app.py --server.port 8501

New in v6.0:
  - Churn Risk panel in Sales Assistant (uses /analytics/doctor/{id}/churn_risk)
  - Employee Skill Decomposition in Employee Reports (uses /analytics/employee/{id}/skill)
  - Feedback logging after AI Playbook (uses POST /feedback/{interaction_id})
  - ML model status badge in sidebar (uses /health)
  - Persona drift flag shown in Doctor Profile
  - All existing sections 100% preserved
"""

import html

import pandas as pd
import requests
import streamlit as st

BASE_URL = "http://localhost:8000"

st.set_page_config(page_title="PatGPT", layout="wide", initial_sidebar_state="expanded")


# ─────────────────────────────────────────────────────────────────────────────
# API HELPERS  (all existing + new ML endpoints)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def get_territories():
    try:
        r = requests.get(f"{BASE_URL}/territories", timeout=10)
        r.raise_for_status()
        return r.json().get("territories", [])
    except Exception:
        return []


@st.cache_data(ttl=300)
def get_doctors_by_territory(territory: str):
    try:
        r = requests.get(f"{BASE_URL}/doctors", params={"territory": territory}, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception:
        return []


@st.cache_data(ttl=300)
def get_products():
    try:
        r = requests.get(f"{BASE_URL}/products", timeout=10)
        r.raise_for_status()
        return r.json().get("products", [])
    except Exception:
        return []


@st.cache_data(ttl=300)
def get_employees(territory=None):
    try:
        params = {"territory": territory} if territory else {}
        r = requests.get(f"{BASE_URL}/employees", params=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception:
        return []


def get_doctor_analytics(doctor_id, time_sec, employee_type):
    try:
        r = requests.get(
            f"{BASE_URL}/analytics/doctor/{doctor_id}",
            params={"time_sec": time_sec, "employee_type": employee_type},
            timeout=30,
        )
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def get_meeting_playbook(doctor_id, time_sec, employee_type):
    try:
        r = requests.get(
            f"{BASE_URL}/llm/meeting_playbook/{doctor_id}",
            params={"time_sec": time_sec, "employee_type": employee_type},
            timeout=60,
        )
        if r.status_code == 503:
            return "❌ AI service unavailable."
        r.raise_for_status()
        return r.json().get("playbook", "")
    except Exception:
        return "❌ Error generating playbook."


def get_product_performance(product=None, territory=None, quarter=None):
    try:
        params = {}
        if product:    params["product"]    = product
        if territory:  params["territory"]  = territory
        if quarter:    params["quarter"]    = quarter
        r = requests.get(f"{BASE_URL}/analytics/product_performance", params=params, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def get_product_aida(doctor_id: str, time_sec: int, employee_type: str):
    try:
        r = requests.get(
            f"{BASE_URL}/analytics/doctor/{doctor_id}/product_aida",
            params={"time_sec": time_sec, "employee_type": employee_type},
        )
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def get_region_matrix(quarter=None):
    try:
        params = {"quarter": quarter} if quarter else {}
        r = requests.get(f"{BASE_URL}/analytics/region_product_matrix", params=params, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def get_doctor_review(doctor_id=None, territory=None):
    try:
        params = {}
        if doctor_id:  params["doctor_id"]  = doctor_id
        if territory:  params["territory"]  = territory
        r = requests.get(f"{BASE_URL}/analytics/doctor_review", params=params, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def get_doctor_overview():
    try:
        r = requests.get(f"{BASE_URL}/analytics/doctor_overview", timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def get_employee_report(employee_id=None, territory=None):
    try:
        params = {}
        if employee_id: params["employee_id"] = employee_id
        if territory:   params["territory"]   = territory
        r = requests.get(f"{BASE_URL}/analytics/employee_report", params=params, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def get_product_performance_summary(territory=None, quarter=None):
    try:
        params = {}
        if territory: params["territory"] = territory
        if quarter:   params["quarter"]   = quarter
        r = requests.get(f"{BASE_URL}/analytics/product_performance/summary", params=params, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def get_product_detail(product_name, territory=None, quarter=None):
    try:
        params = {}
        if territory: params["territory"] = territory
        if quarter:   params["quarter"]   = quarter
        r = requests.get(
            f"{BASE_URL}/analytics/product_performance/detail/{requests.utils.quote(product_name)}",
            params=params,
            timeout=30,
        )
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def get_region_breakdown(product=None, quarter=None):
    try:
        params = {}
        if product: params["product"] = product
        if quarter: params["quarter"] = quarter
        r = requests.get(f"{BASE_URL}/analytics/region_breakdown", params=params, timeout=30)
        r.raise_for_status()
        return r.json().get("regional_breakdown", [])
    except Exception:
        return []


def get_trend_analysis(territory=None, quarter=None):
    try:
        params = {}
        if territory: params["territory"] = territory
        if quarter:   params["quarter"]   = quarter
        r = requests.get(f"{BASE_URL}/analytics/trend_analysis", params=params, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}


def get_product_ai_analysis(product_name, territory=None, quarter=None):
    try:
        params = {}
        if territory: params["territory"] = territory
        if quarter:   params["quarter"]   = quarter
        r = requests.get(
            f"{BASE_URL}/llm/product_ai_analysis/{requests.utils.quote(product_name)}",
            params=params,
            timeout=60,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def get_llm_product_insight(product_name):
    try:
        r = requests.get(f"{BASE_URL}/llm/product_insight/{product_name}", timeout=30)
        r.raise_for_status()
        return r.json().get("explanation", "No explanation available.")
    except Exception as e:
        return f"Error: {str(e)}"


# ── NEW ML endpoints ──────────────────────────────────────────────────────────

def get_churn_risk(doctor_id: str):
    """GET /analytics/doctor/{id}/churn_risk"""
    try:
        r = requests.get(f"{BASE_URL}/analytics/doctor/{doctor_id}/churn_risk", timeout=15)
        if r.status_code == 503:
            return None   # model not loaded yet
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def get_employee_skill(employee_id: str):
    """GET /analytics/employee/{id}/skill"""
    try:
        r = requests.get(f"{BASE_URL}/analytics/employee/{employee_id}/skill", timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def post_feedback(interaction_id: str, playbook_used: str, outcome: str,
                  interest_after: int, notes: str = "") -> bool:
    """POST /feedback/{interaction_id}"""
    try:
        r = requests.post(
            f"{BASE_URL}/feedback/{interaction_id}",
            json={
                "playbook_used":  playbook_used,
                "outcome":        outcome,
                "interest_after": interest_after,
                "notes":          notes,
            },
            timeout=10,
        )
        return r.status_code == 200
    except Exception:
        return False


def get_health():
    try:
        r = requests.get(f"{BASE_URL}/health", timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# SHARED UI HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _trend_badge(trend: str) -> str:
    return {"improving": "🟢 Improving", "declining": "🔴 Declining"}.get(trend, "🟡 Stable")


def _delta_color(val: float) -> str:
    return "normal" if val >= 0 else "inverse"


def _pct(v):  return f"{v:.1%}"
def _int(v):  return f"{int(v):,}"


def _kpi_row(data: dict):
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total Sales",     _int(data.get("total_sales", 0)))
    c2.metric("Interactions",    _int(data.get("total_interactions", 0)))
    c3.metric("Unique Doctors",  _int(data.get("unique_doctors", 0)))
    c4.metric("Conversion Rate", _pct(data.get("conversion_rate", 0)))
    c5.metric("Avg Interest",    f"{data.get('avg_interest', 0):.1f}/5")
    c6.metric(
        "QoQ Growth",
        _pct(data.get("qoq_growth", 0)),
        delta=_pct(data.get("qoq_growth", 0)),
        delta_color=_delta_color(data.get("qoq_growth", 0)),
    )


def _churn_tier_badge(tier: str) -> str:
    return {
        "critical": "🔴 Critical",
        "high":     "🟠 High",
        "medium":   "🟡 Medium",
        "low":      "🟢 Low",
    }.get(str(tier).lower(), "⚪ Unknown")


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — model status + nav
# ─────────────────────────────────────────────────────────────────────────────

def _render_sidebar():
    st.sidebar.title("PatGPT")

    # ML model health badge
    health = get_health()
    if health:
        model_status = health.get("models", {})
        if model_status:
            all_ok = all(v == "ok" for v in model_status.values())
            badge  = "🟢 All ML models loaded" if all_ok else "🟡 Some models missing"
            with st.sidebar.expander(badge, expanded=False):
                for name, status in model_status.items():
                    icon = "✅" if status == "ok" else "❌"
                    st.write(f"{icon} `{name}`")
        st.sidebar.caption(
            f"Rows: {health.get('rows', '—')} | "
            f"Doctors: {health.get('doctors', '—')} | "
            f"Products: {health.get('products', '—')}"
        )

    st.sidebar.divider()
    menu = st.sidebar.radio(
        "Menu",
        ["Sales Assistant", "Product Performance", "Doctor Review", "Employee Reports"],
    )
    return menu


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — SALES ASSISTANT
# ─────────────────────────────────────────────────────────────────────────────

def sales_assistant_section():
    st.header("Sales Assistant")

    for key, default in [
        ("ps_data",          None),
        ("ps_doctor_id",     None),
        ("ps_doctor_name",   None),
        ("ps_employee_type", "mr"),
        ("ps_selected_time", 60),
        ("ps_submitted",     False),
        ("doctor_analytics", None),
        ("product_aida",     None),
        ("churn_data",       None),
        ("playbook_shown",   False),
        ("playbook_text",    ""),
        ("feedback_sent",    False),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    territories = get_territories()
    if not territories:
        st.warning("No territories found. Is the server running?")
        st.code("python api/server.py", language="bash")
        return

    col_t, col_d = st.columns(2)
    with col_t:
        territory = st.selectbox(
            "Select Territory",
            territories,
            index=None,
            placeholder="Choose a territory...",
        )
    with col_d:
        if territory:
            doctors = get_doctors_by_territory(territory)
            if doctors:
                doctor_options = {
                    f"{d['doctor_name']} (ID: {d['doctor_id']})": d["doctor_id"]
                    for d in doctors
                }
                selected_label = st.selectbox(
                    "Select Doctor",
                    list(doctor_options.keys()),
                    index=None,
                    placeholder="Choose a doctor...",
                )
                if selected_label:
                    new_id   = str(doctor_options[selected_label]).strip()
                    new_name = selected_label.split(" (ID:")[0]
                    if new_id != st.session_state.ps_doctor_id:
                        st.session_state.ps_doctor_id     = new_id
                        st.session_state.ps_doctor_name   = new_name
                        st.session_state.doctor_analytics = None
                        st.session_state.product_aida     = None
                        st.session_state.churn_data       = None
                        st.session_state.playbook_shown   = False
                        st.session_state.playbook_text    = ""
                        st.session_state.feedback_sent    = False
                else:
                    st.session_state.ps_doctor_id = None
            else:
                st.info("No doctors in this territory.")
                return
        else:
            st.empty()

    if not st.session_state.ps_doctor_id:
        return

    # Auto-load analytics
    if st.session_state.doctor_analytics is None:
        with st.spinner("Loading doctor insights…"):
            analytics = get_doctor_analytics(
                st.session_state.ps_doctor_id,
                st.session_state.ps_selected_time,
                st.session_state.ps_employee_type,
            )
            paida = get_product_aida(
                st.session_state.ps_doctor_id,
                st.session_state.ps_selected_time,
                st.session_state.ps_employee_type,
            )
            churn = get_churn_risk(st.session_state.ps_doctor_id)

        if analytics:
            st.session_state.doctor_analytics = analytics
            st.session_state.product_aida     = paida
            st.session_state.churn_data       = churn
        else:
            st.error("Could not load doctor data. Check that the server is running.")
            return

    da       = st.session_state.doctor_analytics
    doc_info = da.get("doctor_info", {})
    eng      = da.get("engagement_metrics", {})
    scoring  = da.get("doctor_scoring", {})
    aida     = da.get("aida", {})
    persona  = da.get("persona", {})
    reco     = da.get("recommendations", {})
    lm       = da.get("last_meeting", {})
    top_hist = da.get("top_historical_products", [])
    paida    = st.session_state.product_aida or {}
    churn    = st.session_state.churn_data or {}

    st.divider()

    # ── CARD 1 — Doctor Profile ───────────────────────────────────────────────
    st.markdown("**Doctor Profile**")
    with st.container(border=True):
        dr_name = doc_info.get("doctor_name", "—")
        st.markdown(f"#### Dr. {dr_name}")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Specialty",          doc_info.get("specialty", "N/A"))
        c2.metric("Total Interactions", eng.get("total_interactions", 0))

        avg_sec = eng.get("avg_meeting_duration_sec") or 0
        avg_min = round(avg_sec / 60, 1)
        with c3:
            st.metric("Avg Meeting", f"{avg_min} min")

        # ── NEW: Persona with drift flag ─────────────────────────────────
        with c4:
            persona_label = persona.get("label", "—")
            drift         = persona.get("persona_drift", False)
            drift_badge   = " 🔄 Drifted" if drift else ""
            st.metric("Persona", f"{persona_label}{drift_badge}")

    # ── CARD 2 — Doctor Rating + Churn Risk ──────────────────────────────────
    st.divider()
    with st.container(border=True):
        st.markdown("#### Doctor Rating")
        rc1, rc2, rc3, rc4, rc5 = st.columns(5, gap="medium")

        rc1.metric("Experience",     f"{doc_info.get('experience_years', 0)} yrs")
        rc2.metric("Patient Load",   f"{doc_info.get('patient_load', 0):,}")
        rc3.metric("Followers",      f"{doc_info.get('social_media_reach', 0):,}")
        rc4.metric("Research Papers", doc_info.get("publications_count", 0))
        rc5.metric("Conversion Rate", f"{eng.get('conversion_rate', 0) * 100:.0f}%")

        score     = scoring.get("score", 0)
        tier      = scoring.get("tier", "low").upper()
        tier_color = {"HIGH": "🟢", "MEDIUM": "🟡", "LOW": "🔴"}.get(tier, "⚪")
        score_pct  = int(score * 100)
        bar_color  = "#10b981" if tier == "HIGH" else ("#f59e0b" if tier == "MEDIUM" else "#ef4444")

        st.markdown(
            f"""
            <div style="margin-top:8px;margin-bottom:4px;">
                <div style="display:flex;align-items:center;gap:10px;">
                    <span style="font-size:13px;font-weight:600;min-width:90px;">
                        {tier_color} {tier} TIER
                    </span>
                    <div style="flex:1;background:#e2e8f0;border-radius:8px;height:10px;overflow:hidden;">
                        <div style="width:{score_pct}%;background:{bar_color};
                                    height:100%;border-radius:8px;transition:width 0.4s;"></div>
                    </div>
                    <span style="font-size:13px;color:#64748b;">{score:.2f} / 1.00</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── NEW: Churn Risk Panel ────────────────────────────────────────────────
    if churn:
        st.divider()
        with st.container(border=True):
            st.markdown("#### Disengagement Risk (ML Model)")
            risk_score = churn.get("risk_score", 0)
            risk_tier  = churn.get("risk_tier", "unknown")
            tier_badge = _churn_tier_badge(risk_tier)

            cr1, cr2 = st.columns([1, 3])
            with cr1:
                st.metric("Risk Tier", tier_badge)
                st.metric("Risk Score", f"{risk_score:.0%}")
            with cr2:
                components = churn.get("components", {})
                if components:
                    comp_df = pd.DataFrame(
                        [{"Feature": k, "Value": round(float(v), 3)}
                         for k, v in components.items()]
                    )
                    st.dataframe(comp_df, use_container_width=True, hide_index=True)

            if risk_tier in ("high", "critical"):
                st.warning(
                    f"⚠️ This doctor is showing signs of disengagement. "
                    f"Consider a re-engagement visit within the next 2 weeks."
                )

    # ── CARD 3 — Last Meeting + Top Products ─────────────────────────────────
    st.divider()
    with st.container(border=True):
        st.markdown("#### Meeting History & Top Products")

        if lm and lm.get("date"):
            lm_col, _ = st.columns([3, 1])
            with lm_col:
                interest_stars = "⭐" * int(lm.get("interest_level", 0))
                outcome_icon   = {"positive": "✅", "negative": "❌", "neutral": "⏳"}.get(
                    lm.get("outcome", "neutral"), "⏳"
                )
                lm_time_min = round((lm.get("actual_time_seconds") or 0) / 60, 1)
                st.markdown(
                    f"""
                    <div style="background:#f8fafc;border-left:4px solid #2748f1;
                                padding:10px 14px;border-radius:6px;margin-bottom:12px;">
                        <span style="font-size:12px;color:#2748f1;">🕒 Last Meeting — <b>{lm.get('date')}</b></span><br>
                        <span style="font-size:14px;color:#1e293b;">
                            <b>{lm.get('product','—')}</b> &nbsp;·&nbsp;
                            {outcome_icon} {lm.get('outcome','').title()} &nbsp;·&nbsp;
                            {interest_stars} &nbsp;·&nbsp;
                            ⏱ {lm_time_min} min
                        </span>
                        {f'<br><span style="font-size:12px;color:#64748b;">⚠ Objection: {lm.get("objection")}</span>'
                          if lm.get("objection") and lm.get("objection") not in ("none","nan","") else ""}
                        {f'<br><span style="font-size:12px;color:#475569;">Notes: {html.escape(str(lm.get("meeting_notes",""))[:120])}</span>'
                          if lm.get("meeting_notes") else ""}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.caption("No previous meeting recorded.")

        st.markdown("**Top 3 Products This Doctor Engages With**")
        if top_hist:
            tp_cols = st.columns(min(len(top_hist), 3))
            for idx, p in enumerate(top_hist[:3]):
                prod_avg_min = round((p.get("avg_time_per_presentation") or 0) / 60, 1)
                with tp_cols[idx]:
                    st.markdown(
                        f"""
                        <div style="background:#f1f5f9;border-radius:10px;padding:12px;
                                    text-align:center;border:1px solid #e2e8f0;margin-bottom:12px;">
                            <div style="font-size:13px;font-weight:700;color:#1e293b;
                                        margin-bottom:4px;">{p['product_name']}</div>
                            <div style="font-size:12px;color:#64748b;">
                                Presented <b>{p['times_presented']}×</b>
                            </div>
                            <div style="font-size:20px;font-weight:700;color:#6366f1;
                                        margin:4px 0;">{prod_avg_min}
                                <span style="font-size:12px;">min avg</span>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
        else:
            st.caption("No historical product data.")

    # ── CARD 4 — AIDA per Product ─────────────────────────────────────────────
    st.divider()
    primary_products = reco.get("primary_products", [])

    with st.container(border=True):
        st.markdown("#### AIDA Stage — Recommended Products")

        stage           = aida.get("aida_stage", "awareness")
        stage_label     = aida.get("aida_label", "Awareness")
        aida_color      = aida.get("aida_color", "#64748B")
        confidence      = int((aida.get("aida_confidence", 0)) * 100)
        all_stages      = aida.get("all_stages", ["awareness", "interest", "desire", "action"])
        stage_idx       = aida.get("aida_stage_index", 0)
        stage_emojis    = aida.get("stage_emojis", {
            "awareness": "👁️", "interest": "🔍", "desire": "🔥", "action": "✅"
        })
        stage_labels_map = aida.get("stage_labels", {
            "awareness": "Awareness", "interest": "Interest",
            "desire": "Desire", "action": "Action",
        })
        stage_colors_map = aida.get("stage_colors", {
            "awareness": "#64748B", "interest": "#F59E0B",
            "desire": "#8B5CF6",    "action":  "#10B981",
        })

        funnel_html = '<div style="display:flex;gap:8px;align-items:center;margin-bottom:14px;">'
        for i, s in enumerate(all_stages):
            active    = i <= stage_idx
            dot_color = stage_colors_map.get(s, "#94a3b8") if active else "#e2e8f0"
            fnt_color = "#fff" if active else "#94a3b8"
            funnel_html += (
                f'<div style="background:{dot_color};color:{fnt_color};border-radius:20px;'
                f'padding:4px 12px;font-size:12px;font-weight:600;">'
                f'{stage_emojis.get(s,"")} {stage_labels_map.get(s,"").upper()}</div>'
            )
            if i < len(all_stages) - 1:
                funnel_html += '<span style="color:#cbd5e1;font-size:16px;">›</span>'
        funnel_html += (
            f'</div><p style="font-size:12px;color:#64748b;">'
            f'Overall doctor stage — confidence <b>{confidence}%</b></p>'
        )
        st.markdown(funnel_html, unsafe_allow_html=True)

        product_aida_list = paida.get("product_aida", [])
        if product_aida_list:
            pa_cols = st.columns(min(len(product_aida_list), 3))
            for idx, pa in enumerate(product_aida_list[:3]):
                p_color      = pa.get("aida_color", "#64748B")
                p_emoji      = pa.get("aida_emoji", "👁️")
                p_label      = pa.get("aida_label", "Awareness")
                p_conv       = pa.get("conversion_rate", 0)
                p_interest   = pa.get("avg_interest", 0)
                p_confidence = int(pa.get("aida_confidence", 0) * 100)
                with pa_cols[idx]:
                    st.markdown(
                        f"""
                        <div style="border:1px solid {p_color};border-radius:10px;
                                    padding:12px;text-align:center;margin-bottom:12px;">
                            <div style="font-size:13px;font-weight:700;margin-bottom:6px;">
                                {pa['product_name']}</div>
                            <div style="background:{p_color};color:#fff;border-radius:16px;
                                        padding:3px 10px;display:inline-block;
                                        font-size:12px;font-weight:600;margin-bottom:6px;">
                                {p_emoji} {p_label}
                            </div>
                            <div style="font-size:12px;color:#64748b;">
                                Conv: <b>{p_conv:.0%}</b> &nbsp;·&nbsp;
                                Interest: <b>{p_interest:.1f}/5</b>
                            </div>
                            <div style="font-size:11px;color:#94a3b8;margin-top:3px;">
                                Confidence: {p_confidence}%
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
        elif primary_products:
            pp_cols = st.columns(min(len(primary_products), 3))
            for idx, p in enumerate(primary_products[:3]):
                with pp_cols[idx]:
                    st.markdown(
                        f"""
                        <div style="border:1px solid {aida_color};border-radius:10px;
                                    padding:12px;text-align:center;margin-bottom:12px;">
                            <div style="font-size:13px;font-weight:700;margin-bottom:6px;">
                                {p['product_name']}</div>
                            <div style="background:{aida_color};color:#fff;border-radius:16px;
                                        padding:3px 10px;display:inline-block;
                                        font-size:12px;font-weight:600;margin-bottom:6px;">
                                {aida.get('aida_emoji','👁️')} {stage_label}
                            </div>
                            <div style="font-size:12px;color:#64748b;">
                                Conv: <b>{p.get('conversion_rate',0):.0%}</b> &nbsp;·&nbsp;
                                Interest: <b>{p.get('avg_interest',0):.1f}/5</b>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
        else:
            st.caption("No product recommendations available.")

    # ── CARD 5 — AI Playbook + Feedback ──────────────────────────────────────
    st.divider()
    with st.container(border=True):
        st.markdown("#### Create Personalised AI Playbook?")
        st.markdown(
            f"Want a **personalised AI playbook** for today's meeting with "
            f"**Dr. {doc_info.get('doctor_name', '—')}**? "
            f"Based on their **{stage_label}** stage, persona, past objections, and top products."
        )

        col_btn, col_note = st.columns([1, 3])
        with col_btn:
            generate = st.button("Generate AI Playbook", type="primary")
        with col_note:
            st.caption("Uses Azure OpenAI. Takes ~5-10 seconds.")

        if generate:
            with st.spinner("Generating AI playbook…"):
                playbook = get_meeting_playbook(
                    st.session_state.ps_doctor_id,
                    st.session_state.ps_selected_time,
                    st.session_state.ps_employee_type,
                )
            st.session_state.playbook_text  = playbook
            st.session_state.playbook_shown = True
            st.session_state.feedback_sent  = False

        if st.session_state.playbook_shown and st.session_state.playbook_text:
            st.markdown("---")
            st.markdown("**Today's Meeting Playbook**")
            st.markdown(st.session_state.playbook_text)

            # ── NEW: Feedback collection ──────────────────────────────────
            if not st.session_state.feedback_sent:
                st.divider()
                st.markdown("**How did the visit go?** *(Optional — helps improve AI quality)*")
                fb_col1, fb_col2, fb_col3 = st.columns(3)
                with fb_col1:
                    fb_outcome = st.selectbox(
                        "Visit Outcome",
                        ["positive", "neutral", "negative"],
                        key="fb_outcome",
                    )
                with fb_col2:
                    fb_interest = st.slider(
                        "Doctor Interest After Visit",
                        min_value=1, max_value=5, value=3,
                        key="fb_interest",
                    )
                with fb_col3:
                    fb_notes = st.text_input("Notes (optional)", key="fb_notes")

                if st.button("Submit Feedback", key="fb_submit"):
                    # Use doctor_id as interaction_id proxy (best-effort)
                    sent = post_feedback(
                        interaction_id=st.session_state.ps_doctor_id,
                        playbook_used=st.session_state.playbook_text[:500],
                        outcome=fb_outcome,
                        interest_after=fb_interest,
                        notes=fb_notes,
                    )
                    if sent:
                        st.success("Feedback recorded — thank you!")
                        st.session_state.feedback_sent = True
                    else:
                        st.warning("Could not save feedback (server may be offline).")
            else:
                st.success("Feedback already submitted for this playbook.")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — PRODUCT PERFORMANCE  (unchanged from v5)
# ─────────────────────────────────────────────────────────────────────────────

def product_performance_section():
    st.header("Product Performance")

    try:
        products_list    = requests.get(f"{BASE_URL}/products",    timeout=10).json().get("products",    [])
        territories_list = requests.get(f"{BASE_URL}/territories", timeout=10).json().get("territories", [])
    except Exception:
        products_list    = []
        territories_list = []

    col_p, col_r, col_q = st.columns(3)
    with col_p:
        product_sel = st.selectbox("Product", ["— Overall Summary —"] + sorted(products_list), key="pp_product")
    with col_r:
        territory_sel = st.selectbox("Territory / Region", ["All"] + sorted(territories_list), key="pp_territory")
    with col_q:
        quarter_sel = st.selectbox("Quarter", ["All", "Q1", "Q2", "Q3", "Q4"], key="pp_quarter")

    product_param   = None if product_sel   == "— Overall Summary —" else product_sel
    territory_param = None if territory_sel == "All"                 else territory_sel
    quarter_param   = None if quarter_sel   == "All"                 else quarter_sel

    st.divider()
    if product_param:
        _render_product_detail(product_param, territory_param, quarter_param)
    else:
        _render_overall_summary(territory_param, quarter_param)


def _render_overall_summary(territory_param, quarter_param):
    st.markdown("### Portfolio Overview")
    with st.spinner("Loading portfolio data…"):
        summary_data = get_product_performance_summary(territory=territory_param, quarter=quarter_param)
        trend_data   = get_trend_analysis(territory=territory_param, quarter=quarter_param)
        region_data  = get_region_breakdown(quarter=quarter_param)

    if not summary_data or not summary_data.get("summary"):
        st.info("No data found for the selected filters.")
        return

    df_summary = pd.DataFrame(summary_data["summary"])

    if trend_data:
        port_trend = trend_data.get("portfolio_trend", "stable")
        growing    = trend_data.get("top_growing_products",   [])
        declining  = trend_data.get("top_declining_products", [])
        with st.container(border=True):
            st.markdown(f"**Portfolio Trend: {_trend_badge(port_trend)}**")
            tc1, tc2 = st.columns(2)
            with tc1:
                st.markdown("**Top Growing Products**")
                for p in growing:
                    st.markdown(f"- {p['product']} `slope: {p['slope']:.1f}`")
            with tc2:
                st.markdown("**Top Declining Products**")
                for p in declining:
                    st.markdown(f"- {p['product']} `slope: {p['slope']:.1f}`")
        pm = trend_data.get("portfolio_monthly_sales", [])
        if pm:
            pm_df = pd.DataFrame(pm).set_index("month")
            st.markdown("**Monthly Portfolio Sales**")
            st.line_chart(pm_df["sales_volume"])

    st.divider()
    st.markdown("**Product Summary Table**")

    def _highlight_underperf(row):
        color = "background-color:#fef2f2;color:#000000;" if row.get("underperformance_flag") else ""
        return [color] * len(row)

    display_cols = [c for c in [
        "product", "total_sales", "total_interactions", "unique_doctors",
        "conversion_rate", "avg_interest", "follow_up_rate", "qoq_growth",
        "trend", "top_region", "underperformance_flag",
    ] if c in df_summary.columns]

    st.dataframe(
        df_summary[display_cols].style.apply(_highlight_underperf, axis=1),
        use_container_width=True,
    )

    underperf = (
        df_summary[df_summary["underperformance_flag"] == True]
        if "underperformance_flag" in df_summary.columns else pd.DataFrame()
    )
    if not underperf.empty:
        st.warning(f"⚠️ {len(underperf)} product(s) flagged as underperforming.")
        with st.expander("View underperforming products"):
            for _, row in underperf.iterrows():
                st.markdown(
                    f'<div style="background:#fef2f2;color:#000000;border-left:4px solid #ef4444;'
                    f'padding:8px 12px;border-radius:6px;margin-bottom:6px;">'
                    f'<b>{row["product"]}</b> — Conv: {row.get("conversion_rate",0):.0%}'
                    f' | Trend: {row.get("trend","—")} | QoQ: {row.get("qoq_growth",0):.1%}</div>',
                    unsafe_allow_html=True,
                )

    if region_data:
        st.divider()
        st.markdown("**Regional Breakdown**")
        st.dataframe(pd.DataFrame(region_data), use_container_width=True)

    st.divider()
    st.markdown("**Region × Product Sales Heatmap**")
    matrix = get_region_matrix(quarter=quarter_param)
    if matrix and matrix.get("matrix"):
        df_hm = pd.DataFrame(matrix["matrix"])
        if "region" in df_hm.columns:
            df_hm = df_hm.set_index("region")
            st.dataframe(df_hm.style.background_gradient(cmap="Blues"), use_container_width=True)
        else:
            st.info("No region data in matrix.")
    else:
        st.info("No region-product matrix available.")


def _render_product_detail(product_name, territory_param, quarter_param):
    st.markdown(f"### {product_name} — Individual Report")
    with st.spinner(f"Loading data for {product_name}…"):
        data = get_product_detail(product_name, territory=territory_param, quarter=quarter_param)

    if not data:
        st.error("Could not load product detail.")
        return
    if "error" in data:
        st.error(data["error"])
        return

    with st.container(border=True):
        flag_badge = " ⚠️ **Underperforming**" if data.get("is_underperforming") else " ✅ On Track"
        st.markdown(
            f"**Trend:** {_trend_badge(data.get('trend','stable'))} &nbsp;|&nbsp; "
            f"**Status:** {flag_badge} &nbsp;|&nbsp; "
            f"**Engagement Depth:** {data.get('engagement_depth',0):.1f} visits/doctor &nbsp;|&nbsp; "
            f"**Follow-up Rate:** {data.get('follow_up_rate',0):.0%}"
        )
        _kpi_row(data)

    st.divider()
    tab_trend, tab_doctors, tab_region, tab_funnel, tab_obj, tab_qoq = st.tabs([
        "Trends", "Doctors", "Regions", "Funnel & AIDA", "Objections", "Quarterly"
    ])

    with tab_trend:
        col_ms, col_mc = st.columns(2)
        with col_ms:
            st.markdown("**Monthly Sales Volume**")
            ms = data.get("monthly_sales", [])
            if ms:
                st.line_chart(pd.DataFrame(ms).set_index("month")["sales_volume"])
            else:
                st.info("No monthly sales data.")
        with col_mc:
            st.markdown("**Monthly Conversion Rate**")
            mc = data.get("monthly_conversion", [])
            if mc:
                st.line_chart(pd.DataFrame(mc).set_index("month")["conversion_rate"])
            else:
                st.info("No monthly conversion data.")
        st.markdown("**Product Adoption Trend (Cumulative Unique Doctors)**")
        at = data.get("adoption_trend", [])
        if at:
            st.area_chart(pd.DataFrame(at).set_index("month")["cumulative_doctors"])
        else:
            st.info("No adoption trend data.")

    with tab_doctors:
        col_top, col_low = st.columns(2)
        with col_top:
            st.markdown("**Top Prescribing Doctors**")
            top_docs = data.get("top_doctors", [])
            if top_docs:
                st.dataframe(pd.DataFrame(top_docs), use_container_width=True)
            else:
                st.info("No doctor performance data.")
        with col_low:
            st.markdown("**Lowest Prescribing Doctors**")
            low_docs = data.get("lowest_doctors", [])
            if low_docs:
                st.dataframe(pd.DataFrame(low_docs), use_container_width=True)
            else:
                st.info("No data.")
        st.markdown("**Follow-up Analytics**")
        fu = data.get("follow_up_analytics", [])
        if fu:
            st.dataframe(pd.DataFrame(fu), use_container_width=True)

    with tab_region:
        st.markdown("**Region Performance Comparison**")
        rc = data.get("region_comparison", []) or data.get("regional_breakdown", [])
        if rc:
            df_rc = pd.DataFrame(rc)
            st.dataframe(df_rc, use_container_width=True)
            region_col = "region" if "region" in df_rc.columns else df_rc.columns[0]
            if "total_sales" in df_rc.columns:
                st.bar_chart(df_rc.set_index(region_col)["total_sales"])
        else:
            st.info("No regional data available.")

    with tab_funnel:
        col_f, col_a = st.columns(2)
        with col_f:
            st.markdown("**Conversion Funnel**")
            funnel = data.get("conversion_funnel", [])
            if funnel:
                for row in funnel:
                    st.markdown(
                        f'<div style="background:#f1f5f9;color:#000000;border-left:4px solid #6366f1;'
                        f'padding:8px 14px;border-radius:6px;margin-bottom:6px;">'
                        f'<b>{row["stage"]}</b> — <span style="font-size:18px;color:#6366f1;'
                        f'font-weight:700;">{row["count"]:,}</span></div>',
                        unsafe_allow_html=True,
                    )
        with col_a:
            st.markdown("**AIDA Stage Distribution**")
            aida_dist = data.get("aida_distribution", {})
            _STAGE_COLORS = {"awareness":"#64748B","interest":"#F59E0B","desire":"#8B5CF6","action":"#10B981"}
            _STAGE_EMOJI  = {"awareness":"👁️","interest":"🔍","desire":"🔥","action":"✅"}
            for stage, count in aida_dist.items():
                color = _STAGE_COLORS.get(stage, "#94a3b8")
                emoji = _STAGE_EMOJI.get(stage, "•")
                st.markdown(
                    f'<div style="background:{color}18;border-left:4px solid {color};'
                    f'padding:7px 12px;border-radius:6px;margin-bottom:5px;">'
                    f'{emoji} <b style="color:{color};">{stage.title()}</b> — {count} doctor(s)</div>',
                    unsafe_allow_html=True,
                )

    with tab_obj:
        col_ob, col_comp = st.columns(2)
        with col_ob:
            st.markdown("**All Objections**")
            obj = data.get("objection_breakdown", {})
            if obj:
                df_obj = pd.DataFrame(
                    sorted(obj.items(), key=lambda x: -x[1]),
                    columns=["Objection", "Count"],
                )
                st.dataframe(df_obj, use_container_width=True)
                st.bar_chart(df_obj.set_index("Objection")["Count"])
            else:
                st.info("No objections recorded.")
        with col_comp:
            st.markdown("**Competitor-related Objections**")
            comp_obj = data.get("competitor_objections", {})
            if comp_obj:
                df_comp = pd.DataFrame(
                    sorted(comp_obj.items(), key=lambda x: -x[1]),
                    columns=["Objection", "Count"],
                )
                st.dataframe(df_comp, use_container_width=True)
            else:
                st.info("No competitor objections detected.")

    with tab_qoq:
        st.markdown("**QoQ Performance**")
        qoq = data.get("qoq_performance", [])
        if qoq:
            df_q = pd.DataFrame(qoq)
            st.dataframe(df_q, use_container_width=True)
            if "total_sales" in df_q.columns:
                x_col = "quarter" if "quarter" in df_q.columns else df_q.columns[0]
                st.bar_chart(df_q.set_index(x_col)["total_sales"])
        else:
            st.info("No quarterly data available.")

    st.divider()
    _render_ai_analysis_card(product_name, data.get("is_underperforming", False))


def _render_ai_analysis_card(product_name: str, is_underperforming: bool):
    label = "AI Analysis" + (" — ⚠️ Underperformance Detected" if is_underperforming else "")
    with st.expander(label, expanded=False):
        st.markdown(
            f'<div style="background:#eff6ff;color:#000000;border-left:4px solid #3b82f6;'
            f'padding:10px 14px;border-radius:6px;margin-bottom:12px;font-size:13px;">'
            f'{"This product has been flagged as underperforming." if is_underperforming else "This product appears on track."}'
            f'</div>',
            unsafe_allow_html=True,
        )
        if st.button("Generate AI Analysis", key=f"ai_btn_{product_name}", type="primary"):
            with st.spinner("Running AI analysis…"):
                ai = get_product_ai_analysis(product_name)
            st.session_state[f"ai_analysis_{product_name}"] = ai

        ai = st.session_state.get(f"ai_analysis_{product_name}")
        if not ai:
            return
        if "error" in ai:
            st.error(ai["error"])
            return

        st.markdown("#### Root Causes")
        for rc in ai.get("root_causes", []):
            st.markdown(
                f'<div style="background:#fef2f2;border-left:4px solid #ef4444;'
                f'padding:8px 12px;border-radius:6px;margin-bottom:5px;font-size:13px;">❗ {rc}</div>',
                unsafe_allow_html=True,
            )

        st.markdown("#### Recommendations")
        for i, rec in enumerate(ai.get("recommendations", []), 1):
            st.markdown(
                f'<div style="background:#f0fdf4;border-left:4px solid #22c55e;'
                f'padding:8px 12px;border-radius:6px;margin-bottom:5px;font-size:13px;">'
                f'<b>{i}.</b> {rec}</div>',
                unsafe_allow_html=True,
            )

        col_qw, col_nba = st.columns(2)
        with col_qw:
            st.markdown("#### Quick Win")
            st.info(ai.get("quick_win", "—"))
        with col_nba:
            st.markdown("#### Next Best Action")
            st.success(ai.get("next_best_action", "—"))

        ts = ai.get("territory_suggestion")
        if ts:
            st.markdown("#### Territory-Specific Suggestion")
            st.markdown(
                f'<div style="background:#faf5ff;border-left:4px solid #8b5cf6;'
                f'padding:8px 12px;border-radius:6px;font-size:13px;">{ts}</div>',
                unsafe_allow_html=True,
            )
        if ai.get("llm_explanation"):
            st.markdown("#### LLM Deep-Dive")
            st.markdown(ai["llm_explanation"])


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — DOCTOR REVIEW  (unchanged from v5)
# ─────────────────────────────────────────────────────────────────────────────

def view_nba_hint(analysis):
    stage      = analysis.get("aida_stage", "awareness")
    emoji      = analysis.get("aida_emoji",  "👁️")
    label      = analysis.get("aida_label",  "Awareness")
    confidence = int(analysis.get("aida_confidence", 0) * 100)
    hints = {
        "awareness": "Introduce brand, leave brief.",
        "interest":  "Share clinical data, ask about patients.",
        "desire":    "Reinforce with case study, push for trial.",
        "action":    "Upsell complementary product.",
    }
    return f"{emoji} **{label}** (confidence {confidence}%) — {hints.get(stage, 'Engage appropriately.')}"


def doctor_review_section():
    st.header("Doctor Review")
    tab1, tab2 = st.tabs(["All Doctors", "Doctor Analysis"])

    with tab1:
        territory_sel   = st.selectbox("Filter by Territory", ["All"] + get_territories(), key="dr_territory")
        territory_filter = None if territory_sel == "All" else territory_sel
        overview = get_doctor_overview()
        if overview:
            st.subheader("Specialty Average Conversion")
            st.dataframe(pd.DataFrame(overview.get("specialty_avg_conversion", [])))
            st.subheader("Top 5 Doctors")
            st.dataframe(pd.DataFrame(overview.get("top5_doctors", [])))
            st.subheader("Territory Comparison")
            st.dataframe(pd.DataFrame(overview.get("territory_comparison", [])))
        if st.button("Refresh Doctor List", key="dr_refresh"):
            st.rerun()

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            territory_sel2 = st.selectbox("Territory", ["All"] + get_territories(), key="dr_territory2")
        with c2:
            t = None if territory_sel2 == "All" else territory_sel2
            doctors     = get_doctors_by_territory(t) if t else []
            doctor_opts = {f"{d['doctor_name']} ({d['doctor_id']})": d["doctor_id"] for d in doctors}
            doctor_label = st.selectbox("Doctor", ["Select a doctor"] + list(doctor_opts.keys()), key="dr_doctor")

        if doctor_label != "Select a doctor":
            doctor_id = doctor_opts[doctor_label]
            analysis  = get_doctor_review(doctor_id=doctor_id)
            if analysis:
                st.subheader(analysis.get("doctor_name", ""))
                mc1, mc2, mc3, mc4 = st.columns(4)
                mc1.metric("Conv",     f"{analysis.get('conv_rate',0):.1%}")
                mc2.metric("Interest", f"{analysis.get('avg_interest',0):.1f}/5")
                mc3.metric("Follow-up", f"{analysis.get('follow_up_rate',0):.1%}")
                mc4.metric("LTV",      f"Rs {analysis.get('ltv',0):,}")
                st.caption(
                    f"Score: {analysis.get('doctor_score',0):.2f} ({analysis.get('tier','').upper()}) | "
                    f"Territory Rank: #{analysis.get('rank_in_territory','—')} | "
                    f"Specialty Rank: #{analysis.get('rank_in_specialty','—')}"
                )
                with st.expander("Product Affinity"):
                    st.dataframe(pd.DataFrame(analysis.get("product_affinity", [])))
                with st.expander("Objection Intelligence"):
                    obj = analysis.get("objection_intelligence", {})
                    if obj.get("has_objections"):
                        st.write(obj.get("objection_breakdown", {}))
                    else:
                        st.write("No objections recorded.")
                with st.expander("Next Best Action"):
                    st.info(view_nba_hint(analysis))
            else:
                st.error("Doctor not found.")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — EMPLOYEE REPORTS  (+ NEW: Skill Decomposition)
# ─────────────────────────────────────────────────────────────────────────────

def employee_reports_section():
    st.header("Employee Reports")
    tab1, tab2 = st.tabs(["Team Summary", "Individual Report"])

    with tab1:
        territory_sel = st.selectbox("Territory", ["All"] + get_territories(), key="er_territory")
        t = None if territory_sel == "All" else territory_sel
        team_data = get_employee_report(territory=t)
        if team_data and "team" in team_data:
            st.dataframe(pd.DataFrame(team_data["team"]), use_container_width=True)
        else:
            st.info("No team data.")

    with tab2:
        territory_sel2 = st.selectbox("Territory", ["All"] + get_territories(), key="er_territory2")
        t2 = None if territory_sel2 == "All" else territory_sel2
        employees = get_employees(territory=t2)
        emp_opts  = {
            f"{e.get('employee_name','')} ({e.get('employee_id','')})": e.get("employee_id")
            for e in employees
        }
        emp_label = st.selectbox("Employee", ["Select"] + list(emp_opts.keys()), key="er_emp")

        if emp_label != "Select":
            emp_id = emp_opts[emp_label]
            report = get_employee_report(employee_id=emp_id)

            if report and "error" not in report:
                st.subheader(report.get("employee_name", ""))
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Visits",       report.get("total_visits", 0))
                m2.metric("Conv",         f"{report.get('conv_rate',0):.1%}")
                m3.metric("Avg Duration", f"{report.get('avg_duration_sec',0):.0f}s")
                m4.metric("Score",        f"{report.get('emp_score',0):.2f}")

                st.write(f"**Most Successful Product:** {report.get('most_successful_product','N/A')}")
                st.write("**Improvement Areas:**")
                for s in report.get("improvement_suggestions", []):
                    st.markdown(f"- {s}")

                with st.expander("Peer Comparison"):
                    comp = report.get("comparison", {})
                    st.write(f"Conv vs Avg: {comp.get('conv_vs_avg',0):.3f}")
                    st.write(f"Outperforming: {'Yes' if comp.get('outperforming') else 'No'}")

                # ── NEW: ML Skill Decomposition ───────────────────────────
                st.divider()
                st.markdown("#### ML Skill Decomposition")
                with st.spinner("Loading skill scores…"):
                    skill = get_employee_skill(emp_id)

                if skill:
                    sv = skill.get("skill_vector", [0, 0, 0])
                    sk1, sk2, sk3 = st.columns(3)
                    sk1.metric("Opening Score",  f"{skill.get('opening_score',0):.0%}",
                               help="Percentile rank: early interest lift vs territory peers")
                    sk2.metric("Handling Score", f"{skill.get('handling_score',0):.0%}",
                               help="Percentile rank: objection resolution rate")
                    sk3.metric("Closing Score",  f"{skill.get('closing_score',0):.0%}",
                               help="Percentile rank: desire-to-action conversion rate")

                    if skill.get("coaching_priority"):
                        st.warning(
                            "Coaching Priority: This employee is in the bottom 40% overall. "
                            "Consider scheduling a coaching session."
                        )

                    # Skill radar as a simple bar chart
                    skill_df = pd.DataFrame({
                        "Skill":  ["Opening", "Handling", "Closing"],
                        "Score":  [sv[0] if len(sv) > 0 else 0,
                                   sv[1] if len(sv) > 1 else 0,
                                   sv[2] if len(sv) > 2 else 0],
                    }).set_index("Skill")
                    st.bar_chart(skill_df)
                else:
                    st.info("Skill data not available (model may not be loaded).")
            else:
                st.error("Employee not found.")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    menu = _render_sidebar()

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