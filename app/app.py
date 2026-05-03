"""
app.py — PatGPT AI Sales Assistant
- AIDA progress bar with stage guidance
- Employee type selection buttons (MR / Area Manager / VP / GM)
- Product recommendation cards panel (no graph)
- Conversation guide: What to say / show / avoid / next step
- Next Best Action engine output
- Intent score + Visit success probability
- Doctor persona badge
"""

import streamlit as st
import requests
import html

BASE_URL = "http://localhost:8000"

st.set_page_config(
    page_title="PatGPT",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────
# GLOBAL STYLES
# ─────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@600;700;800&family=DM+Sans:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0F0E17;
    color: #FFFFFE;
}

/* Logo */
.logo {
    font-family: 'Syne', sans-serif;
    font-size: 2.6rem;
    font-weight: 800;
    background: linear-gradient(135deg, #6C63FF 0%, #A855F7 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    letter-spacing: -1px;
    margin-bottom: 0.25rem;
}
.logo-sub {
    text-align: center;
    font-size: 0.85rem;
    color: #94A3B8;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 2.5rem;
}

/* Section label */
.section-label {
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    color: #94A3B8;
    margin-bottom: 0.6rem;
    margin-top: 1.4rem;
}

/* Doctor hero card */
.doc-hero {
    background: linear-gradient(135deg, #1A1744 0%, #2D1B69 100%);
    border: 1px solid rgba(108,99,255,0.35);
    border-radius: 20px;
    padding: 1.5rem 1.6rem 1.3rem;
    margin-bottom: 1rem;
    position: relative;
    overflow: hidden;
}
.doc-hero::before {
    content: '';
    position: absolute;
    top: -40px; right: -40px;
    width: 160px; height: 160px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(168,85,247,0.18) 0%, transparent 70%);
}
.doc-hero-name {
    font-family: 'Syne', sans-serif;
    font-size: 1.4rem;
    font-weight: 800;
    color: #FFFFFE;
    margin-bottom: 0.25rem;
}
.doc-hero-meta {
    font-size: 0.82rem;
    color: #C4B5FD;
    margin-bottom: 1rem;
    line-height: 1.7;
}
.doc-hero-stats {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.7rem;
}
.stat-box {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px;
    padding: 0.65rem 0.5rem;
    text-align: center;
}
.stat-box-val { font-family: 'Syne', sans-serif; font-size: 1.05rem; font-weight: 700; color: #FFFFFE; line-height: 1.2; }
.stat-box-lbl { font-size: 0.62rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.1em; color: #94A3B8; margin-top: 0.2rem; }

/* Badges / pills */
.badge-row { display: flex; flex-wrap: wrap; gap: 0.5rem; margin: 0.6rem 0 1rem; }
.badge {
    display: inline-flex; align-items: center; gap: 0.3rem;
    padding: 0.35rem 0.9rem; border-radius: 99px;
    font-size: 0.8rem; font-weight: 700; letter-spacing: 0.02em;
}
.badge-purple { background: rgba(108,99,255,0.18); border: 1px solid rgba(108,99,255,0.4); color: #A78BFA; }
.badge-green  { background: #DCFCE7; color: #15803D; }
.badge-yellow { background: #FEF9C3; color: #A16207; }
.badge-red    { background: #FEE2E2; color: #DC2626; }
.badge-teal   { background: rgba(16,185,129,0.15); border: 1px solid rgba(16,185,129,0.3); color: #34D399; }
.badge-amber  { background: rgba(245,158,11,0.15); border: 1px solid rgba(245,158,11,0.3); color: #FBBF24; }
.badge-slate  { background: rgba(100,116,139,0.15); border: 1px solid rgba(100,116,139,0.3); color: #94A3B8; }

/* AIDA bar */
.aida-wrap {
    background: #16142A;
    border: 1px solid rgba(108,99,255,0.2);
    border-radius: 16px;
    padding: 1.2rem 1.4rem 1rem;
    margin-bottom: 1rem;
}
.aida-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.9rem; font-weight: 700; color: #FFFFFE; margin-bottom: 0.8rem;
}
.aida-bar {
    display: grid; grid-template-columns: repeat(4, 1fr);
    gap: 0.35rem; margin-bottom: 0.9rem;
}
.aida-step {
    text-align: center; padding: 0.55rem 0.3rem;
    border-radius: 10px; border: 1px solid transparent;
    transition: all 0.2s;
}
.aida-step-active {
    border-color: currentColor;
    box-shadow: 0 0 12px rgba(255,255,255,0.1);
}
.aida-step-inactive { opacity: 0.3; }
.aida-emoji { font-size: 1.1rem; line-height: 1; }
.aida-label { font-size: 0.7rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; margin-top: 0.2rem; }
.aida-guidance {
    background: rgba(255,255,255,0.04);
    border-radius: 10px; padding: 0.85rem 1rem;
    border-left: 3px solid;
}
.aida-guidance-row { margin-bottom: 0.5rem; }
.aida-guidance-row:last-child { margin-bottom: 0; }
.aida-guidance-lbl { font-size: 0.7rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: #94A3B8; margin-bottom: 0.15rem; }
.aida-guidance-val { font-size: 0.83rem; color: #CBD5E1; line-height: 1.5; }

/* Product card panel */
.prod-panel {
    background: #16142A;
    border: 1px solid rgba(108,99,255,0.2);
    border-radius: 16px;
    overflow: hidden;
    margin-bottom: 1rem;
}
.prod-panel-header {
    padding: 0.9rem 1.3rem 0.7rem;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    font-family: 'Syne', sans-serif;
    font-size: 0.9rem; font-weight: 700; color: #FFFFFE;
}
.prod-card {
    padding: 0.95rem 1.3rem;
    border-bottom: 1px solid rgba(255,255,255,0.05);
    display: flex; align-items: center; gap: 1rem;
}
.prod-card:last-child { border-bottom: none; }
.prod-rank {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem; font-weight: 800;
    color: rgba(255,255,255,0.15);
    min-width: 1.5rem; text-align: center;
}
.prod-body { flex: 1; }
.prod-name { font-size: 0.95rem; font-weight: 700; color: #FFFFFE; margin-bottom: 0.2rem; }
.prod-meta { font-size: 0.76rem; color: #64748B; }
.prod-badges { display: flex; flex-wrap: wrap; gap: 0.3rem; margin-top: 0.3rem; }
.prod-badge {
    font-size: 0.68rem; font-weight: 700;
    padding: 0.18rem 0.55rem; border-radius: 99px; letter-spacing: 0.04em;
}
.pb-primary  { background: rgba(108,99,255,0.2); border: 1px solid rgba(108,99,255,0.4); color: #A78BFA; }
.pb-support  { background: rgba(16,185,129,0.12); border: 1px solid rgba(16,185,129,0.3); color: #34D399; }
.pb-closing  { background: rgba(245,158,11,0.12); border: 1px solid rgba(245,158,11,0.3); color: #FBBF24; }
.pb-reminder { background: rgba(100,116,139,0.12); border: 1px solid rgba(100,116,139,0.3); color: #94A3B8; }
.prod-score-bar-wrap { min-width: 3.5rem; text-align: right; }
.prod-score-pct { font-family: 'Syne', sans-serif; font-size: 0.9rem; font-weight: 700; color: #FFFFFE; }
.prod-score-sub { font-size: 0.65rem; color: #64748B; }
.prod-score-bar {
    width: 100%; height: 3px; border-radius: 99px;
    background: rgba(255,255,255,0.07); margin-top: 0.3rem;
    overflow: hidden;
}
.prod-score-fill { height: 100%; border-radius: 99px; }

/* NBA card */
.nba-card {
    background: linear-gradient(135deg, #1A1744 0%, #0F172A 100%);
    border: 1px solid rgba(108,99,255,0.35);
    border-radius: 16px;
    padding: 1.1rem 1.3rem;
    margin-bottom: 1rem;
}
.nba-goal { font-size: 0.72rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em; color: #64748B; margin-bottom: 0.5rem; }
.nba-action { font-size: 1rem; font-weight: 700; color: #FFFFFE; margin-bottom: 0.4rem; }
.nba-cta {
    display: inline-block;
    background: linear-gradient(135deg, #6C63FF, #A855F7);
    color: #FFFFFE; font-size: 0.82rem; font-weight: 700;
    padding: 0.4rem 1.1rem; border-radius: 99px;
    margin-top: 0.3rem;
}
.nba-product { font-size: 0.78rem; color: #64748B; margin-top: 0.5rem; }

/* Conversation guide */
.convo-wrap {
    background: #16142A;
    border: 1px solid rgba(108,99,255,0.2);
    border-radius: 16px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
}
.convo-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.9rem; font-weight: 700; color: #FFFFFE; margin-bottom: 0.9rem;
}
.convo-row {
    display: flex; gap: 0.8rem; align-items: flex-start;
    padding: 0.7rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.05);
}
.convo-row:last-child { border-bottom: none; padding-bottom: 0; }
.convo-icon { font-size: 1.1rem; flex-shrink: 0; margin-top: 0.05rem; }
.convo-lbl { font-size: 0.7rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: #94A3B8; margin-bottom: 0.2rem; }
.convo-val { font-size: 0.83rem; color: #CBD5E1; line-height: 1.5; }

/* Employee buttons override */
div[data-testid="column"] .stButton > button {
    width: 100%;
    border-radius: 10px;
    font-size: 0.82rem;
    font-weight: 700;
    padding: 0.5rem;
    border: 1px solid rgba(108,99,255,0.3);
    background: rgba(108,99,255,0.1);
    color: #A78BFA;
    transition: all 0.15s;
}
div[data-testid="column"] .stButton > button:hover {
    background: rgba(108,99,255,0.25);
    border-color: rgba(108,99,255,0.6);
}

/* Success probability bar */
.success-bar-wrap {
    background: #16142A;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    padding: 0.9rem 1.2rem;
    margin-bottom: 0.8rem;
}
.success-bar-label { font-size: 0.72rem; color: #94A3B8; font-weight: 600; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.5rem; }
.success-bar-row { display: flex; align-items: center; gap: 0.8rem; }
.success-bar { flex: 1; height: 6px; background: rgba(255,255,255,0.08); border-radius: 99px; overflow: hidden; }
.success-bar-fill { height: 100%; border-radius: 99px; }
.success-pct { font-family: 'Syne', sans-serif; font-size: 1rem; font-weight: 800; }

/* Divider */
hr { border: none; border-top: 1px solid #1E293B; margin: 1.4rem 0; }

/* Hide streamlit chrome */
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────
# API HELPERS
# ─────────────────────────────────
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


def get_analytics(doctor_id: str, time_sec: int, employee_type: str):
    try:
        r = requests.get(
            f"{BASE_URL}/analytics/doctor/{doctor_id}",
            params={"time_sec": time_sec, "employee_type": employee_type},
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"Failed to load doctor data: {e}")
        return None


# ─────────────────────────────────
# SESSION STATE
# ─────────────────────────────────
for key, default in [
    ("page", "home"),
    ("selected_territory", None),
    ("selected_doctor_id", None),
    ("selected_doctor_name", None),
    ("employee_type", "MR"),
    ("selected_time", 60),
    ("analytics_data", None),
    ("submitted", False),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ─────────────────────────────────
# HOME PAGE
# ─────────────────────────────────
def home_page():
    st.markdown('<div class="logo">patgpt</div>', unsafe_allow_html=True)
    st.markdown('<div class="logo-sub">AI Sales Assistant · Pharma</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        territories = get_territories()
        if not territories:
            st.warning("No territories found. Please check your backend.")
            return

        territory = st.selectbox(
            "Select Territory", territories,
            index=None, placeholder="Choose a territory…",
        )

        if territory:
            doctors = get_doctors_by_territory(territory)
            if doctors:
                doctor_options = {
                    f"{d['doctor_name']} (ID: {d['doctor_id']})": d["doctor_id"]
                    for d in doctors
                }
                selected_label = st.selectbox(
                    "Select Doctor", list(doctor_options.keys()),
                    index=None, placeholder="Choose a doctor…",
                )
                if selected_label:
                    if st.button("Open Sales Assistant", use_container_width=True):
                        st.session_state.page = "dashboard"
                        st.session_state.selected_territory = territory
                        st.session_state.selected_doctor_id = str(doctor_options[selected_label]).strip()
                        st.session_state.selected_doctor_name = selected_label.split(" (ID:")[0]
                        st.session_state.analytics_data = None
                        st.session_state.submitted = False
                        st.rerun()
            else:
                st.info("No doctors found in this territory.")


# ─────────────────────────────────
# HELPERS
# ─────────────────────────────────
def _product_type_badge(bucket: str) -> str:
    classes = {
        "primary":  "pb-primary",
        "support":  "pb-support",
        "closing":  "pb-closing",
        "reminder": "pb-reminder",
    }
    labels = {
        "primary":  "⭐ Primary",
        "support":  "🔗 Support",
        "closing":  "🎯 Closing",
        "reminder": "📌 Reminder",
    }
    cls = classes.get(bucket, "pb-primary")
    lbl = labels.get(bucket, bucket.capitalize())
    return f'<span class="prod-badge {cls}">{lbl}</span>'


def _category_badge(cat: str) -> str:
    cat_map = {
        "high_performer":               ("🏆 High Performer", "#10B981"),
        "high_interest_low_conversion": ("🔥 High Interest", "#F59E0B"),
        "potential":                    ("🌱 Potential", "#6C63FF"),
        "low_performer":                ("⚠ Low Perform", "#EF4444"),
    }
    label, color = cat_map.get(cat, (cat, "#64748B"))
    return f'<span class="prod-badge" style="background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.1);color:{color}">{label}</span>'


def _score_bar_color(score: float) -> str:
    if score >= 0.6:
        return "#10B981"
    elif score >= 0.35:
        return "#F59E0B"
    return "#EF4444"


def _render_product_card(product: dict, bucket: str, rank: int):
    name    = html.escape(str(product.get("product_name", "—")))
    score   = product.get("score", 0)
    conv    = product.get("conversion_rate", 0)
    interest = product.get("avg_interest", 0)
    cat     = product.get("category", "")
    fill_w  = int(score * 100)
    fill_c  = _score_bar_color(score)

    bucket_badge = _product_type_badge(bucket)
    cat_badge    = _category_badge(cat)

    st.markdown(f"""
    <div class="prod-card">
        <div class="prod-rank">#{rank}</div>
        <div class="prod-body">
            <div class="prod-name">{name}</div>
            <div class="prod-meta">Conv {conv:.0%} · Interest {interest:.1f}/5</div>
            <div class="prod-badges">{bucket_badge}{cat_badge}</div>
        </div>
        <div class="prod-score-bar-wrap">
            <div class="prod-score-pct">{fill_w}%</div>
            <div class="prod-score-sub">score</div>
            <div class="prod-score-bar">
                <div class="prod-score-fill" style="width:{fill_w}%;background:{fill_c};"></div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def _get_product_limit(time_sec: int) -> int:
    """Return number of products to show based on selected time."""
    if time_sec <= 60:
        return 1
    elif time_sec <= 120:
        return 3
    else:
        return 5


# ─────────────────────────────────
# DASHBOARD PAGE
# ─────────────────────────────────
def dashboard_page():
    st.markdown('<div class="logo">patgpt</div>', unsafe_allow_html=True)
    st.markdown('<div class="logo-sub">AI Sales Assistant · Pharma</div>', unsafe_allow_html=True)

    # --- Load analytics if not already loaded (with defaults) ---
    if st.session_state.analytics_data is None:
        with st.spinner("Loading doctor data..."):
            default_time = st.session_state.selected_time
            default_emp = st.session_state.employee_type.lower().replace(" ", "_")
            analytics = get_analytics(st.session_state.selected_doctor_id, default_time, default_emp)
            if analytics:
                st.session_state.analytics_data = analytics
                st.session_state.submitted = True
            else:
                st.error("Could not load doctor data.")
                if st.button("← Back"):
                    st.session_state.page = "home"
                    st.rerun()
                return

    analytics = st.session_state.analytics_data

    # ── 1. DOCTOR CARD (always shown) ──
    doc     = analytics.get("doctor_info", {})
    eng     = analytics.get("engagement_metrics", {})
    aida    = analytics.get("aida", {})
    intent  = analytics.get("intent", {})
    persona = analytics.get("persona", {})
    reco    = analytics.get("recommendations", {})
    nba     = analytics.get("next_best_action", {})
    success = analytics.get("visit_success", {})
    scoring = analytics.get("doctor_scoring", {})
    obj_res = analytics.get("objection_resolution", {})

    doctor_score = scoring.get("score", 0)
    doctor_tier  = scoring.get("tier", "low")

    specialty      = str(doc.get("specialty", "—")).replace("_", " ").title()
    territory_name = str(doc.get("territory", "—")).title()
    exp_years      = doc.get("experience_years", "—")
    patient_load   = doc.get("patient_load", "—")
    publications   = doc.get("publications_count", "—")
    social_reach   = doc.get("social_media_reach", 0)

    st.markdown(f"""
    <div class="doc-hero">
        <div class="doc-hero-name">👨‍⚕️ {html.escape(st.session_state.selected_doctor_name)}</div>
        <div class="doc-hero-meta">{html.escape(specialty)} &nbsp;·&nbsp; {html.escape(territory_name)} Territory</div>
        <div class="doc-hero-stats">
            <div class="stat-box"><div class="stat-box-val">{patient_load}</div><div class="stat-box-lbl">Patients/mo</div></div>
            <div class="stat-box"><div class="stat-box-val">{exp_years}y</div><div class="stat-box-lbl">Experience</div></div>
            <div class="stat-box"><div class="stat-box-val">{publications}</div><div class="stat-box-lbl">Publications</div></div>
            <div class="stat-box"><div class="stat-box-val">{social_reach:,}</div><div class="stat-box-lbl">Social</div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Doctor badge row (always shown)
    tier_cls  = "badge-green" if doctor_tier == "high" else ("badge-yellow" if doctor_tier == "medium" else "badge-red")
    tier_icon = "🌟" if doctor_tier == "high" else ("⚡" if doctor_tier == "medium" else "⚠️")
    tier_text = doctor_tier.capitalize() + " Value"
    intent_cls   = "badge-teal" if intent.get("intent_label") == "High Intent" else ("badge-amber" if "Moderate" in intent.get("intent_label", "") else "badge-slate")
    persona_lbl  = persona.get("label", "—")
    success_emoji = success.get("emoji", "🟡")
    success_lbl   = success.get("label", "—")
    success_pct   = success.get("probability_pct", "—")
    success_cls   = "badge-teal" if success.get("probability", 0) >= 0.6 else ("badge-amber" if success.get("probability", 0) >= 0.35 else "badge-red")

    st.markdown(f"""
    <div class="badge-row">
        <span class="badge {tier_cls}">{tier_icon} {tier_text}</span>
        <span class="badge {intent_cls}">🎯 {html.escape(intent.get('intent_label', '—'))}</span>
        <span class="badge badge-purple">{html.escape(persona_lbl)}</span>
        <span class="badge {success_cls}">{success_emoji} {html.escape(success_lbl)} · {success_pct}</span>
    </div>
    """, unsafe_allow_html=True)

    # ── 2. ROLE SELECTION ──
    st.markdown('<div class="section-label">👤 Your Role</div>', unsafe_allow_html=True)
    emp_cols = st.columns(4)
    emp_types = ["MR", "Area Manager", "VP", "GM"]
    for i, et in enumerate(emp_types):
        with emp_cols[i]:
            if st.button(f"{'✅ ' if st.session_state.employee_type == et else ''}{et}", key=f"emp_{et}"):
                st.session_state.employee_type = et

    # ── 3. TIME SELECTION ──
    st.markdown('<div class="section-label">⏱ Visit Duration</div>', unsafe_allow_html=True)
    time_map = {
        "20 sec": 20, "30 sec": 30, "1 min": 60, "2 min": 120, "5 min": 300,
    }
    selected_label = st.select_slider("", options=list(time_map.keys()), value="1 min")
    selected_time = time_map[selected_label]

    # ── 4. SUBMIT BUTTON ──
    if st.button("Submit & Get Recommendations", use_container_width=True, type="primary"):
        with st.spinner("Updating recommendations..."):
            emp_type_param = st.session_state.employee_type.lower().replace(" ", "_")
            new_analytics = get_analytics(st.session_state.selected_doctor_id, selected_time, emp_type_param)
            if new_analytics:
                st.session_state.analytics_data = new_analytics
                st.session_state.selected_time = selected_time
                st.session_state.submitted = True
                st.rerun()
            else:
                st.error("Failed to refresh recommendations.")

    # Ensure we have submitted state (first load default already set)
    if not st.session_state.submitted:
        st.info("Please select your role and visit duration, then click 'Submit & Get Recommendations'.")
        return

    # Use the updated analytics
    analytics = st.session_state.analytics_data
    reco = analytics.get("recommendations", {})
    nba = analytics.get("next_best_action", {})
    aida = analytics.get("aida", {})
    intent = analytics.get("intent", {})
    success = analytics.get("visit_success", {})
    persona = analytics.get("persona", {})
    obj_res = analytics.get("objection_resolution", {})
    
        # ── 5. PRODUCT RECOMMENDATIONS (with time-based limit) ──
    st.markdown('<div class="section-label">📦 Product Recommendations</div>', unsafe_allow_html=True)

    primary_products  = reco.get("primary_products", [])
    support_products  = reco.get("support_products", [])
    closing_products  = reco.get("closing_products", [])
    reminder_items    = reco.get("reminder_items", [])
    mode              = reco.get("mode", "—")
    event_active      = reco.get("event_active", False)
    event_type        = reco.get("event_type", None)

    # Flatten all recommended products in order
    all_products = []
    for p in primary_products:
        all_products.append((p, "primary"))
    for p in support_products:
        all_products.append((p, "support"))
    for p in closing_products:
        all_products.append((p, "closing"))
    for p in reminder_items:
        all_products.append((p, "reminder"))

    product_limit = _get_product_limit(st.session_state.selected_time)
    limited_products = all_products[:product_limit]

    mode_icon = {"ultra_short": "⚡", "short": "⏩", "medium": "⏱", "long": "🗓️"}.get(mode, "🎯")
    mode_pill = f'<span class="badge badge-purple">{mode_icon} {mode.replace("_"," ").title()} Session</span>'
    event_pill = f'<span class="badge badge-amber">🎉 {html.escape(str(event_type).title())} Event</span>' if event_active else ""
    st.markdown(f'<div class="badge-row">{mode_pill}{event_pill}</div>', unsafe_allow_html=True)

    if limited_products:
        st.markdown('<div class="prod-panel"><div class="prod-panel-header">📋 Recommended Products · Top to Bottom</div>', unsafe_allow_html=True)
        for rank, (prod, bucket) in enumerate(limited_products, start=1):
            _render_product_card(prod, bucket, rank)
        st.markdown("</div>", unsafe_allow_html=True)
        if len(all_products) > product_limit:
            st.caption(f"Showing top {product_limit} product(s) for {st.session_state.selected_time//60} min visit. Change duration and resubmit to see more.")
    else:
        st.info("No products to recommend for this visit duration.")

    # Visit success probability bar (already shown in badge row, but keep standalone for visibility)
    prob = success.get("probability", 0)
    prob_pct = int(prob * 100)
    prob_color = success.get("color", "#F59E0B")
    st.markdown(f"""
    <div class="success-bar-wrap">
        <div class="success-bar-label">🔮 Visit Conversion Probability</div>
        <div class="success-bar-row">
            <div class="success-bar">
                <div class="success-bar-fill" style="width:{prob_pct}%;background:{prob_color};"></div>
            </div>
            <div class="success-pct" style="color:{prob_color};">{prob_pct}%</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # AIDA stage bar
    aida_stage      = aida.get("aida_stage", "awareness")
    aida_idx        = aida.get("aida_stage_index", 0)
    all_stages      = aida.get("all_stages", ["awareness", "interest", "desire", "action"])
    stage_colors    = aida.get("stage_colors", {})
    stage_labels    = aida.get("stage_labels", {})
    stage_emojis    = aida.get("stage_emojis", {})
    aida_color      = aida.get("aida_color", "#6C63FF")
    aida_conf       = aida.get("aida_confidence", 0)
    guidance        = aida.get("stage_guidance", {})

    steps_html = ""
    for i, s in enumerate(all_stages):
        active = i == aida_idx
        color  = stage_colors.get(s, "#64748B")
        lbl    = stage_labels.get(s, s.capitalize())
        emoji  = stage_emojis.get(s, "")
        cls    = "aida-step-active" if active else "aida-step-inactive"
        bg     = f"background:rgba({_hex_to_rgb(color)},0.15); color:{color}; border-color:{color}" if active else ""
        steps_html += f"""
        <div class="aida-step {cls}" style="{bg}">
            <div class="aida-emoji">{emoji}</div>
            <div class="aida-label">{html.escape(lbl)}</div>
        </div>"""

    st.markdown(f"""
    <div class="aida-wrap">
        <div class="aida-title">🧠 AIDA Sales Stage &nbsp;
            <span style="font-size:0.75rem;color:#64748B;font-weight:400;">
                {html.escape(aida.get('aida_label',''))} · {int(aida_conf*100)}% confidence
            </span>
        </div>
        <div class="aida-bar">{steps_html}</div>
        <div class="aida-guidance" style="border-color:{aida_color};">
            <div class="aida-guidance-row">
                <div class="aida-guidance-lbl">💬 What to say</div>
                <div class="aida-guidance-val">{html.escape(guidance.get('what_to_say',''))}</div>
            </div>
            <div class="aida-guidance-row">
                <div class="aida-guidance-lbl">📋 What to show</div>
                <div class="aida-guidance-val">{html.escape(guidance.get('what_to_show',''))}</div>
            </div>
            <div class="aida-guidance-row">
                <div class="aida-guidance-lbl">🚫 What to avoid</div>
                <div class="aida-guidance-val">{html.escape(guidance.get('what_to_avoid',''))}</div>
            </div>
            <div class="aida-guidance-row">
                <div class="aida-guidance-lbl">➡️ Next step</div>
                <div class="aida-guidance-val">{html.escape(guidance.get('next_step',''))}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Next Best Action
    st.markdown('<div class="section-label">🚀 Next Best Action</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="nba-card">
        <div class="nba-goal">Goal · {html.escape(nba.get('goal',''))}</div>
        <div class="nba-action">{html.escape(nba.get('action',''))}</div>
        <div><span class="nba-cta">👉 {html.escape(nba.get('cta',''))}</span></div>
        <div class="nba-product">🎯 Focus product: {html.escape(nba.get('product_focus','—'))}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── 6. CONVERSATION GUIDE (fixed HTML rendering) ──
    st.markdown('<div class="section-label">🗣 Conversation Playbook</div>', unsafe_allow_html=True)
    persona_approach = persona.get("approach", "Engage naturally based on doctor cues.")
    persona_desc     = persona.get("description", "")

    # Inline the conversation rows to avoid nested f-string issues (fixes div error)
    convo_html = f"""
    <div class="convo-wrap">
        <div class="convo-title">🧭 How to Run This Visit</div>
        <div class="convo-row">
            <div class="convo-icon">💬</div>
            <div><div class="convo-lbl">Your Opening</div><div class="convo-val">{html.escape(guidance.get('what_to_say', '—'))}</div></div>
        </div>
        <div class="convo-row">
            <div class="convo-icon">📋</div>
            <div><div class="convo-lbl">What to Show</div><div class="convo-val">{html.escape(guidance.get('what_to_show', '—'))}</div></div>
        </div>
        <div class="convo-row">
            <div class="convo-icon">🚫</div>
            <div><div class="convo-lbl">What to Avoid</div><div class="convo-val">{html.escape(guidance.get('what_to_avoid', '—'))}</div></div>
        </div>
        <div class="convo-row">
            <div class="convo-icon">🎭</div>
            <div><div class="convo-lbl">Doctor Persona</div><div class="convo-val">{html.escape(persona_lbl)}: {html.escape(persona_desc)}</div></div>
        </div>
        <div class="convo-row">
            <div class="convo-icon">⚙️</div>
            <div><div class="convo-lbl">Approach Style</div><div class="convo-val">{html.escape(persona_approach)}</div></div>
        </div>
        <div class="convo-row">
            <div class="convo-icon">➡️</div>
            <div><div class="convo-lbl">Close With</div><div class="convo-val">{html.escape(guidance.get('next_step', '—'))}</div></div>
        </div>
    </div>
    """
    st.markdown(convo_html, unsafe_allow_html=True)

    # ── 7. EXPANDERS (full metrics) ──
    st.markdown("<hr>", unsafe_allow_html=True)
    with st.expander("📋 Full Engagement Metrics"):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Conversion",     f"{eng.get('conversion_rate', 0):.0%}")
        c2.metric("Avg Interest",   f"{eng.get('avg_interest_level', 0):.1f}/5")
        c3.metric("Interactions",   eng.get("total_interactions", 0))
        c4.metric("Follow-up Rate", f"{eng.get('follow_up_rate', 0):.0%}")

    with st.expander("🔬 Intent Score Breakdown"):
        comp = intent.get("components", {})
        i1, i2, i3 = st.columns(3)
        i1.metric("Interest",       f"{comp.get('interest_norm',0):.0%}")
        i2.metric("Follow-up",      f"{comp.get('follow_up_rate',0):.0%}")
        i3.metric("Recent Activity", f"{comp.get('recent_activity',0):.0%}")
        st.write(f"**Overall Intent Score:** {intent.get('intent_score',0):.2f} — *{intent.get('pitch_aggression','balanced').capitalize()} pitch recommended*")

    if obj_res and obj_res.get("has_objections"):
        with st.expander("🛡️ Objection Intelligence"):
            obj_breakdown = obj_res.get("objection_breakdown", {})
            if obj_breakdown:
                for obj, count in list(obj_breakdown.items())[:5]:
                    st.write(f"• **{obj}** — {count} occurrence(s)")
            persistent = obj_res.get("persistent_objections", [])
            if persistent:
                st.warning("⚠️ Persistent: " + ", ".join([p["objection_type"] for p in persistent]))

    # ── Back button ──
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("← Back to Home"):
        st.session_state.page = "home"
        st.session_state.selected_territory   = None
        st.session_state.selected_doctor_id   = None
        st.session_state.selected_doctor_name = None
        st.session_state.analytics_data = None
        st.session_state.submitted = False
        st.rerun()


# ─────────────────────────────────
# UTILITY
# ─────────────────────────────────
def _hex_to_rgb(hex_color: str) -> str:
    """Convert #RRGGBB to 'R,G,B' string for rgba()"""
    h = hex_color.lstrip("#")
    if len(h) == 6:
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f"{r},{g},{b}"
    return "108,99,255"  # fallback purple


# ─────────────────────────────────
# ROUTING
# ─────────────────────────────────
if st.session_state.page == "home":
    home_page()
else:
    dashboard_page()