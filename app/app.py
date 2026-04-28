import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib
from plotly.subplots import make_subplots

BASE_URL = "http://localhost:8000"

st.set_page_config(page_title="Doctor Dashboard", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
    }
    .insight-box {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 10px 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 10px 20px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

st.title("🧠 Enhanced Doctor Analytics Dashboard")

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


def get_trends_data(doctor_id):
    try:
        response = requests.get(f"{BASE_URL}/analytics/trends/doctor/{doctor_id}")
        response.raise_for_status()
        return response.json()
    except:
        return None


def get_competitive_data(doctor_id):
    try:
        response = requests.get(f"{BASE_URL}/analytics/competitive/doctor/{doctor_id}")
        response.raise_for_status()
        return response.json()
    except:
        return None


def get_territory_comparison(doctor_id, territory):
    try:
        response = requests.get(f"{BASE_URL}/analytics/comparison/doctor/{doctor_id}/territory/{territory}")
        response.raise_for_status()
        return response.json()
    except:
        return None


def get_objection_details(doctor_id):
    try:
        response = requests.get(f"{BASE_URL}/analytics/objections/doctor/{doctor_id}")
        response.raise_for_status()
        return response.json()
    except:
        return None


# ============================
# SIDEBAR
# ============================

st.sidebar.header("🔍 Filters")

territory = st.sidebar.selectbox("Select Territory", get_territories())

doctors = get_doctors(territory)

doctor_map = {
    f"{d['doctor_name']} (ID: {d['doctor_id']})": d["doctor_id"]
    for d in doctors
}

doctor_selected = st.sidebar.selectbox(
    "Select Doctor",
    list(doctor_map.keys()),
    index=None,
    placeholder="Select a doctor"
)

if doctor_selected is None:
    st.warning("⚠️ Please select a doctor to view analytics")
    st.stop()

doctor_id = doctor_map[doctor_selected]

# Load manager view toggle
manager_view = st.sidebar.checkbox("📊 Manager View", value=False)

# ============================
# DATA FETCH
# ============================

with st.spinner("Loading analytics..."):
    analytics = get_doctor_data(doctor_id)
    trends = get_trends_data(doctor_id)
    competitive = get_competitive_data(doctor_id)
    territory_comp = get_territory_comparison(doctor_id, territory)
    objection_details = get_objection_details(doctor_id)

doc = analytics["doctor_info"]
eng = analytics["engagement_metrics"]
products = analytics["product_performance"]["product_breakdown"]

# ============================
# HEADER WITH PERFORMANCE TIER
# ============================

col_header1, col_header2 = st.columns([3, 1])

with col_header1:
    st.subheader(f"👨‍⚕️ {doc['doctor_name']} ({doc['specialty']})")
    st.write(f"**Experience:** {doc['experience_years']} years | **Publications:** {doc['publications_count']} | **Social Reach:** {doc['social_media_reach']} followers")
    st.write(f"**Patient Load:** {doc['patient_load']} patients | **Territory:** {doc['territory'].title()}")

with col_header2:
    if territory_comp:
        tier = territory_comp['performance_vs_avg']['performance_tier']
        percentile = territory_comp['performance_vs_avg']['percentile_rank']
        
        tier_colors = {
            "Top Performer": "🟢",
            "Above Average": "🟡",
            "Below Average": "🟠",
            "Needs Attention": "🔴"
        }
        
        # st.metric(
        #     "Performance Tier",
        #     f"{tier_colors.get(tier, '⚪')} {tier}",
        #     f"{percentile:.0f}th percentile"
        # )
        
        color_map = {
    "Top Performer": "#22c55e",
    "Above Average": "#eab308",
    "Below Average": "#f97316",
    "Needs Attention": "#ef4444"
    }

    st.markdown(f"""
    <div style="display:flex; flex-direction:column; align-items:flex-start;">
        <div style="font-size:14px; color:gray;">Performance Tier</div>
        <div style="display:flex; align-items:center; gap:10px;">
            <div style="width:14px; height:14px; border-radius:50%; background:{color_map.get(tier, '#9ca3af')};"></div>
            <div style="font-size:18px; font-weight:600;">{tier}</div>
        </div>
        <div style="font-size:13px; color:#22c55e;">↑ {percentile:.0f}th percentile</div>
    </div>
    """, unsafe_allow_html=True)

# ============================
# KEY METRICS ROW
# ============================

col1, col2, col3, col4 = st.columns(4)

col1.metric("Conversion Rate", f"{eng['conversion_rate']:.0%}")
col2.metric("Avg Interest", f"{eng['avg_interest_level']:.1f}/5")
col3.metric("Total Interactions", eng["total_interactions"])
col4.metric("Follow-up Rate", f"{eng['follow_up_rate']:.0%}")

# ============================
# TABS FOR ORGANIZED VIEW
# ============================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Product Performance",
    "📈 Trends & Analytics", 
    "⚔️ Competitive Intelligence",
    "⚠️ Objection Resolution",
    "🤖 AI Insights"
])

# ============================
# TAB 1: Product Performance
# ============================

with tab1:
    st.markdown("## 📦 Product Performance Overview")
    
    product_df = pd.DataFrame(products)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(
            product_df,
            x="product_name",
            y="conversion_rate",
            title="Conversion Rate per Product",
            color="conversion_rate",
            color_continuous_scale="Blues"
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig2 = px.scatter(
            product_df,
            x="avg_interest",
            y="conversion_rate",
            size="times_presented",
            hover_name="product_name",
            title="Interest vs Conversion",
            color="times_presented",
            color_continuous_scale="Viridis"
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    # Product table
    st.markdown("### Product Details")
    st.dataframe(
        product_df.style.background_gradient(cmap='RdYlGn', subset=['conversion_rate']),
        use_container_width=True
    )

# ============================
# TAB 2: Trends & Analytics
# ============================

# with tab2:
#     st.markdown("## 📈 Trend Analytics")
    
#     if trends:
#         # Trend direction indicators
#         trend_summary = trends.get('trends', {})
#         col1, col2, col3 = st.columns(3)
        
#         with col1:
#             conv_trend = trend_summary.get('conversion', 'stable')
#             trend_emoji = {"improving": "📈", "declining": "📉", "stable": "➡️"}
#             st.metric("Conversion Trend", trend_emoji.get(conv_trend, '➡️') + " " + conv_trend.title())
        
#         with col2:
#             int_trend = trend_summary.get('interest', 'stable')
#             st.metric("Interest Trend", trend_emoji.get(int_trend, '➡️') + " " + int_trend.title())
        
#         with col3:
#             total_months = trends.get('summary', {}).get('total_months_tracked', 0)
#             st.metric("Months Tracked", total_months)
        
#         st.markdown("---")
        
#         # Monthly conversion trend
#         monthly_conv = pd.DataFrame(trends.get('monthly_conversion', []))
#         if not monthly_conv.empty:
#             fig = px.line(
#                 monthly_conv,
#                 x='month',
#                 y='conversion_rate',
#                 title='Conversion Rate Trend Over Time',
#                 markers=True
#             )
#             fig.update_traces(line_color='#667eea', line_width=3)
#             st.plotly_chart(fig, use_container_width=True)
        
#         # Monthly interest trend
#         col1, col2 = st.columns(2)
        
#         with col1:
#             monthly_int = pd.DataFrame(trends.get('monthly_interest', []))
#             if not monthly_int.empty:
#                 fig = px.line(
#                     monthly_int,
#                     x='month',
#                     y='avg_interest',
#                     title='Interest Level Trend',
#                     markers=True
#                 )
#                 fig.update_traces(line_color='#f093fb', line_width=3)
#                 st.plotly_chart(fig, use_container_width=True)
        
#         with col2:
#             interaction_freq = pd.DataFrame(trends.get('interaction_frequency', []))
#             if not interaction_freq.empty:
#                 fig = px.bar(
#                     interaction_freq,
#                     x='month',
#                     y='interaction_count',
#                     title='Interaction Frequency',
#                     color='interaction_count',
#                     color_continuous_scale='Greens'
#                 )
#                 st.plotly_chart(fig, use_container_width=True)
        
#         # Product trends
#         product_trends = analytics.get('product_trends', [])
#         if product_trends:
#             st.markdown("### 📦 Product-Level Trends")
            
#             selected_product = st.selectbox(
#                 "Select Product for Detailed Trend",
#                 [p['product_name'] for p in product_trends]
#             )
            
#             selected_data = next(
#                 (p for p in product_trends if p['product_name'] == selected_product),
#                 None
#             )
            
#             if selected_data:
#                 product_monthly = pd.DataFrame(selected_data['monthly_data'])
                
#                 if not product_monthly.empty:
#                     fig = make_subplots(specs=[[{"secondary_y": True}]])
                    
#                     fig.add_trace(
#                         go.Scatter(
#                             x=product_monthly['month'],
#                             y=product_monthly['conversion_rate'],
#                             name="Conversion Rate",
#                             line=dict(color='#667eea', width=3)
#                         ),
#                         secondary_y=False
#                     )
                    
#                     fig.add_trace(
#                         go.Scatter(
#                             x=product_monthly['month'],
#                             y=product_monthly['avg_interest'],
#                             name="Interest Level",
#                             line=dict(color='#f093fb', width=3)
#                         ),
#                         secondary_y=True
#                     )
                    
#                     fig.update_layout(title=f"Trend: {selected_product}")
#                     fig.update_xaxes(title_text="Month")
#                     fig.update_yaxes(title_text="Conversion Rate", secondary_y=False)
#                     fig.update_yaxes(title_text="Interest Level", secondary_y=True)
                    
#                     st.plotly_chart(fig, use_container_width=True)
    
#     else:
#         st.info("Insufficient data for trend analysis")
    
#     # Territory Comparison
#     if territory_comp and manager_view:
#         st.markdown("## 🎯 Territory Comparison")
        
#         col1, col2 = st.columns(2)
        
#         with col1:
#             st.markdown("### Doctor vs Territory Average")
            
#             doctor_metrics = territory_comp['doctor_metrics']
#             territory_avg = territory_comp['territory_avg']
            
#             comparison_df = pd.DataFrame({
#                 'Metric': ['Conversion Rate', 'Avg Interest'],
#                 'Doctor': [doctor_metrics['conversion_rate'], doctor_metrics['avg_interest']],
#                 'Territory Avg': [territory_avg['conversion_rate'], territory_avg['avg_interest']]
#             })
            
#             fig = go.Figure()
#             fig.add_trace(go.Bar(name='Doctor', x=comparison_df['Metric'], y=comparison_df['Doctor']))
#             fig.add_trace(go.Bar(name='Territory Avg', x=comparison_df['Metric'], y=comparison_df['Territory Avg']))
#             fig.update_layout(barmode='group', title="Performance Comparison")
            
#             st.plotly_chart(fig, use_container_width=True)
        
#         with col2:
#             st.markdown("### Performance Delta")
            
#             perf_vs_avg = territory_comp['performance_vs_avg']
            
#             st.metric(
#                 "Conversion vs Territory",
#                 f"{doctor_metrics['conversion_rate']:.1%}",
#                 f"{perf_vs_avg['conversion_diff']:+.1%}"
#             )
            
#             st.metric(
#                 "Interest vs Territory",
#                 f"{doctor_metrics['avg_interest']:.2f}",
#                 f"{perf_vs_avg['interest_diff']:+.2f}"
#             )
            
#             st.metric(
#                 "Percentile Rank",
#                 f"{perf_vs_avg['percentile_rank']:.0f}th",
#                 perf_vs_avg['performance_tier']
#             )

# ============================
# TAB 3: Competitive Intelligence
# ============================

with tab3:
    st.markdown("## ⚔️ Competitive Intelligence")
    
    if competitive:
        threat_level = competitive.get('threat_level', 'low')
        threat_score = competitive.get('competitor_threat_score', 0)
        
        # Threat level indicator
        threat_colors = {
            'high': '🔴',
            'medium': '🟡',
            'low': '🟢'
        }
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Threat Level",
                f"{threat_colors.get(threat_level, '⚪')} {threat_level.upper()}",
                f"{threat_score*100:.0f}% of objections"
            )
        
        win_loss = competitive.get('win_loss_analysis', {})
        
        with col2:
            st.metric("Win Rate", f"{win_loss.get('win_rate', 0):.0%}")
        
        with col3:
            st.metric("Loss Rate", f"{win_loss.get('loss_rate', 0):.0%}")
        
        st.markdown("---")
        
        # Win/Loss/Neutral breakdown
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### 📊 Outcome Distribution")
            
            outcome_df = pd.DataFrame({
                'Outcome': ['Wins', 'Losses', 'Neutral'],
                'Count': [win_loss.get('wins', 0), win_loss.get('losses', 0), win_loss.get('neutral', 0)]
            })
            
            fig = px.pie(
                outcome_df,
                names='Outcome',
                values='Count',
                color='Outcome',
                color_discrete_map={'Wins': '#4ade80', 'Losses': '#f87171', 'Neutral': '#94a3b8'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 🎯 Products at Risk")
            
            products_at_risk = competitive.get('products_at_risk', [])
            
            if products_at_risk:
                risk_df = pd.DataFrame(products_at_risk)
                
                fig = px.bar(
                    risk_df,
                    x='product_name',
                    y='competitor_objections',
                    color='risk_level',
                    title="Competitor Objections by Product",
                    color_discrete_map={'high': '#f87171', 'medium': '#fbbf24'}
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.success("✅ No products currently at risk from competitors")
        
        # Competitive insights
        st.markdown("### 💡 Competitive Insights")
        insights = competitive.get('competitive_insights', 'No insights available')
        st.info(insights)
        
    else:
        st.info("No competitive data available")

# ============================
# TAB 4: Objection Resolution
# ============================

with tab4:
    st.markdown("## ⚠️ Objection Resolution Tracker")
    
    if objection_details and objection_details.get('has_objections'):
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Objections", objection_details.get('total_objections', 0))
        
        with col2:
            overall_resolution = objection_details.get('overall_resolution_rate', 0)
            st.metric("Overall Resolution Rate", f"{overall_resolution:.0%}")
        
        st.markdown("---")
        
        # Objection breakdown pie chart
        col1 = st.columns([1, 1])[0]
        
        with col1:
            st.markdown("### 📊 Objection Types")
            
            obj_breakdown = objection_details.get('objection_breakdown', {})
            if obj_breakdown:
                obj_df = pd.DataFrame(
                    list(obj_breakdown.items()),
                    columns=['Objection', 'Count']
                )
                
                fig = px.pie(
                    obj_df,
                    names='Objection',
                    values='Count',
                    hole=0.4
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Detailed table
        st.markdown("### 📋 Detailed Objection Analysis")
        resolution_data = objection_details.get('resolution_analysis', [])
        if resolution_data:
            res_display = pd.DataFrame(resolution_data)
            res_display['overcome_rate'] = res_display['overcome_rate'].apply(lambda x: f"{x:.0%}")
            res_display['follow_up_rate'] = res_display['follow_up_rate'].apply(lambda x: f"{x:.0%}")
            
            st.dataframe(res_display, use_container_width=True)
        
        # Recommendations
        st.markdown("### 💡 Recommendations")
        recommendations = objection_details.get('recommendations', [])
        
        if recommendations:
            for rec in recommendations:
                st.markdown(f"- {rec}")
        else:
            st.info("No specific recommendations at this time")
        
        # Persistent objections warning
        persistent = objection_details.get('persistent_objections', [])
        if persistent:
            st.markdown("### 🚨 Persistent Objections (Need Escalation)")
            
            for obj in persistent:
                st.warning(
                    f"**{obj['objection_type'].upper()}** - Raised {obj['occurrence_count']} times, "
                    f"only {obj['overcome_rate']:.0%} success rate"
                )
    
    else:
        st.success("✅ No objections recorded for this doctor")

# ============================
# TAB 5: AI Insights (Enhanced)
# ============================

with tab5:
    st.markdown("## 🤖 AI-Powered Insights")

    if st.button("Generate AI Insights"):
        with st.spinner("Generating..."):
            insights = requests.get(f"{BASE_URL}/insights/doctor/{doctor_id}").json()

            if not insights or "insights" not in insights:
                st.error(insights)
            else:
                data = insights["insights"]

                st.markdown("### Best Product to Promote")
                st.write(data.get("best_product") or "No data")

                st.markdown("### Similar Products")
                st.write(data.get("similar_products") or "No data")

                st.markdown("### Doctor Rating")
                st.write(data.get("doctor_value") or "No data")

                st.markdown("### Suggestion")
                st.write(data.get("suggestion") or "No data")

                st.markdown("### Trend")
                st.write(data.get("trend_narrative") or "No data")

# ============================
# FOOTER
# ============================

st.markdown("---")
st.caption("Enhanced Doctor Analytics Dashboard v2.0 | Powered by Azure OpenAI")