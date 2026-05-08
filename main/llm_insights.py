"""
llm_insights.py
------------------------
Enhanced LLM engine with:
- Strategic narrative generation
- Trend narrative
- Product underperformance explanation (plan §5)
- LLM load minimised — only called via explicit user actions
"""

from typing import Dict, Any, Optional, List
from openai import AzureOpenAI
import config.azure_sate as azure_state


class LLMInsightsEngineEnhanced:
    """Enhanced LLM insights — called only when user explicitly requests AI narrative."""

    def __init__(self, deployment: str = "o4-mini"):
        self.deployment = deployment
        self.client = azure_state.client
        if not self.client:
            raise RuntimeError("Azure OpenAI client not initialized")

    # ─────────────────────────────────────────────
    # PUBLIC: doctor insights
    # ─────────────────────────────────────────────
    def generate_doctor_insights(
        self,
        analytics_data: Dict[str, Any],
        focus_areas: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Generate comprehensive AI insights for a doctor visit."""
        doctor_info     = analytics_data.get("doctor_info", {})
        engagement      = analytics_data.get("engagement_metrics", {})
        recommendations = analytics_data.get("recommendation_engine", {})
        aida            = analytics_data.get("aida", {})
        persona         = analytics_data.get("persona", {})

        basic_insights   = self._generate_recommendation_block(
            doctor_info, engagement, recommendations, aida, persona
        )
        trend_narrative  = self._generate_trend_narrative(analytics_data)

        return {
            **basic_insights,
            "trend_narrative": trend_narrative,
        }

    # ─────────────────────────────────────────────
    # PUBLIC: meeting playbook
    # ─────────────────────────────────────────────
    def generate_meeting_playbook(self, summary: Dict[str, Any]) -> str:
        """
        Called only when user clicks 'Generate AI Playbook'.
        Builds a concise, persona-aware meeting playbook from doctor summary data.
        """
        doctor_info  = summary.get("doctor_info", {})
        aida         = summary.get("aida", {})
        persona      = summary.get("persona", {})
        last_meeting = summary.get("last_meeting", {})
        top_products = summary.get("top_historical_products", [])
        objections   = summary.get("objection_analysis", {}).get("objection_breakdown", {})
        engagement   = summary.get("engagement_metrics", {})

        notes           = last_meeting.get("meeting_notes") or "No notes recorded"
        conv_rate       = engagement.get("conversion_rate", 0) * 100
        avg_duration_min = round((engagement.get("avg_meeting_duration_sec") or 0) / 60, 1)

        top_products_text = "\n".join([
            f"- {p['product_name']}: presented {p['times_presented']} times, "
            f"avg {p.get('avg_time_per_presentation', 0)} sec"
            for p in top_products
        ]) or "No historical product data"

        common_objections = ", ".join(list(objections.keys())[:3]) if objections else "none"

        prompt = f"""
You are a pharma sales coach. Create a concise "Meeting Playbook" for today's visit.

Doctor: {doctor_info.get('doctor_name')} (ID: {doctor_info.get('doctor_id')})
Specialty: {doctor_info.get('specialty')}
AIDA Stage: {aida.get('aida_label', 'Unknown')} — {aida.get('stage_guidance', {}).get('what_to_say', '')}
Persona: {persona.get('label', 'Unknown')} — {persona.get('approach', '')}

Engagement:
- Conversion Rate: {conv_rate:.0f}%
- Avg Meeting Duration: {avg_duration_min} min

Last Meeting (date {last_meeting.get('date', 'N/A')}):
- Product: {last_meeting.get('product', '—')}
- Interest: {last_meeting.get('interest_level', 0)}/5
- Outcome: {last_meeting.get('outcome', '—')}
- Objection: {last_meeting.get('objection') or 'none'}
- Duration: {last_meeting.get('actual_time_seconds', 0)} sec
- Notes: {notes}

Top historically presented products (by time spent):
{top_products_text}

Common objections: {common_objections}

Your task: Write a short, actionable playbook for the MR/manager. Include:
1. Opening line tailored to the AIDA stage and persona.
2. Two key talking points (tie to past objections or meeting notes if present).
3. Suggested product focus (choose from top products above).
4. A closing question to move the doctor forward in the funnel.

Format as bullet points. Keep it under 200 words.
"""
        response = self.client.chat.completions.create(
            model=self.deployment,
            messages=[
                {"role": "system", "content": "You are a concise pharmaceutical sales assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=1,
            max_completion_tokens=800,
        )
        return response.choices[0].message.content or "Unable to generate playbook."

    # ─────────────────────────────────────────────
    # PUBLIC: product underperformance (plan §5)
    # ─────────────────────────────────────────────
    def explain_product_underperformance(
        self,
        product_name: str,
        metrics: Dict[str, Any],
        objections: Dict[str, int],
    ) -> str:
        """
        Called only when user clicks 'Why underperforming?'.
        Feeds product metrics + objection data into a focused prompt.
        Metrics are pre-cleaned by server before calling this method.
        """
        top_objections = ", ".join(
            f"{k} ({v}×)" for k, v in sorted(objections.items(), key=lambda x: -x[1])[:5]
        ) if objections else "None recorded"

        engagement_depth = metrics.get("engagement_depth", "N/A")
        engagement_note = (
            f"{engagement_depth} interactions per doctor on average — "
            + ("high re-engagement but low conversion suggests messaging or fit issue."
               if isinstance(engagement_depth, (int, float)) and engagement_depth > 5
               else "low engagement depth — reps may not be revisiting this product.")
        ) if engagement_depth != "N/A" else "Engagement depth data not available."

        prompt = f"""
You are a pharmaceutical sales strategist. Analyse why this product is underperforming
and provide 3 specific, actionable recommendations.

Product: {product_name}
Total Sales Volume: {metrics.get('total_sales', 0)}
Total Interactions: {metrics.get('total_interactions', 0)}
Conversion Rate: {metrics.get('conversion_rate', 0):.1%}
Average Interest: {metrics.get('avg_interest', 0):.1f}/5
QoQ Growth: {metrics.get('qoq_growth', 0):.1%}
Sales Trend: {metrics.get('trend', 'stable')}
Engagement Depth: {engagement_note}
Top Objections: {top_objections}

Output format:

ROOT CAUSE:
<2-3 sentences on the main reason, referencing specific metrics above>

RECOMMENDATIONS:
1. <specific, measurable action tied to the data>
2. <specific, measurable action tied to the data>
3. <specific, measurable action tied to the data>

QUICK WIN:
<single immediate tactic the MR can use tomorrow>
"""
        response = self.client.chat.completions.create(
            model=self.deployment,
            messages=[
                {"role": "system", "content": "You are a pharmaceutical sales performance analyst. Be specific, data-driven, and concise."},
                {"role": "user", "content": prompt},
            ],
            temperature=1,
            max_completion_tokens=1500,
        )
        return response.choices[0].message.content or "Unable to generate explanation."

    # ─────────────────────────────────────────────
    # PRIVATE: recommendation block
    # ─────────────────────────────────────────────
    def _generate_recommendation_block(
        self,
        doctor_info: Dict,
        engagement: Dict,
        recommendations: Dict,
        aida: Dict,
        persona: Dict,
    ) -> Dict[str, str]:
        """Original recommendation block — enriched with AIDA and persona context."""
        top_products = self._format_top_products(recommendations.get("top_products", []))
        conversion_pct = (engagement.get("conversion_rate", 0) or 0) * 100

        prompt = f"""
You are a pharma sales expert. Your task is to recommend actions, not summarize data.

Doctor Profile:
- Specialty: {doctor_info.get('specialty')}
- Experience: {doctor_info.get('experience_years')} years
- Publications: {doctor_info.get('publications_count')}
- Social Media Reach: {doctor_info.get('social_media_reach')}
- Patient Load: {doctor_info.get('patient_load')}

Engagement:
- Conversion Rate: {conversion_pct:.1f}%
- Interest Level: {engagement.get('avg_interest_level', 0)}/5

AIDA Stage: {aida.get('aida_label', 'Unknown')} (confidence {int((aida.get('aida_confidence', 0))*100)}%)
Doctor Persona: {persona.get('label', 'Unknown')} — {persona.get('description', '')}

Top Products:
{top_products}

Doctor Score: {recommendations.get('doctor_score', 0)}

Your task:
1. Identify the single best product to push
2. Explain why it fits this doctor's AIDA stage and persona
3. Suggest two similar products
4. Evaluate doctor value (High / Medium / Low)
5. Give territory-aware, persona-tailored suggestion

Output format:

BEST PRODUCT:
<product name + reason tied to AIDA stage and persona>

SIMILAR PRODUCTS:
- product 1 (reason)
- product 2 (reason)

DOCTOR VALUE:
<High / Medium / Low + reasoning>

SUGGESTION:
1-2 actionable, persona-aware recommendations on how to engage this doctor effectively
"""
        response = self.client.chat.completions.create(
            model=self.deployment,
            messages=[
                {"role": "system", "content": "You are a pharmaceutical sales decision assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=1,
            max_completion_tokens=5000,
        )

        content = response.choices[0].message.content
        if not content:
            return {"best_product": "", "similar_products": "", "doctor_value": "", "suggestion": ""}

        sections = {"best_product": "", "similar_products": "", "doctor_value": "", "suggestion": ""}
        current = None
        for line in content.split("\n"):
            line = line.strip()
            upper = line.upper()
            if upper.startswith("BEST PRODUCT"):
                current = "best_product"
            elif upper.startswith("SIMILAR PRODUCTS"):
                current = "similar_products"
            elif upper.startswith("DOCTOR VALUE"):
                current = "doctor_value"
            elif upper.startswith("SUGGESTION"):
                current = "suggestion"
            elif current:
                sections[current] += line + "\n"

        return {k: v.strip() for k, v in sections.items()}

    # ─────────────────────────────────────────────
    # PRIVATE: trend narrative
    # ─────────────────────────────────────────────
    def _generate_trend_narrative(self, analytics_data: Dict[str, Any]) -> str:
        """Generate a short human-readable trend narrative (rule-based, no LLM call)."""
        trends = analytics_data.get("trend_analytics")
        if not trends:
            return "Insufficient data for trend analysis."

        trend_summary   = trends.get("trends", {})
        conversion_trend = trend_summary.get("conversion", "stable")
        interest_trend   = trend_summary.get("interest", "stable")
        sales_trend      = trend_summary.get("sales", "stable")
        total_months     = trends.get("summary", {}).get("total_months_tracked", 0)

        parts = []

        icon_map = {"improving": "📈", "declining": "📉", "stable": "➡️"}
        parts.append(f"{icon_map.get(conversion_trend, '➡️')} Conversion {conversion_trend}")

        if interest_trend != "stable":
            parts.append(f"interest {interest_trend}")

        if sales_trend != "stable":
            parts.append(f"sales {sales_trend}")

        if total_months:
            parts.append(f"over {total_months} months")

        return " | ".join(parts) if parts else "Trend data unavailable."

    # ─────────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────────
    def _format_top_products(self, products: list) -> str:
        if not products:
            return "No products available"
        return "\n".join([
            f"- {p['product_name']} (conversion: {p.get('conversion_rate', 0):.1%}, "
            f"interest: {p.get('avg_interest', 0):.1f}/5)"
            for p in products
        ])