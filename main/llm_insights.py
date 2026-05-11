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

    def generate_meeting_playbook(self, summary: Dict[str, Any]) -> str:
        """Builds a concise, persona-aware meeting playbook from doctor summary data."""
        doctor_info  = summary.get("doctor_info", {})
        aida         = summary.get("aida", {})
        persona      = summary.get("persona", {})
        last_meeting = summary.get("last_meeting", {})
        top_products = summary.get("top_historical_products", [])
        objections   = summary.get("objection_analysis", {}).get("objection_breakdown", {})
        engagement   = summary.get("engagement_metrics", {})

        notes           = last_meeting.get("meeting_notes") or "No notes recorded"
        conv_rate       = engagement.get("conversion_rate", 0)
        avg_duration_min = round((engagement.get("avg_meeting_duration_sec") or 0) / 60, 1)

        top_products_text = "\n".join([
            f"- {p.get('product_name', '')}: presented {p.get('times_presented', 0)} times, "
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
- Conversion Rate: {conv_rate:.0%}
- Avg Meeting Duration: {avg_duration_min} min

Last Meeting (date {last_meeting.get('date', 'N/A')}):
- Product: {last_meeting.get('product', '—')}
- Interest: {last_meeting.get('interest_level', 0)}/5
- Outcome: {last_meeting.get('outcome', '—')}
- Objection: {last_meeting.get('objection') or 'none'}
- Duration: {last_meeting.get('actual_time_seconds', 0)} sec
- Notes: {notes}

Top historically presented products:
{top_products_text}

Common objections: {common_objections}

Your task: Write a short, actionable playbook for the MR/manager. Include:
1. Opening line tailored to the AIDA stage and persona.
2. Two key talking points (tie to past objections or meeting notes if present).
3. Suggested product focus (choose from top products above).
4. A closing question to move the doctor forward in the funnel.

Format as bullet points. Keep it under 200 words.
"""
        try:
            response = self.client.chat.completions.create(
                model=self.deployment,
                messages=[
                    {"role": "system", "content": "You are a concise pharmaceutical sales assistant."},
                    {"role": "user", "content": prompt},
                ],
                temperature=1,
                max_completion_tokens=800,
            )
            result = response.choices[0].message.content
            if result and result.strip():
                return result.strip()
        except Exception:
            pass   # fall through to fallback

        # ── Rule‑based fallback ─────────────────────────────────────────
        stage = aida.get("aida_stage", "awareness")
        persona_type = persona.get("persona", "analytical")
        doc_name = doctor_info.get("doctor_name", "the doctor")

        opening_lines = {
            "awareness": "Good morning, Dr. {name}. I’d like to briefly share a new approach that addresses {specialty} challenges.",
            "interest":  "Dr. {name}, I noticed you’ve shown interest in {top_product} – let me share some clinical data that may be useful.",
            "desire":    "Great to see your growing interest, Dr. {name}. I have a patient success story that reinforces why {top_product} fits your practice.",
            "action":    "Thank you for your continued trust, Dr. {name}. Today I’d like to discuss how we can expand your results with complementary options.",
        }

        talking_points = {
            "analytical":    ["Highlight recent peer‑reviewed study results.", "Compare efficacy data vs. alternatives."],
            "emotional":     ["Share a patient story that demonstrates improved adherence.", "Ask about a memorable patient case."],
            "fast_decision": ["Present top‑line value in 30 seconds.", "Offer a limited‑time trial or sample."],
            "resistant":     ["Address the most common objection without being pushy.", "Ask what would need to change for them to reconsider."],
        }

        closing_questions = {
            "awareness": "Would you be open to reviewing a one‑pager before our next visit?",
            "interest":  "Shall I leave a trial sample for you to evaluate with a few patients?",
            "desire":    "Can we schedule a follow‑up to discuss starting a pilot prescription?",
            "action":    "Which of these two complementary products would you like to introduce next?",
        }

        specialty = doctor_info.get("specialty", "your specialty")
        top_product = ", ".join([p.get("product_name", "") for p in top_products[:2]]) or "our key product"

        opening = opening_lines.get(stage, opening_lines["awareness"]).format(name=doc_name, specialty=specialty, top_product=top_product)
        points = talking_points.get(persona_type, talking_points["analytical"])
        closing = closing_questions.get(stage, closing_questions["awareness"])

        playbook = f"""
        **Opening Line**
        {opening}

        **Key Talking Points**
        1. {points[0]}
        2. {points[1]}

        **Suggested Product Focus**  
        {top_product} (based on historical engagement)

        **Closing Question**
        {closing}
        """
        return playbook.strip()

    def explain_product_underperformance(
        self,
        product_name: str,
        metrics: Dict[str, Any],
        objections: Dict[str, int],
    ) -> str:
        """Called only when user clicks 'Why underperforming?'."""
        top_objections = ", ".join(
            f"{k} ({v}×)" for k, v in sorted(objections.items(), key=lambda x: -x[1])[:5]
        ) if objections else "None recorded"

        engagement_depth = metrics.get("engagement_depth", "N/A")
        if isinstance(engagement_depth, (int, float)):
            engagement_note = (
                f"{engagement_depth} interactions per doctor — "
                + ("high re-engagement but low conversion suggests messaging/fit issue."
                   if engagement_depth > 5
                   else "low engagement depth — reps may not be revisiting this product.")
            )
        else:
            engagement_note = "Engagement depth data not available."

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
<2-3 sentences>

RECOMMENDATIONS:
1. <action>
2. <action>
3. <action>

QUICK WIN:
<single tactic>
"""
        try:
            response = self.client.chat.completions.create(
                model=self.deployment,
                messages=[
                    {"role": "system", "content": "You are a pharmaceutical sales performance analyst."},
                    {"role": "user", "content": prompt},
                ],
                temperature=1,
                max_completion_tokens=1500,
            )
            return response.choices[0].message.content or "Unable to generate explanation."
        except Exception as e:
            return f"Explanation failed: {str(e)}"

    def _generate_recommendation_block(self, doctor_info, engagement, recommendations, aida, persona):
        top_products = self._format_top_products(recommendations.get("top_products", []))
        conversion_pct = (engagement.get("conversion_rate", 0) or 0) * 100

        prompt = f"""
You are a pharma sales expert. Recommend actions, not summarize data.

Doctor: {doctor_info.get('specialty')}, {doctor_info.get('experience_years')} yrs exp
Conversion: {conversion_pct:.1f}%
Interest: {engagement.get('avg_interest_level', 0)}/5
AIDA: {aida.get('aida_label', 'Unknown')} (confidence {int((aida.get('aida_confidence', 0))*100)}%)
Persona: {persona.get('label', 'Unknown')} — {persona.get('description', '')}
Top Products: {top_products}
Doctor Score: {recommendations.get('doctor_score', 0)}

Output:
BEST PRODUCT: <name + reason>
SIMILAR PRODUCTS:
- product1 (reason)
- product2 (reason)
DOCTOR VALUE: <High/Medium/Low + reasoning>
SUGGESTION: <1-2 actionable, persona-aware tips>
"""
        try:
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
                if line.upper().startswith("BEST PRODUCT"):
                    current = "best_product"
                elif line.upper().startswith("SIMILAR PRODUCTS"):
                    current = "similar_products"
                elif line.upper().startswith("DOCTOR VALUE"):
                    current = "doctor_value"
                elif line.upper().startswith("SUGGESTION"):
                    current = "suggestion"
                elif current:
                    sections[current] += line + "\n"
            return {k: v.strip() for k, v in sections.items()}
        except:
            return {"best_product": "", "similar_products": "", "doctor_value": "", "suggestion": ""}

    def _generate_trend_narrative(self, analytics_data) -> str:
        trends = analytics_data.get("trend_analytics")
        if not trends:
            return "Insufficient data."
        trend_summary = trends.get("trends", {})
        parts = []
        icon_map = {"improving": "📈", "declining": "📉", "stable": "➡️"}
        if trend_summary.get("conversion"):
            parts.append(f"{icon_map.get(trend_summary['conversion'], '➡️')} Conversion {trend_summary['conversion']}")
        return " | ".join(parts) or "Stable"

    def _format_top_products(self, products: list) -> str:
        if not products:
            return "No products"
        return "\n".join([
            f"- {p.get('product_name', '')} (conv: {p.get('conversion_rate', 0):.1%}, interest: {p.get('avg_interest', 0):.1f}/5)"
            for p in products
        ])