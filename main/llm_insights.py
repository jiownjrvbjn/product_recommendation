"""
llm_insights.py
---------------
LLM-powered text generation and insights engine.
Generates narrative summaries, strategic recommendations, and actionable insights.
"""

from typing import Dict, Any, Optional
from openai import AzureOpenAI
import config.azure_sate as azure_state

class LLMInsightsEngine:
    def __init__(self, deployment: str = "o4-mini"):
        self.deployment = deployment
        self.client = azure_state.client

        if not self.client:
            raise RuntimeError("Azure OpenAI client not initialized")

    def generate_doctor_insights(
        self,
        analytics_data: Dict[str, Any],
        focus_areas: Optional[list] = None
    ) -> Dict[str, str]:

        doctor_info = analytics_data.get("doctor_info", {})
        engagement = analytics_data.get("engagement_metrics", {})
        recommendations = analytics_data.get("recommendation_engine", {})

        return self._generate_recommendation_block(
            doctor_info,
            engagement,
            recommendations
        )

    def _generate_recommendation_block(
        self,
        doctor_info: Dict,
        engagement: Dict,
        recommendations: Dict
    ) -> str:

        top_products = self._format_top_products(
            recommendations.get("top_products", [])
        )

        prompt = f"""
You are a pharma sales expert. Your task is to recommend actions, not summarize data.

Doctor Profile:
- Specialty: {doctor_info.get('specialty')}
- Experience: {doctor_info.get('experience_years')} years
- Publications: {doctor_info.get('publications_count')}
- Social Media Reach: {doctor_info.get('social_media_reach')}
- Patient Load: {doctor_info.get('patient_load')}

Engagement:
- Conversion Rate: {engagement.get('conversion_rate')*100:.1f}%
- Interest Level: {engagement.get('avg_interest_level')}/5

Top Products:
{top_products}

Doctor Score: {recommendations.get('doctor_score')}

Your task:

1. Identify the single best product to push
2. Explain why it fits this doctor
3. Suggest two similar products
4. Evaluate doctor value (High / Medium / Low)
5. Give territory-aware suggestion

Output format:

BEST PRODUCT:
<product name + reason>

SIMILAR PRODUCTS:
- product 1 (reason)
- product 2 (reason)

DOCTOR VALUE:
<High / Medium / Low + reasoning>

SUGGESTION:
1-2 actionable, territory-aware recommendations
"""

        response = self.client.chat.completions.create(
            model=self.deployment,
            messages=[
                {"role": "system", "content": "You are a pharmaceutical sales decision assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=1,
            max_completion_tokens=1000
        )

        content = response.choices[0].message.content
        if not content:
            return {
                "best_product": "",
                "similar_products": "",
                "doctor_value": "",
                "suggestion": ""
            }
        sections = {
            "best_product": "",
            "similar_products": "",
            "doctor_value": "",
            "suggestion": ""
        }
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

    def _format_top_products(self, products: list) -> str:
        if not products:
            return "No products available"

        return "\n".join([
            f"- {p['product_name']} (conversion: {p['conversion_rate']}, interest: {p['avg_interest']})"
            for p in products
        ])