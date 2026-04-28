"""
llm_insights.py
------------------------
Enhanced LLM engine with:
- Strategic narrative generation
"""

from typing import Dict, Any, Optional, List
from openai import AzureOpenAI
import config.azure_sate as azure_state


class LLMInsightsEngineEnhanced:
    """Enhanced LLM insights with sentiment, NLP, and advanced strategies"""
    
    def __init__(self, deployment: str = "o4-mini"):
        self.deployment = deployment
        self.client = azure_state.client
        
        if not self.client:
            raise RuntimeError("Azure OpenAI client not initialized")
    
    def generate_doctor_insights(
        self,
        analytics_data: Dict[str, Any],
        focus_areas: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Generate comprehensive AI insights with all enhancements"""
        
        doctor_info = analytics_data.get('doctor_info', {})
        engagement = analytics_data.get('engagement_metrics', {})
        recommendations = analytics_data.get('recommendation_engine', {})
        
        # Basic recommendation block (existing)
        basic_insights = self._generate_recommendation_block(
            doctor_info,
            engagement,
            recommendations
        )
        trend_narrative = self._generate_trend_narrative(analytics_data)
        
        return {
            **basic_insights, 
            "trend_narrative": trend_narrative
        }
    
    def _generate_recommendation_block(
        self,
        doctor_info: Dict,
        engagement: Dict,
        recommendations: Dict
    ) -> Dict[str, str]:
        """Original recommendation block (unchanged)"""
        
        top_products = self._format_top_products(
            recommendations.get('top_products', [])
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
1-2 actionable, territory-aware product recommendations on how to engage this doctor effectively
"""
        
        response = self.client.chat.completions.create(
            model=self.deployment,
            messages=[
                {"role": "system", "content": "You are a pharmaceutical sales decision assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=1,
            max_completion_tokens=5000
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
    
    def _generate_trend_narrative(self, analytics_data: Dict[str, Any]) -> str:
        """Generate narrative about trends"""
        
        trends = analytics_data.get('trend_analytics')
        if not trends:
            return "Insufficient data for trend analysis"
        
        trend_summary = trends.get('trends', {})
        conversion_trend = trend_summary.get('conversion', 'stable')
        interest_trend = trend_summary.get('interest', 'stable')
        
        total_months = trends.get('summary', {}).get('total_months_tracked', 0)
        
        narrative_parts = []
        
        if conversion_trend == 'improving':
            narrative_parts.append("📈 Conversion trending upward")
        elif conversion_trend == 'declining':
            narrative_parts.append("📉 Conversion declining - needs attention")
        else:
            narrative_parts.append("➡️ Conversion stable")
        
        if interest_trend == 'improving':
            narrative_parts.append("interest growing")
        elif interest_trend == 'declining':
            narrative_parts.append("interest waning")
        
        narrative_parts.append(f"based on {total_months} months of data")
        
        return " | ".join(narrative_parts)
    
    def _extract_objections_from_analytics(self, analytics_data: Dict[str, Any]) -> List[str]:
        """Extract objection list from analytics data"""
        objection_data = analytics_data.get('objection_analysis', {})
        breakdown = objection_data.get('objection_breakdown', {})
        return list(breakdown.keys()) if breakdown else []
    
    def _extract_outcomes_from_analytics(self, analytics_data: Dict[str, Any]) -> List[str]:
        """Extract outcomes from analytics data - approximation"""
        # This is an approximation since we don't have raw outcomes in summary
        engagement = analytics_data.get('engagement_metrics', {})
        conversion = engagement.get('conversion_rate', 0)
        total = engagement.get('total_interactions', 0)
        
        positive_count = int(total * conversion)
        negative_count = total - positive_count
        
        return ['positive'] * positive_count + ['negative'] * negative_count
    
    def _extract_interest_levels_from_analytics(self, analytics_data: Dict[str, Any]) -> List[float]:
        """Extract interest levels from product performance"""
        products = analytics_data.get('product_performance', {}).get('product_breakdown', [])
        
        interest_levels = []
        for product in products:
            avg_interest = product.get('avg_interest', 0)
            times = product.get('times_presented', 1)
            interest_levels.extend([avg_interest] * times)
        
        return interest_levels
    
    def _format_top_products(self, products: list) -> str:
        """Format top products for prompt"""
        if not products:
            return "No products available"
        
        return "\n".join([
            f"- {p['product_name']} (conversion: {p['conversion_rate']}, interest: {p['avg_interest']})"
            for p in products
        ])