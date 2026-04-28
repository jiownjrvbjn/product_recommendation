"""
analytics_engine.py
-----------------------------
Enhanced analytics engine with:
- Trend analytics (time-series)
- Competitive intelligence
- Manager comparison views
- Objection resolution tracking
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta


class TrendAnalytics:
    """Handles time-series analysis and trend computations"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.df['interaction_date'] = pd.to_datetime(self.df['interaction_date'])
    
    def get_doctor_trends(self, doctor_id: int, days: int = 90) -> Dict[str, Any]:
        """Get time-series trends for a specific doctor"""
        doctor_df = self.df[self.df['doctor_id'] == doctor_id].copy()
        
        if doctor_df.empty:
            return None
        
        # Sort by date
        doctor_df = doctor_df.sort_values('interaction_date')
        
        # Monthly aggregation
        doctor_df['month'] = doctor_df['interaction_date'].dt.to_period('M')
        
        # Conversion rate over time
        monthly_conversion = doctor_df.groupby('month').apply(
            lambda x: (x['outcome'].str.lower() == 'positive').sum() / len(x)
        ).reset_index()
        monthly_conversion.columns = ['month', 'conversion_rate']
        monthly_conversion['month'] = monthly_conversion['month'].astype(str)
        
        # Interest level over time
        monthly_interest = doctor_df.groupby('month')['interest_level'].mean().reset_index()
        monthly_interest.columns = ['month', 'avg_interest']
        monthly_interest['month'] = monthly_interest['month'].astype(str)
        
        # Interaction frequency over time
        interaction_frequency = doctor_df.groupby('month').size().reset_index()
        interaction_frequency.columns = ['month', 'interaction_count']
        interaction_frequency['month'] = interaction_frequency['month'].astype(str)
        
        # Product diversity over time
        product_diversity = doctor_df.groupby('month')['product_name'].nunique().reset_index()
        product_diversity.columns = ['month', 'unique_products']
        product_diversity['month'] = product_diversity['month'].astype(str)
        
        # Calculate trends (improving/declining/stable)
        conversion_trend = self._calculate_trend(monthly_conversion['conversion_rate'].tolist())
        interest_trend = self._calculate_trend(monthly_interest['avg_interest'].tolist())
        
        return {
            "monthly_conversion": monthly_conversion.to_dict(orient='records'),
            "monthly_interest": monthly_interest.to_dict(orient='records'),
            "interaction_frequency": interaction_frequency.to_dict(orient='records'),
            "product_diversity": product_diversity.to_dict(orient='records'),
            "trends": {
                "conversion": conversion_trend,
                "interest": interest_trend
            },
            "summary": {
                "total_months_tracked": len(monthly_conversion),
                "avg_interactions_per_month": round(doctor_df.groupby('month').size().mean(), 1),
                "most_active_month": interaction_frequency.loc[
                    interaction_frequency['interaction_count'].idxmax(), 'month'
                ] if not interaction_frequency.empty else None
            }
        }
    
    def get_product_trends(self, doctor_id: int) -> List[Dict[str, Any]]:
        """Get product-specific trends for a doctor"""
        doctor_df = self.df[self.df['doctor_id'] == doctor_id].copy()
        
        if doctor_df.empty:
            return []
        
        doctor_df['month'] = pd.to_datetime(doctor_df['interaction_date']).dt.to_period('M')
        
        products = []
        for product in doctor_df['product_name'].unique():
            p_df = doctor_df[doctor_df['product_name'] == product].copy()
            
            # Monthly product performance
            monthly = p_df.groupby('month').agg({
                'interest_level': 'mean',
                'outcome': lambda x: (x.str.lower() == 'positive').sum() / len(x)
            }).reset_index()
            
            monthly.columns = ['month', 'avg_interest', 'conversion_rate']
            monthly['month'] = monthly['month'].astype(str)
            
            products.append({
                "product_name": product,
                "monthly_data": monthly.to_dict(orient='records'),
                "trend": self._calculate_trend([m['conversion_rate'] for m in monthly.to_dict(orient='records')])
            })
        
        return products
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction from a list of values"""
        if not values or len(values) < 2:
            return "stable"
        
        # Simple linear regression slope
        x = np.arange(len(values))
        y = np.array(values)
        
        # Handle NaN values
        mask = ~np.isnan(y)
        if mask.sum() < 2:
            return "stable"
        
        x_clean = x[mask]
        y_clean = y[mask]
        
        slope = np.polyfit(x_clean, y_clean, 1)[0]
        
        if slope > 0.05:
            return "improving"
        elif slope < -0.05:
            return "declining"
        else:
            return "stable"


class CompetitiveIntelligence:
    """Analyzes competitive positioning and threats"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.df['objection'] = self.df['objection'].str.strip().str.lower()
        self.df['outcome'] = self.df['outcome'].str.strip().str.lower()
    
    def get_competitive_analysis(self, doctor_id: int) -> Dict[str, Any]:
        """Analyze competitive threats and positioning"""
        doctor_df = self.df[self.df['doctor_id'] == doctor_id]
        
        if doctor_df.empty:
            return None
        
        # Competitor loyalty objections
        comp_loyalty = doctor_df[
            doctor_df['objection'].str.contains('competitor', na=False)
        ]
        
        competitor_threat_score = len(comp_loyalty) / len(doctor_df) if len(doctor_df) > 0 else 0
        
        # Products losing to competitors
        products_at_risk = []
        for product in doctor_df['product_name'].unique():
            p_df = doctor_df[doctor_df['product_name'] == product]
            comp_obj = p_df[p_df['objection'].str.contains('competitor', na=False)]
            
            if len(comp_obj) > 0:
                products_at_risk.append({
                    "product_name": product,
                    "competitor_objections": len(comp_obj),
                    "risk_level": "high" if len(comp_obj) / len(p_df) > 0.3 else "medium"
                })
        
        # Win rate analysis
        total_interactions = len(doctor_df)
        wins = len(doctor_df[doctor_df['outcome'] == 'positive'])
        losses = len(doctor_df[doctor_df['outcome'] == 'negative'])
        neutral = total_interactions - wins - losses
        
        return {
            "competitor_threat_score": round(competitor_threat_score, 3),
            "threat_level": self._get_threat_level(competitor_threat_score),
            "products_at_risk": products_at_risk,
            "win_loss_analysis": {
                "wins": wins,
                "losses": losses,
                "neutral": neutral,
                "win_rate": round(wins / total_interactions, 2) if total_interactions > 0 else 0,
                "loss_rate": round(losses / total_interactions, 2) if total_interactions > 0 else 0
            },
            "competitive_insights": self._generate_competitive_insights(
                competitor_threat_score, products_at_risk, wins, losses
            )
        }
    
    def _get_threat_level(self, score: float) -> str:
        """Determine threat level from competitor score"""
        if score > 0.4:
            return "high"
        elif score > 0.2:
            return "medium"
        else:
            return "low"
    
    def _generate_competitive_insights(
        self, threat_score: float, at_risk: List, wins: int, losses: int
    ) -> str:
        """Generate human-readable competitive insights"""
        insights = []
        
        if threat_score > 0.3:
            insights.append(f"⚠️ High competitor presence ({threat_score*100:.0f}% of objections)")
        
        if at_risk:
            products = ", ".join([p['product_name'] for p in at_risk[:2]])
            insights.append(f"🎯 Products at risk: {products}")
        
        if wins > losses * 2:
            insights.append("✅ Strong win rate - maintain momentum")
        elif losses > wins:
            insights.append("⚠️ Losing more than winning - needs intervention")
        
        return " | ".join(insights) if insights else "No significant competitive threats"


class ManagerComparison:
    """Provides territory and team-level comparisons"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.df.columns = self.df.columns.str.strip().str.lower()
        self.df['territory'] = self.df['territory'].str.strip().str.lower()
        self.df['outcome'] = self.df['outcome'].str.strip().str.lower()
    
    def get_territory_benchmarks(self, territory: str) -> Dict[str, Any]:
        """Get territory-wide performance benchmarks"""
        territory = territory.strip().lower()
        territory_df = self.df[self.df['territory'] == territory]
        
        if territory_df.empty:
            return None
        
        # Doctor-level aggregation
        doctor_stats = []
        for doctor_id in territory_df['doctor_id'].unique():
            doc_df = territory_df[territory_df['doctor_id'] == doctor_id]
            
            conversion = (doc_df['outcome'] == 'positive').sum() / len(doc_df)
            avg_interest = doc_df['interest_level'].mean()
            
            doctor_stats.append({
                'doctor_id': int(doctor_id),
                'conversion_rate': conversion,
                'avg_interest': avg_interest,
                'total_interactions': len(doc_df)
            })
        
        df_stats = pd.DataFrame(doctor_stats)
        
        return {
            "territory_name": territory,
            "total_doctors": len(doctor_stats),
            "avg_conversion_rate": round(df_stats['conversion_rate'].mean(), 3),
            "avg_interest_level": round(df_stats['avg_interest'].mean(), 2),
            "top_performer": {
                "doctor_id": int(df_stats.loc[df_stats['conversion_rate'].idxmax(), 'doctor_id']),
                "conversion_rate": round(df_stats['conversion_rate'].max(), 3)
            },
            "bottom_performer": {
                "doctor_id": int(df_stats.loc[df_stats['conversion_rate'].idxmin(), 'doctor_id']),
                "conversion_rate": round(df_stats['conversion_rate'].min(), 3)
            },
            "percentiles": {
                "p25": round(df_stats['conversion_rate'].quantile(0.25), 3),
                "p50": round(df_stats['conversion_rate'].quantile(0.50), 3),
                "p75": round(df_stats['conversion_rate'].quantile(0.75), 3),
                "p90": round(df_stats['conversion_rate'].quantile(0.90), 3)
            }
        }
    
    def compare_doctor_to_territory(
        self, doctor_id: int, territory: str
    ) -> Dict[str, Any]:
        """Compare a specific doctor against territory benchmarks"""
        
        # Get doctor metrics
        doctor_df = self.df[self.df['doctor_id'] == doctor_id]
        if doctor_df.empty:
            return None
        
        doc_conversion = (doctor_df['outcome'] == 'positive').sum() / len(doctor_df)
        doc_interest = doctor_df['interest_level'].mean()
        
        # Get territory benchmarks
        benchmarks = self.get_territory_benchmarks(territory)
        if not benchmarks:
            return None
        
        # Calculate percentile rank
        territory_df = self.df[self.df['territory'] == territory.strip().lower()]
        all_conversions = []
        
        for doc_id in territory_df['doctor_id'].unique():
            doc_temp = territory_df[territory_df['doctor_id'] == doc_id]
            conv = (doc_temp['outcome'] == 'positive').sum() / len(doc_temp)
            all_conversions.append(conv)
        
        percentile_rank = (sum(1 for x in all_conversions if x < doc_conversion) / 
                          len(all_conversions)) * 100 if all_conversions else 50
        
        return {
            "doctor_metrics": {
                "conversion_rate": round(doc_conversion, 3),
                "avg_interest": round(doc_interest, 2)
            },
            "territory_avg": {
                "conversion_rate": benchmarks['avg_conversion_rate'],
                "avg_interest": benchmarks['avg_interest_level']
            },
            "performance_vs_avg": {
                "conversion_diff": round(doc_conversion - benchmarks['avg_conversion_rate'], 3),
                "interest_diff": round(doc_interest - benchmarks['avg_interest_level'], 2),
                "percentile_rank": round(percentile_rank, 1),
                "performance_tier": self._get_performance_tier(percentile_rank)
            }
        }
    
    def _get_performance_tier(self, percentile: float) -> str:
        """Classify doctor into performance tier"""
        if percentile >= 75:
            return "Top Performer"
        elif percentile >= 50:
            return "Above Average"
        elif percentile >= 25:
            return "Below Average"
        else:
            return "Needs Attention"
    
    def get_specialty_comparison(self, doctor_id: int) -> Dict[str, Any]:
        """Compare doctor against others in same specialty"""
        doctor_df = self.df[self.df['doctor_id'] == doctor_id]
        
        if doctor_df.empty:
            return None
        
        specialty = doctor_df['specialty'].iloc[0]
        specialty_df = self.df[self.df['specialty'] == specialty]
        
        # Doctor metrics
        doc_conversion = (doctor_df['outcome'] == 'positive').sum() / len(doctor_df)
        
        # Specialty metrics
        specialty_doctors = []
        for doc_id in specialty_df['doctor_id'].unique():
            doc_temp = specialty_df[specialty_df['doctor_id'] == doc_id]
            conv = (doc_temp['outcome'] == 'positive').sum() / len(doc_temp)
            specialty_doctors.append(conv)
        
        specialty_avg = np.mean(specialty_doctors)
        
        rank = sorted(specialty_doctors, reverse=True).index(doc_conversion) + 1 if doc_conversion in specialty_doctors else 0
        
        return {
            "specialty": specialty,
            "total_doctors_in_specialty": len(specialty_doctors),
            "doctor_conversion": round(doc_conversion, 3),
            "specialty_avg_conversion": round(specialty_avg, 3),
            "rank_in_specialty": rank,
            "percentile_in_specialty": round((1 - rank/len(specialty_doctors)) * 100, 1) if rank > 0 else 0
        }


class ObjectionResolutionTracker:
    """Tracks objection patterns and resolution strategies"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.df['objection'] = self.df['objection'].str.strip().str.lower()
        self.df['outcome'] = self.df['outcome'].str.strip().str.lower()
        self.df['follow_up'] = self.df['follow_up'].str.strip().str.lower()
    
    def get_objection_analysis(self, doctor_id: int) -> Dict[str, Any]:
        """Comprehensive objection analysis for a doctor"""
        doctor_df = self.df[self.df['doctor_id'] == doctor_id]
        
        if doctor_df.empty:
            return None
        
        # Filter rows with objections
        objections_df = doctor_df[doctor_df['objection'].notna() & (doctor_df['objection'] != '')]
        
        if objections_df.empty:
            return {
                "has_objections": False,
                "total_objections": 0,
                "message": "No objections recorded for this doctor"
            }
        
        # Objection types
        objection_types = objections_df['objection'].value_counts().to_dict()
        
        # Resolution analysis
        resolution_data = []
        for obj_type in objections_df['objection'].unique():
            obj_df = objections_df[objections_df['objection'] == obj_type]
            
            # How often this objection was overcome
            overcome_rate = (obj_df['outcome'] == 'positive').sum() / len(obj_df)
            follow_up_rate = (obj_df['follow_up'] == 'yes').sum() / len(obj_df)
            
            resolution_data.append({
                "objection_type": obj_type,
                "occurrence_count": len(obj_df),
                "overcome_rate": round(overcome_rate, 3),
                "follow_up_rate": round(follow_up_rate, 3),
                "resolution_difficulty": self._get_resolution_difficulty(overcome_rate)
            })
        
        # Sort by occurrence
        resolution_data.sort(key=lambda x: x['occurrence_count'], reverse=True)
        
        # Persistent objections (raised multiple times, low resolution)
        persistent = [r for r in resolution_data if r['occurrence_count'] >= 2 and r['overcome_rate'] < 0.3]
        
        return {
            "has_objections": True,
            "total_objections": len(objections_df),
            "objection_breakdown": objection_types,
            "resolution_analysis": resolution_data,
            "persistent_objections": persistent,
            "overall_resolution_rate": round(
                (objections_df['outcome'] == 'positive').sum() / len(objections_df), 3
            ),
            "recommendations": self._generate_objection_recommendations(resolution_data, persistent)
        }
    
    def _get_resolution_difficulty(self, overcome_rate: float) -> str:
        """Classify objection resolution difficulty"""
        if overcome_rate > 0.6:
            return "easy"
        elif overcome_rate > 0.3:
            return "moderate"
        else:
            return "difficult"
    
    def _generate_objection_recommendations(
        self, resolution_data: List[Dict], persistent: List[Dict]
    ) -> List[str]:
        """Generate actionable recommendations for objection handling"""
        recommendations = []
        
        if persistent:
            obj_names = ", ".join([p['objection_type'] for p in persistent[:2]])
            recommendations.append(
                f"🎯 Focus on persistent objections: {obj_names} - consider escalation or new approach"
            )
        
        # Find easiest to overcome
        easy_ones = [r for r in resolution_data if r['resolution_difficulty'] == 'easy' and r['occurrence_count'] > 0]
        if easy_ones:
            recommendations.append(
                f"✅ '{easy_ones[0]['objection_type']}' objections are being handled well - document winning approach"
            )
        
        # Find difficult ones
        difficult = [r for r in resolution_data if r['resolution_difficulty'] == 'difficult']
        if difficult:
            recommendations.append(
                f"⚠️ '{difficult[0]['objection_type']}' objections need new strategy - current approach not working"
            )
        
        return recommendations


class RecommendationEngine:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def normalize(self, x, min_val, max_val):
        return (x - min_val) / (max_val - min_val + 1e-5)

    def score_products(self, doctor_df: pd.DataFrame):
        results = []

        for product in doctor_df["product_name"].unique():
            p_df = doctor_df[doctor_df["product_name"] == product]

            conversion = (p_df["outcome"] == "Positive").sum() / len(p_df)
            interest = p_df["interest_level"].mean()
            follow_up = (p_df["follow_up"] == "Yes").sum() / len(p_df)

            score = (0.5 * conversion) + (0.3 * (interest / 5)) + (0.2 * follow_up)

            if interest >= 3.5 and conversion >= 0.5:
                category = "high_performer"
            elif interest >= 3.5:
                category = "high_interest_low_conversion"
            elif conversion < 0.3:
                category = "low_performer"
            else:
                category = "potential"

            results.append({
                "product_name": product,
                "score": round(score, 3),
                "conversion_rate": round(conversion, 2),
                "avg_interest": round(interest, 2),
                "category": category
            })

        return sorted(results, key=lambda x: x["score"], reverse=True)

    def score_doctor(self, doctor_info):
        return (
            0.3 * self.normalize(doctor_info["patient_load"], 0, 100) +
            0.2 * self.normalize(doctor_info["experience_years"], 0, 40) +
            0.2 * self.normalize(doctor_info["publications_count"], 0, 50) +
            0.3 * self.normalize(doctor_info["social_media_reach"], 0, 10000)
        )

    def generate_strategy(self, doctor_info, top_products):
        strategy = []

        if doctor_info["experience_years"] > 10:
            strategy.append("clinical_data")
        else:
            strategy.append("benefit_selling")

        if doctor_info["social_media_reach"] > 5000:
            strategy.append("brand_products")

        if doctor_info["patient_load"] > 50:
            strategy.append("fast_moving_products")

        strategy.append(f"pitch_{min(2, len(top_products))}_products")

        return strategy


class DoctorAnalyticsEnhanced:
    """Main analytics engine with all enhanced features"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.df.columns = self.df.columns.str.strip().str.lower()
        self.df['territory'] = self.df['territory'].str.strip().str.lower()
        self.df['doctor_name'] = self.df['doctor_name'].str.strip()
        
        # Initialize all sub-engines
        self.reco_engine = RecommendationEngine(self.df)
        self.trend_engine = TrendAnalytics(self.df)
        self.competitive_engine = CompetitiveIntelligence(self.df)
        self.manager_engine = ManagerComparison(self.df)
        self.objection_engine = ObjectionResolutionTracker(self.df)
    
    def get_doctor_summary(self, doctor_id: int) -> Optional[Dict[str, Any]]:
        """Enhanced doctor summary with all analytics"""
        doctor_df = self.df[self.df['doctor_id'] == doctor_id]
        
        if doctor_df.empty:
            return None
        
        # Basic doctor info
        doctor_info = {
            "doctor_id": int(doctor_id),
            "doctor_name": doctor_df['doctor_name'].iloc[0],
            "territory": doctor_df['territory'].iloc[0],
            "specialty": doctor_df['specialty'].iloc[0],
            "patient_load": int(doctor_df['patient_load'].iloc[0]),
            "experience_years": int(doctor_df['experience_years'].iloc[0]),
            "publications_count": int(doctor_df['publications_count'].iloc[0]),
            "social_media_reach": int(doctor_df['social_media_reach'].iloc[0]),
        }
        
        doctor_df['outcome'] = doctor_df['outcome'].str.strip().str.lower()
        doctor_df['follow_up'] = doctor_df['follow_up'].str.strip().str.lower()
        
        # Engagement metrics
        engagement = {
            "total_interactions": doctor_df['interaction_id'].nunique(),
            "avg_interest_level": round(doctor_df['interest_level'].mean(), 2),
            "conversion_rate": round((doctor_df['outcome'] == 'positive').sum() / len(doctor_df), 2),
            "follow_up_rate": round((doctor_df['follow_up'] == 'yes').sum() / len(doctor_df), 2)
        }
        
        # Product performance
        product_list = []
        for p in doctor_df['product_name'].unique():
            p_df = doctor_df[doctor_df['product_name'] == p]
            p_df['outcome'] = p_df['outcome'].str.strip().str.lower()
            product_list.append({
                "product_name": p,
                "avg_interest": round(p_df['interest_level'].mean(), 2),
                "conversion_rate": round((p_df['outcome'] == 'positive').sum() / len(p_df), 2),
                "times_presented": len(p_df)
            })
        
        # Objection analysis (basic)
        doctor_df['objection'] = doctor_df['objection'].str.strip().replace('', None)
        objection_counts = (
            doctor_df['objection']
            .dropna()
            .value_counts()
            .to_dict()
        )
        
        objection_analysis = {
            "total_objections": int(doctor_df['objection'].notna().sum()),
            "objection_breakdown": objection_counts,
            "has_objections": bool(objection_counts)
        }
        
        # Recommendation engine
        scored_products = self.reco_engine.score_products(doctor_df)
        doctor_score = self.reco_engine.score_doctor(doctor_info)
        strategy = self.reco_engine.generate_strategy(doctor_info, scored_products)
        
        # NEW: Enhanced analytics
        trends = self.trend_engine.get_doctor_trends(doctor_id)
        product_trends = self.trend_engine.get_product_trends(doctor_id)
        competitive = self.competitive_engine.get_competitive_analysis(doctor_id)
        territory_comparison = self.manager_engine.compare_doctor_to_territory(
            doctor_id, doctor_info['territory']
        )
        specialty_comparison = self.manager_engine.get_specialty_comparison(doctor_id)
        objection_deep_dive = self.objection_engine.get_objection_analysis(doctor_id)
        
        return {
            "doctor_info": doctor_info,
            "engagement_metrics": engagement,
            "product_performance": {"product_breakdown": product_list},
            "objection_analysis": objection_analysis,
            "recommendation_engine": {
                "top_products": scored_products[:3],
                "all_products_ranked": scored_products,
                "doctor_score": round(doctor_score, 3),
                "strategy": strategy
            },
            # NEW SECTIONS
            "trend_analytics": trends,
            "product_trends": product_trends,
            "competitive_intelligence": competitive,
            "territory_comparison": territory_comparison,
            "specialty_comparison": specialty_comparison,
            "objection_resolution": objection_deep_dive
        }
    
    def get_territory_overview(self, territory: str) -> Dict[str, Any]:
        """Get complete territory overview for managers"""
        benchmarks = self.manager_engine.get_territory_benchmarks(territory)
        
        if not benchmarks:
            return None
        
        return {
            "territory_benchmarks": benchmarks,
            "doctor_list": self._get_territory_doctor_list(territory)
        }
    
    def _get_territory_doctor_list(self, territory: str) -> List[Dict[str, Any]]:
        """Get list of all doctors in territory with key metrics"""
        territory = territory.strip().lower()
        territory_df = self.df[self.df['territory'] == territory]
        
        doctors = []
        for doctor_id in territory_df['doctor_id'].unique():
            doc_df = territory_df[territory_df['doctor_id'] == doctor_id]
            
            conversion = (doc_df['outcome'].str.lower() == 'positive').sum() / len(doc_df)
            interest = doc_df['interest_level'].mean()
            
            doctors.append({
                'doctor_id': int(doctor_id),
                'doctor_name': doc_df['doctor_name'].iloc[0],
                'specialty': doc_df['specialty'].iloc[0],
                'conversion_rate': round(conversion, 3),
                'avg_interest': round(interest, 2),
                'total_interactions': len(doc_df)
            })
        
        return sorted(doctors, key=lambda x: x['conversion_rate'], reverse=True)