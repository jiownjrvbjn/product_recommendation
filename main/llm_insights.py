"""
main/llm_insights.py
------------------------
PatGPT LLM Layer — v2.0

Upgrades vs v1:
  1. JSON mode on all API calls  (response_format={"type":"json_object"})
     → No more string-split parsing. All output is structured JSON.
  2. Chain-of-thought scaffold on every prompt
     → Step 1: barrier/opportunity  Step 2: best action  Step 3: draft output
  3. Prompt versioning + JSONL call log (logs/llm_calls.jsonl)
  4. Few-shot injection in generate_meeting_playbook
     → Loads data/few_shot_examples.json, selects 2 by cosine similarity
  5. Doctor narrative memory (_build_doctor_narrative)
     → 150-200 word prose summary cached by doctor_id + last_interaction_date
     → Passed as system message context
  6. Objection RAG
     → Loads data/objection_embeddings.faiss + data/objection_metadata.json
     → Retrieves top-3 similar past objections at call time
  7. Exponential backoff (3 retries, 2s base) on every API call
  8. Structured fallback dicts — never a plain error string

Public method signatures UNCHANGED:
  generate_doctor_insights(analytics_data, focus_areas) → Dict
  generate_meeting_playbook(summary)                     → str
  explain_product_underperformance(product_name, metrics, objections) → str
"""

from __future__ import annotations

import json
import logging
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

import config.azure_sate as azure_state

logger = logging.getLogger("patgpt.llm_insights")

# ── Prompt version — bump when prompts change ─────────────────────────────────
PROMPT_VERSION = "2.0.0"

# ── Paths ─────────────────────────────────────────────────────────────────────
_LOGS_DIR          = Path("logs")
_DATA_DIR          = Path("data")
_LLM_LOG           = _LOGS_DIR / "llm_calls.jsonl"
_FEW_SHOT_PATH     = _DATA_DIR / "few_shot_examples.json"
_OBJ_FAISS_PATH    = _DATA_DIR / "objection_embeddings.faiss"
_OBJ_META_PATH     = _DATA_DIR / "objection_metadata.json"

_LOGS_DIR.mkdir(parents=True, exist_ok=True)

# ── Chain-of-thought scaffold (injected into every prompt) ────────────────────
_COT_PREFIX = (
    "Think step by step before answering:\n"
    "Step 1: What is the single biggest barrier or opportunity for this doctor right now?\n"
    "Step 2: What is the one action that would most move the needle?\n"
    "Step 3: Draft the output below, strictly following the JSON schema provided.\n\n"
)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _cosine_sim(a: List[float], b: List[float]) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _log_call(
    doctor_id: str,
    method: str,
    latency_ms: float,
    token_count: int,
    success: bool,
) -> None:
    record = {
        "timestamp":     datetime.utcnow().isoformat(),
        "doctor_id":     doctor_id,
        "method":        method,
        "prompt_version": PROMPT_VERSION,
        "latency_ms":    round(latency_ms, 1),
        "token_count":   token_count,
        "outcome_logged": False,
        "success":       success,
    }
    try:
        with open(_LLM_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception:
        pass


def _call_with_backoff(
    client,
    deployment: str,
    messages: List[Dict],
    max_tokens: int = 1500,
    use_json_mode: bool = True,
    retries: int = 3,
    base_delay: float = 2.0,
) -> Optional[str]:
    """
    Call Azure OpenAI with exponential backoff.
    Returns the raw content string, or None after all retries fail.
    JSON mode is on by default — prompts MUST instruct the model to return JSON.
    """
    kwargs: Dict[str, Any] = {
        "model":    deployment,
        "messages": messages,
        "max_completion_tokens": max_tokens,
        "temperature": 1,
    }
    if use_json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(**kwargs)
            content = resp.choices[0].message.content
            return content if content else None
        except Exception as e:
            wait = base_delay * (2 ** attempt)
            logger.warning(f"LLM call failed (attempt {attempt+1}/{retries}): {e}. Retrying in {wait}s…")
            if attempt < retries - 1:
                time.sleep(wait)

    logger.error("LLM call failed after all retries.")
    return None


def _parse_json(raw: Optional[str], fallback: Dict) -> Dict:
    """Safely parse JSON response; return fallback on any failure."""
    if not raw:
        return fallback
    try:
        # Strip accidental markdown fences
        clean = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        return json.loads(clean)
    except Exception:
        return fallback


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class LLMInsightsEngineEnhanced:
    """
    Enhanced LLM insights.
    Called only when user explicitly requests AI narrative.
    All three public method signatures are unchanged.
    """

    PROMPT_VERSION = PROMPT_VERSION

    def __init__(self, deployment: str = "o4-mini"):
        self.deployment = deployment
        self.client = azure_state.client
        if not self.client:
            raise RuntimeError("Azure OpenAI client not initialized")

        # ── Few-shot library ──────────────────────────────────────────────
        self._few_shot_examples: List[Dict] = []
        if _FEW_SHOT_PATH.exists():
            try:
                with open(_FEW_SHOT_PATH, encoding="utf-8") as f:
                    self._few_shot_examples = json.load(f)
                logger.info(f"Loaded {len(self._few_shot_examples)} few-shot examples")
            except Exception as e:
                logger.warning(f"Few-shot load failed: {e}")

        # ── Objection FAISS index ─────────────────────────────────────────
        self._obj_index = None
        self._obj_meta: List[Dict] = []
        if _OBJ_FAISS_PATH.exists() and _OBJ_META_PATH.exists():
            try:
                import faiss  # type: ignore
                self._obj_index = faiss.read_index(str(_OBJ_FAISS_PATH))
                with open(_OBJ_META_PATH, encoding="utf-8") as f:
                    self._obj_meta = json.load(f)
                logger.info(f"Loaded objection FAISS index ({self._obj_index.ntotal} vectors)")
            except Exception as e:
                logger.warning(f"Objection FAISS load failed: {e}")

        # ── Narrative cache: {doctor_id + last_date → narrative_str} ─────
        self._narrative_cache: Dict[str, str] = {}

    # ──────────────────────────────────────────────────────────────────────────
    # PUBLIC METHOD 1 — generate_doctor_insights
    # ──────────────────────────────────────────────────────────────────────────

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
        doctor_id       = str(doctor_info.get("doctor_id", "unknown"))

        t0 = time.time()
        basic_insights  = self._generate_recommendation_block(
            doctor_id, doctor_info, engagement, recommendations, aida, persona
        )
        trend_narrative = self._generate_trend_narrative(analytics_data)
        latency = (time.time() - t0) * 1000
        _log_call(doctor_id, "generate_doctor_insights", latency, 0, True)

        return {
            **basic_insights,
            "trend_narrative": trend_narrative,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # PUBLIC METHOD 2 — generate_meeting_playbook
    # ──────────────────────────────────────────────────────────────────────────

    def generate_meeting_playbook(self, summary: Dict[str, Any]) -> str:
        """
        Builds a concise, persona-aware meeting playbook.

        Enhancements vs v1:
          - JSON mode output → structured playbook dict → formatted string
          - Chain-of-thought prefix
          - Doctor narrative summary injected as system message
          - 2 few-shot examples injected by cosine similarity
          - Top-3 similar past objections from FAISS injected into prompt
        """
        doctor_info  = summary.get("doctor_info", {})
        aida         = summary.get("aida", {})
        persona      = summary.get("persona", {})
        last_meeting = summary.get("last_meeting", {})
        top_products = summary.get("top_historical_products", [])
        objections   = summary.get("objection_analysis", {}).get("objection_breakdown", {})
        engagement   = summary.get("engagement_metrics", {})
        doctor_id    = str(doctor_info.get("doctor_id", "unknown"))

        notes            = last_meeting.get("meeting_notes") or "No notes recorded"
        conv_rate        = engagement.get("conversion_rate", 0)
        avg_duration_min = round((engagement.get("avg_meeting_duration_sec") or 0) / 60, 1)

        top_products_text = "\n".join([
            f"- {p.get('product_name', '')}: presented {p.get('times_presented', 0)} times, "
            f"avg {p.get('avg_time_per_presentation', 0)} sec"
            for p in top_products
        ]) or "No historical product data"

        common_objections = ", ".join(list(objections.keys())[:3]) if objections else "none"

        # ── Doctor narrative (system message) ────────────────────────────
        narrative = self._get_or_build_narrative(summary)

        # ── Objection RAG ────────────────────────────────────────────────
        rag_block = self._retrieve_similar_objections(list(objections.keys()))

        # ── Few-shot examples ────────────────────────────────────────────
        few_shot_turns = self._select_few_shot_examples(summary, n=2)

        # ── Schema ───────────────────────────────────────────────────────
        schema = json.dumps({
            "opening_line":     "<string — opening sentence tailored to AIDA stage and persona>",
            "talking_point_1":  "<string — first key talking point>",
            "talking_point_2":  "<string — second key talking point>",
            "product_focus":    "<string — product name and one-line reason>",
            "closing_question": "<string — question to move doctor forward in funnel>",
        }, indent=2)

        user_content = (
            _COT_PREFIX
            + f"""You are a pharma sales coach. Return ONLY a JSON object matching this schema:
{schema}

Doctor: {doctor_info.get('doctor_name')} (ID: {doctor_id})
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

{rag_block}

Return ONLY the JSON object. No extra text."""
        )

        messages: List[Dict] = [
            {"role": "system", "content": f"You are a concise pharmaceutical sales assistant.\n\n{narrative}"},
        ]
        messages.extend(few_shot_turns)
        messages.append({"role": "user", "content": user_content})

        t0      = time.time()
        raw     = _call_with_backoff(self.client, self.deployment, messages, max_tokens=800)
        latency = (time.time() - t0) * 1000
        _log_call(doctor_id, "generate_meeting_playbook", latency, 0, raw is not None)

        fallback_dict = self._rule_based_playbook(summary)
        parsed  = _parse_json(raw, fallback_dict)

        return self._format_playbook(parsed)

    # ──────────────────────────────────────────────────────────────────────────
    # PUBLIC METHOD 3 — explain_product_underperformance
    # ──────────────────────────────────────────────────────────────────────────

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

        schema = json.dumps({
            "root_cause":       "<2-3 sentences explaining why this product underperforms>",
            "recommendation_1": "<specific actionable recommendation>",
            "recommendation_2": "<specific actionable recommendation>",
            "recommendation_3": "<specific actionable recommendation>",
            "quick_win":        "<single tactic implementable this week>",
        }, indent=2)

        user_content = (
            _COT_PREFIX
            + f"""You are a pharmaceutical sales strategist. Return ONLY a JSON object matching this schema:
{schema}

Product: {product_name}
Total Sales Volume: {metrics.get('total_sales', 0)}
Total Interactions: {metrics.get('total_interactions', 0)}
Conversion Rate: {metrics.get('conversion_rate', 0):.1%}
Average Interest: {metrics.get('avg_interest', 0):.1f}/5
QoQ Growth: {metrics.get('qoq_growth', 0):.1%}
Sales Trend: {metrics.get('trend', 'stable')}
Engagement Depth: {engagement_note}
Top Objections: {top_objections}

Return ONLY the JSON object. No extra text."""
        )

        messages = [
            {"role": "system", "content": "You are a pharmaceutical sales performance analyst."},
            {"role": "user",   "content": user_content},
        ]

        t0      = time.time()
        raw     = _call_with_backoff(self.client, self.deployment, messages, max_tokens=1000)
        latency = (time.time() - t0) * 1000
        _log_call("product:" + product_name, "explain_product_underperformance", latency, 0, raw is not None)

        fallback = {
            "root_cause":       "Insufficient data to determine root cause.",
            "recommendation_1": "Increase visit frequency for this product.",
            "recommendation_2": "Address top objections with clinical evidence.",
            "recommendation_3": "Retrain reps on product positioning.",
            "quick_win":        "Schedule a product-focused refresher session this week.",
        }
        parsed = _parse_json(raw, fallback)

        # Return formatted string (backward-compatible with frontend)
        return (
            f"**Root Cause**\n{parsed.get('root_cause', '')}\n\n"
            f"**Recommendations**\n"
            f"1. {parsed.get('recommendation_1', '')}\n"
            f"2. {parsed.get('recommendation_2', '')}\n"
            f"3. {parsed.get('recommendation_3', '')}\n\n"
            f"**Quick Win**\n{parsed.get('quick_win', '')}"
        )

    # ──────────────────────────────────────────────────────────────────────────
    # PRIVATE — recommendation block (used by generate_doctor_insights)
    # ──────────────────────────────────────────────────────────────────────────

    def _generate_recommendation_block(
        self, doctor_id, doctor_info, engagement, recommendations, aida, persona
    ) -> Dict[str, Any]:
        top_products   = self._format_top_products(recommendations.get("top_products", []))
        conversion_pct = (engagement.get("conversion_rate", 0) or 0) * 100

        schema = json.dumps({
            "best_product":    "<product name + one-line reason>",
            "similar_products": "<bullet list of 1-2 alternative products with reasons>",
            "doctor_value":    "<High | Medium | Low + 1-sentence reasoning>",
            "suggestion":      "<1-2 actionable, persona-aware tips as a string>",
        }, indent=2)

        user_content = (
            _COT_PREFIX
            + f"""You are a pharma sales expert. Recommend actions, do NOT summarise data.
Return ONLY a JSON object matching this schema:
{schema}

Doctor: {doctor_info.get('specialty')}, {doctor_info.get('experience_years')} yrs exp
Conversion: {conversion_pct:.1f}%
Interest: {engagement.get('avg_interest_level', 0)}/5
AIDA: {aida.get('aida_label', 'Unknown')} (confidence {int((aida.get('aida_confidence', 0))*100)}%)
Persona: {persona.get('label', 'Unknown')} — {persona.get('description', '')}
Top Products: {top_products}
Doctor Score: {recommendations.get('doctor_score', 0)}

Return ONLY the JSON object. No extra text."""
        )

        messages = [
            {"role": "system", "content": "You are a pharmaceutical sales decision assistant."},
            {"role": "user",   "content": user_content},
        ]

        raw = _call_with_backoff(self.client, self.deployment, messages, max_tokens=800)
        fallback = {
            "best_product": "", "similar_products": "",
            "doctor_value": "", "suggestion": "",
        }
        return _parse_json(raw, fallback)

    # ──────────────────────────────────────────────────────────────────────────
    # PRIVATE — trend narrative (pure rule-based, no LLM call needed)
    # ──────────────────────────────────────────────────────────────────────────

    def _generate_trend_narrative(self, analytics_data: Dict[str, Any]) -> str:
        trends = analytics_data.get("trend_analytics")
        if not trends:
            return "Insufficient data."
        trend_summary = trends.get("trends", {})
        icon_map = {"improving": "📈", "declining": "📉", "stable": "➡️"}
        parts = []
        for metric in ("conversion", "interest", "sales"):
            val = trend_summary.get(metric)
            if val:
                parts.append(f"{icon_map.get(val, '➡️')} {metric.capitalize()} {val}")
        return " | ".join(parts) or "Stable"

    # ──────────────────────────────────────────────────────────────────────────
    # PRIVATE — doctor narrative memory
    # ──────────────────────────────────────────────────────────────────────────

    def _get_or_build_narrative(self, summary: Dict[str, Any]) -> str:
        """
        Return a 150-200 word prose narrative for the doctor.
        Cached by doctor_id + last_interaction_date.
        """
        doctor_id   = str(summary.get("doctor_info", {}).get("doctor_id", "unknown"))
        last_date   = str(summary.get("last_meeting", {}).get("date", ""))
        cache_key   = f"{doctor_id}::{last_date}"

        if cache_key in self._narrative_cache:
            return self._narrative_cache[cache_key]

        narrative = self._build_doctor_narrative(summary)
        self._narrative_cache[cache_key] = narrative
        return narrative

    def _build_doctor_narrative(self, summary: Dict[str, Any]) -> str:
        """
        Build a 150-200 word narrative covering:
          - AIDA progression history
          - Recurring objection themes
          - Product acceptance pattern
          - Last commitment made
        Falls back to rule-based summary if LLM call fails.
        """
        doctor_info  = summary.get("doctor_info", {})
        aida         = summary.get("aida", {})
        persona      = summary.get("persona", {})
        engagement   = summary.get("engagement_metrics", {})
        last_meeting = summary.get("last_meeting", {})
        objections   = summary.get("objection_resolution", {})
        top_products = summary.get("top_historical_products", [])

        top_obj_list = list(
            (objections.get("objection_breakdown") or {}).keys()
        )[:3]
        top_prod_names = [p.get("product_name", "") for p in top_products[:2]]
        last_commitment = last_meeting.get("meeting_notes") or "none recorded"

        schema = json.dumps({"narrative": "<150-200 word prose narrative>"}, indent=2)

        user_content = (
            f"Write a 150-200 word internal doctor narrative summary for a pharma sales rep.\n"
            f"Return ONLY a JSON object: {schema}\n\n"
            f"Doctor: {doctor_info.get('doctor_name')}, {doctor_info.get('specialty')}\n"
            f"AIDA stage: {aida.get('aida_label', 'Unknown')} "
            f"(confidence {int(aida.get('aida_confidence', 0)*100)}%)\n"
            f"Persona: {persona.get('label', 'Unknown')}\n"
            f"Conversion rate: {engagement.get('conversion_rate', 0):.0%}\n"
            f"Total interactions: {engagement.get('total_interactions', 0)}\n"
            f"Recurring objections: {', '.join(top_obj_list) or 'none'}\n"
            f"Top products by engagement: {', '.join(top_prod_names) or 'none'}\n"
            f"Last meeting notes / commitment: {last_commitment}\n\n"
            f"Cover: AIDA progression, recurring objections, product acceptance pattern, "
            f"last commitment. Return ONLY the JSON object."
        )

        messages = [
            {"role": "system", "content": "You are a CRM analyst writing internal doctor summaries."},
            {"role": "user",   "content": user_content},
        ]

        raw = _call_with_backoff(self.client, self.deployment, messages, max_tokens=400)
        parsed = _parse_json(raw, {})
        narrative = parsed.get("narrative", "")

        if not narrative:
            # Rule-based fallback
            narrative = (
                f"Dr. {doctor_info.get('doctor_name', 'Unknown')} ({doctor_info.get('specialty', '')}) "
                f"is currently in the {aida.get('aida_label', 'Awareness')} stage with "
                f"{engagement.get('conversion_rate', 0):.0%} conversion across "
                f"{engagement.get('total_interactions', 0)} interactions. "
                f"Persona: {persona.get('label', 'Analytical')}. "
                f"Top recurring objections: {', '.join(top_obj_list) or 'none recorded'}. "
                f"Strongest product engagement on {', '.join(top_prod_names) or 'our portfolio'}. "
                f"Last commitment: {last_commitment}."
            )

        return narrative

    # ──────────────────────────────────────────────────────────────────────────
    # PRIVATE — few-shot injection
    # ──────────────────────────────────────────────────────────────────────────

    def _select_few_shot_examples(
        self, summary: Dict[str, Any], n: int = 2
    ) -> List[Dict]:
        """
        Select n examples from few_shot_examples.json by cosine similarity
        of the current doctor's feature vector to each example's feature vector.

        Each example in the JSON should have:
          {
            "doctor_features": {"aida_stage": ..., "persona": ..., ...},
            "ideal_playbook": { <same schema as generate_meeting_playbook output> }
          }

        Returns a list of {role, content} message turns for injection.
        """
        if not self._few_shot_examples:
            return []

        current_vec = self._summary_to_feature_vec(summary)
        scored = []
        for ex in self._few_shot_examples:
            ex_vec = self._summary_to_feature_vec(ex.get("doctor_features", {}))
            sim    = _cosine_sim(current_vec, ex_vec)
            scored.append((sim, ex))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = [ex for _, ex in scored[:n]]

        turns: List[Dict] = []
        for ex in top:
            playbook = ex.get("ideal_playbook", {})
            turns.append({
                "role":    "user",
                "content": f"Example doctor context: {json.dumps(ex.get('doctor_features', {}))}\nGenerate playbook JSON:",
            })
            turns.append({
                "role":    "assistant",
                "content": json.dumps(playbook),
            })
        return turns

    def _summary_to_feature_vec(self, summary: Any) -> List[float]:
        """
        Convert a summary dict (or doctor_features dict) to a simple
        numeric feature vector for cosine similarity.
        Works on both full summary dicts and flat doctor_features dicts.
        """
        aida_map = {"awareness": 0, "interest": 1, "desire": 2, "action": 3}
        persona_map = {"analytical": 0, "emotional": 1, "fast_decision": 2, "resistant": 3, "balanced": 4}

        if isinstance(summary, dict) and "doctor_info" in summary:
            # Full summary dict
            aida_stage = summary.get("aida", {}).get("aida_stage", "awareness")
            persona    = summary.get("persona", {}).get("persona", "analytical")
            conv       = summary.get("engagement_metrics", {}).get("conversion_rate", 0)
            interest   = summary.get("engagement_metrics", {}).get("avg_interest_level", 0)
        else:
            # Flat dict (from few_shot_examples.json doctor_features)
            aida_stage = str(summary.get("aida_stage", "awareness")).lower()
            persona    = str(summary.get("persona", "analytical")).lower()
            conv       = float(summary.get("conversion_rate", 0))
            interest   = float(summary.get("avg_interest", 0))

        return [
            float(aida_map.get(str(aida_stage).lower(), 0)) / 3.0,
            float(persona_map.get(str(persona).lower(), 0)) / 4.0,
            min(float(conv), 1.0),
            min(float(interest) / 5.0, 1.0),
        ]

    # ──────────────────────────────────────────────────────────────────────────
    # PRIVATE — objection RAG
    # ──────────────────────────────────────────────────────────────────────────

    def _retrieve_similar_objections(self, current_objections: List[str]) -> str:
        """
        Retrieve top-3 semantically similar past objections from FAISS.
        Returns a formatted string to inject into the prompt, or empty string
        if FAISS index is not loaded or current_objections is empty.
        """
        if self._obj_index is None or not self._obj_meta or not current_objections:
            return ""

        query_text = " | ".join(current_objections[:3])
        embedding  = self._embed_text(query_text)
        if embedding is None:
            return ""

        try:
            import faiss  # type: ignore
            query_vec = np.array([embedding], dtype=np.float32)
            faiss.normalize_L2(query_vec)
            _, indices = self._obj_index.search(query_vec, 3)

            results = []
            for idx in indices[0]:
                if 0 <= idx < len(self._obj_meta):
                    m = self._obj_meta[idx]
                    results.append(
                        f"- Objection: \"{m.get('objection', '')}\" → "
                        f"Resolution: \"{m.get('resolution', '')}\" "
                        f"(outcome: {m.get('outcome', 'unknown')})"
                    )

            if results:
                return (
                    "Past objections similar to this doctor's history and how they were resolved:\n"
                    + "\n".join(results) + "\n"
                )
        except Exception as e:
            logger.warning(f"Objection RAG retrieval failed: {e}")

        return ""

    def _embed_text(self, text: str) -> Optional[List[float]]:
        """Embed text using Azure OpenAI text-embedding-3-small."""
        try:
            resp = self.client.embeddings.create(
                model="text-embedding-3-small",
                input=text,
            )
            return resp.data[0].embedding
        except Exception as e:
            logger.warning(f"Embedding failed: {e}")
            return None

    # ──────────────────────────────────────────────────────────────────────────
    # PRIVATE — helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _format_playbook(self, parsed: Dict[str, Any]) -> str:
        """Convert structured JSON playbook dict to formatted string."""
        return (
            f"**Opening Line**\n{parsed.get('opening_line', '—')}\n\n"
            f"**Key Talking Points**\n"
            f"1. {parsed.get('talking_point_1', '—')}\n"
            f"2. {parsed.get('talking_point_2', '—')}\n\n"
            f"**Suggested Product Focus**\n{parsed.get('product_focus', '—')}\n\n"
            f"**Closing Question**\n{parsed.get('closing_question', '—')}"
        )

    def _rule_based_playbook(self, summary: Dict[str, Any]) -> Dict[str, Any]:
        """Structured fallback dict — same schema as JSON mode output."""
        doctor_info  = summary.get("doctor_info", {})
        aida         = summary.get("aida", {})
        persona      = summary.get("persona", {})
        top_products = summary.get("top_historical_products", [])

        stage        = aida.get("aida_stage", "awareness")
        persona_type = persona.get("persona", "analytical")
        doc_name     = doctor_info.get("doctor_name", "the doctor")
        specialty    = doctor_info.get("specialty", "your specialty")
        top_product  = ", ".join([p.get("product_name", "") for p in top_products[:2]]) or "our key product"

        opening_lines = {
            "awareness": f"Good morning, Dr. {doc_name}. I'd like to briefly share a new approach that addresses {specialty} challenges.",
            "interest":  f"Dr. {doc_name}, I noticed you've shown interest in {top_product} – let me share some clinical data that may be useful.",
            "desire":    f"Great to see your growing interest, Dr. {doc_name}. I have a patient success story that reinforces why {top_product} fits your practice.",
            "action":    f"Thank you for your continued trust, Dr. {doc_name}. Today I'd like to discuss how we can expand your results with complementary options.",
        }
        talking = {
            "analytical":    ("Highlight recent peer-reviewed study results.", "Compare efficacy data vs. alternatives."),
            "emotional":     ("Share a patient story demonstrating improved adherence.", "Ask about a memorable patient case."),
            "fast_decision": ("Present top-line value in 30 seconds.", "Offer a limited-time trial or sample."),
            "resistant":     ("Address the most common objection without being pushy.", "Ask what would need to change for them to reconsider."),
            "balanced":      ("Present balanced evidence and build rapport.", "Offer a sample and ask for feedback."),
        }
        closing = {
            "awareness": "Would you be open to reviewing a one-pager before our next visit?",
            "interest":  "Shall I leave a trial sample for you to evaluate with a few patients?",
            "desire":    "Can we schedule a follow-up to discuss starting a pilot prescription?",
            "action":    "Which of these two complementary products would you like to introduce next?",
        }

        tp = talking.get(persona_type, talking["analytical"])
        return {
            "opening_line":     opening_lines.get(stage, opening_lines["awareness"]),
            "talking_point_1":  tp[0],
            "talking_point_2":  tp[1],
            "product_focus":    f"{top_product} (based on historical engagement)",
            "closing_question": closing.get(stage, closing["awareness"]),
        }

    def _format_top_products(self, products: list) -> str:
        if not products:
            return "No products"
        return "\n".join([
            f"- {p.get('product_name', '')} "
            f"(conv: {p.get('conversion_rate', 0):.1%}, interest: {p.get('avg_interest', 0):.1f}/5)"
            for p in products
        ])