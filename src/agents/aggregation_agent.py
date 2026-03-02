"""
LLM Aggregation Agent 


Behavior:
1) Hard veto: if ANY agent output has action == "NO_TRADE" -> return NO_TRADE (no LLM call)
2) Otherwise call LLM to aggregate and return STRICT JSON:
   {action, direction, confidence, explanation}
3) If LLM returns non-JSON, fall back to a simple deterministic rule:
   pick highest-score BUY vote; else NO_TRADE

Final return shape matches your app:
{
  "action": "BUY"|"NO_TRADE",
  "direction": "YES"|"NO"|None,
  "confidence": 0..1,
  "explanation": "...",
  "agents": agent_outputs
}
"""

import os
import asyncio
import json
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

load_dotenv()

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient


# ----------------------------
# Model + Agent setup
# ----------------------------

_model_client = OpenAIChatCompletionClient(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
)

_aggregation_agent = AssistantAgent(
    name="AggregationAgent",
    model_client=_model_client,
    system_message=(
        "You are a prediction market trading analyst. All safety vetoes have already been applied "
        "before you receive this data — every agent here has cleared guardrail checks.\n\n"

        "Your job: decide whether the directional evidence is strong enough to BUY, and if so, "
        "which direction (YES or NO). Your goal is profit within a 1-week horizon. "
        "When in doubt, NO_TRADE is the correct answer — a missed opportunity is better than a bad trade.\n\n"

        "=== UNDERSTANDING THE AGENTS ===\n\n"

        "RulesAgent (clarity signal, not a directional voter):\n"
        "  score = how clearly the market rules define resolution (0..1).\n"
        "  Use this as a confidence multiplier. score < 0.85 = ambiguous rules = reduce your confidence.\n"
        "  Check signals.ambiguity_flags — any flags present means extra caution.\n\n"

        "RiskAgent (market quality signal, not a directional voter):\n"
        "  score = composite of spread tightness, volume, and pricing sanity (0..1).\n"
        "  score < 0.60 = thin or illiquid market = reduce confidence significantly.\n"
        "  Check signals.flags — 'wide_spread_warn' or 'inactive_market' are meaningful cautions.\n\n"

        "TrendCandlesAgent (PRIMARY directional signal, weight ~0.50):\n"
        "  score range when active: ~0.55–0.95. Only trusts signals with clear momentum + MA agreement.\n"
        "  Key signals to check:\n"
        "    mom_long: % price move over ~72h. abs(mom_long) < 0.02 = no meaningful trend.\n"
        "    ma_gap: short MA minus long MA. abs(ma_gap) < 0.01 = flat, no trend.\n"
        "    vol_ratio: recent vs prior 24h volume. >= 1.20 confirms the move; < 1.0 is suspicious.\n"
        "    volatility: std of returns. >= 0.03 = noisy market, penalize confidence.\n"
        "  If this agent abstained (action=null), that is a strong signal to NO_TRADE.\n\n"

        "NewsEvidenceAgent (SECONDARY directional signal, weight ~0.35):\n"
        "  score = abs(p_yes - 0.50) * 2. Example: p_yes=0.65 → score=0.30 (weak).\n"
        "  Only meaningful when: score > 0.25 AND articles_found >= 3.\n"
        "  p_yes near 0.50 = neutral, treat as abstain.\n"
        "  Use to corroborate TrendCandlesAgent. Do not let it override trend.\n\n"

        "PricingBaselineAgent (WEAK tie-breaker only, weight ~0.15):\n"
        "  Picks the cheaper side to buy using ask prices. No predictive power on its own.\n"
        "  Only use this to break a tie between equally strong signals.\n"
        "  A large ask gap (gap > 0.15) may indicate the market already priced in the move — caution.\n\n"

        "=== HOW TO DECIDE ===\n\n"

        "1. Compute a quality multiplier from guardrail agents:\n"
        "   quality = min(RulesAgent.score, RiskAgent.score)  [use 0.70 if agent is absent]\n"
        "   If quality < 0.55 → NO_TRADE (market conditions too poor regardless of signals).\n\n"

        "2. Compute weighted directional confidence:\n"
        "   For each directional agent voting BUY on direction D, accumulate:\n"
        "     weighted(D) += base_weight * agent.score\n"
        "   Winning direction = argmax(weighted). Edge = weighted(YES) - weighted(NO).\n\n"

        "3. Apply confidence threshold:\n"
        "   raw_confidence = weighted(winner) / sum_of_base_weights\n"
        "   final_confidence = raw_confidence * quality\n"
        "   If final_confidence < 0.52 → NO_TRADE.\n"
        "   If edge < 0.08 (signals are close) → NO_TRADE.\n\n"

        "4. Check for direction conflict:\n"
        "   If TrendCandlesAgent and NewsEvidenceAgent both have score > 0.25 but disagree on direction → NO_TRADE.\n\n"

        "5. Adjust for time horizon:\n"
        "   Check close_time. If market resolves in < 24h → require final_confidence >= 0.65.\n"
        "   If market resolves in > 7 days → reduce confidence slightly (trend may not persist).\n\n"

        "6. Final output:\n"
        "   If all checks pass → BUY with winning direction.\n"
        "   Otherwise → NO_TRADE.\n\n"

        "=== OUTPUT FORMAT ===\n"
        "Return STRICT JSON ONLY. No markdown, no extra text.\n"
        "{\n"
        "  \"action\": \"BUY\" or \"NO_TRADE\",\n"
        "  \"direction\": \"YES\" or \"NO\" or null,\n"
        "  \"confidence\": <number 0..1>,\n"
        "  \"explanation\": <string, max 280 chars, name which agents drove the call and the key signals>\n"
        "}"
    ),
)


# ----------------------------
# Small helpers
# ----------------------------

def _clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except (TypeError, ValueError):
        return default


def _shorten(s: str, max_len: int = 280) -> str:
    s = (s or "").strip()
    if len(s) <= max_len:
        return s
    return s[: max_len - 1].rstrip() + "…"


def _prune_agent_outputs(agent_outputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Keep agent outputs mostly intact, but prune giant raw article payloads
    so the aggregator prompt doesn't explode.
    """
    pruned: List[Dict[str, Any]] = []
    for a in agent_outputs:
        a2 = dict(a)
        raw = a2.get("raw")

        # common case: NewsEvidenceAgent raw contains "articles" list
        if isinstance(raw, dict) and isinstance(raw.get("articles"), list):
            arts = raw.get("articles") or []
            a2["raw"] = {
                **raw,
                "articles": [
                    {
                        "title": x.get("title"),
                        "source": x.get("source"),
                        "publishedAt": x.get("publishedAt"),
                        "url": x.get("url"),
                    }
                    for x in arts[:6]
                ],
            }
        pruned.append(a2)
    return pruned


def _fallback_deterministic(agent_outputs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    If LLM fails / returns junk:
    - choose the BUY vote with the highest score
    - else NO_TRADE
    """
    buy_votes = [
        a for a in agent_outputs
        if a.get("action") == "BUY" and a.get("direction") in ("YES", "NO")
    ]

    if not buy_votes:
        conf = min(_safe_float(a.get("score"), 1.0) for a in agent_outputs)
        expl = " | ".join(a.get("reason", "") for a in agent_outputs if a.get("reason"))
        return {
            "action": "NO_TRADE",
            "direction": None,
            "confidence": _clip01(conf),
            "explanation": _shorten(expl or "No agent recommended a trade."),
            "agents": agent_outputs,
        }

    best = max(buy_votes, key=lambda a: _safe_float(a.get("score"), 0.0))
    direction = best.get("direction")

    conf = min(_safe_float(a.get("score"), 1.0) for a in agent_outputs)
    expl = " | ".join(a.get("reason", "") for a in agent_outputs if a.get("reason"))

    return {
        "action": "BUY",
        "direction": direction,
        "confidence": _clip01(conf),
        "explanation": _shorten(expl or f"Direction chosen by {best.get('agent','a directional agent')}."),
        "agents": agent_outputs,
    }


def _validate_llm_obj(obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Validate and normalize LLM JSON output.
    """
    if not isinstance(obj, dict):
        return None

    action = str(obj.get("action") or "").upper().strip()
    direction = obj.get("direction")
    if isinstance(direction, str):
        direction = direction.upper().strip()

    confidence = _clip01(_safe_float(obj.get("confidence"), 0.0))
    explanation = _shorten(str(obj.get("explanation") or "").strip())

    if action not in ("BUY", "NO_TRADE"):
        return None

    if action == "NO_TRADE":
        direction = None
    else:
        if direction not in ("YES", "NO"):
            return None

    return {
        "action": action,
        "direction": direction,
        "confidence": confidence,
        "explanation": explanation or "Aggregated recommendation.",
    }


# ----------------------------
# Main API (async + sync)
# ----------------------------

async def aggregate_recommendation(ctx: Dict[str, Any], agent_outputs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Main aggregator: ctx + agent_outputs -> FINAL recommendation dict
    """

    # 1) Hard veto: any NO_TRADE => final NO_TRADE (no LLM call)
    vetoes = [a for a in agent_outputs if a.get("action") == "NO_TRADE"]
    if vetoes:
        conf = min(_safe_float(a.get("score"), 0.0) for a in vetoes)
        expl = " | ".join(a.get("reason", "") for a in vetoes if a.get("reason"))
        return {
            "action": "NO_TRADE",
            "direction": None,
            "confidence": _clip01(conf),
            "explanation": _shorten(expl or "One or more agents vetoed this market."),
            "agents": agent_outputs,
        }

    # 2) Build compact snapshot for LLM
    market_snapshot = {
        "ticker": ctx.get("ticker"),
        "title": ctx.get("title"),
        "status": ctx.get("status"),
        "close_time": ctx.get("close_time"),
        "yes_ask": ctx.get("yes_ask"),
        "no_ask": ctx.get("no_ask"),
        "yes_spread": ctx.get("yes_spread"),
        "no_spread": ctx.get("no_spread"),
        "volume_24h": ctx.get("volume_24h"),
        "open_interest": ctx.get("open_interest"),
        "semantics": ctx.get("semantics"),
    }

    payload = {
        "market_snapshot": market_snapshot,
        "agents": _prune_agent_outputs(agent_outputs),
    }

    prompt = (
        "Aggregate the following JSON into a final recommendation.\n"
        "Return STRICT JSON only.\n\n"
        "INPUT JSON:\n"
        + json.dumps(payload, indent=2, ensure_ascii=False)
    )

    # 3) Call LLM
    result = await _aggregation_agent.run(task=[TextMessage(content=prompt, source="user")])
    text = result.messages[-1].content

    # 4) Parse JSON (or fallback)
    try:
        obj = json.loads(text)
        norm = _validate_llm_obj(obj)
        if norm is None:
            raise ValueError("LLM JSON schema invalid.")
        return {
            **norm,
            "agents": agent_outputs,
        }
    except Exception:
        # fallback deterministic
        fb = _fallback_deterministic(agent_outputs)
        fb["explanation"] = _shorten(f"{fb['explanation']} (fallback: non-JSON or invalid LLM output)")
        return fb


def aggregate_recommendation_sync(ctx: Dict[str, Any], agent_outputs: List[Dict[str, Any]]) -> Dict[str, Any]:
    return asyncio.run(aggregate_recommendation(ctx, agent_outputs))