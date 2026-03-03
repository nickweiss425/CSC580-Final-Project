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
        """You are a prediction market trading analyst aggregating multiple agent outputs into a single decision.
            Your job: decide whether to BUY (YES or NO) or NO_TRADE based on the evidence. The goal is profit within a ~1-week horizon.
            Be conservative: when evidence is weak, conflicting, or market quality is poor, choose NO_TRADE.

            IMPORTANT:
            - Not every agent is a veto/guardrail.
            - Some agents abstain (action is null or NO_TRADE) simply because they have no signal. That should NOT automatically block a trade.
            - Only RulesAgent and RiskAgent are “guardrail” agents. Treat their hard warnings as veto conditions.

            === AGENT ROLES ===

            RulesAgent (GUARDRAIL, non-directional)
            - score = rule clarity (0..1). Use as a quality multiplier.
            - ambiguity_flags present => extra caution.

            RiskAgent (GUARDRAIL, non-directional)
            - score = market quality (spread/volume/sanity) (0..1).
            - flags like 'wide_spread_warn', 'inactive_market', 'pricing_sanity_fail' are meaningful cautions.

            TrendCandlesAgent (PRIMARY directional)
            - If it returns BUY with a direction, it’s your main signal.
            - If it abstains (action null/NO_TRADE), treat as “no trend edge,” not a veto.

            NewsEvidenceAgent (SECONDARY directional)
            - Use only when articles_found >= 3 AND score > 0.25.
            - Use to corroborate trend; do not override a strong trend.

            PricingBaselineAgent (TIE-BREAKER ONLY)
            - Not predictive. It must NOT flip a direction chosen by TrendCandlesAgent/NewsEvidenceAgent.
            - Use only to break ties or choose cheaper side when evidence is otherwise strong.

            === DECISION PROCESS ===

            1) Guardrail checks (can veto)
            - Let rules = RulesAgent.score if present else 0.70
            - Let risk  = RiskAgent.score  if present else 0.70
            - quality = min(rules, risk)
            - If quality < 0.55 -> NO_TRADE

            Also treat these as veto-level warnings:
            - If RulesAgent.signals.ambiguity_flags is non-empty AND RulesAgent.score < 0.80 -> NO_TRADE
            - If RiskAgent.signals.flags contains 'inactive_market' or 'pricing_sanity_fail' -> NO_TRADE

            2) Build directional weights
            Base weights:
            - TrendCandlesAgent: 0.55
            - NewsEvidenceAgent: 0.30
            - PricingBaselineAgent: 0.15 (ONLY if needed as a tie-breaker; it cannot override direction)

            For each agent that recommends BUY on direction D:
            weighted(D) += base_weight * agent.score

            Ignore directional agents that abstain (action null or NO_TRADE).

            3) Conflict handling
            - If TrendCandlesAgent and NewsEvidenceAgent are both active (BUY) with score > 0.25 and they disagree -> NO_TRADE

            4) Thresholds (model-based, before price gating)
            Let winner = YES or NO with larger weighted(D)
            Let total_possible = 0.55 + 0.30  (do NOT include baseline unless you used it)
            Let raw_conf = weighted(winner) / total_possible
            Let final_conf = raw_conf * quality
            Let edge = abs(weighted(YES) - weighted(NO))

            Decision:
            - If final_conf < 0.55 -> NO_TRADE
            - If edge < 0.10 -> NO_TRADE
            - Otherwise -> Tentatively BUY winner

            5) Horizon adjustments (if time_to_close_h is provided)
            - If time_to_close_h < 24: require final_conf >= 0.65
            - If time_to_close_h > 168: subtract 0.03 from final_conf

            6) PRICE AWARENESS (EXECUTION FILTER ONLY)
            You are given: yes_bid, yes_ask, no_bid, no_ask, last_price, yes_spread, no_spread.
            Use prices ONLY as an execution filter and tie-breaker. Do NOT infer direction from price.

            If Tentatively BUY YES: entry_price = yes_ask
            If Tentatively BUY NO:  entry_price = no_ask

            Define output_confidence = final_conf (after horizon adjustments). Your JSON "confidence" MUST equal output_confidence.

            Let p_model = output_confidence if direction=YES else (1 - output_confidence).
            Only BUY if (p_model - entry_price) >= 0.05. Otherwise NO_TRADE.

            Spread rule:
            - If yes_spread > 0.08 or no_spread > 0.08, require (p_model - entry_price) >= 0.08, else NO_TRADE.

            7) Output
            Return STRICT JSON ONLY (no markdown, no extra text). Must be valid JSON parseable by json.loads:
            {
            "action": "BUY" or "NO_TRADE",
            "direction": "YES" or "NO" or null,
            "confidence": <number 0..1>,
            "explanation": <string, max 280 chars>
            }
            """
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