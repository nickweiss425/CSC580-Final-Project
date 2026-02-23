"""
Simple NewsEvidenceAgent (Directional) — from scratch

Workflow:
1) Use LLM to create a NewsAPI query from ctx (title + rules).
2) Fetch recent articles with NewsAPI.
3) Use LLM to estimate P(YES) from those articles.
4) Map P(YES) to BUY YES/NO or abstain.

Standardized AgentOutput:
{
  "agent": "NewsEvidenceAgent",
  "action": "BUY" or None,
  "direction": "YES" | "NO" | None,
  "score": 0..1,
  "reason": str,
  "signals": dict,
  "raw": dict
}
"""

from __future__ import annotations

import os
import re
import json
import asyncio
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta

from dotenv import load_dotenv
from newsapi import NewsApiClient

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient

load_dotenv()


# -----------------------------
# Small helpers
# -----------------------------

def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _get_str(ctx: Dict[str, Any], key: str) -> str:
    v = ctx.get(key, "")
    return str(v).strip() if v is not None else ""


def _strip_code_fences(text: str) -> str:
    t = (text or "").strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    return t.strip()


def _safe_json_loads(text: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(_strip_code_fences(text))
    except Exception:
        return None


def _decision_from_p_yes(p_yes: float) -> Tuple[Optional[str], Optional[str]]:
    if p_yes >= 0.55:
        return "BUY", "YES"
    if p_yes <= 0.45:
        return "BUY", "NO"
    return None, None


def _edge_score(p_yes: float) -> float:
    # directional edge strength
    return _clamp(abs(p_yes - 0.5) * 2.0)


# -----------------------------
# Clients
# -----------------------------

_model_client = OpenAIChatCompletionClient(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
)

_news_client = NewsApiClient(api_key=os.getenv("NEWSAPI_KEY"))


# -----------------------------
# LLM Agents
# -----------------------------

_query_builder = AssistantAgent(
    name="NewsQueryBuilder",
    model_client=_model_client,
    system_message=(
        "You generate a NewsAPI query string for the `q` parameter (NewsAPI /v2/everything).\n"
        "Goal: retrieve SOME relevant articles even if exact wording varies.\n\n"
        "Input: market title + rules. Output: STRICT JSON only:\n"
        "{ \"query\": string }\n\n"
        "Guidelines (follow all):\n"
        "- Be BROAD enough to return results. Avoid stacking many long quoted phrases.\n"
        "- Use at most ONE quoted phrase total.\n"
        "- Prefer this pattern: <main_entity> AND (<syn1> OR <syn2> OR <syn3>)\n"
        "- Include 2–5 short keywords total (not sentences). Prefer OR groups for synonyms.\n"
        "- Avoid generic filler words (e.g., 'will', 'strictly', 'greater', 'not', dates unless critical).\n"
        "- If the market is about sports championships, include synonyms like: NCAA OR \"March Madness\" OR \"national title\".\n"
        "- If about weather totals, include: snow OR snowfall OR storm OR forecast OR inches.\n"
        "- If about a person saying a word/phrase, include the person + (speech OR transcript OR remarks) + the target word(s).\n"
        "- Keep query under 120 characters if possible.\n"
        "- Return STRICT JSON only. No extra keys.\n"
    ),
)

_evidence_model = AssistantAgent(
    name="NewsEvidenceModel",
    model_client=_model_client,
    system_message=(
        "You are an evidence-based analyst for binary markets.\n"
        "You will be given market title + rules and a list of news articles.\n"
        "Estimate the probability the market resolves YES based ONLY on the provided articles.\n\n"
        "Return STRICT JSON only:\n"
        "{\n"
        "  \"p_yes\": number (0..1),\n"
        "  \"confidence\": number (0..1),\n"
        "  \"summary\": string\n"
        "}\n\n"
        "Calibration rules:\n"
        "- If articles are mostly indirect (rankings, odds, punditry, general discussion), keep p_yes near 0.50 and confidence low.\n"
        "- Use high p_yes (>0.65) ONLY when articles contain strong, direct evidence tied to the resolution criteria.\n"
        "- For long-horizon outcomes (e.g., season champion), even the favorite is usually not \"very likely\"; avoid extreme probabilities.\n"
        "- If evidence is weak/unrelated, set p_yes≈0.50 and confidence low.\n"
    ),
)

# -----------------------------
# News fetch
# -----------------------------

async def _fetch_news_articles(query: str, max_articles: int = 10, lookback_days: int = 7) -> List[Dict[str, Any]]:
    try:
        from_date = (datetime.utcnow() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        resp = _news_client.get_everything(
            q=query,
            from_param=from_date,
            language="en",
            sort_by="relevancy",
            page_size=max_articles,
        )
        raw = resp.get("articles", []) or []
        out: List[Dict[str, Any]] = []
        for a in raw:
            src = a.get("source") or {}
            out.append({
                "title": a.get("title", "") or "",
                "description": a.get("description", "") or "",
                "content": a.get("content", "") or "",
                "url": a.get("url", "") or "",
                "publishedAt": a.get("publishedAt", "") or "",
                "source": (src.get("name") or "") if isinstance(src, dict) else str(src),
            })
        return out
    except Exception as e:
        print(f"Error fetching news: {e}")
        return []


# -----------------------------
# Main agent
# -----------------------------

async def run_news_evidence_agent(ctx: Dict[str, Any]) -> Dict[str, Any]:
    title = _get_str(ctx, "title")
    rules_primary = _get_str(ctx, "rules_primary")
    rules_secondary = _get_str(ctx, "rules_secondary")
    event_ticker = _get_str(ctx, "event_ticker")

    if not title:
        return {
            "agent": "NewsEvidenceAgent",
            "action": None,
            "direction": None,
            "score": 0.0,
            "reason": "Missing title; cannot run news analysis.",
            "signals": {},
            "raw": {},
        }

    # 1) Ask LLM to build query
    qb_payload = {
        "title": title,
        "event_ticker": event_ticker,
        "rules_primary": rules_primary,
        "rules_secondary": rules_secondary,
    }
    qb_prompt = (
        "Create a NewsAPI query for this market.\n"
        "Return STRICT JSON only.\n\n"
        f"{json.dumps(qb_payload, ensure_ascii=False)}"
    )

    query = title  # fallback
    qb_raw: Dict[str, Any] = {}
    try:
        qb_result = await _query_builder.run(task=[TextMessage(content=qb_prompt, source='user')])
        qb_text = qb_result.messages[-1].content if qb_result.messages else ""
        qb_data = _safe_json_loads(qb_text) or {}
        qb_raw = qb_data if isinstance(qb_data, dict) else {}
        if isinstance(qb_raw.get("query", None), str) and qb_raw["query"].strip():
            query = qb_raw["query"].strip()
    except Exception as e:
        qb_raw = {"error": str(e)}

    # 2) Fetch articles
    articles = await _fetch_news_articles(query, max_articles=10, lookback_days=7)

    if not articles:
        return {
            "agent": "NewsEvidenceAgent",
            "action": None,
            "direction": None,
            "score": 0.0,
            "reason": "No relevant news articles found; abstaining.",
            "signals": {"query": query, "articles_found": 0},
            "raw": {"query_builder": qb_raw, "articles": []},
        }

    # 3) Ask LLM to estimate P(YES)
    ev_payload = {
        "title": title,
        "rules_primary": rules_primary,
        "rules_secondary": rules_secondary,
        "articles": articles,
    }
    ev_prompt = (
        "Estimate P(YES) for this market using only the provided articles.\n"
        "Return STRICT JSON only.\n\n"
        f"{json.dumps(ev_payload, ensure_ascii=False)}"
    )

    p_yes = 0.5
    conf = 0.0
    summary = ""
    ev_raw: Dict[str, Any] = {}

    try:
        ev_result = await _evidence_model.run(task=[TextMessage(content=ev_prompt, source='user')])
        ev_text = ev_result.messages[-1].content if ev_result.messages else ""
        ev_data = _safe_json_loads(ev_text) or {}
        ev_raw = ev_data if isinstance(ev_data, dict) else {}

        p_yes = _clamp(float(ev_raw.get("p_yes", 0.5)))
        conf = _clamp(float(ev_raw.get("confidence", 0.0)))
        summary = str(ev_raw.get("summary", "") or "")
    except Exception as e:
        ev_raw = {"error": str(e)}
        summary = "Evidence analysis failed."

    action, direction = _decision_from_p_yes(p_yes)
    score = _edge_score(p_yes)

    if action is None:
        reason = f"P(YES)≈{p_yes:.2f} (no clear edge). confidence={conf:.2f}. {summary}".strip()
    else:
        reason = f"P(YES)≈{p_yes:.2f}; BUY {direction}. edge_score={score:.2f}, confidence={conf:.2f}. {summary}".strip()

    return {
        "agent": "NewsEvidenceAgent",
        "action": action,
        "direction": direction,
        "score": float(score),
        "reason": reason,
        "signals": {
            "query": query,
            "p_yes": float(p_yes),
            "confidence": float(conf),
            "articles_found": len(articles),
        },
        "raw": {
            "query_builder": qb_raw,
            "articles": articles,
            "evidence": ev_raw,
            "rule": "LLM builds query; LLM estimates P(YES) from fetched articles; threshold to BUY/abstain.",
        },
    }


def run_news_evidence_agent_sync(ctx: Dict[str, Any]) -> Dict[str, Any]:
    return asyncio.run(run_news_evidence_agent(ctx))