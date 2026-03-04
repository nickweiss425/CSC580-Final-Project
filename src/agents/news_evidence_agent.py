"""
NewsEvidenceAgent (Directional) — from scratch

Workflow:
1) Use LLM to create a NewsAPI query from ctx (title + rules + close_time).
2) Sanitize the query to remove special characters that break NewsAPI.
3) Fetch recent articles with NewsAPI (dynamic lookback based on market timing).
4) Use LLM to filter irrelevant articles, then estimate P(YES).
5) Map P(YES) to BUY YES/NO or abstain.

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

# Sanitize NewsAPI query to remove special characters that break NewsAPI
def _sanitize_query(query: str) -> str:
    # Remove special characters
    cleaned = re.sub(r'[^\w\s"()+\-]', ' ', query)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()

    # Truncate to NewsAPI's 500-char limit
    if len(cleaned) > 500:
        cleaned = cleaned[:500].rsplit(' ', 1)[0]

    return cleaned

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
        "You generate a NewsAPI query string for the `q` parameter of the /v2/everything endpoint.\n"
        "Goal: build a SHORT, BROAD query that will retrieve relevant English-language news articles.\n\n"

        "=== NEWSAPI QUERY SYNTAX (follow exactly) ===\n"
        "Allowed operators: AND, OR, NOT, double-quotes for exact phrases, + (must appear), - (must not appear), parentheses for grouping.\n"
        "FORBIDDEN characters (will break the API): #, $, %, @, &, *, {, }, [, ], |, \\, ;, <, >, /, !, ^, ~, `, :, comma.\n"
        "Do NOT include any URLs, dates, or timestamps in the query string.\n"
        "Max length: 500 characters. Aim for under 120.\n\n"

        "=== QUERY CONSTRUCTION GUIDELINES ===\n"
        "1. Extract the CORE SUBJECT from the market title (person, team, event, or topic).\n"
        "2. Use this pattern: <core_subject> AND (<keyword1> OR <keyword2> OR <keyword3>)\n"
        "3. Use 2-5 short, concrete keywords. Prefer common nouns and verbs journalists would use.\n"
        "4. Use at most ONE quoted phrase (for a proper name or specific term). Keep it short (2-3 words max).\n"
        "5. Do NOT use full sentences, questions, or generic filler words like 'will', 'could', 'should', 'market', 'prediction'.\n"
        "6. Do NOT stack multiple quoted phrases — this is the #1 cause of zero results.\n"
        "7. For sports: include the sport name + league + team/player. E.g.: basketball AND NBA AND Lakers\n"
        "8. For weather/climate: include location + weather terms. E.g.: Chicago AND (snow OR snowfall OR storm)\n"
        "9. For politics/policy: include the person or body + topic. E.g.: \"Federal Reserve\" AND (rate OR interest OR hike)\n"
        "10. For elections/polls: include candidate or race + election terms. E.g.: Trump AND (poll OR election OR vote)\n\n"

        "=== LOOKBACK DAYS ===\n"
        "You will be given `time_to_close_h` (hours until the market closes). Use it to choose how far back to search:\n"
        "- If time_to_close_h <= 48:   lookback_days = 3  (focus on very recent news)\n"
        "- If time_to_close_h <= 168:  lookback_days = 7\n"
        "- If time_to_close_h <= 720:  lookback_days = 14\n"
        "- Otherwise:                  lookback_days = 14\n"
        "If time_to_close_h is null or missing, default to 7.\n\n"

        "=== OUTPUT FORMAT ===\n"
        "Return STRICT JSON only. No markdown, no explanation.\n"
        "{\n"
        "  \"query\": string,\n"
        "  \"lookback_days\": int\n"
        "}\n\n"

        "=== EXAMPLES ===\n"
        "GOOD: {\"query\": \"\\\"Elon Musk\\\" AND (Tesla OR SpaceX OR CEO)\", \"lookback_days\": 7}\n"
        "GOOD: {\"query\": \"NCAA AND basketball AND (championship OR tournament OR \\\"March Madness\\\")\", \"lookback_days\": 14}\n"
        "BAD:  {\"query\": \"Will Elon Musk say 'Mars' in his next speech??\", \"lookback_days\": 7}  ← sentence, special chars\n"
        "BAD:  {\"query\": \"\\\"exact phrase one\\\" AND \\\"exact phrase two\\\" AND \\\"exact phrase three\\\"\", \"lookback_days\": 7}  ← too many quoted phrases\n"
    ),
)

_evidence_model = AssistantAgent(
    name="NewsEvidenceModel",
    model_client=_model_client,
    system_message=(
        "You are an evidence-based analyst for binary prediction markets.\n"
        "You receive a market title, resolution rules, and a list of news articles.\n\n"

        "=== YOUR TASK (two steps) ===\n\n"

        "STEP 1 — RELEVANCE FILTER:\n"
        "For each article, determine if it is DIRECTLY relevant to the market's resolution criteria.\n"
        "An article is relevant if it discusses the specific subject, event, or metric that the market resolves on.\n"
        "Discard articles that are:\n"
        "  - About a different subject/entity even if superficially similar\n"
        "  - General background or opinion pieces with no factual bearing on the outcome\n"
        "  - Paywalled stubs, error pages, or articles with no meaningful content\n"
        "  - Duplicates of another article already counted\n"
        "Count how many articles you keep as relevant vs. the total provided.\n\n"

        "STEP 2 — PROBABILITY ESTIMATION:\n"
        "Using ONLY the relevant articles from Step 1, estimate P(YES) — the probability the market resolves YES.\n\n"

        "=== CALIBRATION RULES ===\n"
        "- If zero articles are relevant → p_yes = 0.50, confidence = 0.0\n"
        "- If only 1-2 articles are relevant with indirect evidence → keep p_yes within 0.40–0.60, confidence low (0.1–0.3)\n"
        "- Use p_yes > 0.65 or < 0.35 ONLY when multiple relevant articles provide strong, direct, factual evidence\n"
        "- For long-horizon outcomes (season champions, annual records), be conservative — even strong favorites rarely warrant p_yes > 0.75\n"
        "- For short-horizon outcomes (next 24–48h), direct evidence can push p_yes further from 0.50\n"
        "- Confidence reflects how much you trust your estimate: 0.0 = pure guess, 1.0 = outcome is essentially confirmed by articles\n"
        "- Rankings, odds, and punditry are WEAK evidence — they should move p_yes only slightly from 0.50\n\n"

        "=== OUTPUT FORMAT ===\n"
        "Return STRICT JSON only. No markdown, no explanation.\n"
        "{\n"
        "  \"relevant_count\": int,\n"
        "  \"total_count\": int,\n"
        "  \"p_yes\": number (0..1),\n"
        "  \"confidence\": number (0..1),\n"
        "  \"summary\": string (1-2 sentences explaining your reasoning)\n"
        "}\n"
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
    close_time = _get_str(ctx, "close_time")
    time_to_close_h = ctx.get("time_to_close_h")

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
        "close_time": close_time,
        "time_to_close_h": round(time_to_close_h, 1) if time_to_close_h is not None else None,
    }
    qb_prompt = (
        "Create a NewsAPI query for this market.\n"
        "Return STRICT JSON only.\n\n"
        f"{json.dumps(qb_payload, ensure_ascii=False)}"
    )

    query = title  # fallback
    lookback_days = 7  # default
    qb_raw: Dict[str, Any] = {}
    try:
        qb_result = await _query_builder.run(task=[TextMessage(content=qb_prompt, source='user')])
        qb_text = qb_result.messages[-1].content if qb_result.messages else ""
        qb_data = _safe_json_loads(qb_text) or {}
        qb_raw = qb_data if isinstance(qb_data, dict) else {}
        if isinstance(qb_raw.get("query", None), str) and qb_raw["query"].strip():
            query = qb_raw["query"].strip()
        if isinstance(qb_raw.get("lookback_days", None), (int, float)):
            lookback_days = max(1, min(14, int(qb_raw["lookback_days"])))
    except Exception as e:
        qb_raw = {"error": str(e)}

    # 2) Sanitize query to remove special characters that break NewsAPI
    query = _sanitize_query(query)

    # 3) Fetch articles using dynamic lookback
    articles = await _fetch_news_articles(query, max_articles=10, lookback_days=lookback_days)

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
    
    # 4) Ask LLM to filter for relevance, then estimate P(YES)
    ev_payload = {
        "title": title,
        "rules_primary": rules_primary,
        "rules_secondary": rules_secondary,
        "articles": articles,
    }
    ev_prompt = (
        "First filter the articles for relevance to the market, then estimate P(YES).\n"
        "Return STRICT JSON only.\n\n"
        f"{json.dumps(ev_payload, ensure_ascii=False)}"
    )

    p_yes = 0.5
    conf = 0.0
    summary = ""
    relevant_count = 0
    total_count = len(articles)
    ev_raw: Dict[str, Any] = {}

    try:
        ev_result = await _evidence_model.run(task=[TextMessage(content=ev_prompt, source='user')])
        ev_text = ev_result.messages[-1].content if ev_result.messages else ""
        ev_data = _safe_json_loads(ev_text) or {}
        ev_raw = ev_data if isinstance(ev_data, dict) else {}

        p_yes = _clamp(float(ev_raw.get("p_yes", 0.5)))
        conf = _clamp(float(ev_raw.get("confidence", 0.0)))
        summary = str(ev_raw.get("summary", "") or "")
        relevant_count = int(ev_raw.get("relevant_count", 0) or 0)
        total_count = int(ev_raw.get("total_count", total_count) or total_count)
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
            "relevant_count": relevant_count,
            "total_count": total_count,
        },
        "raw": {
            "query_builder": qb_raw,
            "articles": articles,
            "evidence": ev_raw,
            "rule": "LLM builds query; LLM filters for relevance then estimates P(YES) from articles; threshold to BUY/abstain.",
        },
    }


def run_news_evidence_agent_sync(ctx: Dict[str, Any]) -> Dict[str, Any]:
    return asyncio.run(run_news_evidence_agent(ctx))