"""
Deterministic TrendCandlesAgent (Candlestick Trend / Momentum)

Pass in:
  - ctx: MarketContext (must include at least "ticker"; optional: yes_ask/no_ask etc.)
  - candles: list of dicts from your fetch_candlesticks() return value

Returns standardized AgentOutput:
{
  "agent": "TrendCandlesAgent",
  "action": "BUY" or None,
  "direction": "YES" or "NO" or None,
  "score": 0..1,
  "reason": "...",
  "signals": {...},
  "raw": {...}
}
"""


from __future__ import annotations
from datetime import datetime, timezone
import os
import asyncio
import json
import re
from typing import Any, Dict, List
from dotenv import load_dotenv

load_dotenv()

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient

MAX_CANDLES = 180

# set up the model connection using ChatGPT 4o mini as model
_model_client = OpenAIChatCompletionClient(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
)

import json
import time
from typing import Any, Dict, List, Optional, Tuple

import requests

from src.kalshi.client import BASE_URL


# ----------------------------
# Helpers
# ----------------------------
def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except (TypeError, ValueError):
        return None


def _mean(xs: List[float]) -> Optional[float]:
    if not xs:
        return None
    return sum(xs) / len(xs)


def _std(xs: List[float]) -> Optional[float]:
    if len(xs) < 2:
        return None
    m = sum(xs) / len(xs)
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return var ** 0.5


def _pct_return(a: float, b: float) -> Optional[float]:
    if b is None or a is None or b == 0:
        return None
    return (a / b) - 1.0


def _extract_series(candles: List[Dict[str, Any]]) -> Tuple[List[int], List[float], List[int], List[int]]:
    """
    Input candle format (your fetcher):
      {
        "end_ts": int,
        "open": float|None, "high": float|None, "low": float|None, "close": float|None,
        "volume": int, "open_interest": int
      }
    Returns (end_ts_list, closes, volumes, open_interests), dropping missing close.
    """
    ts: List[int] = []
    close: List[float] = []
    vol: List[int] = []
    oi: List[int] = []

    for i, c in enumerate(candles):
        c_next = candles[i + 1] if i + 1 < len(candles) else None
        c_next_end_ts = c_next.get("end_period_ts") if c_next else None
        interval = c_next_end_ts - c.get("end_period_ts") if c_next_end_ts and c.get("end_period_ts") else None
        dups = interval/60 if interval else 0
        for _ in range(int(dups)):
            cl = _safe_float(c.get("price").get("close"))
            if cl is None:
                cl = _safe_float(c.get("price").get("previous"))
                if cl is None:
                    continue
            ts.append(int(c.get("end_period_ts")))
            close.append(float(cl))
            vol.append(int(c.get("volume") or 0))
            oi.append(int(c.get("open_interest") or 0))

    return ts, close, vol, oi


def get_trend_candlesticks(ctx, hours = 3, increment = 1):
    BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"

    # find the series ticker using get event
    url = f"{BASE_URL}/events/{ctx.get('event_ticker')}"
    resp = requests.get(url, timeout=1)
    if resp.status_code != 200:
        series_ticker = ""
    else:
        event_data = resp.json()
        series_ticker = event_data.get("event", {}).get("series_ticker", "")

    ticker = ctx.get("ticker")
    params = {}

    params["start_ts"] = int(time.time() - hours * 60*60)
    params["end_ts"] = int(time.time())
    # print(params)
    params["period_interval"] = increment
    url = f"{BASE_URL}/series/{series_ticker}/markets/{ticker}/candlesticks"
    resp = requests.get(url, params=params, timeout=1)
    candles = resp.json().get("candlesticks", []) 
    return candles



_candlestick_agent = AssistantAgent(
    name="CandlestickAgent",
    model_client=_model_client,
    system_message=(
        "You are a financial candlestick pattern analyst.\n"
        "Given candlestick data, analyze the overall trend and patterns.\n"
        "Return STRICT JSON only with keys:\n"
        "trend (string: 'uptrend', 'downtrend', 'sideways'), "
        "clarity (0..1 number), "
        "flags (list of strings with detected patterns or issues), "
        "notes (string with any additional observations).\n"
        "Focus on how the recent price action impacts the likelihood of the event outcome."
    ),
)

# ----------------------------
# Main agent
# ----------------------------


def prepare_candle_info(candles):
    filtered_candles = []
    for i in range(MAX_CANDLES):
        if i >= len(candles):
            break
        filtered_candles.append(
            {
                "end_ts": candles[i].get("end_period_ts"),
                "open": candles[i].get("price", {}).get("open"),
                "high": candles[i].get("price", {}).get("high"),
                "low": candles[i].get("price", {}).get("low"),
                "close": candles[i].get("price", {}).get("close"),
                "volume": candles[i].get("volume"),
                "open_interest": candles[i].get("open_interest"),
            }

        )
    return filtered_candles

async def run_trend_candles_agent_gpt(ctx):
    
    # determine the interval of candles we should look at
    close_time = ctx.get("close_time")
    if isinstance(close_time, str):
        close_time = datetime.fromisoformat(close_time)

    if close_time is not None and close_time.tzinfo is None:
        close_time = close_time.replace(tzinfo=timezone.utc)

    now = datetime.now(timezone.utc)
    T = (close_time - now).total_seconds()
    
    if T > (6 * 60 * 60): 
        candles = get_trend_candlesticks(ctx, hours=6, increment=60)
    else:
        candles = get_trend_candlesticks(ctx, hours=3, increment=1)
    print(len(candles))
    filtered_candles = prepare_candle_info(candles)
    prompt = "Analyze the following candlestick data and identify any trends or patterns:\n" + json.dumps(filtered_candles, indent=2)
    # Run sentiment analysis
    # print(prompt)
    try:
        candlestick_agent = _candlestick_agent
        result = await candlestick_agent.run(task=[TextMessage(content=prompt, source="user")])
        text = result.messages[-1].content
        print(text)
        
        # Parse LLM response
        clean = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
        clean = re.sub(r"\s*```$", "", clean)

        sem = json.loads(clean)
        # sem = json.loads(text)
        trend = sem.get('trend', 'sideways')
        direction = "YES" if trend == "uptrend" else "NO" if trend == "downtrend" else None
        clarity = sem.get('clarity', 0.5)
        flags = sem.get('flags', [])
        notes = sem.get('notes', '')
        veto = clarity < 0.6 or trend == 'sideways'  # example veto condition

    except Exception as e:
        print(f"Error in candlestick analysis: {e}")
        sem = None
        veto = None
        trend = None
        direction = None
        clarity = 0.0
        flags = []
        notes = str(e)

    return {
        "agent": "CandlestickAgent",
        "action": "BUY" if direction in ["YES", "NO"] and not veto else None,
        "direction": direction,                      # never chooses YES/NO
        "score": clarity,                       # interpret as "rules clarity"
        "reason": f"Rules are clear." if not veto else f"Ambiguity/low clarity: {flags} (score={clarity:.2f})",
        "signals": {
            "trend": trend,
            "clarity": clarity,
            "flags": flags,
            "notes": notes,
        },
        "raw": {
            "candles_analyzed": len(candles),
            "time_interval_used": "hr" if T > (6 * 60 * 60) else "min",
        }
    }
     
def run_trend_candles_agent_gpt_sync(ctx: Dict[str, Any]) -> Dict[str, Any]:
    return asyncio.run(run_trend_candles_agent_gpt(ctx))