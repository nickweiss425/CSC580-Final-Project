import sys
from pathlib import Path
import json


# add project root to Python path
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))


import streamlit as st
from src.kalshi.client import (
    fetch_markets,
    normalize_markets,
    search_markets_progressive,
    fetch_market_by_ticker,   
)

from src.agents.market_context import build_market_context
from src.agents.rules_agent import run_rules_agent_sync
from src.agents.risk_agent import run_risk_agent
from src.agents.pricing_baseline_agent import run_pricing_baseline_agent
from src.agents.news_event_agent import run_news_event_agent_sync

from src.agents.candlestick_agent import run_trend_candles_agent
from src.agents.candlestick_agent_gpt import run_trend_candles_agent_gpt_sync
from src.agents.historical_agent import run_historical_agent

from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback
from app.Home import build_market_context 


def run_test_agent(agent_name, ticker):
    raw = fetch_market_by_ticker(ticker)
    if "market" in raw:
        market = raw["market"]
    else:
        market = raw  
    # display the market of the ticker entered
    selected_market = (normalize_markets([market]))[0]
    ctx = build_market_context(selected_market)

    if agent_name == "RulesAgent":
        return run_rules_agent_sync(ctx)
    elif agent_name == "RiskAgent":
        return run_risk_agent(ctx)
    elif agent_name == "PricingBaselineAgent":
        return run_pricing_baseline_agent(ctx)
    elif agent_name == "NewsEventAgent":
        return run_news_event_agent_sync(ctx)
    elif agent_name == "CandlestickAgent":
        return run_trend_candles_agent(ctx)
    elif agent_name == "CandlestickAgentGPT":
        return run_trend_candles_agent_gpt_sync(ctx)
    elif agent_name == "HistoricalAgent":
        return run_historical_agent(ctx)
    else:
        raise ValueError(f"Unknown agent name: {agent_name}")
    
if __name__ == "__main__":
    agent_name = "CandlestickAgent"  # Change this to test different agents
    ticker = "KXHIGHNY-26MAR03-T39"  # Change this to test different tickers


    try:
        result = run_test_agent(agent_name, ticker)
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"Error running {agent_name} for ticker {ticker}: {e}")
        traceback.print_exc()
