import json

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]  # .../CSC580-Final-Project
sys.path.append(str(ROOT_DIR))

from src.kalshi.client import fetch_market_by_ticker
from src.agents.market_context import build_market_context
from src.agents.pricing_agent import run_pricing_agent_sync

ticker = "KXPRESNOMD-28-GN"
raw = fetch_market_by_ticker(ticker)
market = raw.get("market", raw)

ctx = build_market_context(market)
out = run_pricing_agent_sync(ctx)
print(out)
