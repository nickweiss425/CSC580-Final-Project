# Streamlit app (UI improvements for nicer results display)
# Key changes:
# - Store ctx_enriched in session_state so it can be shown
# - Show a clear "Final Decision" card (action/direction/confidence + explanation)
# - Add agent breakdown as readable cards with signals + optional raw JSON
# - Keep raw JSON in an expander for debugging

import sys
from pathlib import Path
import json
import time

# add project root to Python path
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

import streamlit as st
from src.kalshi.client import (
    fetch_markets,
    normalize_markets,
    search_markets_progressive,
    fetch_market_by_ticker,
    fetch_candlesticks,
)

from src.agents.market_context import build_market_context
from src.agents.rules_agent import run_rules_agent_sync
from src.agents.risk_agent import run_risk_agent
from src.agents.pricing_baseline_agent import run_pricing_baseline_agent
from src.agents.news_evidence_agent import run_news_evidence_agent_sync
from src.agents.candlestick_agent import run_trend_candles_agent
from src.agents.aggregation_agent import aggregate_recommendation_sync


# ----------------------------
# State helpers
# ----------------------------
def init_state() -> None:
    if "selected_market_id" not in st.session_state:
        st.session_state.selected_market_id = None

    if "market_results" not in st.session_state:
        st.session_state.market_results = get_markets_cached(limit=200)

    if "recommendation" not in st.session_state:
        st.session_state.recommendation = None

    # store last ctx for display/debug
    if "last_ctx" not in st.session_state:
        st.session_state.last_ctx = None


def reset_selection_and_recommendation() -> None:
    st.session_state.selected_market_id = None
    st.session_state.recommendation = None
    st.session_state.last_ctx = None


def set_market_results(results: list[dict]) -> None:
    st.session_state.market_results = results
    reset_selection_and_recommendation()


@st.cache_data(ttl=60)
def get_markets_cached(limit: int = 200):
    raw = fetch_markets(limit=limit, mve_filter="exclude")
    return normalize_markets(raw.get("markets", []))


# ----------------------------
# UI: Left panel
# ----------------------------
def render_market_browser() -> None:
    st.subheader("Market Browser")

    ticker = st.text_input("Load by ticker", placeholder="e.g. KXPRESPERSON-28-JVAN")
    load_clicked = st.button("Load ticker", use_container_width=True)

    if load_clicked:
        try:
            raw = fetch_market_by_ticker(ticker)
            market = raw["market"] if "market" in raw else raw
            set_market_results(normalize_markets([market]))
        except Exception as e:
            st.error(f"Could not load ticker: {e}")

    query = st.text_input("Search markets", placeholder="Type keywords…")
    col_a, col_b = st.columns([1, 1])
    with col_a:
        search_clicked = st.button("Search", use_container_width=True)
    with col_b:
        clear_clicked = st.button("Clear", use_container_width=True)

    if clear_clicked:
        markets = get_markets_cached(limit=200)
        set_market_results(markets)

    if search_clicked:
        search_results = search_markets_progressive(query, mve_filter="exclude", target_results=20)
        set_market_results(normalize_markets(search_results))

    results = st.session_state.market_results
    if not results:
        st.info("No markets match your search.")
        return

    options = [m["ticker"] for m in results]

    current_selection = st.session_state.selected_market_id
    if current_selection not in options:
        st.session_state.selected_market_id = options[0]
        st.session_state.recommendation = None
        st.session_state.last_ctx = None

    def fmt(mid: str) -> str:
        for m in results:
            if m["ticker"] == mid:
                return m.get("list_title") or m.get("title") or mid
        return mid

    chosen_id = st.radio(
        label="Select a market",
        options=options,
        index=options.index(st.session_state.selected_market_id),
        format_func=fmt,
    )
    st.session_state.selected_market_id = chosen_id


# ----------------------------
# UI: Right panel
# ----------------------------
def get_selected_market() -> dict | None:
    selected_id = st.session_state.get("selected_market_id")
    if not selected_id:
        return None
    results = st.session_state.get("market_results", [])
    return next((m for m in results if m.get("ticker") == selected_id), None)


def render_market_details(selected_market: dict) -> None:
    st.subheader("Market Details")

    st.markdown(f"**Market ID:** `{selected_market.get('ticker')}`")
    st.markdown(f"**Title:** {selected_market.get('title')}")
    st.caption(selected_market.get("list_title", ""))

    st.divider()

    c1, c2 = st.columns(2)
    c1.metric("Status", selected_market.get("status", "—"))
    c2.metric("Close Time", selected_market.get("close_time", "—"))

    rules = selected_market.get("rules_primary")
    if rules:
        st.markdown("#### Resolution Rules")
        st.write(rules)

    st.markdown("#### Current Prices")
    p1, p2 = st.columns(2)
    p1.metric("Cost to buy YES", selected_market.get("yes_ask_dollars", "—"))
    p2.metric("Cost to buy NO", selected_market.get("no_ask_dollars", "—"))


def _fmt_dir(d: object) -> str:
    return d if isinstance(d, str) and d else "—"


def _fmt_conf(x: object) -> str:
    try:
        return f"{int(float(x) * 100)}%"
    except Exception:
        return "—"


def _chip_action(action: str) -> str:
    # lightweight "badge" using markdown
    if action == "BUY":
        return "✅ **BUY**"
    if action == "NO_TRADE":
        return "🛑 **NO_TRADE**"
    return f"**{action or '—'}**"


def _agent_card(a: dict) -> None:
    agent = a.get("agent", "UnknownAgent")
    action = a.get("action")
    direction = a.get("direction")
    score = a.get("score")

    header_cols = st.columns([1.4, 1, 1, 1])
    header_cols[0].markdown(f"**{agent}**")
    header_cols[1].markdown(f"Action: `{action or '—'}`")
    header_cols[2].markdown(f"Dir: `{direction or '—'}`")
    header_cols[3].markdown(f"Score: `{score if score is not None else '—'}`")

    reason = a.get("reason") or ""
    if reason:
        st.caption(reason)

    signals = a.get("signals") or {}
    if isinstance(signals, dict) and signals:
        with st.expander("Signals", expanded=False):
            st.json(signals)

    raw = a.get("raw")
    if raw is not None:
        with st.expander("Raw (debug)", expanded=False):
            st.json(raw)


def generate_recommendation() -> dict:
    selected_market = get_selected_market()
    if not selected_market:
        return {
            "action": "NO_TRADE",
            "direction": None,
            "confidence": 0.0,
            "explanation": "No market selected.",
            "agents": [],
        }

    ctx = build_market_context(selected_market)
    agent_outputs = []

    # RulesAgent
    rules_out = run_rules_agent_sync(ctx)
    agent_outputs.append(rules_out)

    # ctx enriched
    rules_sig = rules_out.get("signals", {}) or {}
    ctx_enriched = dict(ctx)
    ctx_enriched["semantics"] = {
        "yes_means": rules_sig.get("yes_means", ""),
        "no_means": rules_sig.get("no_means", ""),
        "clarity_score": float(rules_out.get("score", 0.0) or 0.0),
        "ambiguity_flags": rules_sig.get("ambiguity_flags", []) or [],
        "notes": rules_sig.get("notes", ""),
    }

    # store context for UI
    st.session_state.last_ctx = ctx_enriched

    # Candles -> TrendCandlesAgent
    end_ts = int(time.time())
    start_ts = end_ts - 14 * 24 * 3600
    candles = fetch_candlesticks(
        market_ticker=ctx["ticker"],
        start_ts=start_ts,
        end_ts=end_ts,
        period_interval=60,
    )
    candlestick_agent_out = run_trend_candles_agent(ctx, candles)
    agent_outputs.append(candlestick_agent_out)

    # RiskAgent
    risk_out = run_risk_agent(ctx)
    agent_outputs.append(risk_out)

    # PricingBaselineAgent
    pricing_out = run_pricing_baseline_agent(ctx)
    agent_outputs.append(pricing_out)

    # NewsEvidenceAgent
    news_out = run_news_evidence_agent_sync(ctx)
    agent_outputs.append(news_out)

    # Aggregation (LLM)
    final = aggregate_recommendation_sync(ctx_enriched, agent_outputs)
    return final


def render_recommendation_panel() -> None:
    st.divider()
    st.subheader("Recommendation")

    get_rec = st.button("Get recommendation", type="primary", use_container_width=True)

    if get_rec:
        with st.spinner("Running agents..."):
            st.session_state.recommendation = generate_recommendation()

    rec = st.session_state.recommendation
    if rec is None:
        st.caption("Click **Get recommendation** to generate an output.")
        return

    # --- Final decision summary ---
    st.markdown("### Final Decision")
    s1, s2, s3 = st.columns([1, 1, 1])
    s1.markdown(_chip_action(rec.get("action")))
    s2.metric("Direction", _fmt_dir(rec.get("direction")))
    s3.metric("Confidence", _fmt_conf(rec.get("confidence")))

    explanation = rec.get("explanation") or ""
    if explanation:
        st.info(explanation)

    # --- Agent breakdown ---
    agents = rec.get("agents") or []
    st.markdown("### Agent Breakdown")
    if not agents:
        st.caption("No agent outputs attached.")
    else:
        # show in a stable order (optional)
        # If you want strict ordering, you can define a preferred list.
        for a in agents:
            with st.container(border=True):
                _agent_card(a)

    # --- Debug: context + full JSON ---
    with st.expander("Debug (context + full JSON)", expanded=False):
        st.markdown("**Market Context (ctx_enriched)**")
        st.json(st.session_state.get("last_ctx"))
        st.markdown("**Full System Output**")
        st.json(rec)


def render_right_panel() -> None:
    selected_market = get_selected_market()
    if not selected_market:
        st.subheader("Market Details")
        st.warning("Select a market on the left to view details.")
        return

    render_market_details(selected_market)
    render_recommendation_panel()


# ----------------------------
# App entry
# ----------------------------
def main() -> None:
    st.set_page_config(page_title="Prediction Market Assistant (Demo)", layout="wide")

    init_state()

    st.title("Prediction Market Decision Support")
    st.caption("Browse markets, view details, then request a recommendation.")

    left, right = st.columns([1.1, 1.9], gap="large")
    with left:
        render_market_browser()
    with right:
        render_right_panel()


if __name__ == "__main__":
    main()