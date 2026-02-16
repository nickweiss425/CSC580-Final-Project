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

# ----------------------------
# State helpers
# ----------------------------
def init_state() -> None:
    """Initialize Streamlit session state defaults"""

    # don't override keys if they already exist
    # previous values are preserved

    # selected market id: which market radio button is selected
    if "selected_market_id" not in st.session_state:
        st.session_state.selected_market_id = None

    # market_results: list currently shown in the left panel (after search)
    if "market_results" not in st.session_state:
        st.session_state.market_results = get_markets_cached(limit=200)    

    # recommendation: output of agent system from button press
    if "recommendation" not in st.session_state:
        st.session_state.recommendation = None



def reset_selection_and_recommendation() -> None:
    """
        When the market list changes (due to searching), the previously selected
        market might no longer be valid. Reset selection and wipe recommendations.
    """
    st.session_state.selected_market_id = None
    st.session_state.recommendation = None


def set_market_results(results: list[dict]) -> None:
    """
        Updates results list. 
    """
    st.session_state.market_results = results
    reset_selection_and_recommendation()


@st.cache_data(ttl=60)
def get_markets_cached(limit: int = 200):
    raw = fetch_markets(
        limit=limit,
        mve_filter="exclude",
    )
    return normalize_markets(raw.get("markets", []))



# ----------------------------
# UI: Left panel
# ----------------------------


def render_market_browser() -> None:
    """
        Render left part of the UI, which shows list of markets with search
        function. 
    """

    # setup header
    st.subheader("Market Browser")

    # direct load by ticker
    ticker = st.text_input("Load by ticker", placeholder="e.g. KXNCAAMBTOTAL-26FEB04PEPPSEA-151")
    load_clicked = st.button("Load ticker", use_container_width=True)

    # option to fetch market by ticker id
    if load_clicked:
        try:
            # get the market info by ticker
            raw = fetch_market_by_ticker(ticker)
            if "market" in raw:
                market = raw["market"]
            else:
                market = raw  
            # display the market of the ticker entered
            set_market_results(normalize_markets([market]))
        except Exception as e:
            st.error(f"Could not load ticker: {e}")

    # search input for market names
    query = st.text_input("Search markets", placeholder="Type keywords…")
    col_a, col_b = st.columns([1, 1])

    # setup two buttons side by side
    with col_a:
        search_clicked = st.button("Search", use_container_width=True)
    with col_b:
        clear_clicked = st.button("Clear", use_container_width=True)

    # reset to default list
    if clear_clicked:
        markets = get_markets_cached(limit=200)
        set_market_results(markets)

    # search behavior
    if search_clicked:
        search_results = search_markets_progressive(query,  mve_filter="exclude", target_results=20)
        search_results_norm = normalize_markets(search_results)
        set_market_results(search_results_norm)

    # get results from state
    results = st.session_state.market_results

    # if no results (empty list), display message
    if not results:
        st.info("No markets match your search.")
        # exit so we dont try to build ui components that will be useless
        return

    # build list of selectable ids
    # these are the options shown in the search results
    options = []
    for m in results:
        options.append(m["ticker"])

    # ensure selection is valid
    current_selection = st.session_state.selected_market_id
    if current_selection not in options:
        # default to top option
        current_selection = options[0]
        st.session_state.selected_market_id = current_selection
        st.session_state.recommendation = None

    def fmt(mid: str) -> str:
        # find title for this market id (fallback to id)
        for m in results:
            if m["ticker"] == mid:
                list_title = m["list_title"]
                break
        return f"{list_title}"

    default_index = options.index(st.session_state.selected_market_id)

    chosen_id = st.radio(
        label="Select a market",
        options=options,
        index=default_index,
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


    # -----------------------------------

    # --- Header ---
    st.markdown(f"**Market ID:** `{selected_market.get('ticker')}`")
    st.markdown(f"**Title:** {selected_market.get('title')}")
    st.caption(selected_market.get("list_title", ""))

    st.divider()

    # --- Core details ---
    c1, c2, c3 = st.columns(3)
    c1.metric("Status", selected_market.get("status", "—"))
    c2.metric("Close Time", selected_market.get("close_time", "—"))
    c3.metric("Liquidity", f"${selected_market.get('liquidity_dollars', '—')}")

    # --- Resolution rules ---
    rules = selected_market.get("rules_primary")
    if rules:
        st.markdown("#### Resolution Rules")
        st.write(rules)


    # --- Prices (lightweight orderbook snapshot) ---
    st.markdown("#### Current Prices")
    c1, c2 = st.columns(2)
    c1.metric("Cost to buy YES", selected_market.get("yes_ask_dollars", "—"))
    c2.metric("Cost to buy NO", selected_market.get("no_ask_dollars", "—"))





def generate_recommendation() -> dict:
    """
    Run the end-to-end recommendation pipeline for the selected market.

    Steps:
      1) Build MarketContext (ctx)
      2) Run RulesAgent (semantic clarity + possible veto)
      3) Enrich ctx with semantics (ctx_enriched)
      4) Run RiskAgent (deterministic execution-risk veto)
      5) Run other specialized directional agents (Pricing/Trend/Evidence)
      6) Aggregate all agent outputs into a final recommendation
    """
    selected_market = get_selected_market()
    if not selected_market:
        return {
            "action": "NO_TRADE",
            "direction": None,
            "confidence": 0.0,
            "explanation": "No market selected.",
            "agents": [],
        }

    # build standardized context
    ctx = build_market_context(selected_market)

    agent_outputs = []

    # RulesAgent (LLM): semantics + clarity gate
    rules_out = run_rules_agent_sync(ctx)
    agent_outputs.append(rules_out)

    # enrich context for downstream semantic agents
    rules_sig = rules_out.get("signals", {}) or {}
    ctx_enriched = dict(ctx)
    ctx_enriched["semantics"] = {
        "yes_means": rules_sig.get("yes_means", ""),
        "no_means": rules_sig.get("no_means", ""),
        "clarity_score": float(rules_out.get("score", 0.0) or 0.0),
        "ambiguity_flags": rules_sig.get("ambiguity_flags", []) or [],
        "notes": rules_sig.get("notes", ""),
    }

    # RiskAgent (deterministic)
    risk_out = run_risk_agent(ctx)
    agent_outputs.append(risk_out)

    # Specialized agents (ADD NEW AGENTS HERE!!!!)
    # Example:
    # pricing_out = run_pricing_agent_sync(ctx_enriched)
    # agent_outputs.append(pricing_out)

    # simple aggregation logic
    vetoes = [o for o in agent_outputs if o.get("action") == "NO_TRADE"]
    if vetoes:
        confidence = min(float(o.get("score", 0.0) or 0.0) for o in vetoes)
        explanation = " | ".join(o.get("reason", "") for o in vetoes if o.get("reason"))
        return {
            "action": "NO_TRADE",
            "direction": None,
            "confidence": confidence,
            "explanation": explanation or "One or more agents vetoed this market.",
            "agents": agent_outputs,
        }

    # no directional agents yet. CHANGE THIS TO IMPLEMENT MORE COMPLEX AGGREAGTION LOGIC
    # OR HAVE AN LLM AGENT AGGREGATE
    return {
        "action": "READY",  
        "direction": None,
        "confidence": min(float(o.get("score", 1.0) or 1.0) for o in agent_outputs),
        "explanation": "Passed Rules + Risk checks. Add a directional agent next (Pricing/Trend/Evidence).",
        "agents": agent_outputs,
    }







def render_recommendation_panel() -> None:
    st.divider()
    st.subheader("Recommendation")

    get_rec = st.button("Get recommendation", type="primary")

    if get_rec:
        st.session_state.recommendation = generate_recommendation()

    rec = st.session_state.recommendation
    if rec is None:
        st.caption("Click **Get recommendation** to generate an output (stubbed for now).")
        return

    st.markdown("### Agent Input (Market Context)")
    st.code(json.dumps(rec, indent=2))

    # c1, c2, c3 = st.columns(3)
    # c1.metric("Action", rec["action"])
    # c2.metric("Direction", rec["direction"])
    # c3.metric("Confidence", f"{int(rec['confidence'] * 100)}%")

    # st.write(rec["explanation"])

    # with st.expander("Agent breakdown", expanded=True):
    #     for a in rec["agents"]:
    #         st.markdown(f"**{a['agent']}** — {a['output']}")
    #         st.caption(a["reason"])


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
    st.set_page_config(
        page_title="Prediction Market Assistant (Demo)",
        layout="wide",
    )

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
