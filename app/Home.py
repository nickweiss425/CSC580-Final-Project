import sys
from pathlib import Path

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
            raw = fetch_market_by_ticker(ticker)
            if "market" in raw:
                market = raw["market"]
            else:
                market = raw  
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
        title = mid
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

    # --- YES / NO semantics ---
    # yes_label = selected_market.get("yes_sub_title")
    # no_label = selected_market.get("no_sub_title")

    # if yes_label or no_label:
    #     st.markdown("#### Contract Semantics")
    #     if yes_label:
    #         st.markdown(f"**YES resolves if:** {yes_label}")
    #     if no_label:
    #         st.markdown(f"**NO resolves if:** {no_label}")

    # --- Prices (lightweight orderbook snapshot) ---
    st.markdown("#### Current Prices")
    c1, c2 = st.columns(2)
    c1.metric("YES Bid", selected_market.get("yes_bid_dollars", "—"))
    c1.metric("YES Ask", selected_market.get("yes_ask_dollars", "—"))
    c2.metric("NO Bid", selected_market.get("no_bid_dollars", "—"))
    c2.metric("NO Ask", selected_market.get("no_ask_dollars", "—"))



def generate_recommendation() -> dict:
    # placeholder until the agent system is wired
    return {
        "action": "DON'T_BUY",
        "direction": "YES",
        "confidence": 0.55,
        "explanation": "Stub output. Agents will be connected after Kalshi integration.",
        "agents": [
            {"agent": "Market Baseline", "output": "YES @ ~61%", "reason": "Derived from best YES bid."},
            {"agent": "Momentum", "output": "NO (weak)", "reason": "Placeholder momentum signal."},
        ],
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

    c1, c2, c3 = st.columns(3)
    c1.metric("Action", rec["action"])
    c2.metric("Direction", rec["direction"])
    c3.metric("Confidence", f"{int(rec['confidence'] * 100)}%")

    st.write(rec["explanation"])

    with st.expander("Agent breakdown", expanded=True):
        for a in rec["agents"]:
            st.markdown(f"**{a['agent']}** — {a['output']}")
            st.caption(a["reason"])


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
