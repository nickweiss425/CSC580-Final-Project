from __future__ import annotations

from typing import Any, Dict, List, Optional
import requests

BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"


def fetch_markets(
    status: str = "open",
    limit: int = 100,
    cursor: Optional[str] = None,
    series_ticker: Optional[str] = None,
    mve_filter: Optional[str] = "exclude", 
    timeout_s: int = 10,
) -> Dict[str, Any]:
    """
    Fetch markets from Kalshi (public market data).

    Returns the raw JSON response (dict). The markets list is typically in response["markets"].
    """
    url = f"{BASE_URL}/markets"
    params: Dict[str, Any] = {"limit": limit}

    if status:
        params["status"] = status
    if mve_filter:
        params["mve_filter"] = mve_filter
    if cursor:
        params["cursor"] = cursor
    if series_ticker:
        params["series_ticker"] = series_ticker

    resp = requests.get(url, params=params, timeout=timeout_s)
    resp.raise_for_status()
    return resp.json()

def normalize_markets(raw_markets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized = []
    for m in raw_markets:
        normalized.append({
            "id": m.get("ticker"),
            "title": m.get("title", ""),
            "list_title": build_list_title(m),
        })
    return normalized




def search_markets_all(
    query: str,
    mve_filter: str = "exclude",
    page_limit: int = 200,
    max_pages: int = 200,
    max_results: int = 500,
    status: Optional[str] = None,
) -> List[Dict[str, Any]]:
    
    """
    Search across all markets by paging through /markets and returning those whose
    title contains query (case-insensitive). Applies Kalshi's mve_filter server-side.
    Returns a list of raw market dicts (not normalized).
    """
    q = (query or "").strip().lower()
    if not q:
        return []

    results: List[Dict[str, Any]] = []
    cursor: Optional[str] = None
    pages = 0

    while pages < max_pages and len(results) < max_results:
        raw = fetch_markets(
            limit=page_limit,
            cursor=cursor,
            status=status,
            mve_filter=mve_filter,
        )

        markets = raw.get("markets", [])
        for m in markets:
            title = (m.get("title") or "").lower()
            ticker = (m.get("ticker") or "").lower()
            if q in title or q in ticker:
                results.append(m)
                if len(results) >= max_results:
                    break

        # pagination
        cursor = raw.get("cursor") or raw.get("next_cursor")
        pages += 1

        if not cursor:
            break

    return results

    
def search_markets_progressive(
    query: str,
    mve_filter: str = "exclude",
    page_limit: int = 200,
    initial_pages: int = 5,
    max_pages: int = 100,
    target_results: int = 100,
    status: str = "open",
) -> list[dict]:

    q = query.strip().lower()
    if not q:
        return []
    
    results = []
    cursor = None
    pages_checked = 0

    # phase 1: scan first few pages
    while pages_checked < initial_pages:
        raw = fetch_markets(limit=page_limit, cursor=cursor,
                             status=status, mve_filter=mve_filter)
        markets = raw.get("markets", [])
        for m in markets:
            if q in (m.get("title","").lower() + m.get("ticker","").lower()):
                results.append(m)
                if len(results) >= target_results:
                    return results
        cursor = raw.get("cursor") or raw.get("next_cursor")
        pages_checked += 1
        if not cursor:
            return results

    # phase 2: continue if needed
    while pages_checked < max_pages and len(results) < target_results:
        raw = fetch_markets(limit=page_limit, cursor=cursor,
                             status=status, mve_filter=mve_filter)
        markets = raw.get("markets", [])
        for m in markets:
            if q in (m.get("title","").lower() + m.get("ticker","").lower()):
                results.append(m)
                if len(results) >= target_results:
                    return results
        cursor = raw.get("cursor") or raw.get("next_cursor")
        pages_checked += 1
        if not cursor:
            break

    return results



def fetch_market_by_ticker(
    ticker: str,
    timeout_s: int = 10,
) -> Dict[str, Any]:
    """
    Fetch a single market by its ticker.
    Returns raw JSON (dict). The market dict is usually in response["market"].
    """
    t = (ticker or "").strip()
    if not t:
        raise ValueError("ticker is empty")

    url = f"{BASE_URL}/markets/{t}"
    resp = requests.get(url, timeout=timeout_s)
    resp.raise_for_status()
    return resp.json()


def build_list_title(m: dict) -> str:
    base = (m.get("title") or "").strip()

    strike = m.get("floor_strike")
    strike_type = m.get("strike_type")

    # Numeric thresholds
    if strike is not None and strike_type in {"greater", "less", "greater_equal", "less_equal"}:
        sym = {
            "greater": ">",
            "less": "<",
            "greater_equal": "≥",
            "less_equal": "≤",
        }[strike_type]
        return f"{base} — {sym} {strike}"

    # If YES label clarifies the market, append it
    yes_label = (m.get("yes_sub_title") or "").strip()
    if yes_label:
        return f"{base}    {yes_label}"

    return base

