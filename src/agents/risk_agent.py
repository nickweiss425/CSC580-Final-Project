"""
Deterministic RiskAgent (Market Quality / Execution Risk)

This agent does NOT use an LLM. It applies simple, transparent rules to decide
whether a market is tradable right now (spreads, activity, staleness, sanity checks).

Output format matches the standardized AgentOutput used by other agents:
{
  "agent": "RiskAgent",
  "action": "NO_TRADE" or None,
  "direction": None,
  "score": 0..1,
  "reason": "...",
  "signals": {...},
  "raw": {...}  # optional debug details
}
"""

from typing import Any, Dict, List, Tuple, Optional


# ----------------------------
# Tunable Thresholds for Rules 
# ----------------------------
DEFAULT_THRESHOLDS = {
    # spread thresholds (in probability units: 0.05 == 5 cents)
    "spread_warn": 0.05,
    "spread_veto": 0.10,

    # market activity/depth
    "min_volume_24h": 1,          # 0 means "completely inactive"
    "min_open_interest": 1,       # 0 means "no positions exist"
    "min_liquidity_dollars": 1000.0,

    # stale quote thresholds (seconds)
    "quote_age_warn_s": 600,      # 10 minutes
    "quote_age_veto_s": 1800,     # 30 minutes

    # pricing sanity check
    "ask_sum_veto": 1.10,         # if yes_ask + no_ask is far above 1, book is bad
}


# ----------------------------
# Helper functions
# ----------------------------
def _get_float(ctx: Dict[str, Any], key: str) -> Optional[float]:
    """Safely read a float from ctx; returns None if missing/unparseable."""
    val = ctx.get(key, None)
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _get_int(ctx: Dict[str, Any], key: str) -> Optional[int]:
    """Safely read an int from ctx; returns None if missing/unparseable."""
    val = ctx.get(key, None)
    if val is None:
        return None
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def _check_spreads(ctx: Dict[str, Any], t: Dict[str, Any]) -> Tuple[float, List[str]]:
    """
    Returns (risk_score_component, reasons).
    risk_score_component is in [0,1] where 1 is good and 0 is bad.
    Spread is basically how much you lose instantly due to the bid/ask gap
    tight spread = market is liquid, has fair pricing
    wide spring = market is thin, you overpay pretty hard to get a share
    """
    yes_spread = _get_float(ctx, "yes_spread")
    no_spread = _get_float(ctx, "no_spread")

    reasons: List[str] = []
    # if missing, don't veto; just be cautious
    if yes_spread is None or no_spread is None:
        reasons.append("Spread data missing.")
        return 0.7, reasons

    worst = max(yes_spread, no_spread)

    if worst >= t["spread_veto"]:
        reasons.append(f"Spreads are very wide (worst_spread={worst:.2f}).")
        return 0.0, reasons

    if worst >= t["spread_warn"]:
        reasons.append(f"Spreads are somewhat wide (worst_spread={worst:.2f}).")
        # Map warn..veto linearly to 0.6..0.1
        span = t["spread_veto"] - t["spread_warn"]
        frac = (worst - t["spread_warn"]) / span if span > 0 else 1.0
        return max(0.1, 0.6 - 0.5 * frac), reasons

    reasons.append(f"Spreads are tight (worst_spread={worst:.2f}).")
    return 1.0, reasons


def _check_activity(ctx: Dict[str, Any], t: Dict[str, Any]) -> Tuple[float, List[str]]:
    """
    Checks volume_24h and open_interest
    we want to know if anyone is actually trading or holding positions
    volume_24h == 0 -> nobody traded recently
    open_interest == 0 -> nobody holds positions right now
    """
    vol_24h = _get_int(ctx, "volume_24h")
    oi = _get_int(ctx, "open_interest")

    reasons: List[str] = []

    # missing data -> mild caution
    if vol_24h is None or oi is None:
        reasons.append("Activity data missing (volume_24h/open_interest).")
        return 0.7, reasons

    # hard red flag: completely inactive
    if vol_24h < t["min_volume_24h"] and oi < t["min_open_interest"]:
        reasons.append("Zero recent volume and zero open interest (inactive market).")
        return 0.0, reasons

    # one of them is zero -> still risky, but not always veto
    score = 1.0
    if vol_24h < t["min_volume_24h"]:
        reasons.append("Zero 24h volume (thin trading).")
        score = min(score, 0.4)
    if oi < t["min_open_interest"]:
        reasons.append("Zero open interest (no positions exist).")
        score = min(score, 0.4)

    if score == 1.0:
        reasons.append("Market shows recent activity (volume/open interest).")

    return score, reasons


def _check_liquidity(ctx: Dict[str, Any], t: Dict[str, Any]) -> Tuple[float, List[str]]:
    """
    Checks liquidity_dollars
    Liquidity = how much money sitting in the book
    Low liquidity means your trade might move the price, fills are worse
    """
    liq = _get_float(ctx, "liquidity_dollars")
    reasons: List[str] = []

    if liq is None:
        reasons.append("Liquidity data missing.")
        return 0.7, reasons

    # if low liquidity, be cautious
    if liq < t["min_liquidity_dollars"]:
        reasons.append(f"Low liquidity (${liq:.2f}).")
        return 0.4, reasons

    # high score for high liquidity
    reasons.append(f"Liquidity looks ok (${liq:.2f}).")
    return 1.0, reasons


def _check_staleness(ctx: Dict[str, Any], t: Dict[str, Any]) -> Tuple[float, List[str]]:
    """
    Checks quote_age_s (how stale the quote is)
    If quotes are old, price might be out of date
    """
    age = _get_float(ctx, "quote_age_s")
    reasons: List[str] = []

    if age is None:
        reasons.append("Quote age missing.")
        return 0.7, reasons

    # if quote too old, low score
    if age >= t["quote_age_veto_s"]:
        reasons.append(f"Quotes are stale (quote_age_s={age:.0f}s).")
        return 0.0, reasons

    # quote age in warning zone
    if age >= t["quote_age_warn_s"]:
        reasons.append(f"Quotes somewhat old (quote_age_s={age:.0f}s).")
        # scores decline as qupte age gets closer to veto threshold
        span = t["quote_age_veto_s"] - t["quote_age_warn_s"]
        frac = (age - t["quote_age_warn_s"]) / span if span > 0 else 1.0
        return max(0.1, 0.6 - 0.5 * frac), reasons

    # score of 1 for recent quptes
    reasons.append(f"Quotes are fresh (quote_age_s={age:.0f}s).")
    return 1.0, reasons


def _check_pricing_sanity(ctx: Dict[str, Any], t: Dict[str, Any]) -> Tuple[float, List[str]]:
    """
    Basic sanity check: if yes_ask + no_ask is far above 1, the book is bad.
    If that sum is huge (like 1.6–2.0), it usually indicates:
        extreme illiquidity
        broken/empty book
        you’d be massively overpaying
    """
    yes_ask = _get_float(ctx, "yes_ask")
    no_ask = _get_float(ctx, "no_ask")
    reasons: List[str] = []

    if yes_ask is None or no_ask is None:
        reasons.append("Ask prices missing (yes_ask/no_ask).")
        return 0.7, reasons

    ask_sum = yes_ask + no_ask
    if ask_sum > t["ask_sum_veto"]:
        reasons.append(f"Pricing sanity failed (yes_ask+no_ask={ask_sum:.2f} > {t['ask_sum_veto']:.2f}).")
        return 0.0, reasons

    reasons.append(f"Pricing sanity ok (yes_ask+no_ask={ask_sum:.2f}).")
    return 1.0, reasons


# ----------------------------
# Main agent function
# ----------------------------
def run_risk_agent(ctx: Dict[str, Any], thresholds: Dict[str, Any] = DEFAULT_THRESHOLDS) -> Dict[str, Any]:
    """
    Evaluate market quality / execution risk using deterministic rules.

    Returns a standardized AgentOutput dict. This agent typically either:
      - vetoes with action="NO_TRADE", or
      - abstains with action=None (meaning "no objections").
    """
    t = dict(DEFAULT_THRESHOLDS)
    t.update(thresholds or {})

    # run checks
    s_spread, r_spread = _check_spreads(ctx, t)
    s_act, r_act = _check_activity(ctx, t)
    s_liq, r_liq = _check_liquidity(ctx, t)
    s_stale, r_stale = _check_staleness(ctx, t)
    s_sanity, r_sanity = _check_pricing_sanity(ctx, t)

    # combine scores conservatively (worst case dominates)
    score = min(s_spread, s_act, s_liq, s_stale, s_sanity)

    # if any check hits 0.0, we veto
    veto = score <= 0.0

    # build reasons
    reasons = []
    reasons.extend(r_sanity)
    reasons.extend(r_spread)
    reasons.extend(r_act)
    reasons.extend(r_liq)
    reasons.extend(r_stale)

    # keep the final reason concise, include only the most important lines
    key_reasons = [r for r in reasons if "failed" in r.lower() or "very wide" in r.lower() or "zero" in r.lower() or "stale" in r.lower()]
    if not key_reasons:
        key_reasons = reasons[:2]  # fallback

    reason = " | ".join(key_reasons[:2])  # limit to 2 short reasons

    return {
        "agent": "RiskAgent",
        "action": "NO_TRADE" if veto else None,
        "direction": None,  # RiskAgent never picks YES/NO
        "score": float(score),
        "reason": reason if reason else ("Execution risk too high." if veto else "Market quality looks acceptable."),
        "signals": {
            "pricing_sanity_score": s_sanity,
            "spread_score": s_spread,
            "activity_score": s_act,
            "liquidity_score": s_liq,
            "staleness_score": s_stale,
            "thresholds": {
                "spread_warn": t["spread_warn"],
                "spread_veto": t["spread_veto"],
                "min_volume_24h": t["min_volume_24h"],
                "min_open_interest": t["min_open_interest"],
                "min_liquidity_dollars": t["min_liquidity_dollars"],
                "quote_age_warn_s": t["quote_age_warn_s"],
                "quote_age_veto_s": t["quote_age_veto_s"],
                "ask_sum_veto": t["ask_sum_veto"],
            },
            "observed": {
                "yes_spread": _get_float(ctx, "yes_spread"),
                "no_spread": _get_float(ctx, "no_spread"),
                "volume_24h": _get_int(ctx, "volume_24h"),
                "open_interest": _get_int(ctx, "open_interest"),
                "liquidity_dollars": _get_float(ctx, "liquidity_dollars"),
                "quote_age_s": _get_float(ctx, "quote_age_s"),
                "yes_ask": _get_float(ctx, "yes_ask"),
                "no_ask": _get_float(ctx, "no_ask"),
            },
            "flags": _collect_flags(ctx, t),
        },
        "raw": {
            "checks": {
                "spread": {"score": s_spread, "reasons": r_spread},
                "activity": {"score": s_act, "reasons": r_act},
                "liquidity": {"score": s_liq, "reasons": r_liq},
                "staleness": {"score": s_stale, "reasons": r_stale},
                "pricing_sanity": {"score": s_sanity, "reasons": r_sanity},
            }
        },
    }


def _collect_flags(ctx: Dict[str, Any], t: Dict[str, Any]) -> List[str]:
    """Return a short list of human-readable risk flags."""
    flags: List[str] = []

    yes_spread = _get_float(ctx, "yes_spread")
    no_spread = _get_float(ctx, "no_spread")
    if yes_spread is not None and no_spread is not None:
        if max(yes_spread, no_spread) >= t["spread_veto"]:
            flags.append("wide_spread_veto")
        elif max(yes_spread, no_spread) >= t["spread_warn"]:
            flags.append("wide_spread_warn")

    vol_24h = _get_int(ctx, "volume_24h")
    oi = _get_int(ctx, "open_interest")
    if vol_24h is not None and oi is not None and vol_24h < t["min_volume_24h"] and oi < t["min_open_interest"]:
        flags.append("inactive_market")

    liq = _get_float(ctx, "liquidity_dollars")
    if liq is not None and liq < t["min_liquidity_dollars"]:
        flags.append("low_liquidity")

    age = _get_float(ctx, "quote_age_s")
    if age is not None:
        if age >= t["quote_age_veto_s"]:
            flags.append("stale_quote_veto")
        elif age >= t["quote_age_warn_s"]:
            flags.append("stale_quote_warn")

    yes_ask = _get_float(ctx, "yes_ask")
    no_ask = _get_float(ctx, "no_ask")
    if yes_ask is not None and no_ask is not None:
        if (yes_ask + no_ask) > t["ask_sum_veto"]:
            flags.append("pricing_sanity_failed")

    return flags
