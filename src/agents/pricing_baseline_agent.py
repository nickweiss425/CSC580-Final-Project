"""
Deterministic PricingBaselineAgent

This agent chooses a trade direction (YES/NO) using only the current ask prices.
Because the system only recommends BUY (never SELL), the ask prices represent
the true cost to enter each side.

Output format matches the standardized AgentOutput dict:
{
  "agent": "PricingBaselineAgent",
  "action": "BUY" or None,
  "direction": "YES" | "NO" | None,
  "score": 0..1,
  "reason": str,
  "signals": dict,
  "raw": dict
}
"""

from typing import Any, Dict, Optional


def _get_float(ctx: Dict[str, Any], key: str) -> Optional[float]:
    """Safely read a float from ctx; returns None if missing/unparseable."""
    val = ctx.get(key, None)
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _gap_to_score(gap: float) -> float:
    """
    Convert ask-price gap into a simple confidence score in [0, 1].

    Bigger separation between YES and NO asks => stronger directional signal.
    """
    # Very small separation => weak signal
    if gap < 0.02:
        return 0.30
    if gap < 0.05:
        return 0.50
    if gap < 0.10:
        return 0.65
    if gap < 0.20:
        return 0.80
    return 0.90


def run_pricing_baseline_agent(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """
    Pick a buy direction based on which side is cheaper to buy (lower ask).

    Notes:
    - This is a baseline direction agent. It does not consider truth/probability.
    - Risk/Rules gates should run separately to prevent bad trades.
    """
    yes_ask = _get_float(ctx, "yes_ask")
    no_ask = _get_float(ctx, "no_ask")

    # if missing prices, abstain
    if yes_ask is None or no_ask is None:
        return {
            "agent": "PricingBaselineAgent",
            "action": None,
            "direction": None,
            "score": 0.0,
            "reason": "Missing yes_ask or no_ask; cannot choose a direction.",
            "signals": {"yes_ask": yes_ask, "no_ask": no_ask},
            "raw": {},
        }

    gap = abs(yes_ask - no_ask)

    # if essentially equal, treat as no signal
    if gap < 1e-6:
        return {
            "agent": "PricingBaselineAgent",
            "action": None,
            "direction": None,
            "score": 0.2,
            "reason": "YES and NO asks are equal; no clear price-based direction.",
            "signals": {"yes_ask": yes_ask, "no_ask": no_ask, "gap": gap},
            "raw": {},
        }

    # choose the cheaper side to buy
    if yes_ask < no_ask:
        direction = "YES"
        cheaper = yes_ask
        more_expensive = no_ask
    else:
        direction = "NO"
        cheaper = no_ask
        more_expensive = yes_ask

    score = _gap_to_score(gap)

    return {
        "agent": "PricingBaselineAgent",
        "action": "BUY",
        "direction": direction,
        "score": float(score),
        "reason": (
            f"{direction} is cheaper to buy (ask {cheaper:.2f} vs {more_expensive:.2f}); "
            f"price gap={gap:.2f}."
        ),
        "signals": {
            "yes_ask": yes_ask,
            "no_ask": no_ask,
            "gap": gap,
            "cheaper_side": direction,
        },
        "raw": {
            "rule": "Pick the side with the lower ask price (cheaper to buy).",
        },
    }
