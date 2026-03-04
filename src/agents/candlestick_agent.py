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

from typing import Any, Dict, List, Optional, Tuple


DEFAULT_THRESHOLDS = {
    # minimum candles to compute signals
    "min_candles": 48,          # ~2 days of hourly candles

    # momentum windows (in candles)
    "mom_short_n": 24,          # ~24h
    "mom_long_n": 72,           # ~72h

    # moving average windows (in candles)
    "ma_short_n": 12,           # ~12h
    "ma_long_n": 48,            # ~48h

    # decision thresholds
    "mom_long_min_abs": 0.02,   # require >= 2% move over long horizon
    "trend_min_abs": 0.01,      # require MA gap >= 1 cent in prob space

    # volume confirmation
    "vol_confirm_ratio": 1.20,  # last 24h volume vs previous 24h volume

    # if volatility is extreme, downweight confidence
    "volatility_window": 48,
    "volatility_warn": 0.03,    # 3% std of returns over window is “noisy”
}


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

    for c in candles:
        cl = _safe_float(c.get("close"))
        if cl is None:
            continue
        ts.append(int(c.get("end_ts")))
        close.append(float(cl))
        vol.append(int(c.get("volume") or 0))
        oi.append(int(c.get("open_interest") or 0))

    return ts, close, vol, oi


def _compute_momentum(closes: List[float], n: int) -> Optional[float]:
    if len(closes) <= n:
        return None
    return _pct_return(closes[-1], closes[-1 - n])


def _compute_ma_gap(closes: List[float], short_n: int, long_n: int) -> Optional[float]:
    if len(closes) < max(short_n, long_n):
        return None
    ma_s = _mean(closes[-short_n:])
    ma_l = _mean(closes[-long_n:])
    if ma_s is None or ma_l is None:
        return None
    return ma_s - ma_l


def _compute_vol_ratio(volumes: List[int], window: int) -> Optional[float]:
    # compare last window volume to previous window volume
    if len(volumes) < 2 * window:
        return None
    v_last = sum(volumes[-window:])
    v_prev = sum(volumes[-2 * window:-window])
    if v_prev <= 0:
        return None
    return v_last / v_prev


def _compute_volatility(closes: List[float], window: int) -> Optional[float]:
    if len(closes) < window + 1:
        return None
    rets: List[float] = []
    for i in range(len(closes) - window, len(closes)):
        r = _pct_return(closes[i], closes[i - 1])
        if r is not None:
            rets.append(r)
    return _std(rets)


# ----------------------------
# Main agent
# ----------------------------
def run_trend_candles_agent(
    ctx: Dict[str, Any],
    candles: List[Dict[str, Any]],
    thresholds: Dict[str, Any] = DEFAULT_THRESHOLDS,
) -> Dict[str, Any]:
    t = dict(DEFAULT_THRESHOLDS)
    t.update(thresholds or {})

    ticker = (ctx.get("ticker") or "").strip()
    if not ticker:
        return {
            "agent": "TrendCandlesAgent",
            "action": None,
            "direction": None,
            "score": 0.0,
            "reason": "Missing market ticker.",
            "signals": {"missing": ["ticker"]},
        }

    if not candles:
        return {
            "agent": "TrendCandlesAgent",
            "action": None,
            "direction": None,
            "score": 0.3,
            "reason": "No candlestick data provided.",
            "signals": {"n": 0},
        }

    ts, closes, volumes, open_interests = _extract_series(candles)

    if len(closes) < int(t["min_candles"]):
        return {
            "agent": "TrendCandlesAgent",
            "action": None,
            "direction": None,
            "score": 0.4,
            "reason": f"Insufficient candlestick history (n={len(closes)}).",
            "signals": {"n": len(closes), "min_candles": int(t["min_candles"])},
            "raw": {"candles_n_raw": len(candles)},
        }

    # compute indicators
    mom_short = _compute_momentum(closes, int(t["mom_short_n"]))
    mom_long = _compute_momentum(closes, int(t["mom_long_n"]))
    ma_gap = _compute_ma_gap(closes, int(t["ma_short_n"]), int(t["ma_long_n"]))
    vol_ratio = _compute_vol_ratio(volumes, int(t["mom_short_n"]))  # use 24h window by default
    volat = _compute_volatility(closes, int(t["volatility_window"]))

    if mom_long is None or ma_gap is None:
        return {
            "agent": "TrendCandlesAgent",
            "action": None,
            "direction": None,
            "score": 0.5,
            "reason": "Not enough data to compute momentum/trend reliably.",
            "signals": {"mom_long": mom_long, "ma_gap": ma_gap, "n": len(closes)},
        }

    # require “strength”
    if abs(mom_long) < float(t["mom_long_min_abs"]) or abs(ma_gap) < float(t["trend_min_abs"]):
        return {
            "agent": "TrendCandlesAgent",
            "action": None,
            "direction": None,
            "score": 0.55,
            "reason": "No strong directional trend in recent candles.",
            "signals": {
                "mom_short": mom_short,
                "mom_long": mom_long,
                "ma_gap": ma_gap,
                "vol_ratio": vol_ratio,
                "volatility": volat,
                "n": len(closes),
            },
        }

    # direction from consistent momentum + MA regime
    if mom_long > 0 and ma_gap > 0:
        direction = "YES"
        base_reason = f"Uptrend: mom_long={mom_long:.2%}, ma_gap={ma_gap:.2f}."
    elif mom_long < 0 and ma_gap < 0:
        direction = "NO"
        base_reason = f"Downtrend: mom_long={mom_long:.2%}, ma_gap={ma_gap:.2f}."
    else:
        return {
            "agent": "TrendCandlesAgent",
            "action": None,
            "direction": None,
            "score": 0.55,
            "reason": "Signals conflict (momentum vs moving averages).",
            "signals": {
                "mom_short": mom_short,
                "mom_long": mom_long,
                "ma_gap": ma_gap,
                "vol_ratio": vol_ratio,
                "volatility": volat,
                "n": len(closes),
            },
        }

    # score (0..1): strength + volume confirmation - volatility penalty
    mom_strength = min(1.0, abs(mom_long) / 0.10)   # 10% move -> strong
    gap_strength = min(1.0, abs(ma_gap) / 0.05)     # 5c MA gap -> strong
    strength = 0.5 * mom_strength + 0.5 * gap_strength

    score = 0.55 + 0.35 * strength

    reasons: List[str] = [base_reason]

    if vol_ratio is not None:
        if vol_ratio >= float(t["vol_confirm_ratio"]):
            score += 0.05
            reasons.append(f"Volume confirms move (vol_ratio={vol_ratio:.2f}).")
        else:
            reasons.append(f"Volume not strongly confirming (vol_ratio={vol_ratio:.2f}).")

    if volat is not None and volat >= float(t["volatility_warn"]):
        score -= 0.07
        reasons.append(f"High volatility (std_ret={volat:.2%}) lowers confidence.")

    score = max(0.0, min(1.0, float(score)))
    reason = " | ".join(reasons[:2])  # keep concise

    return {
        "agent": "TrendCandlesAgent",
        "action": "BUY",
        "direction": direction,
        "score": float(score),
        "reason": reason,
        "signals": {
            "mom_short": mom_short,
            "mom_long": mom_long,
            "ma_gap": ma_gap,
            "vol_ratio": vol_ratio,
            "volatility": volat,
            "thresholds": {
                "min_candles": int(t["min_candles"]),
                "mom_short_n": int(t["mom_short_n"]),
                "mom_long_n": int(t["mom_long_n"]),
                "ma_short_n": int(t["ma_short_n"]),
                "ma_long_n": int(t["ma_long_n"]),
                "mom_long_min_abs": float(t["mom_long_min_abs"]),
                "trend_min_abs": float(t["trend_min_abs"]),
                "vol_confirm_ratio": float(t["vol_confirm_ratio"]),
                "volatility_warn": float(t["volatility_warn"]),
            },
            "observed": {
                "last_close": closes[-1],
                "first_close": closes[0],
                "n_candles": len(closes),
                "last_open_interest": open_interests[-1] if open_interests else None,
                "last_end_ts": ts[-1] if ts else None,
            },
        },
        "raw": {
            "last_5_closes": closes[-5:],
            "last_5_volumes": volumes[-5:],
        },
    }