from datetime import datetime, timezone


from datetime import datetime

def _parse_iso_z(ts: str | None):
    if not ts:
        return None

    # normalize Z to +00:00
    ts = ts.replace("Z", "+00:00")

    # Fix fractional seconds if present and not 6 digits
    # Examples it should handle:
    # 2026-02-15T21:04:39.29951+00:00   (5 digits)
    # 2026-02-15T21:04:39.299510+00:00  (6 digits)
    # 2026-02-15T21:04:39+00:00         (no fractional)
    if "." in ts:
        head, tail = ts.split(".", 1)          # head = ...T21:04:39
        frac, tz = tail.split("+", 1) if "+" in tail else tail.split("-", 1)
        sign = "+" if "+" in tail else "-"
        tz = sign + tz                         # tz like +00:00

        # keep only digits in frac, then pad/truncate to 6
        frac_digits = "".join(ch for ch in frac if ch.isdigit())
        frac_digits = (frac_digits + "000000")[:6]

        ts = f"{head}.{frac_digits}{tz}"

    return datetime.fromisoformat(ts)



def _prob_from_cents(value: int | None):
    if value is None:
        return None
    return value / 100.0


def build_market_context(raw: dict) -> dict:
    """
    convert a raw Kalshi market dict into a simple, flat dictionary
    for use by agents.
    """

    now = datetime.now(timezone.utc)

    # --- parse times ---
    close_dt = _parse_iso_z(raw.get("close_time"))
    updated_dt = _parse_iso_z(raw.get("updated_time"))

    time_to_close_h = None
    if close_dt:
        time_to_close_h = (close_dt - now).total_seconds() / 3600.0

    # --- normalize pricing (cents → probability) ---
    yes_ask = _prob_from_cents(raw.get("yes_ask"))
    yes_bid = _prob_from_cents(raw.get("yes_bid"))
    no_ask = _prob_from_cents(raw.get("no_ask"))
    no_bid = _prob_from_cents(raw.get("no_bid"))

    # --- spreads ---
    yes_spread = None
    if yes_ask is not None and yes_bid is not None:
        yes_spread = yes_ask - yes_bid

    no_spread = None
    if no_ask is not None and no_bid is not None:
        no_spread = no_ask - no_bid

    # --- build final context dict ---
    context = {
        # identifiers
        "ticker": raw.get("ticker"),
        "event_ticker": raw.get("event_ticker"),
        "title": raw.get("title"),
        "subtitle": raw.get("subtitle"),
        "status": raw.get("status"),
        "market_type": raw.get("market_type"),

        # semantics
        "strike_type": raw.get("strike_type"),
        "custom_strike": raw.get("custom_strike") or {},
        "yes_label": raw.get("yes_sub_title"),
        "no_label": raw.get("no_sub_title"),

        # rules
        "rules_primary": raw.get("rules_primary"),
        "rules_secondary": raw.get("rules_secondary"),

        # prices (probabilities 0–1)
        "yes_ask": yes_ask,
        "yes_bid": yes_bid,
        "no_ask": no_ask,
        "no_bid": no_bid,
        "last_price": _prob_from_cents(raw.get("last_price")),

        # market stats
        "volume": raw.get("volume"),
        "volume_24h": raw.get("volume_24h"),
        "open_interest": raw.get("open_interest"),

        # timing
        "close_time": raw.get("close_time"),
        "updated_time": raw.get("updated_time"),
        "time_to_close_h": time_to_close_h,

        # simple derived quality metrics
        "yes_spread": yes_spread,
        "no_spread": no_spread,
    }

    return context
