from __future__ import annotations

import re
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple, Optional

import numpy as np

# If you don't have sklearn installed, tell me and I'll swap to a Jaccard/Okapi BM25 fallback.
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import duckdb

TRADES = 'data/kalshi/trades/*.parquet'
MARKETS = 'data/kalshi/markets/*.parquet' 

STOPWORDS = {
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "at", "by",
    "is", "will", "be", "are", "was", "were", "as", "with", "from"
}

RESULT_YES = {"YES", "Y", "TRUE", "1", "YES_WINS"}
RESULT_NO = {"NO", "N", "FALSE", "0", "NO_WINS"}


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _to_prob(price_int: Any) -> Optional[float]:
    """
    Kalshi often stores prices in cents (0-100). Your schema shows BIGINT.
    Returns probability in [0,1].
    """
    if price_int is None:
        return None
    try:
        v = float(price_int)
    except Exception:
        return None
    # Handle both 0..100 and already 0..1 just in case
    if v > 1.0:
        return max(0.0, min(1.0, v / 100.0))
    return max(0.0, min(1.0, v))


def _normalize_text(s: str) -> str:
    s = (s or "").lower()
    # Replace numbers with a token so "151" and "147" still match the template
    s = re.sub(r"\d+(\.\d+)?", " <num> ", s)
    s = re.sub(r"[^a-z0-9<> ]+", " ", s)
    tokens = [t for t in s.split() if t and t not in STOPWORDS]
    return " ".join(tokens)


def _series_prefix(event_ticker: str) -> str:
    if not event_ticker:
        return ""
    return event_ticker.split("-", 1)[0]


def _bucketize(value: float, edges: List[float]) -> int:
    """
    Return bucket index based on edges (sorted ascending).
    """
    for i, e in enumerate(edges):
        if value <= e:
            return i
    return len(edges)


def _time_to_close_hours(close_time: Optional[datetime]) -> Optional[float]:
    if close_time is None:
        return None
    now = _utcnow()
    dt = (close_time - now).total_seconds() / 3600.0
    return dt


def _parse_result_to_yes_win(result: Optional[str]) -> Optional[bool]:
    if not result:
        return None
    r = str(result).strip().upper()
    if r in RESULT_YES:
        return True
    if r in RESULT_NO:
        return False
    # Some Kalshi exports use "yes"/"no" or other strings
    if r == "YES":
        return True
    if r == "NO":
        return False
    return None


STOPWORDS = {
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "at", "by",
    "is", "will", "be", "are", "was", "were", "as", "with", "from"
}


def _normalize_text(s: str) -> str:
    s = (s or "").lower()
    # Replace numbers with a stable token so thresholds/dates don't break similarity too much
    s = re.sub(r"\d+(\.\d+)?", " <num> ", s)
    s = re.sub(r"[^a-z0-9<> ]+", " ", s)
    toks = [t for t in s.split() if t and t not in STOPWORDS]
    return " ".join(toks)


def _event_prefix(event_ticker: str) -> str:
    if not event_ticker:
        return ""
    return event_ticker.split("-", 1)[0]


def _ticker_tokens(ticker: str) -> str:
    """
    Kalshi tickers often embed structured hints.
    We don't assume a universal format; we just split on '-' and add tokens.
    """
    if not ticker:
        return ""
    parts = ticker.split("-")
    # Add shorter “family” tokens, but avoid dumping every long segment
    keep = []
    for p in parts[:4]:  # cap to first few segments
        if not p:
            continue
        if len(p) > 24:
            keep.append(p[:24])
        else:
            keep.append(p)
    return " ".join(keep)


@dataclass
class SimilarMarket:
    ticker: str
    event_ticker: str
    title: str
    score: float


def find_similar_markets(
    con,
    ctx, 
    *,
    hard_limit: int = 20000,
    top_k: int = 100,
) -> Tuple[Dict[str, Any], List[SimilarMarket]]:
    """
    Find similar markets using only:
      - ticker
      - event_ticker
      - title

    Steps:
      1) Load target market (ticker/event_ticker/title)
      2) Pull candidates via event_ticker exact match, else prefix match
      3) Rank candidates via TF-IDF cosine similarity on normalized title (+ small ticker token help)
    """
    target_ticker = ctx.get("ticker")
    target = ctx 
    # print(target)
    ev = target["event_ticker"]
    ev_pref = _event_prefix(ev)

    # --- Candidate fetch: exact event_ticker first ---
    candidates = con.execute(
        f"""
        SELECT ticker, event_ticker, title
        FROM '{MARKETS}'
        WHERE ticker != ?
          AND event_ticker = ?
        LIMIT ?
        """,
        [target_ticker, ev, hard_limit],
    ).fetchall()

    # --- Fallback: same event prefix (series-ish) ---
    if not candidates and ev_pref:
        candidates = con.execute(
            f"""
            SELECT ticker, event_ticker, title
            FROM '{MARKETS}'
            WHERE ticker != ?
              AND split_part(event_ticker, '-', 1) = ?
            LIMIT ?
            """,
            [target_ticker, ev_pref, hard_limit],
        ).fetchall()

    # --- Last fallback: broad pool (careful—can be large) ---
    if not candidates:
        candidates = con.execute(
            f"""
            SELECT ticker, event_ticker, title
            FROM '{MARKETS}'
            WHERE ticker != ?
            LIMIT ?
            """,
            [target_ticker, hard_limit],
        ).fetchall()

    # Build docs for TF-IDF
    # Use title as primary, and lightly include ticker tokens to help clustering
    target_doc = _normalize_text(target["title"]) + " " + _normalize_text(_ticker_tokens(target["ticker"]))

    docs = [target_doc]
    for (tick, ev_t, title) in candidates:
        doc = _normalize_text(title or "") + " " + _normalize_text(_ticker_tokens(tick or ""))
        docs.append(doc)

    # If the candidate set is tiny, TF-IDF can be unstable; still works fine.
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=30000)
    X = vec.fit_transform(docs)

    sims = cosine_similarity(X[0:1], X[1:]).ravel()

    # Take top_k by similarity
    idx = np.argsort(-sims)[:top_k]

    # Apply a filter to ensure we only return markets with some minimal similarity (e.g. >0.1)
    idx = idx[sims[idx] > 0.1]

    out: List[SimilarMarket] = []
    for i in idx:
        tick, ev_t, title = candidates[int(i)]
        out.append(
            SimilarMarket(
                ticker=tick,
                event_ticker=ev_t or "",
                title=title or "",
                score=float(sims[int(i)]),
            )
        )

    meta = {
        "target": target,
        "candidate_count": len(candidates),
        "returned": len(out),
        "event_prefix": ev_pref,
        "used": {
            "fields": ["ticker", "event_ticker", "title"],
            "hard_limit": hard_limit,
            "top_k": top_k,
            "strategy": "event_ticker exact -> event_ticker prefix -> global fallback; TFIDF(title+ticker_tokens)",
        },
    }
    return meta, out


import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, brier_score_loss, classification_report
import numpy as np

def train_mini_model(cohort):
    con = duckdb.connect()
    
    candidate_tickers = [m.ticker for m in cohort]  
    print(f"Training model on {len(candidate_tickers)} similar markets...")

    query = f"""
        WITH trade_metrics AS (
            SELECT 
                t.ticker,
                (epoch(m.close_time::TIMESTAMP) - epoch(t.created_time::TIMESTAMP)) as time_to_close,
                t.yes_price,
                (t.yes_price + t.no_price) as trade_unit_value, 
                SUM(t.yes_price + t.no_price) OVER (
                    PARTITION BY t.ticker 
                    ORDER BY t.created_time 
                    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                ) as rolling_liquidity,
                CASE WHEN t.taker_side = 'buy' THEN 1 ELSE -1 END as flow_direction,
                m.result
            FROM '{TRADES}' t
            JOIN '{MARKETS}' m ON t.ticker = m.ticker
            WHERE t.ticker IN (SELECT unnest($candidate_tickers))
              AND m.result IN ('yes', 'no')
        )
        SELECT 
            yes_price as price,
            ln(COALESCE(rolling_liquidity, 1) + 1) as log_liq,
            ln((time_to_close / 3600.0) + 1) as log_ttc,
            flow_direction,
            CASE WHEN result = 'yes' THEN 1 ELSE 0 END as label
        FROM trade_metrics
        WHERE time_to_close > 0
    """
    
    df = con.execute(query, {"candidate_tickers": candidate_tickers}).df()

    if df.empty or len(df) < 20:
        print("!!! Insufficient data to train/test model.")
        return None, 0

    # Ensure we have both classes in our target
    if len(df['label'].unique()) < 2:
        print("!!! Data contains only one outcome class. Cannot validate.")
        return None, len(df)

    # 1. Feature Selection
    X = df[['price', 'log_liq', 'log_ttc']]
    y = df['label']

    # 2. Train/Test Split (80% Train, 20% Test)
    # We use shuffle=True, but in production, you might want to split by time (Temporal Split)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

    # 3. Training
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    # 4. Validation
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] # Probability of 'Yes'

    acc = accuracy_score(y_test, y_pred)
    brier = brier_score_loss(y_test, y_prob)

    print("\n" + "="*30)
    print("MODEL VALIDATION RESULTS")
    print(f"Test Accuracy: {acc:.2%}")
    print(f"Brier Score:   {brier:.4f} (Lower is better)")
    print("-" * 30)
    # Shows precision/recall for both Yes and No outcomes
    print(classification_report(y_test, y_pred))
    print("="*30 + "\n")
    
    # Re-train on full dataset before returning for maximum production performance
    model.fit(X, y)
    
    return model, len(X_train), len(X_test), acc, brier
# 3. Predict for the Current Market
def get_model_signal(model, current_price, current_liq, current_ttc):
    # Prepare the input (must match training transforms)
    input_data = pd.DataFrame([[
        current_price, 
        np.log(current_liq + 1), 
        np.log(current_ttc + 1)
    ]], columns=['price', 'log_liq', 'log_ttc'])
    
    # Get probability of 'Yes' winning
    prob_yes = model.predict_proba(input_data)[0][1]
    
    return prob_yes








def evaluate_edge(
    target_market: Dict[str, Any],
    cohort: List[SimilarMarket],
    *,
    min_samples: int = 40,
) -> Dict[str, Any]:
    """
    Compare current (price/liquidity/time_to_close) to history in cohort.
    Returns BUY YES/NO if edge is positive, else NO_TRADE.
    """

    # Current state from markets table values
    yes_ask = _to_prob(target_market.get("yes_ask"))
    no_ask = _to_prob(target_market.get("no_ask"))
    liquidity = float(target_market.get("liquidity")) if target_market.get("liquidity") is not None else None
    close_time = target_market.get("close_time")
    close_time_dt = datetime.fromisoformat(close_time.replace('Z', '+00:00'))
    ttc_h = _time_to_close_hours(close_time_dt)

    if yes_ask is None or no_ask is None or liquidity is None or ttc_h is None:
        return {
            "agent": "HistoricalAgent",
            "action": "NO_TRADE",
            "direction": None,
            "score": 0.0,
            "reason": "Missing current yes/no ask, liquidity, or close_time for evaluation.",
            "signals": {},
        }
    model, train_count, test_count, acc, brier = train_mini_model(cohort)
    prob_yes = get_model_signal(model, yes_ask, liquidity, ttc_h)
    return prob_yes, len(cohort), train_count, test_count, acc, brier


def run_historical_agent(ctx: dict) -> Dict[str, Any]:
    ticker = ctx.get("ticker")  
    con = duckdb.connect()
    meta, cohort = find_similar_markets(con, ctx)

    # print a couple of market tickers
    print("Similar markets found:")
    for m in cohort[:5]:
        print(f" - {m.ticker} (score: {m.score:.3f})")

    if "error" in meta:
        return {
            "agent": "HistoricalAgent",
            "action": "NO_TRADE",
            "direction": None,
            "score": 0.0,
            "reason": meta["error"],
            "signals": {},
        }

    target = {
        "ticker": ctx.get("ticker"),
        "event_ticker": ctx.get("event_ticker"),
        "title": ctx.get("title"),
        "yes_ask": ctx.get("yes_ask"),
        "no_ask": ctx.get("no_ask"),
        "liquidity": ctx.get("volume"),
        "close_time": ctx.get("close_time"),
    }

    print(len(cohort), "similar markets found for", ticker)
    prob_yes, sim_count, train_count, test_count, acc, brier = evaluate_edge(target, cohort)
    
    return {
        "agent": "HistoricalAgent",
        "action": "BUY" if prob_yes > 0.1 else "NO_TRADE",
        "direction": "YES" if prob_yes > 0.5 else "NO" if prob_yes > 0.1 else None,
        "score": prob_yes,
        "reason": f"Model trained on {train_count} samples from {sim_count} similar markets. Tested on {test_count} samples. Accuracy: {acc:.2f}, Brier Score: {brier:.2f}",
        "signals": {
            "current_price": target["yes_ask"],
            "current_liquidity": target["liquidity"],
            "current_time_to_close_h": _time_to_close_hours(datetime.fromisoformat(target["close_time"].replace('Z', '+00:00'))),
            "prob_yes_win": prob_yes,
            "similar_markets_evaluated": sim_count,
            "training_samples": train_count,
            "prefix_match": meta.get("event_prefix"),
        },
    }


