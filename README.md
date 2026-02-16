# CSC580-Final-Project

## Run App

streamlit run app/Home.py

---

# System Overview

This project implements a modular, multi-agent decision support system for prediction markets.  
Each agent specializes in a specific task and returns a standardized output.  
A deterministic aggregation step combines all agent outputs into a final recommendation.

The system currently supports:

- Guardrail agents (veto / safety checks)  
- Directional agents (recommend YES/NO to buy)  

---

# Implemented Agents

## RulesAgent (LLM-Based)  
Type: Guardrail Agent  

Purpose:  
Analyzes the market resolution rules to determine whether the contract is clearly defined and unambiguous.

Main Functionality:
- Extracts what YES and NO mean from the rules.
- Flags ambiguous or risky resolution language.
- Produces a clarity score (0–1).
- Can veto the trade (action = "NO_TRADE") if clarity is low.

Justification:  
Prevents trading on poorly defined or ambiguous contracts.  
Acts as a semantic validation layer before any directional reasoning.

---

## RiskAgent (Deterministic)  
Type: Guardrail Agent  

Purpose:  
Evaluates market and execution quality.

Main Functionality:
- Checks bid/ask spreads.
- Checks 24h volume and open interest.
- Checks liquidity depth.
- Checks quote staleness.
- Performs pricing sanity checks (e.g., yes_ask + no_ask).

Returns:
- action = "NO_TRADE" if market conditions are unsafe.
- Otherwise abstains.

Justification:  
Prevents trades in illiquid, inactive, or structurally broken markets.  
Ensures the system does not recommend trades that are not realistically executable.

---

## PricingBaselineAgent (Deterministic)  
Type: Directional Agent  

Purpose:  
Provides a baseline directional recommendation based on current ask prices.

Main Functionality:
- Compares yes_ask vs no_ask.
- Recommends buying the cheaper side.
- Generates a confidence score based on price separation.

Returns:
- action = "BUY"
- direction = "YES" or "NO"

Justification:  
Provides a simple, transparent directional baseline.  
Acts as a market-implied reference signal before more advanced agents (Trend/Evidence) are added.

---

# Aggregation Logic

The system combines agent outputs deterministically:

1. If any guardrail agent returns NO_TRADE -> Final decision is NO_TRADE.
2. Otherwise, directional agents vote.
3. The strongest directional signal determines the final direction.
4. Confidence is computed conservatively from agent scores.

This design ensures:
- Safety-first decision making  
- Deterministic and explainable aggregation  
- Easy extensibility for additional agents  

---

# Planned Extensions

The architecture supports adding additional specialized agents such as:

- TrendAgent (candlestick momentum analysis)  
- EvidenceAgent (LLM-based external reasoning)  
- Additional risk or statistical agents  

New agents only need to return a standardized output dictionary to integrate into the pipeline.
