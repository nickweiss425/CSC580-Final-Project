import os
import asyncio
import json
from typing import Any, Dict
from dotenv import load_dotenv

load_dotenv()

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient


# set up the model connection using ChatGPT 4o mini as model
_model_client = OpenAIChatCompletionClient(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
)


# set up agent, connect with model defined above
_rules_agent = AssistantAgent(
    name="RulesAgent",
    model_client=_model_client,
    system_message=(
        "You are a prediction market rules analyst.\n"
        "Given a market context JSON, return STRICT JSON only with keys:\n"
        "yes_means (string), no_means (string), ambiguity_flags (list), "
        "clarity_score (0..1 number), notes (string).\n"
        "Use rules_primary/secondary as truth. If NO isn't stated, infer opposite of YES."
    ),
)


async def run_rules_agent(ctx: Dict[str, Any]) -> Dict[str, Any]:
    # build the promp using the market context dictionary
    prompt = "Market context JSON:\n" + json.dumps(ctx, indent=2) + "\n\nReturn STRICT JSON only."

    # run the agent with the prompt
    result = await _rules_agent.run(task=[TextMessage(content=prompt, source="user")])

    # get last message, which should be model output
    text = result.messages[-1].content

    # parse output (or fall back safely)
    try:
        sem = json.loads(text)
    except json.JSONDecodeError:
        sem = {
            "yes_means": "",
            "no_means": "",
            "ambiguity_flags": ["non_json_output"],
            "clarity_score": 0.0,
            "notes": f"Non-JSON output:\n{text}",
        }

    # get clarity score and any flags
    clarity = float(sem.get("clarity_score", 0.0) or 0.0)
    flags = sem.get("ambiguity_flags", []) or []


    # decide whether or not to veto (if clarity is too low)
    veto = (clarity < 0.8) or (len(flags) > 0)

    
    # wrap into standardized AgentOutput
    return {
        "agent": "RulesAgent",
        "action": "NO_TRADE" if veto else None, # gatekeeper: veto or abstain
        "direction": None,                      # never chooses YES/NO
        "score": clarity,                       # interpret as "rules clarity"
        "reason": "Rules are clear." if not veto else f"Ambiguity/low clarity: {flags} (score={clarity:.2f})",
        "signals": {
            "yes_means": sem.get("yes_means", ""),
            "no_means": sem.get("no_means", ""),
            "ambiguity_flags": flags,
            "notes": sem.get("notes", ""),
        },
        "raw": sem,
    }

# need async function because the agent.run() makes a network call, need to wait
# for a response
def run_rules_agent_sync(ctx: Dict[str, Any]) -> Dict[str, Any]:
    return asyncio.run(run_rules_agent(ctx))
