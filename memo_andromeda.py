"""
memo.py

NEXT STEP after scoring.py.

You now have:
    - unified_claims.json  (Step 5 - all 23 claims, flat table)
    - risk_metrics.json    (Step 6 - loss ratio, trend, maturity caveats, etc.)

This script does STEP 7: hands both files to the LLM and asks it to write
a plain-English underwriting memo - the kind of write-up an underwriter
would actually attach to a submission file.

THE CRITICAL RULE HERE: the LLM is writing PROSE ONLY. Every number that
appears in the memo must already exist in risk_metrics.json - the model
is told explicitly not to calculate, re-derive, or "helpfully" round
anything itself. This matters because Step 6 already did the arithmetic
correctly in plain Python; letting the LLM touch numbers again here would
reintroduce exactly the risk we spent Steps 5/6 avoiding - an LLM
silently getting a sum wrong. So this step is narrative-generation, not
data-generation.

Orchestration note: the direct `ollama.chat(...)` call has been replaced
with an Andromeda `Agent` (same `Agent`/`AgentConfig`/`ModelConfig` used by
structuring.py), and the step is wired through a one-node WorkflowBuilder
pipeline so it plugs into the same pipeline shape scoring.py uses. Prompt
text, payload construction, file I/O, and console output are unchanged.
"""

import json
import os

from andromeda import HumanMessage
from andromeda.config import AgentConfig, ModelConfig
from andromeda.core.agent import Agent
from andromeda.core.workflow import WorkflowBuilder


OUTPUT_PATH = "/home/rchanda/TestFolder/POCs_Folder/Training_POC_Insurance/Data/Output"
UNIFIED_CLAIMS_IN = os.path.join(OUTPUT_PATH, "unified_claims.json")
RISK_METRICS_IN = os.path.join(OUTPUT_PATH, "risk_metrics.json")
MEMO_OUT = os.path.join(OUTPUT_PATH, "underwriting_memo.md")

MODEL_NAME = "gpt-oss:20b"   # same text model used in structuring.py

MEMO_PROMPT = """You are an underwriting assistant. You will be given the already-computed \
risk metrics and claims data for a commercial insurance submission. Write a professional \
underwriting memo summarizing the risk picture, as if attaching it to the submission file for \
an underwriter to review before pricing the account.

CRITICAL RULES - follow these exactly:
1. Do NOT calculate, re-derive, round differently, or "correct" any number. Every dollar \
   figure, percentage, and count in your memo must be copied EXACTLY from the data provided - \
   your only job is to explain what the numbers mean in plain English, not to produce new ones.
2. If the data includes "maturity_caveats", you MUST include these in your memo, close to \
   verbatim - they are important warnings about which numbers are still developing and \
   should not be treated as final. Do not soften or omit them.
3. Do NOT make an underwriting recommendation (e.g. "we should decline this risk" or "this is \
   a good account to write") - your job is to summarize the facts clearly so a human \
   underwriter can make that judgment themselves. Present the picture, don't decide for them.
4. If a metric is null or missing, say so plainly rather than guessing a value.

Structure the memo with these sections:
    ## Executive Summary
        (2-3 sentences: overall loss experience picture, in plain terms)
    ## Loss Experience by Policy Year
        (a short line per year: premium, incurred, loss ratio)
    ## Notable Claims
        (large losses, if any; otherwise state none were flagged)
    ## Cause of Loss Patterns
        (from top_causes_of_loss - what's repeating and how often)
    ## Trend
        (oldest vs newest year loss ratio, and the stated direction)
    ## Data Quality & Maturity Notes
        (the maturity_caveats, presented clearly - this section should make the underwriter
        pause before treating any flagged year's loss ratio as final)

Write in clear, professional prose - short paragraphs and bullet points where useful, not just
a data dump. Use Markdown formatting."""


def build_memo_agent() -> Agent:
    return Agent(
        AgentConfig(
            name="memo_writer",
            model=ModelConfig(name=MODEL_NAME, provider="ollama"),
            prompt=MEMO_PROMPT,
        )
    )


def step_memo(state: dict) -> dict:
    unified_claims = state["unified_claims"]
    risk_metrics = state["risk_metrics"]

    print(f"Loaded unified claims ({unified_claims['submission_summary']['total_claims']} claims) "
          f"and risk metrics for {unified_claims['submission_summary']['insured_name']}")

    # We hand the model the submission summary + risk metrics (the numbers
    # it must quote exactly) plus the claims list (for context on what
    # actually happened, e.g. specific causes/dates it might reference).
    # We do NOT ask it to look at raw PDFs or re-extract anything - by
    # this point all the reading and math is already done.
    payload = {
        "submission_summary": unified_claims["submission_summary"],
        "risk_metrics": risk_metrics,
        "claims": unified_claims["unified_claims"],
    }

    user_message = (
        "Here is the computed data for this submission. Write the underwriting memo "
        "using ONLY these numbers - do not calculate anything new.\n\n"
        f"{json.dumps(payload, indent=2)}"
    )

    print("Generating memo...")
    agent = state.get("agent") or build_memo_agent()
    reply = agent.invoke([HumanMessage(content=user_message)])
    memo_text = reply[-1].content

    return {"memo_text": memo_text}


def build_memo_pipeline() -> WorkflowBuilder:
    pipeline = WorkflowBuilder(name="MemoPipeline")
    pipeline.start("memo").run(step_memo)
    return pipeline


def generate_memo():
    with open(UNIFIED_CLAIMS_IN, "r") as f:
        unified_claims = json.load(f)

    with open(RISK_METRICS_IN, "r") as f:
        risk_metrics = json.load(f)

    pipeline = build_memo_pipeline()
    result = pipeline.execute(
        state={"unified_claims": unified_claims, "risk_metrics": risk_metrics}
    )
    memo_text = result["memo_text"]

    with open(MEMO_OUT, "w") as f:
        f.write(memo_text)

    print(f"\nMemo saved to: {MEMO_OUT}")
    print("\n--- Preview ---\n")
    print(memo_text[:500] + ("..." if len(memo_text) > 500 else ""))


if __name__ == "__main__":
    generate_memo()
