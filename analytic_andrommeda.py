"""
analytics_report_llm.py

ALTERNATIVE to analytics_report.py - instead of Python building fixed
tables, this hands the LLM ALL the computed data plus a detailed
system prompt describing what an underwriter needs to see, and lets the
model decide the structure, tables, and emphasis itself.

THE ONE RULE THAT DOESN'T CHANGE: the LLM can decide HOW to present the
data (which tables, what order, how to phrase things) but every number
it writes must be copied exactly from the JSON it's given. It is never
allowed to calculate, re-derive, or round a number itself - Step 6
(scoring.py) already did that in plain, auditable Python. This prompt
just gives it the freedom analytics_report.py's hardcoded templates
didn't.

TRADEOFF vs analytics_report.py (worth knowing before choosing between
them): this version is more flexible and adapts its structure to
whatever the data actually looks like, but it is NOT deterministic -
run it twice on the same data and the report's exact wording, table
choices, and layout can come out differently each time, and there's no
compiler-style guarantee it won't accidentally paraphrase a number
wrong. analytics_report.py is rigid but always produces the exact same
correct numbers, verifiably. Which one to use as the "official" report
is your call - this file exists to make that comparison possible.

Orchestration note: the direct `ollama.chat(...)` call is replaced with
an Andromeda `Agent` (same `Agent`/`AgentConfig`/`ModelConfig` used by
structuring.py/memo.py), and the step is wired through a one-node
WorkflowBuilder pipeline, matching the shape every other converted file
in this pipeline uses. Prompt text, payload construction, and file I/O
are unchanged.
"""

import json
import os

import scoring
import memo

from andromeda import HumanMessage
from andromeda.config import AgentConfig, ModelConfig
from andromeda.core.agent import Agent
from andromeda.core.workflow import WorkflowBuilder


LLM_REPORT_OUT = os.path.join(scoring.OUTPUT_PATH, "analytics_report_llm.md")

MODEL_NAME = "gpt-oss:20b"

SYSTEM_PROMPT = """You are an underwriting analytics assistant. You will be given the complete, \
already-computed claims data and risk metrics for a commercial insurance submission. Your job is \
to write a single, complete, well-organized analytics report that gives an underwriter the full \
risk picture at a glance.

CRITICAL RULE - READ CAREFULLY:
You must NOT calculate, re-derive, round differently, sum, average, or otherwise compute ANY \
number yourself. Every dollar figure, percentage, ratio, and count that appears anywhere in your \
report must be copied EXACTLY from the data provided below. If you need a number that isn't \
directly present in the data, do not estimate it - state that it isn't available. Your job is \
entirely presentation and interpretation of numbers that already exist, never generation of new \
ones. This is the single most important rule in this task - a wrong number in an underwriting \
report is a serious, costly error.

THE KEY QUESTIONS AN UNDERWRITER NEEDS ANSWERED, IN ROUGHLY THIS PRIORITY ORDER:
1. Overall risk picture - is this a good or bad account, in plain numeric terms (loss ratio,
   total claims, total incurred vs premium)?
2. Year-by-year breakdown - for EACH prior policy year: which carrier, premium, total incurred,
   loss ratio, claim count, and whether that year was profitable for the prior carrier
   (loss ratio under 100% = profitable for them, over 100% = they lost money).
3. Trend - is the loss experience getting better or worse over time, and by how much?
4. What KINDS of claims are happening - cause-of-loss patterns, and whether the same type of
   loss keeps repeating (a signal of a fixable operational hazard vs random bad luck).
5. Large/notable individual claims - anything that stands out as an outlier.
6. Open exposure - how much reserve is still sitting on unresolved claims, since this money
   hasn't been paid yet but could still be owed.
7. DATA MATURITY - this is critical and must not be skipped or softened: some policy years have
   a high percentage of still-open claims, which means their loss ratios are not final and could
   still rise. You MUST include the maturity_caveats from the data, clearly and prominently, so
   the underwriter doesn't mistake a still-developing year's number for a final verdict.

FORMAT:
Use Markdown. Use tables wherever you're presenting multiple rows of comparable data (e.g. one
row per policy year, one row per cause of loss) - tables are much faster for an underwriter to
scan than prose. Use short paragraphs or bullets for interpretation and context. Organize the
report in whatever section order best tells this particular account's risk story - you are not
required to use a fixed template, but do make sure all 7 questions above are answered somewhere
in the report.

Do NOT make an underwriting recommendation (e.g. "decline this risk" or "this is a good account
to write"). Present the facts and their meaning clearly enough that a human underwriter can make
that call themselves."""


def build_report_agent() -> Agent:
    return Agent(
        AgentConfig(
            name="analytics_report_writer",
            model=ModelConfig(name=MODEL_NAME, provider="ollama"),
            prompt=SYSTEM_PROMPT,
        )
    )


def step_llm_report(state: dict) -> dict:
    with open(scoring.UNIFIED_CLAIMS_OUT, "r") as f:
        unified = json.load(f)
    with open(scoring.RISK_METRICS_OUT, "r") as f:
        risk_metrics = json.load(f)

    payload = {
        "submission_summary": unified["submission_summary"],
        "risk_metrics": risk_metrics,
        "claims": unified["unified_claims"],
    }

    user_message = (
        "Here is the complete computed data for this submission. Write the full analytics "
        "report using ONLY these numbers - do not calculate anything new.\n\n"
        f"{json.dumps(payload, indent=2)}"
    )

    print("Generating LLM-driven analytics report (this may take a bit longer than the "
          "template version, since the model is deciding structure as well as writing)...")

    agent = state.get("agent") or build_report_agent()
    reply = agent.invoke([HumanMessage(content=user_message)])
    report_text = reply[-1].content

    with open(LLM_REPORT_OUT, "w") as f:
        f.write(report_text)

    print(f"LLM-driven report saved to: {LLM_REPORT_OUT}")
    return {"report_text": report_text, "report_path": LLM_REPORT_OUT}


def build_llm_report_pipeline() -> WorkflowBuilder:
    pipeline = WorkflowBuilder(name="AnalyticsReportLLMPipeline")
    pipeline.start("llm_report").run(step_llm_report)
    return pipeline


def generate_llm_report():
    pipeline = build_llm_report_pipeline()
    result = pipeline.execute(state={})
    return result["report_text"]


if __name__ == "__main__":
    generate_llm_report()
