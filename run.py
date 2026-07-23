"""
run_pipeline.py

STEP 8 - the final piece. Ties everything together.

Your full pipeline, in order:

    processing.py    ->  PDF -> .md files                    (you already built this)
    structuring.py   ->  .md files -> structured JSON          (Step 3/4)
    scoring.py        ->  structured JSON -> unified claims
                          + risk metrics                        (Step 5/6)
    memo.py            ->  risk metrics -> underwriting memo     (Step 7)
    analytics_report.py -> everything -> one human-readable report (Step 8)

Orchestration note: structuring.py, scoring.py, memo.py, and
analytics_report.py were each already converted to run their own work
through a small Andromeda WorkflowBuilder. This file is the one place
that used to just call their entrypoints back to back as plain Python -
now it's a single top-level WorkflowBuilder ("UWTriagePipeline") with one
node per stage, and each node's body is still exactly the same one-line
call to that module's own entrypoint (structuring.structure_files(),
scoring.main(), memo.generate_memo(), analytics_report.build_report()).
Nothing about what each stage does, reads, or writes has changed - only
the thing that sequences them is now an Andromeda pipeline instead of
four bare function calls in main().

At the end, it prints a short, human-readable summary of what came out -
the numbers an underwriter would actually want to see first, before
opening any of the individual output files.
"""

import json
import os

# These three imports assume structuring.py, scoring.py, memo.py, and
# analytics_report.py are in the same folder as this script. Each one
# still works fine on its own if run directly - this script just calls
# them back to back, now as steps in one WorkflowBuilder pipeline.
import structuring
import scoring
import memo
import analytics_report

from andromeda.core.workflow import WorkflowBuilder


def print_final_summary():
    """
    Reads the final output files and prints a short human-readable recap -
    the numbers an underwriter would look at first.
    """
    with open(scoring.UNIFIED_CLAIMS_OUT, "r") as f:
        unified = json.load(f)
    with open(scoring.RISK_METRICS_OUT, "r") as f:
        metrics = json.load(f)

    summary = unified["submission_summary"]
    aggregate = metrics["aggregate"]

    print("\n" + "=" * 72)
    print(f"  UNDERWRITING RISK TRIAGE - FINAL SUMMARY")
    print(f"  {summary['insured_name']}")
    print("=" * 72)
    print(f"  Policy years covered:      {summary['policy_years_covered']}")
    print(f"  Total claims:              {summary['total_claims']}")
    print(f"  Total premium (all years): ${summary['total_premium_all_years']:,.2f}")
    print(f"  Total incurred (all years):${aggregate['total_incurred']:,.2f}")
    print(f"  Overall loss ratio:        {aggregate['overall_loss_ratio_pct']}%")
    print(f"  Avg claims per year:       {aggregate['avg_claims_per_year']}")

    if metrics["trend"]:
        t = metrics["trend"]
        print(f"  Trend:                     {t['direction']} "
              f"({t['oldest_year_loss_ratio_pct']}% -> {t['newest_year_loss_ratio_pct']}%)")

    print(f"  Open exposure (reserves):  ${metrics['open_exposure']['total_outstanding_reserve']:,.2f}")
    print(f"  Large losses flagged:      {len(metrics['large_losses'])}")

    if metrics["maturity_caveats"]:
        print(f"\n  ⚠ {len(metrics['maturity_caveats'])} data maturity caveat(s) - see risk_metrics.json")

    print("\n  Output files:")
    print(f"    - {analytics_report.ANALYTICS_REPORT_OUT}   <- START HERE (full underwriter report)")
    print(f"    - {memo.MEMO_OUT}")
    print(f"    - {scoring.UNIFIED_CLAIMS_OUT}")
    print(f"    - {scoring.RISK_METRICS_OUT}")
    print("=" * 72)


# =============================================================================
# PIPELINE STAGES
# =============================================================================
# Each node is a one-line delegate into that module's own entrypoint, which
# itself already runs as an Andromeda WorkflowBuilder internally (see
# structuring.py/scoring.py/memo.py/analytics_report.py). This master
# pipeline is what sequences them, replacing the plain back-to-back calls
# main() used to make.
# =============================================================================

def step_structuring(state: dict) -> dict:
    print("### STEP 3/4 - Structuring .md files into JSON ###\n")
    structuring.structure_files()
    return {}


def step_scoring(state: dict) -> dict:
    print("\n### STEP 5/6 - Normalizing + scoring risk ###\n")
    scoring.main()
    return {}


def step_memo(state: dict) -> dict:
    print("\n### STEP 7 - Generating underwriting memo ###\n")
    memo.generate_memo()
    return {}


def step_report(state: dict) -> dict:
    print("\n### STEP 8 - Building final analytics report ###\n")
    analytics_report.build_report()
    return {}


def build_run_pipeline() -> WorkflowBuilder:
    pipeline = WorkflowBuilder(name="UWTriagePipeline")
    (
        pipeline
        .start("structuring").run(step_structuring)
        .then("scoring").run(step_scoring)
        .then("memo").run(step_memo)
        .finish("report").run(step_report)
    )
    return pipeline


def main():
    pipeline = build_run_pipeline()
    pipeline.execute(state={})
    print_final_summary()


if __name__ == "__main__":
    main()
