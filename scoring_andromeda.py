"""
scoring.py

NEXT STEP after structuring.py.

You now have Data/Structured/structured_all_files.json - a list of 4
separate LossRun objects (one per PDF), each holding its own list of
claims. That's not yet usable for risk analysis, because the claims are
still split by source document instead of being one flat table spanning
all years.

This script does two things, in order, wired together as an Andromeda
WorkflowBuilder pipeline (the same orchestration primitive structuring.py's
Agent calls presumably already run inside):

    STEP 5 - NORMALIZE / MERGE
        Flatten all 4 LossRun.claims lists into ONE table of claims,
        tagging every row with which policy year/carrier it came from.
        Also builds a submission-level summary (total premium, years
        covered, etc).

    STEP 6 - RISK SCORING
        Pure Python math over that flat table - no LLM involved here at
        all. Computes the standard underwriting metrics: loss ratio,
        frequency, severity, trend, open exposure, large-loss flags.

Why no LLM in this file: everything here is arithmetic on numbers that
were already read and validated in the previous step. LLMs are
unreliable at multi-step math, so once a number has been transcribed
correctly, every calculation on it from here on is done in plain Python,
which is deterministic and always gives the same answer for the same
input.

Why WorkflowBuilder here at all, given neither step calls an LLM: so this
file plugs into the same pipeline object structuring.py's steps do,
instead of being a bare script called out-of-band. The two functions'
logic is unchanged from a plain-script version -- only the orchestration
shell around them changed.
"""

import json
import os
from datetime import datetime

from andromeda.core.workflow import WorkflowBuilder

STRUCTURED_PATH = "/home/rchanda/TestFolder/POCs_Folder/Training_POC_Insurance/Data/Structured"
COMBINED_INPUT = os.path.join(STRUCTURED_PATH, "structured_all_files.json")

OUTPUT_PATH = "/home/rchanda/TestFolder/POCs_Folder/Training_POC_Insurance/Data/Output"
UNIFIED_CLAIMS_OUT = os.path.join(OUTPUT_PATH, "unified_claims.json")
RISK_METRICS_OUT = os.path.join(OUTPUT_PATH, "risk_metrics.json")

# A single claim above this % of the per-occurrence limit gets flagged as
# a "large loss" - worth an underwriter's individual attention rather than
# being averaged into normal claim activity.
LARGE_LOSS_THRESHOLD_PCT = 0.50


# =============================================================================
# STEP 5 - NORMALIZE / MERGE
# =============================================================================

def normalize_submission(loss_runs: list[dict]) -> dict:
    """
    Takes the list of per-file LossRun dicts (one per PDF) and produces:
      - a flat list of every claim across all years, each tagged with
        which policy year/carrier it came from
      - a submission-level summary (total premium, years covered, etc)

    This is a pure restructuring step - no numbers are recalculated here,
    we're just reshaping "4 documents, each with its own claims list"
    into "1 table of all claims, each row knowing its source document".
    """
    unified_claims = []

    for loss_run in loss_runs:
        policy_year_label = f"{loss_run.get('policy_start', '?')} - {loss_run.get('policy_end', '?')}"

        for claim in loss_run.get("claims", []):
            # Copy the claim and attach which document/policy year it's from.
            # This is what lets Step 6 compute "loss ratio BY YEAR" as well
            # as an aggregate across the whole submission.
            row = dict(claim)
            row["source_file"] = loss_run.get("source_file")
            row["carrier_name"] = loss_run.get("carrier_name")
            row["policy_year"] = policy_year_label
            row["policy_start"] = loss_run.get("policy_start")
            row["policy_end"] = loss_run.get("policy_end")
            row["annual_premium"] = loss_run.get("annual_premium")
            row["per_occurrence_limit"] = loss_run.get("per_occurrence_limit")
            unified_claims.append(row)

    submission_summary = {
        "insured_name": loss_runs[0].get("insured_name") if loss_runs else None,
        "policy_years_covered": len(loss_runs),
        "total_premium_all_years": sum(lr.get("annual_premium") or 0 for lr in loss_runs),
        "total_claims": len(unified_claims),
        "source_files": [lr.get("source_file") for lr in loss_runs],
    }

    return {
        "submission_summary": submission_summary,
        "unified_claims": unified_claims,
    }


# =============================================================================
# STEP 6 - RISK SCORING (pure math, no LLM)
# =============================================================================

def compute_risk_metrics(normalized: dict, loss_runs: list[dict]) -> dict:
    """
    Computes the standard underwriting metrics an underwriter actually
    looks at. Everything here is plain arithmetic over already-validated
    numbers - nothing is estimated or asked of an LLM.
    """
    claims = normalized["unified_claims"]

    # --- per-policy-year metrics -------------------------------------------
    # Group claims by which document/policy year they came from, so we can
    # compute a loss ratio and frequency PER YEAR, not just overall - this
    # is what lets us see a TREND (is it getting better or worse over time).
    by_year = {}
    for loss_run in loss_runs:
        year_label = f"{loss_run.get('policy_start', '?')} - {loss_run.get('policy_end', '?')}"
        year_claims = [c for c in claims if c["policy_year"] == year_label]

        premium = loss_run.get("annual_premium") or 0
        total_incurred = sum(c.get("total_incurred", 0) for c in year_claims)
        total_reserve = sum(c.get("reserve", 0) for c in year_claims)
        open_count = len([c for c in year_claims if c.get("status") in ("Open", "Reopened")])

        by_year[year_label] = {
            "carrier_name": loss_run.get("carrier_name"),
            "premium": premium,
            "claim_count": len(year_claims),
            "total_incurred": round(total_incurred, 2),
            # Loss ratio = total incurred / premium. This is THE core
            # underwriting metric - <50% is great, 50-80% acceptable,
            # >100% means the carrier paid out more than it collected.
            "loss_ratio_pct": round((total_incurred / premium) * 100, 1) if premium else None,
            # Severity = average cost per claim - distinguishes "many
            # small claims" from "one bad outlier".
            "avg_severity": round(total_incurred / len(year_claims), 2) if year_claims else 0,

            # --- MATURITY INDICATORS ---------------------------------------
            # A loss ratio is only as trustworthy as the claims behind it
            # are settled. An open claim's total_incurred is an ESTIMATE
            # (paid so far + a reserve guess) - it can still grow before the
            # claim closes. A closed claim's total_incurred is final. Two
            # years can show the same loss ratio today and mean very
            # different things: one is a verdict, the other is a snapshot
            # mid-development. These fields make that visible instead of
            # burying it inside a single "loss ratio" number.
            "open_claim_count": open_count,
            "pct_claims_open": round((open_count / len(year_claims)) * 100, 1) if year_claims else 0,
            # What fraction of this year's total_incurred is still just a
            # reserve (unpaid estimate) rather than money actually paid out.
            # High % here = this year's loss ratio should be read as a
            # FLOOR, not a final number - it can still climb.
            "reserve_share_of_incurred_pct": round((total_reserve / total_incurred) * 100, 1) if total_incurred else 0,
        }

    # --- aggregate (whole submission, all years combined) ------------------
    total_premium = sum(y["premium"] for y in by_year.values())
    total_incurred_all = sum(y["total_incurred"] for y in by_year.values())
    total_claim_count = len(claims)

    aggregate = {
        "total_premium": round(total_premium, 2),
        "total_incurred": round(total_incurred_all, 2),
        "overall_loss_ratio_pct": round((total_incurred_all / total_premium) * 100, 1) if total_premium else None,
        "total_claim_count": total_claim_count,
        # Frequency = average claims per policy year - a high number
        # signals a systemic operational problem, not just bad luck.
        "avg_claims_per_year": round(total_claim_count / len(by_year), 2) if by_year else 0,
    }

    # --- trend: is loss ratio improving or worsening over time? ------------
    # Sort years chronologically by their ACTUAL policy_start date (parsed
    # as a real date, not just string/premium sorting - a string sort on
    # "06/01/2023" doesn't reliably order by year, and sorting by premium
    # is not the same thing as sorting by time at all).
    def parse_date_safe(date_str):
        try:
            return datetime.strptime(date_str, "%m/%d/%Y")
        except (ValueError, TypeError):
            return None

    years_with_dates = [
        (year_label, metrics, parse_date_safe(loss_runs[i].get("policy_start")))
        for i, (year_label, metrics) in enumerate(by_year.items())
        if metrics["loss_ratio_pct"] is not None
    ]
    years_sorted = sorted(
        [item for item in years_with_dates if item[2] is not None],
        key=lambda item: item[2]  # sort by actual parsed date, oldest first
    )
    trend = None
    if len(years_sorted) >= 2:
        oldest_ratio = years_sorted[0][1]["loss_ratio_pct"]
        newest_ratio = years_sorted[-1][1]["loss_ratio_pct"]
        direction = "improving" if newest_ratio < oldest_ratio else "worsening" if newest_ratio > oldest_ratio else "flat"
        trend = {
            "oldest_year": years_sorted[0][0],
            "oldest_year_loss_ratio_pct": oldest_ratio,
            "newest_year": years_sorted[-1][0],
            "newest_year_loss_ratio_pct": newest_ratio,
            "direction": direction,
        }

    # --- open exposure: reserves still sitting on open/reopened claims -----
    # This money hasn't been paid out yet but could still grow - it's
    # unresolved risk that a closed-claims-only view would miss entirely.
    open_claims = [c for c in claims if c.get("status") in ("Open", "Reopened")]
    open_exposure = {
        "open_claim_count": len(open_claims),
        "total_outstanding_reserve": round(sum(c.get("reserve", 0) for c in open_claims), 2),
    }

    # --- large-loss flags ----------------------------------------------------
    # Any single claim that ate up a big chunk of the per-occurrence limit
    # deserves individual underwriter attention, not just being averaged
    # into the overall severity number.
    large_losses = []
    for c in claims:
        limit = c.get("per_occurrence_limit")
        if limit and c.get("total_incurred", 0) >= limit * LARGE_LOSS_THRESHOLD_PCT:
            large_losses.append({
                "claim_number": c.get("claim_number"),
                "policy_year": c.get("policy_year"),
                "total_incurred": c.get("total_incurred"),
                "per_occurrence_limit": limit,
                "pct_of_limit": round((c.get("total_incurred", 0) / limit) * 100, 1),
            })

    # --- cause-of-loss concentration -----------------------------------------
    # Repeated causes across claims suggest a fixable operational hazard,
    # rather than random bad luck - worth surfacing as its own signal.
    cause_counts = {}
    for c in claims:
        cause = c.get("cause_of_loss", "Unknown")
        cause_counts[cause] = cause_counts.get(cause, 0) + 1
    top_causes = sorted(cause_counts.items(), key=lambda x: x[1], reverse=True)[:5]

    # --- data maturity caveats ------------------------------------------------
    # Explicit, human-readable warnings for anything that could make the
    # numbers above misleading if read at face value. This is what Step 7
    # (the memo) should quote directly rather than re-deriving on its own -
    # keeping the caveat logic here, in deterministic Python, means the
    # LLM never has to judge "is this year mature enough to trust" itself.
    maturity_caveats = []
    for year_label, metrics in by_year.items():
        if metrics["pct_claims_open"] >= 30:
            maturity_caveats.append(
                f"{year_label}: {metrics['pct_claims_open']}% of claims still open "
                f"({metrics['reserve_share_of_incurred_pct']}% of incurred is still reserve, not paid) - "
                f"this year's loss ratio ({metrics['loss_ratio_pct']}%) is likely to rise further before "
                f"this policy year is fully developed."
            )

    if trend and years_sorted:
        oldest_open_pct = years_sorted[0][1]["pct_claims_open"]
        newest_open_pct = years_sorted[-1][1]["pct_claims_open"]
        if abs(newest_open_pct - oldest_open_pct) >= 20:
            maturity_caveats.append(
                f"Trend comparison caution: {trend['oldest_year']} is {oldest_open_pct}% open vs. "
                f"{trend['newest_year']} at {newest_open_pct}% open - these two years are at different "
                f"stages of development, so the trend direction ('{trend['direction']}') may partly "
                f"reflect that maturity gap rather than a genuine change in underlying risk."
            )

    return {
        "by_policy_year": by_year,
        "aggregate": aggregate,
        "trend": trend,
        "open_exposure": open_exposure,
        "large_losses": large_losses,
        "top_causes_of_loss": [{"cause": c, "count": n} for c, n in top_causes],
        "maturity_caveats": maturity_caveats,
    }


# =============================================================================
# ANDROMEDA WORKFLOW WIRING
# =============================================================================
# Both step functions below are thin adapters: `state in, state out`, calling
# straight into the two functions above with zero logic changes. This is the
# same shape structuring.py's own steps use, so scoring.py's output can later
# be fused into one end-to-end WorkflowBuilder pipeline (structuring -> this)
# instead of being invoked as a disconnected second script.

def step_normalize(state: dict) -> dict:
    normalized = normalize_submission(state["loss_runs"])
    return {"normalized": normalized}


def step_score(state: dict) -> dict:
    risk_metrics = compute_risk_metrics(state["normalized"], state["loss_runs"])
    return {"risk_metrics": risk_metrics}


def build_scoring_pipeline() -> WorkflowBuilder:
    pipeline = WorkflowBuilder(name="ScoringPipeline")
    (
        pipeline
        .start("normalize").run(step_normalize)
        .finish("score").run(step_score)
    )
    return pipeline


# =============================================================================
# MAIN
# =============================================================================

def main():
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    with open(COMBINED_INPUT, "r") as f:
        loss_runs = json.load(f)  # list of the 4 LossRun dicts from structuring.py

    print(f"Loaded {len(loss_runs)} structured loss run(s) from {COMBINED_INPUT}\n")

    pipeline = build_scoring_pipeline()
    result = pipeline.execute(state={"loss_runs": loss_runs})

    normalized = result["normalized"]
    risk_metrics = result["risk_metrics"]

    # --- Step 5 output ---
    with open(UNIFIED_CLAIMS_OUT, "w") as f:
        json.dump(normalized, f, indent=2)
    print(f"Step 5 complete: {normalized['submission_summary']['total_claims']} claims "
          f"unified across {normalized['submission_summary']['policy_years_covered']} policy year(s)")
    print(f"  -> saved to {UNIFIED_CLAIMS_OUT}\n")

    # --- Step 6 output ---
    with open(RISK_METRICS_OUT, "w") as f:
        json.dump(risk_metrics, f, indent=2)
    print("Step 6 complete: risk metrics computed")
    print(f"  Overall loss ratio: {risk_metrics['aggregate']['overall_loss_ratio_pct']}%")
    print(f"  Total claims: {risk_metrics['aggregate']['total_claim_count']}")
    print(f"  Open exposure (outstanding reserves): ${risk_metrics['open_exposure']['total_outstanding_reserve']:,.2f}")
    print(f"  Large losses flagged: {len(risk_metrics['large_losses'])}")
    if risk_metrics["trend"]:
        print(f"  Trend: {risk_metrics['trend']['direction']}")
    if risk_metrics["maturity_caveats"]:
        print("\n  ⚠ DATA MATURITY CAVEATS:")
        for caveat in risk_metrics["maturity_caveats"]:
            print(f"    - {caveat}")
    print(f"  -> saved to {RISK_METRICS_OUT}")


if __name__ == "__main__":
    main()
