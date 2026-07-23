"""
=============================================================================
 structuring/llm_extractor.py

 STEP 3 / STEP 4 OF THE PIPELINE:  RAW TEXT  ->  STRUCTURED, VALIDATED JSON
=============================================================================

WHAT THIS FILE DOES
--------------------
This is the single place in the whole pipeline where an LLM reads an
unstructured loss-run document and turns it into structured data. It exposes
ONE function - extract_structured() - and that same function is called for
EVERY file in a submission, regardless of which carrier produced it or how
that carrier formatted its report.

THIS IS THE CORE DESIGN IDEA OF THE WHOLE PIPELINE, SO IT'S WORTH BEING
EXPLICIT ABOUT WHY:

  In the real world, an underwriter's submission might contain a loss run
  from Meridian Mutual (a clean table), one from Continental Assurance
  (narrative paragraph blocks, one per claim), one from Summit Underwriters
  (a dense machine-exported grid with abbreviated column codes), and one
  from Pinnacle Casualty that's a faxed scan with no digital text at all.
  Every carrier in the industry has its own template - there is no
  standard.

  A traditional approach would need a hand-written parser (regex, fixed
  column offsets, etc.) PER CARRIER FORMAT. That doesn't scale: every new
  prior carrier the underwriter encounters would require new parsing code,
  and any subtle formatting change breaks it silently.

  An LLM, by contrast, reads the document the way a person would - it
  doesn't care that Summit uses "IncurredTotal" where Meridian uses
  "Incurred" and Continental spells it out as "Total Incurred:". It infers
  the meaning from context and slots it into the same schema every time.
  This function is written to have ZERO awareness of which carrier or
  format it's looking at - that is the entire point.

WHAT THIS FILE DELIBERATELY DOES NOT DO
----------------------------------------
  - It never calculates anything (no totals, no ratios, no sums). The LLM's
    job is transcription only. If a "Total Incurred" is printed on the
    document, we transcribe that number; we do not ask the model to add
    paid_loss + paid_expense + reserve itself, because LLMs are unreliable
    at multi-step arithmetic and a silent math error here would corrupt
    every risk metric computed later in Step 6. All arithmetic in this
    pipeline happens in plain Python, on already-validated numbers.
  - It never guesses at data that isn't in the text. Missing fields become
    null, not a plausible-looking fabrication.
  - It never accepts unvalidated output. Every response is checked against
    a strict schema before it's allowed to leave this file.

HOW CORRECTNESS IS ENFORCED (Pydantic)
---------------------------------------
  The LLM returns a plain string. Strings can say anything - the model
  could hallucinate a field, use the wrong type, or return prose instead
  of JSON. We do not trust that string on its own. We parse it as JSON and
  then run it through a Pydantic model (defined in models.py). Pydantic:
    - rejects the response outright if a required field is missing
    - rejects it if a type is wrong (e.g. a string where a number belongs)
    - rejects it if a constraint is violated (e.g. a negative dollar
      amount, since ge=0 is set on every money field in the schema)
  If validation fails, we don't silently patch around it - we send the
  exact validation error back to the model and ask it to correct itself,
  up to MAX_RETRIES times. If it still can't produce valid output, we
  report a clean failure rather than forcing bad data downstream.
=============================================================================
"""

import json
import logging

from pydantic import ValidationError

from models import ExtractedLossRun  # the Pydantic schema every response must satisfy

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

# Local model served by Ollama. "ollama/" prefix tells LiteLLM which backend
# to route the call to - swapping models later (e.g. to a larger local model,
# or even a hosted one) is a one-line change here, nowhere else in the
# pipeline needs to know or care which model is doing the extraction.
MODEL_NAME = "ollama/gpt-oss:20b"

# Ollama's default local server address. Nothing leaves the machine.
OLLAMA_API_BASE = "http://localhost:11434"

# How many times we'll let the model try to fix its own validation errors
# before we give up and report the file as failed. 3 is enough for the
# model to correct a typo/formatting slip without masking a genuinely
# unreadable document behind endless retries.
MAX_RETRIES = 3


# ---------------------------------------------------------------------------
# THE PROMPT
#
# This is the ONLY place in the codebase that tells the model what a loss
# run "means". It is intentionally generic - it describes the schema and
# the transcription rules, and nothing about any specific carrier's layout.
# That genericness is what lets one prompt handle a clean table, a
# narrative block format, a dense export, and OCR'd scan text equally well.
# ---------------------------------------------------------------------------

EXTRACTION_SYSTEM_PROMPT = """You are a data extraction assistant for a P&C insurance underwriting \
system. You will be given raw text extracted from a commercial lines loss-run report. This text may \
come from a clean table, a narrative paragraph-style report, a dense system export, or an OCR'd scan \
of a faxed document - you should not assume any particular layout; different carriers format these \
reports very differently, and your job is to find the information regardless of layout.

Your ONLY job is to transcribe the information already present in the text into the JSON schema \
below. Follow these rules strictly:

1. DO NOT calculate, infer, or estimate any numbers that are not explicitly stated in the text. If a \
   field is not present, use null (for optional fields) - never invent a plausible-looking value.
2. DO NOT compute total_incurred yourself by adding other fields - if it is printed in the document, \
   transcribe the printed value exactly, even if it looks like it doesn't perfectly add up (that \
   discrepancy itself may be meaningful and should NOT be silently corrected by you).
3. Dates should be normalized to MM/DD/YYYY where the source format allows it to be determined \
   unambiguously; if a date is genuinely unclear, use null rather than guessing.
4. If the same conceptual field appears under different labels in different documents (e.g. "Paid \
   Indemnity" vs "Paid Loss" vs "PdLoss" all mean the same thing), map it to the correct schema field \
   regardless of the label used in the source text.
5. Respond with ONLY a single JSON object matching the schema below - no markdown code fences, no \
   explanation, no commentary before or after the JSON.

SCHEMA:
{
  "source_file": "string - the filename provided to you, copy it exactly",
  "carrier_name": "string or null - the insurance carrier that issued this report",
  "insured_name": "string - the named insured on the policy",
  "policy_number": "string or null",
  "policy_start": "MM/DD/YYYY or null - policy term start date",
  "policy_end": "MM/DD/YYYY or null - policy term end date",
  "annual_premium": "number or null - annual written premium in dollars",
  "per_occurrence_limit": "number or null - per-occurrence coverage limit in dollars",
  "aggregate_limit": "number or null - aggregate coverage limit in dollars",
  "deductible": "number or null - per-claim deductible in dollars",
  "claims": [
    {
      "claim_number": "string - the claim ID as printed",
      "loss_date": "MM/DD/YYYY - date the loss occurred",
      "report_date": "MM/DD/YYYY or null - date the claim was reported to the carrier",
      "close_date": "MM/DD/YYYY or null - date the claim closed, null if still open",
      "status": "one of: Open, Closed, Reopened",
      "cause_of_loss": "string - description of what caused the loss",
      "state": "2-letter US state code or null",
      "paid_loss": "number - paid indemnity/loss amount in dollars, 0 if none",
      "paid_expense": "number - paid expense/ALAE amount in dollars, 0 if none",
      "reserve": "number - outstanding reserve amount in dollars, 0 if none/closed",
      "total_incurred": "number - total incurred as printed in the document",
      "cat_indicator": "true or false - whether this is flagged as a catastrophe loss",
      "subrogation": "number - subrogation recovery amount in dollars, 0 if none"
    }
  ]
}"""


# ---------------------------------------------------------------------------
# INTERNAL HELPER 1: the actual network call to the local model
# ---------------------------------------------------------------------------

def _call_llm(raw_text: str, source_file: str, retry_context: str | None = None) -> str:
    """
    Sends one request to the local LLM and returns its raw text response.

    Parameters
    ----------
    raw_text : the extracted document text (from Step 2/3 - text-PDF path
               or OCR path, it makes no difference to this function, which
               is exactly the point: this function has no idea, and does
               not need to know, where raw_text came from).
    source_file : the original PDF's filename, passed through so the model
                  can echo it back into the "source_file" field, keeping
                  the output traceable to its source document.
    retry_context : if this is a retry after a failed validation attempt,
                    this carries the exact Pydantic error message, so the
                    model can see specifically what it got wrong and fix
                    just that, instead of guessing blindly on a second try.
    """
    # Imported here (rather than at module top) so that environments without
    # litellm installed can still import this module for its schema/prompt
    # constants without crashing - only the actual call requires the package.
    from litellm import completion

    user_prompt = f"Source filename: {source_file}\n\nDocument text:\n{raw_text}"

    if retry_context:
        user_prompt += (
            f"\n\nYOUR PREVIOUS RESPONSE FAILED VALIDATION with this error:\n{retry_context}\n"
            "Please fix the issue and respond again with ONLY the corrected JSON object."
        )

    response = completion(
        model=MODEL_NAME,
        api_base=OLLAMA_API_BASE,
        messages=[
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        # temperature=0 because this is a transcription task, not a creative
        # one - we want the same document to produce the same extraction
        # every time it's run, not stochastic variation.
        temperature=0,
    )
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# INTERNAL HELPER 2: defensive cleanup of the model's raw text response
# ---------------------------------------------------------------------------

def _strip_json_fences(text: str) -> str:
    """
    Even though the prompt explicitly says "no markdown code fences", local
    models sometimes wrap their JSON in ```json ... ``` anyway out of habit
    from their training data. Rather than let that break json.loads(), we
    defensively strip a leading/trailing fence if present. This is a
    tolerance measure, not a substitute for the schema validation that
    follows - malformed content inside the fences still gets caught below.
    """
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        # drop the opening fence line
        lines = lines[1:] if lines[0].startswith("```") else lines
        # drop the closing fence line, if present
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines)
    return text.strip()


# ---------------------------------------------------------------------------
# THE PUBLIC FUNCTION - this is the ONE function the rest of the pipeline
# calls. It is used identically for the clean-table PDF, the narrative-block
# PDF, the dense-export PDF, and the OCR'd scanned PDF. Nothing about its
# signature or internal logic changes based on the source format - the LLM
# behind _call_llm() is what absorbs that variability, not this function.
# ---------------------------------------------------------------------------

def extract_structured(raw_text: str, source_file: str) -> tuple[ExtractedLossRun | None, str | None]:
    """
    Runs the extraction agent on ONE document's raw text and returns a
    validated ExtractedLossRun, with automatic retry on validation failure.

    This function is deliberately format-agnostic: it does not branch on
    source_file, does not inspect raw_text for carrier-specific markers,
    and has no per-carrier code paths anywhere. Every document, regardless
    of its original layout, goes through exactly this same sequence:

        raw text  ->  LLM call  ->  strip fences  ->  JSON parse
                  ->  Pydantic validation  ->  (retry on failure)  ->  result

    Parameters
    ----------
    raw_text : the document's extracted text, from either the pdfplumber
               text path or the Tesseract/vision-model OCR path (Step 2/3).
    source_file : original PDF filename, for traceability and error
                  messages.

    Returns
    -------
    (result, error) tuple:
        - On success: (ExtractedLossRun instance, None)
        - On permanent failure after MAX_RETRIES: (None, error message)
          - this is a clean, explicit failure, never a silently-dropped
            file and never a best-effort/partial object passed downstream.
    """
    last_error: str | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        # --- Step A: call the model -----------------------------------
        try:
            raw_response = _call_llm(raw_text, source_file, retry_context=last_error)
        except Exception as e:
            # Covers connection errors (Ollama not running), timeouts,
            # malformed API responses, etc. We log and retry rather than
            # crash the whole pipeline over one bad file.
            last_error = f"LLM call failed: {e}"
            logger.warning(
                "Extraction attempt %d/%d for %s: %s", attempt, MAX_RETRIES, source_file, last_error
            )
            continue

        # --- Step B: defensive cleanup ----------------------------------
        cleaned = _strip_json_fences(raw_response)

        # --- Step C: parse as JSON ---------------------------------------
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as e:
            last_error = f"Response was not valid JSON: {e}"
            logger.warning(
                "Extraction attempt %d/%d for %s: %s", attempt, MAX_RETRIES, source_file, last_error
            )
            continue

        # Make sure the source_file we already know is authoritative,
        # in case the model echoed it back slightly wrong.
        parsed["source_file"] = source_file

        # --- Step D: validate against the Pydantic schema ----------------
        try:
            result = ExtractedLossRun(**parsed)
        except ValidationError as e:
            last_error = f"Schema validation failed: {e}"
            logger.warning(
                "Extraction attempt %d/%d for %s: %s", attempt, MAX_RETRIES, source_file, last_error
            )
            continue

        # --- Success ------------------------------------------------------
        logger.info(
            "Extraction succeeded for %s on attempt %d/%d (%d claim(s) found)",
            source_file, attempt, MAX_RETRIES, len(result.claims),
        )
        return result, None

    # All retries exhausted - report failure honestly rather than guessing.
    logger.error(
        "Extraction FAILED for %s after %d attempts. Last error: %s",
        source_file, MAX_RETRIES, last_error,
    )
    return None, last_error
