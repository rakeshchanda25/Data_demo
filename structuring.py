"""
structuring.py

NEXT STEP after processing.py.

Your processing.py already did the hard part: PDF -> text/OCR -> saved as
.md files in Data/Processed/. This script picks up exactly there.

What this does, in order:
    1. Read every .md file in Data/Processed/
    2. Send its text to the local LLM (via Ollama) with instructions to
       return structured JSON matching our schema
    3. Validate that JSON with Pydantic - if it's malformed or missing
       fields, ask the model to fix it and retry (up to 3 times)
    4. Save each file's structured JSON, plus one combined JSON with
       everything

This file is self-contained - no imports from any other file you've
written. Just standard libraries + pydantic + ollama's python client.
"""

import os
import json

from pydantic import BaseModel, ValidationError
import ollama


# =============================================================================
# PART 1 - THE SCHEMA (what shape we want the JSON in)
# =============================================================================
# One claim = one row in the loss run. One LossRun = one whole document
# (policy info + a list of its claims).
# =============================================================================

class Claim(BaseModel):
    claim_number: str
    loss_date: str                    # MM/DD/YYYY
    status: str                        # "Open", "Closed", or "Reopened"
    cause_of_loss: str
    state: str | None = None
    paid_loss: float = 0.0
    paid_expense: float = 0.0
    reserve: float = 0.0
    total_incurred: float = 0.0
    cat_indicator: bool = False
    subrogation: float = 0.0


class LossRun(BaseModel):
    source_file: str
    insured_name: str
    carrier_name: str | None = None
    policy_number: str | None = None
    policy_start: str | None = None
    policy_end: str | None = None
    annual_premium: float | None = None
    per_occurrence_limit: float | None = None
    aggregate_limit: float | None = None
    deductible: float | None = None
    claims: list[Claim] = []


# =============================================================================
# PART 2 - CONFIG
# =============================================================================

PROCESSED_PATH = "/home/rchanda/TestFolder/POCs_Folder/Training_POC_Insurance/Data/Processed"
STRUCTURED_PATH = "/home/rchanda/TestFolder/POCs_Folder/Training_POC_Insurance/Data/Structured"

MODEL_NAME = "gpt-oss:20b"   # change this to whatever model you've pulled in Ollama
MAX_RETRIES = 3

EXTRACTION_PROMPT = """You are reading a commercial insurance loss-run report (already \
converted to text/markdown). Extract the information into this JSON shape, and respond \
with ONLY the JSON, nothing else - no markdown fences, no explanation:

{
  "source_file": "string",
  "insured_name": "string",
  "carrier_name": "string or null",
  "policy_number": "string or null",
  "policy_start": "MM/DD/YYYY or null",
  "policy_end": "MM/DD/YYYY or null",
  "annual_premium": number or null,
  "per_occurrence_limit": number or null,
  "aggregate_limit": number or null,
  "deductible": number or null,
  "claims": [
    {
      "claim_number": "string",
      "loss_date": "MM/DD/YYYY",
      "status": "Open, Closed, or Reopened",
      "cause_of_loss": "string",
      "state": "2-letter code or null",
      "paid_loss": number,
      "paid_expense": number,
      "reserve": number,
      "total_incurred": number,
      "cat_indicator": true or false,
      "subrogation": number
    }
  ]
}

Rules:
- Only use information that is actually present in the text below.
- If a field isn't in the text, use null (or 0 for a required number if it's clearly zero).
- Never calculate totals yourself - copy the numbers exactly as printed in the document.
- This document could be in any format (table, narrative, dense export, OCR'd scan) -
  read it carefully regardless of layout and map it to the schema above."""


# =============================================================================
# PART 3 - THE ONE COMMON EXTRACTION FUNCTION
# =============================================================================
# This function is called identically for every .md file, no matter which
# original PDF or carrier it came from. It doesn't inspect the filename or
# branch on format - the LLM behind it handles that variability.
# =============================================================================

def extract_structured(raw_text: str, source_file: str) -> tuple[LossRun | None, str | None]:
    """
    Sends raw_text to the local LLM and validates the response against the
    LossRun schema. Retries up to MAX_RETRIES times if validation fails,
    feeding the exact error back to the model each time.

    Returns (result, error):
        - success: (LossRun instance, None)
        - failure after all retries: (None, error message) - never a guess
    """
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        user_message = f"Source file: {source_file}\n\nDocument text:\n{raw_text}"
        if last_error:
            user_message += (
                f"\n\nYour previous answer failed validation with this error:\n{last_error}\n"
                "Please fix it and respond again with ONLY the corrected JSON."
            )

        try:
            response = ollama.chat(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": EXTRACTION_PROMPT},
                    {"role": "user", "content": user_message},
                ],
            )
            answer = response["message"]["content"].strip()

            # Models sometimes wrap the JSON in ```json fences despite instructions - strip it
            if answer.startswith("```"):
                lines = answer.split("\n")
                answer = "\n".join(lines[1:-1]) if lines[-1].strip().startswith("```") else "\n".join(lines[1:])

            data = json.loads(answer)
            data["source_file"] = source_file   # keep this authoritative, don't trust model's echo

            result = LossRun(**data)             # <- Pydantic validation happens here
            return result, None

        except json.JSONDecodeError as e:
            last_error = f"Response was not valid JSON: {e}"
        except ValidationError as e:
            last_error = f"Schema validation failed: {e}"
        except Exception as e:
            last_error = f"LLM call failed: {e}"

        print(f"  Attempt {attempt}/{MAX_RETRIES} failed: {last_error}")

    return None, last_error


# =============================================================================
# PART 4 - RUN IT ACROSS ALL .md FILES
# =============================================================================

def structure_files():
    os.makedirs(STRUCTURED_PATH, exist_ok=True)

    md_files = [name for name in os.listdir(PROCESSED_PATH) if name.endswith(".md")]
    print(f"Found {len(md_files)} .md file(s) in {PROCESSED_PATH}\n")

    all_results = []
    failures = []

    for md_file in md_files:
        md_path = os.path.join(PROCESSED_PATH, md_file)
        print(f"Structuring {md_file}...")

        with open(md_path, "r", encoding="utf-8") as f:
            raw_text = f.read()

        # THE ONE COMMON CALL - same function, same shape, every file
        result, error = extract_structured(raw_text, source_file=md_file)

        if result:
            all_results.append(result.model_dump())

            out_name = md_file.replace(".md", ".json")
            out_path = os.path.join(STRUCTURED_PATH, out_name)
            with open(out_path, "w") as f:
                json.dump(result.model_dump(), f, indent=2)
            print(f"  -> {len(result.claims)} claim(s) extracted, saved to {out_name}\n")
        else:
            failures.append({"file": md_file, "error": error})
            print(f"  -> FAILED after {MAX_RETRIES} attempts: {error}\n")

    combined_path = os.path.join(STRUCTURED_PATH, "structured_all_files.json")
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"Done. {len(all_results)}/{len(md_files)} files structured successfully.")
    if failures:
        print("\n⚠ FAILED FILES (excluded from output, not silently dropped):")
        for fail in failures:
            print(f"  - {fail['file']}: {fail['error']}")
    print(f"\nCombined output saved to: {combined_path}")


if __name__ == "__main__":
    structure_files()
