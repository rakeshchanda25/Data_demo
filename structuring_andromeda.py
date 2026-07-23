"""
structuring.py

NEXT STEP after processing.py.

Your processing.py already did the hard part: PDF -> text/OCR -> saved as
.md files in Data/Processed/. This script picks up exactly there.

What this does, in order:
    1. Read every .md file in Data/Processed/
    2. Send its text to the local LLM (via Andromeda's Agent, backed by
       Ollama) with instructions to return structured JSON matching our
       schema
    3. Validate that JSON with Pydantic - if it's malformed or missing
       fields, ask the model to fix it and retry (up to 3 times)
    4. Save each file's structured JSON, plus one combined JSON with
       everything

Orchestration note: the direct `ollama.chat(...)` call is replaced with an
Andromeda `Agent` (same `Agent`/`AgentConfig`/`ModelConfig` used by
scoring.py/memo.py's conversions), and the whole per-file loop is wired
through a one-node WorkflowBuilder pipeline, same shape as memo.py. Every
other line -- the schema, the prompt, the manual JSON-fence stripping, the
retry loop, the file I/O -- is unchanged.
"""

import os
import json

from pydantic import BaseModel, ValidationError, field_validator, Field

from andromeda import HumanMessage
from andromeda.config import AgentConfig, ModelConfig
from andromeda.core.agent import Agent
from andromeda.core.workflow import WorkflowBuilder


# =============================================================================
# PART 1 - THE SCHEMA (what shape we want the JSON in)
# =============================================================================
# One claim = one row in the loss run. One LossRun = one whole document
# (policy info + a list of its claims).
# =============================================================================
def clean_number(value):
    """
    Convert formatted monetary strings into numbers.

    Examples:
      "48,500.00" -> 48500.00
      "$1,000,000.00" -> 1000000.00
      None -> 0.0
      "" -> 0.0
    """
    if value is None:
        return 0.0

    if isinstance(value, str):
        value = value.replace(",", "").replace("$", "").strip()

        if value == "":
            return 0.0

    return value


def clean_string(value):
    """
    Normalize missing text fields.
    """
    if value is None:
        return ""

    return str(value).strip()


class Claim(BaseModel):
    claim_number: str = ""
    loss_date: str = ""
    status: str = ""
    cause_of_loss: str = ""
    state: str | None = None

    paid_loss: float = 0.0
    paid_expense: float = 0.0
    reserve: float = 0.0
    total_incurred: float = 0.0
    cat_indicator: bool = False
    subrogation: float = 0.0


    @field_validator(
        "paid_loss",
        "paid_expense",
        "reserve",
        "total_incurred",
        "subrogation",
        mode="before"
    )
    @classmethod
    def clean_claim_amounts(cls, value):
        return clean_number(value)


    @field_validator(
        "claim_number",
        "loss_date",
        "status",
        "cause_of_loss",
        mode="before"
    )
    @classmethod
    def clean_claim_text(cls, value):
        return clean_string(value)


    @field_validator("cat_indicator", mode="before")
    @classmethod
    def clean_cat_indicator(cls, value):

        if value is None:
            return False

        if isinstance(value, bool):
            return value

        if isinstance(value, str):
            value = value.strip().lower()

            if value in ["yes", "y", "true", "t", "1"]:
                return True

            if value in ["no", "n", "false", "f", "0", ""]:
                return False

        return False


class LossRun(BaseModel):
    source_file: str = ""
    insured_name: str = ""
    carrier_name: str | None = None
    policy_number: str | None = None
    policy_start: str | None = None
    policy_end: str | None = None

    annual_premium: float | None = None
    per_occurrence_limit: float | None = None
    aggregate_limit: float | None = None
    deductible: float | None = None

    claims: list[Claim] = Field(default_factory=list)


    @field_validator(
        "annual_premium",
        "per_occurrence_limit",
        "aggregate_limit",
        "deductible",
        mode="before"
    )
    @classmethod
    def clean_policy_amounts(cls, value):
        if value is None:
            return None

        return clean_number(value)


    @field_validator(
        "source_file",
        "insured_name",
        mode="before"
    )
    @classmethod
    def clean_header_text(cls, value):
        return clean_string(value)


# =============================================================================
# PART 2 - CONFIG
# =============================================================================

PROCESSED_PATH = "/home/rakeshchanda/TrainingFolder/POC_Insurance/Data/Processed"
STRUCTURED_PATH = "/home/rakeshchanda/TrainingFolder/POC_Insurance/Data/Structured"

MODEL_NAME = "llama3.2:3b"   # change this to whatever model you've pulled in Ollama
MAX_RETRIES = 1

EXTRACTION_PROMPT = """You are reading a commercial insurance loss-run report (already \
converted to text/markdown). Extract the information into this JSON shape, and respond \
with ONLY the JSON, nothing else - no markdown fences, no explanation.

IMPORTANT:
- Respond with ONLY valid JSON.
- Do NOT include markdown fences.
- Do NOT include explanations or comments.
- Do NOT add fields that are not in the schema.
- Preserve numbers exactly as printed in the document.
- Do not calculate or infer values.

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

EXTRACTION RULES:

1. GENERAL RULES
- Extract only information explicitly present in the document.
- Do not guess missing information.
- Do not create claims that are not present.
- Keep the original values from the document.

2. TEXT FIELDS
- If a required text field is missing, return an empty string "".
- Do not return null for:
  - claim_number
  - loss_date
  - status
  - cause_of_loss

3. NUMERIC FIELDS
- All monetary fields must be returned as numbers.
- Remove commas and currency symbols.

Examples:
Correct:
48500.00

Incorrect:
"48,500.00"
"$48,500.00"

- If a numeric value is missing or blank in the document, return 0.
- Do not return null for numeric claim fields.

4. BOOLEAN FIELDS
- cat_indicator must always be true or false.
- If the document indicates catastrophe/CAT, return true.
- If not mentioned or unclear, return false.
- Never return null.

5. CLAIM STATUS
Use only:
- Open
- Closed
- Reopened

If status is missing, return an empty string "".

6. DATES
- Preserve dates exactly in MM/DD/YYYY format when available.
- If a claim loss date is missing, return an empty string "".

7. FINANCIAL VALUES
- Never calculate totals.
- Never add paid loss + expense + reserve.
- Copy total_incurred exactly as printed.
- Preserve decimal precision.

8. TABLE HANDLING
The document may contain:
- tables
- exported reports
- OCR text
- scanned layouts
- multi-page claim listings

Read all sections carefully and map rows to claims correctly.

9. OUTPUT VALIDATION
Before responding:
- Ensure the output is valid JSON.
- Ensure all numeric fields contain numbers only.
- Ensure all boolean fields contain true or false only.
- Ensure no claim field contains null.

"""


def build_extractor_agent() -> Agent:
    return Agent(
        AgentConfig(
            name="loss_run_structurer",
            model=ModelConfig(name=MODEL_NAME, provider="ollama"),
            prompt=EXTRACTION_PROMPT,
        )
    )


# =============================================================================
# PART 3 - THE ONE COMMON EXTRACTION FUNCTION
# =============================================================================
# This function is called identically for every .md file, no matter which
# original PDF or carrier it came from. It doesn't inspect the filename or
# branch on format - the LLM behind it handles that variability.
# =============================================================================

def extract_structured(
    raw_text: str, source_file: str, agent: Agent | None = None
) -> tuple[LossRun | None, str | None]:
    """
    Sends raw_text to the local LLM (via an Andromeda Agent) and validates
    the response against the LossRun schema. Retries up to MAX_RETRIES
    times if validation fails, feeding the exact error back to the model
    each time.

    Returns (result, error):
        - success: (LossRun instance, None)
        - failure after all retries: (None, error message) - never a guess
    """
    agent = agent or build_extractor_agent()
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        user_message = f"Source file: {source_file}\n\nDocument text:\n{raw_text}"
        if last_error:
            user_message += (
                f"\n\nYour previous answer failed validation with this error:\n{last_error}\n"
                "Please fix it and respond again with ONLY the corrected JSON."
            )

        try:
            reply = agent.invoke([HumanMessage(content=user_message)])
            answer = reply[-1].content.strip()

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

def step_structure_all(state: dict) -> dict:
    processed_path = state["processed_path"]
    structured_path = state["structured_path"]
    agent = state.get("agent") or build_extractor_agent()

    os.makedirs(structured_path, exist_ok=True)

    md_files = [name for name in os.listdir(processed_path) if name.endswith(".md")]
    print(f"Found {len(md_files)} .md file(s) in {processed_path}\n")

    all_results = []
    failures = []

    for md_file in md_files:
        md_path = os.path.join(processed_path, md_file)
        print(f"Structuring {md_file}...")

        with open(md_path, "r", encoding="utf-8") as f:
            raw_text = f.read()

        # THE ONE COMMON CALL - same function, same shape, every file
        result, error = extract_structured(raw_text, source_file=md_file, agent=agent)

        if result:
            all_results.append(result.model_dump())

            out_name = md_file.replace(".md", ".json")
            out_path = os.path.join(structured_path, out_name)
            with open(out_path, "w") as f:
                json.dump(result.model_dump(), f, indent=2)
            print(f"  -> {len(result.claims)} claim(s) extracted, saved to {out_name}\n")
        else:
            failures.append({"file": md_file, "error": error})
            print(f"  -> FAILED after {MAX_RETRIES} attempts: {error}\n")

    combined_path = os.path.join(structured_path, "structured_all_files.json")
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"Done. {len(all_results)}/{len(md_files)} files structured successfully.")
    if failures:
        print("\n⚠ FAILED FILES (excluded from output, not silently dropped):")
        for fail in failures:
            print(f"  - {fail['file']}: {fail['error']}")
    print(f"\nCombined output saved to: {combined_path}")

    return {"structured_results": all_results, "failures": failures, "combined_path": combined_path}


def build_structuring_pipeline() -> WorkflowBuilder:
    pipeline = WorkflowBuilder(name="StructuringPipeline")
    pipeline.start("structure").run(step_structure_all)
    return pipeline


def structure_files():
    pipeline = build_structuring_pipeline()
    pipeline.execute(
        state={"processed_path": PROCESSED_PATH, "structured_path": STRUCTURED_PATH}
    )


if __name__ == "__main__":
    structure_files()
