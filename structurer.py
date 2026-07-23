"""
=============================================================================
 structuring/structurer.py

 ORCHESTRATOR: applies the ONE common extraction function to every file in
 a submission, with no per-file or per-format special-casing anywhere.
=============================================================================

Notice what this file does NOT contain: there is no `if carrier == "Meridian"`,
no filename check, no format detection. Every IngestedFile in the submission -
whether it came from the clean-table PDF, the narrative-block PDF, the
dense-export PDF, or the OCR'd scan - is handed to the exact same
extract_structured() call. The loop below is the entire orchestration logic;
all the intelligence that lets this work across wildly different document
layouts lives inside the LLM prompt in llm_extractor.py, not here.

This is deliberate: it's the proof that the pipeline scales to a new prior
carrier with a completely different report template without a single code
change in this file or in llm_extractor.py. Only the mock-data generator or
a real-world new PDF would differ - the extraction code stays the same.
"""

import logging

from models import Submission, ExtractedLossRun
from structuring.llm_extractor import extract_structured

logger = logging.getLogger(__name__)


def structure_submission(submission: Submission) -> tuple[list[ExtractedLossRun], list[dict]]:
    """
    Runs Step 3/4 across every file in the submission.

    Parameters
    ----------
    submission : the Submission object produced by Steps 1-2, with
                 raw_text already populated on every IngestedFile by
                 Step 2's extraction (text-PDF or OCR path - irrelevant
                 here, both paths produce the same raw_text field).

    Returns
    -------
    (results, failures):
        results  - list of successfully validated ExtractedLossRun objects,
                   one per file that extracted cleanly. Ready to hand
                   straight to Step 5 (normalization/merge).
        failures - list of {"file_name": ..., "error": ...} dicts for any
                   file that could not be structured after retries. These
                   are surfaced explicitly (see triage.py's warning banner)
                   rather than silently excluded with no trace.
    """
    results: list[ExtractedLossRun] = []
    failures: list[dict] = []

    # --- THE ENTIRE ORCHESTRATION LOGIC IS THIS LOOP ---
    # Every file, regardless of its original PDF format, takes exactly the
    # same path through this function. There is nothing here that inspects
    # which carrier produced the file or how it was formatted.
    for f in submission.files:

        # A file with no extracted text (e.g. Step 2 totally failed on it)
        # can't be sent to the LLM at all - report it honestly rather than
        # sending an empty prompt and getting a nonsense/hallucinated result.
        if not f.raw_text or not f.raw_text.strip():
            failures.append({"file_name": f.file_name, "error": "no extracted text available"})
            continue

        # THE ONE CALL. Same function, same arguments shape, every file.
        result, error = extract_structured(raw_text=f.raw_text, source_file=f.file_name)

        if result:
            results.append(result)
        else:
            failures.append({"file_name": f.file_name, "error": error})

    return results, failures
