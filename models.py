"""
models.py
Core data structures used across the pipeline. Keeping these in one place
so every stage (ingestion, extraction, normalization, scoring, memo)
agrees on the same shape of data.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class DocType(str, Enum):
    TEXT_PDF = "text_pdf"       # normal PDF, text is extractable directly
    SCANNED_PDF = "scanned_pdf" # image-only PDF, needs OCR


@dataclass
class IngestedFile:
    """One raw loss-run PDF that has been loaded and classified."""
    file_path: Path
    file_name: str
    doc_type: DocType
    page_count: int
    extractable_char_count: int   # how many text chars pdfplumber found
    # populated by later stages (extraction step) - kept here so the
    # whole record travels together through the pipeline
    raw_text: Optional[str] = None
    extraction_method: Optional[str] = None   # "pdfplumber" | "vision_model" | "tesseract_layout"
    is_degraded: bool = False                  # True = NOT verified by vision model, treat with caution
    extraction_error: Optional[str] = None      # reason for degradation, if any


@dataclass
class Submission:
    """A single underwriting submission = one insured, N loss-run PDFs
    (possibly from different prior carriers, in different formats)."""
    submission_id: str
    source_dir: Path
    files: list[IngestedFile] = field(default_factory=list)

    def summary(self) -> str:
        lines = [f"Submission: {self.submission_id}  ({len(self.files)} file(s))"]
        for f in self.files:
            flag = "  ⚠ UNVERIFIED OCR" if f.is_degraded else ""
            lines.append(
                f"  - {f.file_name:45s} type={f.doc_type.value:12s} "
                f"pages={f.page_count:<3d} chars={f.extractable_char_count}{flag}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Step 3: LLM extraction output schema
#
# These are Pydantic models (not dataclasses) because they're the contract
# the LLM's JSON output gets validated against. If the model returns
# something that doesn't fit this shape - wrong type, missing required
# field, malformed number - Pydantic raises a ValidationError and the
# extraction agent retries, rather than letting bad data flow downstream
# silently. The LLM's ONLY job is populating these fields by reading the
# document; no field here is ever computed/derived by the LLM (loss ratios,
# totals, etc. are Step 5's job, done in pure Python).
# ---------------------------------------------------------------------------

class ClaimStatus(str, Enum):
    OPEN = "Open"
    CLOSED = "Closed"
    REOPENED = "Reopened"


class ExtractedClaim(BaseModel):
    """One claim line-item as read directly off a loss-run document."""
    claim_number: str = Field(..., description="The claim ID/number as printed on the document")
    loss_date: str = Field(..., description="Date of loss, format MM/DD/YYYY")
    report_date: Optional[str] = Field(None, description="Date claim was reported, MM/DD/YYYY")
    close_date: Optional[str] = Field(None, description="Date claim closed, MM/DD/YYYY, or null if still open")
    status: ClaimStatus
    cause_of_loss: str = Field(..., description="Cause/description of the loss as printed")
    state: Optional[str] = Field(None, description="2-letter state code where loss occurred")
    paid_loss: float = Field(..., ge=0, description="Paid loss amount in dollars")
    paid_expense: float = Field(0.0, ge=0, description="Paid expense/ALAE amount in dollars")
    reserve: float = Field(0.0, ge=0, description="Outstanding reserve amount in dollars")
    total_incurred: float = Field(..., ge=0, description="Total incurred = paid_loss + paid_expense + reserve")
    cat_indicator: bool = Field(False, description="True if flagged as a catastrophe loss")
    subrogation: float = Field(0.0, ge=0, description="Subrogation recovery amount in dollars, 0 if none")

    @field_validator("state")
    @classmethod
    def uppercase_state(cls, v):
        return v.upper() if v else v


class ExtractedLossRun(BaseModel):
    """The full structured extraction for ONE source document (one prior carrier/year)."""
    source_file: str = Field(..., description="Original PDF filename this was extracted from")
    carrier_name: Optional[str] = Field(None, description="Name of the carrier that issued this loss run")
    insured_name: str = Field(..., description="Named insured on the policy")
    policy_number: Optional[str] = Field(None, description="Policy number as printed")
    policy_start: Optional[str] = Field(None, description="Policy term start date, MM/DD/YYYY")
    policy_end: Optional[str] = Field(None, description="Policy term end date, MM/DD/YYYY")
    annual_premium: Optional[float] = Field(None, ge=0, description="Annual written premium in dollars")
    per_occurrence_limit: Optional[float] = Field(None, ge=0, description="Per-occurrence limit in dollars")
    aggregate_limit: Optional[float] = Field(None, ge=0, description="Aggregate limit in dollars")
    deductible: Optional[float] = Field(None, ge=0, description="Per-claim deductible in dollars")
    claims: list[ExtractedClaim] = Field(default_factory=list)
