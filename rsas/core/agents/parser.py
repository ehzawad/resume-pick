"""Parser Agent - extracts and normalizes resume structure from PDFs."""

import os
from pathlib import Path
from typing import Type

from pydantic import Field

from ..models.base import AgentContext, AgentResult, RSASBaseModel
from ..models.candidate_profile import ParsedResume
from ..models.enums import AgentType
from ...observability.logger import get_logger
from .base import BaseAgent

logger = get_logger(__name__)


class ResumeParseInput(RSASBaseModel):
    """Input for Parser Agent."""

    candidate_id: str = Field(..., description="Candidate identifier")
    job_id: str = Field(..., description="Job identifier")
    resume_path: str = Field(..., description="Path to resume PDF")
    content_hash: str | None = Field(None, description="SHA-256 of the resume file for idempotency")


class ParserAgent(BaseAgent[ResumeParseInput, ParsedResume]):
    """Agent that parses PDF resumes into structured format."""

    @property
    def agent_type(self) -> AgentType:
        return AgentType.PARSER

    @property
    def output_schema(self) -> Type[ParsedResume]:
        return ParsedResume

    def _hash_input(self, input_data: ResumeParseInput) -> str:
        """Hash input using content hash (or fallback to path) for idempotency across location moves."""
        import hashlib
        key = f"{input_data.job_id}:{input_data.candidate_id}:{input_data.content_hash or input_data.resume_path}"
        return hashlib.sha256(key.encode()).hexdigest()

    async def process(
        self, input_data: ResumeParseInput, context: AgentContext
    ) -> AgentResult[ParsedResume]:
        """Parse resume PDF and extract structured information.

        Args:
            input_data: Resume parse input
            context: Execution context

        Returns:
            AgentResult with ParsedResume
        """
        if os.getenv("RSAS_TEST_MODE"):
            dummy = ParsedResume(
                candidate_id=input_data.candidate_id,
                job_id=input_data.job_id,
                raw_text="mock resume text",
                explicit_skills=["python", "sql"],
                experience=[],
                education=[],
                parse_confidence=1.0,
            )
            return AgentResult(success=True, data=dummy, confidence=1.0, metadata={"mock": True})

        # Extract text from PDF
        raw_text, pdf_library = await self._extract_pdf_text(input_data.resume_path)

        if not raw_text or len(raw_text.strip()) < 50:
            logger.warning(
                "pdf_extraction_minimal",
                candidate_id=input_data.candidate_id,
                text_length=len(raw_text) if raw_text else 0,
            )
            return AgentResult(
                success=False,
                data=None,
                error="PDF extraction failed or produced minimal text",
                confidence=0.0,
            )

        # Build prompt with extracted text
        prompt = self._build_prompt_with_text(raw_text, input_data, context)

        # Call Response API
        output, metadata = await self._call_response_api(prompt, context)

        # Set required fields
        output.candidate_id = input_data.candidate_id
        output.job_id = input_data.job_id
        output.raw_text = raw_text
        output.pdf_library = pdf_library

        return AgentResult(
            success=True,
            data=output,
            confidence=output.parse_confidence,
            tokens_used=metadata.get("tokens_total", 0),
            metadata=metadata,
        )

    def _build_prompt(self, input_data: ResumeParseInput, context: AgentContext) -> str:
        """Build prompt (not used directly - use _build_prompt_with_text)."""
        # This won't be called directly; we extract text first
        return ""

    def _build_prompt_with_text(
        self, raw_text: str, input_data: ResumeParseInput, context: AgentContext
    ) -> str:
        """Build prompt with extracted text.

        Args:
            raw_text: Extracted resume text
            input_data: Input data
            context: Execution context

        Returns:
            Prompt string
        """
        return f"""Parse this resume text into a structured format with high precision.

RESUME TEXT:
{raw_text}

Extract the following sections:

1. **Contact Information**:
   - Full name
   - Email address
   - Phone number
   - Location (city, state/country)
   - LinkedIn URL
   - GitHub URL
   - Portfolio/website URL

2. **Education History** (chronological, most recent first):
   - For each education entry:
     - Degree type (Bachelor, Master, PhD, etc.)
     - Field of study
     - Institution name
     - Start and end dates (or "Present" if current)
     - Honors, awards, GPA if mentioned

3. **Work Experience** (chronological, most recent first):
   - For each job:
     - Job title
     - Company name
     - Location
     - Start and end dates (or "Present" if current)
     - Whether currently employed
     - Bullet points of responsibilities and achievements

4. **Skills** (as explicitly listed in a "Skills" section):
   - Technical skills
   - Tools and technologies
   - Languages
   - Certifications

5. **Projects** (if any):
   - Project name, description, technologies used

6. **Publications** (if any):
   - Title, venue, year

7. **Certifications and Licenses**:
   - Name, issuing organization, date

**Parsing Quality**:
- Assign a confidence score (0.0-1.0) for the overall parsing quality
- Note any missing sections (e.g., "no education section found")
- Normalize dates to ISO format (YYYY-MM-DD or YYYY-MM)
- Preserve original text where ambiguous

Be precise and extract only what is clearly present in the text."""

    async def _extract_pdf_text(self, resume_path: str) -> tuple[str, str]:
        """Extract text from PDF using pdfplumber (primary) or PyPDF2 (fallback).

        Args:
            resume_path: Path to PDF file

        Returns:
            Tuple of (extracted text, library used)
        """
        path = Path(resume_path)

        if not path.exists():
            logger.error("pdf_not_found", resume_path=resume_path)
            return "", "error"

        # Try pdfplumber first
        try:
            import pdfplumber

            with pdfplumber.open(path) as pdf:
                text_parts = []
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)

                text = "\n\n".join(text_parts)

                if text and len(text.strip()) > 50:
                    logger.info(
                        "pdf_extracted_pdfplumber",
                        resume_path=resume_path,
                        text_length=len(text),
                        pages=len(pdf.pages),
                    )
                    return text, "pdfplumber"

        except Exception as e:
            logger.warning(
                "pdfplumber_failed",
                resume_path=resume_path,
                error=str(e),
            )

        # Fallback to PyPDF2
        try:
            import PyPDF2

            with open(path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                text_parts = []

                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)

                text = "\n\n".join(text_parts)

                logger.info(
                    "pdf_extracted_pypdf2",
                    resume_path=resume_path,
                    text_length=len(text),
                    pages=len(reader.pages),
                )
                return text, "pypdf2"

        except Exception as e:
            logger.error(
                "pypdf2_failed",
                resume_path=resume_path,
                error=str(e),
                exc_info=True,
            )
            return "", "error"
