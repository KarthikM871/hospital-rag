from pathlib import Path

import pdfplumber
from pydantic import BaseModel


class ParsedPage(BaseModel):
    page_number: int
    text: str


def parse_pdf(path: Path) -> list[ParsedPage]:
    pages: list[ParsedPage] = []
    with pdfplumber.open(path) as pdf:
        for number, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            pages.append(ParsedPage(page_number=number, text=text))
    return pages