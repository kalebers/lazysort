from __future__ import annotations

from dataclasses import dataclass
from typing import Any


SUPPORTED_DOCUMENT_TYPES = {"pdf", "txt", "md"}
LOW_CONFIDENCE_THRESHOLD = 0.6


@dataclass(frozen=True)
class Classification:
    document_type: str
    suggested_filename: str
    suggested_category: str
    short_summary: str
    confidence: float
    reasoning: str

    @classmethod
    def from_dict(cls, value: Any) -> "Classification":
        if not isinstance(value, dict):
            raise ValueError("Model response must be a JSON object")
        required = {
            "document_type",
            "suggested_filename",
            "suggested_category",
            "short_summary",
            "confidence",
            "reasoning",
        }
        missing = required - value.keys()
        if missing:
            raise ValueError(f"Missing fields: {', '.join(sorted(missing))}")
        text_fields = required - {"confidence"}
        if any(not isinstance(value[key], str) or not value[key].strip() for key in text_fields):
            raise ValueError("Text fields must be non-empty strings")
        confidence = value["confidence"]
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise ValueError("Confidence must be a number")
        if not 0 <= confidence <= 1:
            raise ValueError("Confidence must be between 0 and 1")
        return cls(
            document_type=value["document_type"].strip().lower(),
            suggested_filename=value["suggested_filename"].strip(),
            suggested_category=value["suggested_category"].strip(),
            short_summary=value["short_summary"].strip(),
            confidence=float(confidence),
            reasoning=value["reasoning"].strip(),
        )


@dataclass
class PlanItem:
    source: str
    destination: str
    classification: Classification | None
    error: str | None = None

    @property
    def confidence(self) -> float | None:
        return self.classification.confidence if self.classification else None

    @property
    def is_low_confidence(self) -> bool:
        return self.confidence is not None and self.confidence < LOW_CONFIDENCE_THRESHOLD
