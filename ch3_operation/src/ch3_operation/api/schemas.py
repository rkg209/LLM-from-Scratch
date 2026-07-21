"""Wire shapes for O0's `/review` and `/health` routes. No schema validation here — that
is O2. This is the JSON that goes over the wire, not the reviewer output contract."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ReviewRequest(BaseModel):
    code: str = Field(min_length=1)
    context: str = ""


class ReviewResponse(BaseModel):
    raw: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
