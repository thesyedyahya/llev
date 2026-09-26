from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, Field, field_validator

MAX_OPTIONS = 26  # one letter per option; one token per label keeps the read exact


class _Q(BaseModel):
    instructions: str = Field(min_length=1, max_length=4000)
    # Stable task id (e.g. "ticket.category"). Enables few-shot learning from feedback.
    task: str | None = Field(default=None, max_length=120, pattern=r"^[\w.\-:]+$")
    # fast: small model first, escalate if unsure. accurate: always the escalation model
    # (use for safety / money decisions, where a small model's confident mistakes are unacceptable).
    tier: Literal["fast", "accurate"] = "fast"


class ChoiceQ(_Q):
    type: Literal["choice"] = "choice"
    criteria: dict[str, str] = Field(min_length=2, max_length=MAX_OPTIONS)


class ScoreQ(_Q):
    type: Literal["score"] = "score"
    criteria: list[str] = Field(min_length=2, max_length=10)  # ordered lowest -> highest


class NoulQ(_Q):
    type: Literal["noul"] = "noul"


class MultiQ(_Q):
    """Multi-label: every option that applies (independent yes/no per option)."""

    type: Literal["multi"] = "multi"
    criteria: dict[str, str] = Field(min_length=1, max_length=MAX_OPTIONS)
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)


Question = Annotated[Union[ChoiceQ, ScoreQ, NoulQ, MultiQ], Field(discriminator="type")]


class DecideRequest(BaseModel):
    state: str | dict[str, Any] | list[Any]
    questions: dict[str, Question] = Field(min_length=1)
    escalate: bool = True  # allow the escalation model for low-confidence answers
    fewshot: bool = True  # allow learned examples

    @field_validator("questions")
    @classmethod
    def _keys(cls, v: dict[str, Any]) -> dict[str, Any]:
        for k in v:
            if not k or len(k) > 64:
                raise ValueError("question keys must be 1-64 chars")
        return v


class Answer(BaseModel):
    type: str
    confidence: float  # 0..1, comparable across types - use it to gate automation
    mass: float  # prob. mass the model put on valid labels; low = model confused by format
    engine: str
    escalated: bool = False
    fewshot: int = 0  # learned examples used
    # type-specific
    choice: str | None = None
    score: float | None = None
    level: int | None = None
    noul: float | None = None
    selected: list[str] | None = None
    probabilities: dict[str, float] | list[float] | None = None


class DecideResponse(BaseModel):
    id: str
    model: str
    answers: dict[str, Answer]
    latency_ms: float


class FeedbackRequest(BaseModel):
    id: str
    key: str
    # choice: option key | score: level index | noul: bool | multi: list of option keys
    label: Any
    source: str | None = Field(default=None, max_length=120)


class FeedbackResponse(BaseModel):
    ok: bool
    learned: bool
    detail: str | None = None
