from __future__ import annotations

from pydantic import BaseModel, Field

from shared.schemas import MoveLabel


class LlmMoveItem(BaseModel):
    block_id: str
    label: MoveLabel
    confidence: float = Field(ge=0.0, le=1.0)


class LlmMovesOutput(BaseModel):
    moves: list[LlmMoveItem] = Field(default_factory=list)


class LlmBlueprintRule(BaseModel):
    title: str
    description: str
    supporting_block_ids: list[str] = Field(default_factory=list)


class LlmParagraphPlanItem(BaseModel):
    paragraph_index: int = Field(ge=1)
    label: MoveLabel
    description: str
    supporting_block_ids: list[str] = Field(default_factory=list)


class LlmIntroBlueprintOutput(BaseModel):
    paragraph_plan: list[LlmParagraphPlanItem] = Field(default_factory=list)
    story_rules: list[LlmBlueprintRule] = Field(default_factory=list)
    claim_rules: list[LlmBlueprintRule] = Field(default_factory=list)
    checklist: list[LlmBlueprintRule] = Field(default_factory=list)


class LlmStoryboardItem(BaseModel):
    figure_role: str
    recommended_position: str
    caption_formula: str | None = None
    supporting_block_ids: list[str] = Field(default_factory=list)


class LlmStoryboardOutput(BaseModel):
    items: list[LlmStoryboardItem] = Field(default_factory=list)

