from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Document(BaseModel):
    pdf_id: str
    file_hash: str
    title: str | None = None
    num_pages: int
    created_at: str


BlockType = Literal["paragraph", "heading", "caption", "table", "reference", "other"]


class TextBlock(BaseModel):
    block_id: str
    pdf_id: str
    page_index: int = Field(ge=1)
    text: str
    bbox: list[float] | None = None  # [x0,y0,x1,y1] in page coordinate (top-left origin)
    bbox_list: list[list[float]] | None = None  # optional multi-bbox (same coordinate system as bbox)
    raw_text: str | None = None
    clean_text: str | None = None
    block_type: BlockType = "other"
    section_path: list[str] = Field(default_factory=list)


class EvidencePointer(BaseModel):
    evidence_id: str
    pdf_id: str
    page_index: int = Field(ge=1)
    bbox: list[float] | None = None  # [x0,y0,x1,y1] in page coordinate (top-left origin)
    bbox_norm: list[float] | None = None  # [x0,y0,x1,y1] normalized to [0,1]
    bbox_norm_list: list[list[float]] | None = None
    block_id: str | None = None
    excerpt: str | None = None
    reason: str | None = None
    kind: Literal["intro_paragraph", "caption", "other"] = "other"
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)


MoveLabel = Literal[
    "Context",
    "Problem",
    "Gap",
    "Approach",
    "Contribution",
    "Roadmap",
    "RelatedWorkHook",
    "Claim",
    "Limitation",
    "Other",
]


class RhetoricalMove(BaseModel):
    move_id: str
    label: MoveLabel
    block_id: str
    confidence: float = Field(ge=0.0, le=1.0)


class StyleFeatures(BaseModel):
    sentence_len_stats: dict = Field(default_factory=dict)
    hedging_profile: dict = Field(default_factory=dict)
    connector_profile: dict = Field(default_factory=dict)
    voice_profile: dict = Field(default_factory=dict)
    citation_density: dict = Field(default_factory=dict)


class BlueprintRule(BaseModel):
    rule_id: str
    title: str
    description: str
    supporting_evidence: list[str] = Field(default_factory=list)  # evidence_id list


class IntroBlueprint(BaseModel):
    paragraph_plan: list[dict] = Field(default_factory=list)
    story_rules: list[BlueprintRule] = Field(default_factory=list)
    claim_rules: list[BlueprintRule] = Field(default_factory=list)
    checklist: list[BlueprintRule] = Field(default_factory=list)


class Template(BaseModel):
    template_id: str
    template_type: Literal["IntroOpening", "GapApproach", "ContributionsRoadmap"]
    text_with_slots: str
    slot_schema: dict = Field(default_factory=dict)
    do_rules: list[str] = Field(default_factory=list)
    dont_rules: list[str] = Field(default_factory=list)
    supporting_evidence: list[str] = Field(default_factory=list)


class StoryboardItem(BaseModel):
    item_id: str
    figure_role: str
    recommended_position: str
    caption_formula: str | None = None
    supporting_evidence: list[str] = Field(default_factory=list)


class QualityReport(BaseModel):
    evidence_coverage: float = Field(ge=0.0, le=1.0)
    structure_strength: float = Field(ge=0.0, le=1.0)
    template_slot_score: float = Field(ge=0.0, le=1.0)
    plagiarism_max_similarity: float = Field(ge=0.0, le=1.0, default=0.0)
    plagiarism_flagged: list[dict] = Field(default_factory=list)
    ocr_used: bool = False
    intro_blocks_count_by_pdf: dict = Field(default_factory=dict)
    caption_count_by_pdf: dict = Field(default_factory=dict)
    weak_items: list[dict] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    skill_lib_used: bool = False
    skill_lib_fingerprint: str | None = None
    skill_topics_used: dict = Field(default_factory=dict)
    skill_rules_adopted: int = 0


class SkillPack(BaseModel):
    pack_id: str
    pack_name: str
    pdf_ids: list[str]
    intro_blueprint: IntroBlueprint
    templates: list[Template]
    storyboard: list[StoryboardItem]
    patterns: list[dict] = Field(default_factory=list)
    version: str = "v0"
    build_metadata: dict = Field(default_factory=dict)
    quality_report: QualityReport


class EvidenceIndex(BaseModel):
    pack_id: str
    evidence: list[EvidencePointer]
