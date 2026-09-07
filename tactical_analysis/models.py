"""Shared data contracts used by every tactical-analysis stage."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0
HALF_LENGTH = PITCH_LENGTH / 2.0
HALF_WIDTH = PITCH_WIDTH / 2.0


@dataclass(frozen=True)
class Observation:
    frame: int
    time_sec: float
    track_id: int
    role: str
    team: str | None
    pitch_x: float | None
    pitch_y: float | None
    bbox: dict[str, float] | None = None
    confidence: float | None = None

    @property
    def has_pitch_position(self) -> bool:
        return self.pitch_x is not None and self.pitch_y is not None

    @property
    def is_player(self) -> bool:
        return self.role in {"player", "goalkeeper"} and self.team in {"left", "right"}


@dataclass
class FrameState:
    frame: int
    time_sec: float
    observations: list[Observation] = field(default_factory=list)

    @property
    def players(self) -> list[Observation]:
        return [item for item in self.observations if item.is_player]

    @property
    def balls(self) -> list[Observation]:
        return [item for item in self.observations if item.role == "ball"]

    @property
    def ball(self) -> Observation | None:
        calibrated = [item for item in self.balls if item.has_pitch_position]
        return calibrated[0] if calibrated else (self.balls[0] if self.balls else None)


@dataclass
class AnalysisContext:
    source_json: Path
    source_video: Path | None
    fps: float
    frames: list[FrameState]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_sec(self) -> float:
        return self.frames[-1].time_sec if self.frames else 0.0


@dataclass
class Evidence:
    frame: int | None
    time_sec: float | None
    description_zh: str
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class Finding:
    finding_id: str
    analyzer: str
    priority: str
    category: str
    title_zh: str
    summary_zh: str
    confidence: float
    start_sec: float | None = None
    end_sec: float | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    evidence: list[Evidence] = field(default_factory=list)
    limitations_zh: list[str] = field(default_factory=list)


@dataclass
class TimelineEvent:
    event_id: str
    event_type: str
    frame: int
    time_sec: float
    team: str | None
    actor_track_id: int | None
    target_track_id: int | None
    confidence: float
    label_zh: str
    metrics: dict[str, Any] = field(default_factory=dict)
    period: int | None = None
    event_subtype: str | None = None
    outcome: str = "unknown"
    start_x: float | None = None
    start_y: float | None = None
    end_x: float | None = None
    end_y: float | None = None
    evidence_start_sec: float | None = None
    evidence_end_sec: float | None = None
    source: str = "vision"
    review_status: str = "candidate"
    related_event_ids: list[str] = field(default_factory=list)


@dataclass
class AnalyzerOutput:
    analyzer: str
    priority: str
    summary: dict[str, Any] = field(default_factory=dict)
    findings: list[Finding] = field(default_factory=list)
    events: list[TimelineEvent] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class AnalysisReport:
    schema_version: str
    generated_at: str
    source: dict[str, Any]
    config: dict[str, Any]
    analyzer_outputs: list[AnalyzerOutput]
    provenance: dict[str, Any]
    limitations_zh: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def findings(self) -> list[Finding]:
        return [finding for output in self.analyzer_outputs for finding in output.findings]

    @property
    def events(self) -> list[TimelineEvent]:
        return [event for output in self.analyzer_outputs for event in output.events]
