"""Analyzer interface and registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable

from .models import AnalysisContext, AnalyzerOutput


ANALYZER_REGISTRY: dict[str, type["TacticalAnalyzer"]] = {}


def register_analyzer(name: str) -> Callable[[type["TacticalAnalyzer"]], type["TacticalAnalyzer"]]:
    def decorator(cls: type["TacticalAnalyzer"]) -> type["TacticalAnalyzer"]:
        if name in ANALYZER_REGISTRY:
            raise ValueError(f"Analyzer already registered: {name}")
        cls.name = name
        ANALYZER_REGISTRY[name] = cls
        return cls

    return decorator


class TacticalAnalyzer(ABC):
    name = "base"
    priority = "P4"
    description_zh = ""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}

    @abstractmethod
    def analyze(self, context: AnalysisContext) -> AnalyzerOutput:
        raise NotImplementedError
