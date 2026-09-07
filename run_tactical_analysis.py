#!/usr/bin/env python3
"""Run the modular football tactical-analysis framework on SoccerNetGSR output."""

from __future__ import annotations

import argparse
from pathlib import Path

from tactical_analysis.base import ANALYZER_REGISTRY
from tactical_analysis.pipeline import TacticalAnalysisPipeline
from tactical_analysis.reporting import write_report_bundle


PROJECT_ROOT = Path(__file__).resolve().parent


def resolve_path(value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return (PROJECT_ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run evidence-based football tactical analysis from SoccerNetGSR JSON."
    )
    parser.add_argument("--json-path", help="SoccerNetGSR prediction JSON.")
    parser.add_argument("--input-video", help="Optional source video, used to infer FPS and retain provenance.")
    parser.add_argument(
        "--output-dir",
        default="soccer_input_dataset/outputs/tactical_framework",
        help="Directory for JSON, Markdown and CSV reports.",
    )
    parser.add_argument(
        "--config",
        default="configs/tactical_analysis.yaml",
        help="Framework YAML configuration.",
    )
    parser.add_argument("--fps", type=float, help="Override FPS when no source video is available.")
    parser.add_argument(
        "--analyzers",
        help="Comma-separated analyzer names. Default uses the configured priority chain.",
    )
    parser.add_argument("--list-analyzers", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.list_analyzers:
        for name, analyzer_cls in ANALYZER_REGISTRY.items():
            print(f"{name:22s} {analyzer_cls.priority} {analyzer_cls.description_zh}")
        return 0
    if not args.json_path:
        raise SystemExit("--json-path is required unless --list-analyzers is used")

    json_path = resolve_path(args.json_path)
    video_path = resolve_path(args.input_video)
    output_dir = resolve_path(args.output_dir)
    config_path = resolve_path(args.config)
    if json_path is None or not json_path.exists():
        raise FileNotFoundError(f"GSR JSON not found: {json_path}")
    if video_path is not None and not video_path.exists():
        raise FileNotFoundError(f"Input video not found: {video_path}")
    if config_path is not None and not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    assert output_dir is not None

    selected = [item.strip() for item in args.analyzers.split(",") if item.strip()] if args.analyzers else None
    pipeline = TacticalAnalysisPipeline.from_config_file(config_path)
    report = pipeline.run(
        json_path=json_path,
        video_path=video_path,
        fps=args.fps,
        selected_analyzers=selected,
    )
    paths = write_report_bundle(report, output_dir)
    print(f"Completed {len(report.analyzer_outputs)} analyzers.")
    print(f"Findings: {len(report.findings)}, timeline events: {len(report.events)}")
    for label, path in paths.items():
        print(f"{label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
