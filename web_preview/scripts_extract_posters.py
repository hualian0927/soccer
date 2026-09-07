"""Extract deterministic poster frames for the local web preview."""

from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "web_preview" / "public" / "posters"
VIDEOS = {
    "match-overview.jpg": ROOT / "soccer_input_dataset/outputs/test_5min_current/test_5min_clean_board_preview.mp4",
    "offensive-analysis.jpg": ROOT / "soccer_input_dataset/outputs/test_30s_offensive_analysis_zh.mp4",
    "individual-technique.jpg": ROOT / "soccer_input_dataset/outputs/goal_closeup_individual_technique_zh_bestshot.mp4",
    "formation-analysis.jpg": ROOT / "soccer_input_dataset/outputs/test_30s_formation_clean_board_smoke.mp4",
}


def extract(source: Path, target: Path) -> None:
    capture = cv2.VideoCapture(str(source))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(frame_count * 0.35)))
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"Cannot decode {source}")

    height, width = frame.shape[:2]
    target_ratio = 16 / 9
    if width / height > target_ratio:
        crop_width = int(height * target_ratio)
        left = (width - crop_width) // 2
        frame = frame[:, left : left + crop_width]
    else:
        crop_height = int(width / target_ratio)
        top = (height - crop_height) // 2
        frame = frame[top : top + crop_height, :]
    frame = cv2.resize(frame, (960, 540), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(target), frame, [cv2.IMWRITE_JPEG_QUALITY, 88])


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for filename, source in VIDEOS.items():
        extract(source, OUTPUT / filename)
        print(OUTPUT / filename)


if __name__ == "__main__":
    main()
