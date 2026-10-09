"""Read-only dependency/resource diagnostics. Never prints credential values."""
import importlib.metadata as metadata
import json
from pathlib import Path
import platform
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    packages = ["numpy","PyYAML","torch","torchvision","opencv-contrib-python","pandas",
                "ultralytics","SoccerNet","rfdetr","sahi","jsonschema","torchreid"]
    versions = {}
    for name in packages:
        try: versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError: versions[name] = "missing"
    weights = ["checkpoints/yolox_soccernet.pth.tar","checkpoints/sports_model.pth.tar-60",
               "checkpoints/SoccernetGSR_EfficientNet_Best.pth","checkpoints/CLIP_Jersey.pth",
               "API/model/rfdetr_ball_best_ema.pth","API/model/tcn_ball_completion_best.pt"]
    print(json.dumps({"python":sys.version.split()[0],"system":platform.system(),
        "packages":versions,"commands":{name:shutil.which(name) for name in ("conda","ffmpeg","node","npm")},
        "weights":{p:(ROOT/p).is_file() for p in weights},
        "font":(ROOT/"assets/fonts/NotoSansCJKsc-Regular.otf").is_file(),
        "note":"Missing optional weights are normal for UI/structured-only use. This does not validate GPU inference."},indent=2))


if __name__=="__main__":main()
