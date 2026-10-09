"""Check the Git index for accidental credentials, media, weights and large files."""
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = [re.compile(rb"sk-[A-Za-z0-9_-]{20,}"),re.compile(rb"gh[pousr]_[A-Za-z0-9]{20,}"),
            re.compile(rb"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----")]
FORBIDDEN = {".mp4",".mkv",".mov",".avi",".webm",".pt",".pth",".onnx",".safetensors",".ckpt",".zip",".pem",".key"}


def main():
    names=subprocess.check_output(["git","ls-files","-z"],cwd=ROOT).decode().split("\0")
    problems=[]
    for name in filter(None,names):
        p=Path(name)
        if p.suffix.lower() in FORBIDDEN or ".pth." in name or (p.name.startswith(".env") and p.name!=".env.example"):
            problems.append(f"forbidden artifact: {name}")
        data=subprocess.check_output(["git","show",f":{name}"],cwd=ROOT)
        if len(data)>40*1024*1024:problems.append(f"over 40 MiB: {name}")
        if any(pattern.search(data) for pattern in PATTERNS):problems.append(f"possible secret (value hidden): {name}")
    print("\n".join(problems) if problems else f"PASS: {len(list(filter(None,names)))} indexed files; no matching secrets/media/weights/oversize artifacts.")
    return bool(problems)


if __name__=="__main__":sys.exit(main())
