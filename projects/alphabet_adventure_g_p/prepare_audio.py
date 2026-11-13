"""Generate the approved G–P lesson locally, including exact response pauses."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from img2vid.audio import parse_script


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    segments = parse_script((HERE / "SUNO_SCRIPT.md").read_text(), default_pause=2)
    script = []
    for item in segments:
        if item["type"] == "speech":
            script.append({"section": item["section"], "text": item["text"]})
        else:
            duration = 3 if "Quiz" in item["section"] else item["seconds"]
            script.append({"section": item["section"], "pause": duration})
    target = HERE / "local_script.json"
    target.write_text(json.dumps(script, indent=2, ensure_ascii=False) + "\n")
    command = [
        sys.executable, "-m", "img2vid", "speak", "-f", str(target),
        "-o", str(HERE / "audio" / "g_p_lesson.mp3"),
        "--voice", "af_heart", "--speed", "0.86", "--spell-letters", "--music", "gentle",
    ]
    if args.force:
        command.append("--force")
    subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
