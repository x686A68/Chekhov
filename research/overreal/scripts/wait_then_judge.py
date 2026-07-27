"""Block until the image pilot has produced all 252 images, then run the VLM judge.

Exists so the judge can be launched as a single foreground step instead of polling the
manifest by hand.
"""
import glob
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG = os.path.join(ROOT, "pilot", "images")


def count():
    n = set()
    for fn in glob.glob(os.path.join(IMG, "manifest*.jsonl")):
        with open(fn) as f:
            for l in f:
                if l.strip():
                    r = json.loads(l)
                    n.add((r["id"], r["condition"]))
    return len(n)


def main():
    target = int(sys.argv[1]) if len(sys.argv) > 1 else 252
    while True:
        c = count()
        print(f"{c}/{target} images", flush=True)
        if c >= target:
            break
        time.sleep(30)
    print("generation complete, launching judge", flush=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="5")
    return subprocess.call([sys.executable, os.path.join(ROOT, "scripts", "judge_images.py")], env=env)


if __name__ == "__main__":
    sys.exit(main())
