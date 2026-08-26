"""Export annotation tasks from overreal_v1 pending rows.

Creates a staging dir for HF upload:
  staging/images/<sha1(image_id)[:16]>.<ext>   (hardlinks, no disk cost)
  staging/tasks.jsonl                          image_id, file (hashed), family,
                                               prompt, target
Filenames are hashed so annotators cannot see the generator or prompt
condition — the annotation is blind to provenance.
"""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
DS = ROOT / "data" / "overreal_v1"
STAGING = ROOT / "data" / "annotation_staging"


def main():
    (STAGING / "images").mkdir(parents=True, exist_ok=True)
    n = 0
    with open(STAGING / "tasks.jsonl", "w", encoding="utf-8") as out:
        for line in open(DS / "metadata.jsonl", encoding="utf-8"):
            r = json.loads(line)
            if r.get("annotated") or r.get("refused") or not r.get("file_name"):
                continue
            src = DS / r["file_name"]
            ext = src.suffix.lower()
            h = hashlib.sha1(r["image_id"].encode()).hexdigest()[:16]
            dst = STAGING / "images" / f"{h}{ext}"
            if not dst.exists():
                os.link(src, dst)
            out.write(json.dumps({
                "image_id": r["image_id"], "file": f"images/{h}{ext}",
                "family": r["family"], "prompt": r["prompt"],
                "target": r["target"],
            }, ensure_ascii=False) + "\n")
            n += 1
    print(f"{n} tasks staged -> {STAGING}")


if __name__ == "__main__":
    main()
