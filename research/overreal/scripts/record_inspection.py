"""Record direct-inspection verdicts into pilot/images/inspection.jsonl.

Reads a JSON object on stdin mapping "<item_id>_<condition>" to the option letter that
is visibly the case, optionally with a note:

    {"2_attribution_00_S": "A", "4b_legibility_03_S": ["B", "letter is edge-on"]}

Family and image path are filled in from the manifest, so the inspection file always
lines up with what the judge was asked.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG = os.path.join(ROOT, "pilot", "images")


def main():
    manifest = {}
    for fn in sorted(os.listdir(IMG)):
        if fn.startswith("manifest") and fn.endswith(".jsonl"):
            for l in open(os.path.join(IMG, fn)):
                if l.strip():
                    r = json.loads(l)
                    manifest[f"{r['id']}_{r['condition']}"] = r

    verdicts = json.load(sys.stdin)
    path = os.path.join(IMG, "inspection.jsonl")
    existing = {}
    if os.path.exists(path):
        for l in open(path):
            if l.strip():
                r = json.loads(l)
                existing[f"{r['id']}_{r['condition']}"] = r

    added = 0
    for key, val in verdicts.items():
        if key not in manifest:
            print(f"unknown image key: {key}", file=sys.stderr)
            continue
        letter, note = (val, "") if isinstance(val, str) else (val[0], val[1])
        m = manifest[key]
        existing[key] = {"id": m["id"], "condition": m["condition"], "family": m["family"],
                         "entity": m["entity"], "path": m["path"],
                         "human_letter": letter, "note": note}
        added += 1

    with open(path, "w") as f:
        for k in sorted(existing):
            f.write(json.dumps(existing[k], ensure_ascii=False) + "\n")
    print(f"recorded {added} verdicts, {len(existing)} total in inspection.jsonl")


if __name__ == "__main__":
    main()
