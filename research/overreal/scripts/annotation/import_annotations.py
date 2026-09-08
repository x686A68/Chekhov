"""Import double-blind annotations from the HF annotations dataset into
overreal_v1/metadata.jsonl.

Per (image, annotator) the latest record wins (revisits overwrite). For each
image with two annotators, chronological order assigns the slots:
  label_1 = first annotator's labels, label_2 = second annotator's labels
  agreement = the two label sets are identical (exact); annotated = True
  annotators = pseudonymized ids, first slot first
Images with a single annotation get label_1 filled but stay annotated=False.
`included` is left untouched (adjudication rule to be decided later).

Idempotent: re-running replaces annotation-derived fields from scratch.
"""
import collections
import json
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[4]
META = ROOT / "data" / "overreal_v1" / "metadata.jsonl"
PSEUDO = json.load(open(ROOT / "data" / "annotator_map.private.json"))["name_to_id"]
ANN_REPO = "huangjh16/overreal-annotations"


def load_annotations():
    snap = snapshot_download(ANN_REPO, repo_type="dataset", allow_patterns=["*.jsonl"])
    rows = []
    for f in sorted(Path(snap).rglob("*.jsonl")):
        rows += [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
    latest = {}   # (image_id, annotator) -> record, latest ts wins
    for r in sorted(rows, key=lambda r: r.get("ts", "")):
        latest[(r["image_id"], r["annotator"])] = r
    per_image = collections.defaultdict(list)
    for (iid, _), r in latest.items():
        per_image[iid].append(r)
    for iid in per_image:   # chronological slot order
        per_image[iid].sort(key=lambda r: r["ts"])
    return per_image


def main():
    per_image = load_annotations()
    out, n_full, n_half = [], 0, 0
    for line in open(META, encoding="utf-8"):
        r = json.loads(line)
        anns = per_image.get(r["image_id"])
        if anns and not r.get("refused") and r.get("source_folder") is None:
            a1 = anns[0]
            r["label_1"] = a1["labels"]
            r["annotators"] = [PSEUDO[a1["annotator"]]]
            if len(anns) >= 2:
                a2 = anns[1]
                r["label_2"] = a2["labels"]
                r["agreement"] = set(a1["labels"]) == set(a2["labels"])
                r["annotators"].append(PSEUDO[a2["annotator"]])
                r["annotated"] = True
                n_full += 1
            else:
                n_half += 1
        out.append(r)
    with open(META, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"imported: {n_full} doubly annotated, {n_half} single (kept pending)")


if __name__ == "__main__":
    main()
