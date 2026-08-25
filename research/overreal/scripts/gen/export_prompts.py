"""Export the generation prompt list from overreal_v1 metadata.

One line per included item: {item_id, family, prompt, target}.
"""
import sys

from common import DATASET, PROMPTS, append_jsonl, read_jsonl


def main():
    items = {}
    for r in read_jsonl(DATASET / "metadata.jsonl"):
        if r["included"] and r["item_id"] not in items:
            items[r["item_id"]] = {
                "item_id": r["item_id"],
                "family": r["family"],
                "prompt": r["prompt"],
                "target": r["target"],
            }
    missing = [i for i, r in items.items() if not r["prompt"] or not r["target"]]
    if missing:
        sys.exit(f"items with missing prompt/target: {missing}")
    PROMPTS.unlink(missing_ok=True)
    for r in items.values():
        append_jsonl(PROMPTS, r)
    print(f"wrote {len(items)} prompts -> {PROMPTS}")


if __name__ == "__main__":
    main()
