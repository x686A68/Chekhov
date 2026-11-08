"""Build the explicit-constraint prompts (reviewer KXAW, Q2).

Every mental-state (attribution) item and a fixed random half of the
figurative items get the sentence "No <target> should appear in the image."
appended to the original prompt. Output: data/generation/constraint_prompts.jsonl
  {item_id, family, target, prompt, text}
cond "constraint" in run_local.py reads it.
"""
import random

from common import GEN, PROMPTS, append_jsonl, read_jsonl

OUT = GEN / "constraint_prompts.jsonl"
SUFFIX = " No {target} should appear in the image."


def main():
    rows = read_jsonl(PROMPTS)
    att = [r for r in rows if r["family"] == "attribution"]
    fig = sorted((r for r in rows if r["family"] == "figurative"), key=lambda r: r["item_id"])
    random.Random(0).shuffle(fig)
    fig = sorted(fig[: len(fig) // 2], key=lambda r: r["item_id"])
    OUT.unlink(missing_ok=True)
    for r in att + fig:
        p = r["prompt"].rstrip()
        if p[-1] not in ".!?":
            p += "."
        append_jsonl(OUT, {"item_id": r["item_id"], "family": r["family"],
                           "target": r["target"], "prompt": r["prompt"],
                           "text": p + SUFFIX.format(target=r["target"])})
    print(f"wrote {len(att)} attribution + {len(fig)} figurative -> {OUT}")


if __name__ == "__main__":
    main()
