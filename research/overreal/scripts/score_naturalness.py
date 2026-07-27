"""Measure the naturalness of every item's prompt (DECISIONS.md #24).

The pilot found items that are hard because they are odd rather than because the
phenomenon is hard — *the auditor entering the office, as hungry as a wolf*. Naturalness
therefore has to be measured and reported next to difficulty, and the P5 difficulty loop
needs a floor it can check candidates against. This is the measurement.

Method: length-normalised negative log-likelihood of the prompt under a plain causal LM
(no instruction tuning, no chat template), i.e. per-token perplexity. Lower is more
natural. Two design points:

  - **The baseline is P, not A.** Per-token NLL falls with length — later tokens are
    easier to predict — so comparing S against the much shorter A condition measures
    length, not naturalness. (Measured: excess over A was negative for ten of eleven
    family cells, which is the length effect, not S being *more* natural than the bare
    scenario.) S and P are length-matched by construction (GOAL.md rule 1) and differ only
    in the licensing device, so S − P isolates what the device costs in naturalness.
  - Token counts are reported alongside, so any residual length difference is visible
    rather than buried in the score.
  - The score is only comparable within a family. Across families the scenarios differ in
    length and register, so a global ranking would mostly recover topic.

Usage: CUDA_VISIBLE_DEVICES=5 python scripts/score_naturalness.py [--model ...]
Writes pilot/naturalness.json and pilot/naturalness.md.
"""
import argparse
import json
import os
from collections import defaultdict

os.environ.setdefault("HF_HOME", "/data/users/jiahao_huang/hf")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ITEMS = os.path.join(ROOT, "pilot", "items")
OUT = os.path.join(ROOT, "pilot")

# A base (non-instruct) model is the right instrument: an instruction-tuned model scores
# imperative prompts as natural because it was trained on them.
DEFAULT_MODEL = "Qwen/Qwen3-8B"


def load_items():
    rows = []
    for fn in sorted(os.listdir(ITEMS)):
        if fn.endswith(".jsonl"):
            with open(os.path.join(ITEMS, fn)) as f:
                rows += [json.loads(l) for l in f if l.strip()]
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
                                                 device_map="cuda")
    model.eval()

    @torch.no_grad()
    def nll(text):
        ids = tok(text, return_tensors="pt").input_ids.to("cuda")
        if ids.shape[1] < 2:
            return None, 0
        out = model(ids, labels=ids)
        return float(out.loss), int(ids.shape[1])  # mean NLL per token, length

    items = load_items()
    rows = []
    for it in items:
        # score the image prompt where there is one: it is the bare stimulus, without the
        # task instruction, so it is the thing whose naturalness we actually care about
        prompts = it.get("image_prompts") or {
            k: " ".join(t["content"] for t in v) for k, v in it["prompts"].items()}
        got = {c: nll(p) for c, p in prompts.items()}
        scored = {c: v[0] for c, v in got.items()}
        ntok = {c: v[1] for c, v in got.items()}
        base = scored.get("P")
        rows.append({
            "id": it["id"], "family": it["family"], "entity": it["entity"],
            "scenario": it["scenario"], "nll": scored, "n_tokens": ntok,
            "cost_vs_P": {c: (round(v - base, 4) if (v is not None and base is not None) else None)
                          for c, v in scored.items()},
            "prompts": prompts,
        })

    by_family = defaultdict(list)
    for r in rows:
        by_family[r["family"]].append(r)

    summary = {}
    for fam, frows in sorted(by_family.items()):
        conds = [c for c in frows[0]["nll"] if c != "P"]
        summary[fam] = {
            "n_items": len(frows),
            "mean_nll_P": round(sum(r["nll"]["P"] for r in frows) / len(frows), 4),
            "mean_tokens_P": round(sum(r["n_tokens"]["P"] for r in frows) / len(frows), 1),
            **{f"mean_cost_{c}": round(
                sum(r["cost_vs_P"][c] for r in frows) / len(frows), 4) for c in conds},
            **{f"mean_tokens_{c}": round(
                sum(r["n_tokens"][c] for r in frows) / len(frows), 1) for c in conds},
        }

    with open(os.path.join(OUT, "naturalness.json"), "w") as f:
        json.dump({"model": args.model, "summary": summary, "items": rows}, f, indent=2)

    # the least natural items in each family, which is what the difficulty loop needs
    lines = ["# Prompt naturalness", "",
             f"Per-token NLL under `{args.model}` (a base LM, not an instruct model — an",
             "instruction-tuned model rates imperative prompts as natural because it was",
             "trained on them). Lower is more natural.",
             "",
             "`cost` is the S-condition NLL minus the same item's **P**-condition NLL. P is the",
             "baseline rather than A because S and P are length-matched by construction while A",
             "is much shorter, and per-token NLL falls with length. Positive cost = the",
             "suppression device makes the prompt less natural than the licensed version.", ""]
    for fam, frows in sorted(by_family.items()):
        conds = [c for c in frows[0]["nll"] if c.startswith("S")]
        c0 = conds[0]
        worst = sorted(frows, key=lambda r: -(r["cost_vs_P"][c0] or 0))[:3]
        best = sorted(frows, key=lambda r: (r["cost_vs_P"][c0] or 0))[:3]
        lines += [f"## {fam}", "", f"mean P-condition NLL {summary[fam]['mean_nll_P']} over {summary[fam]['mean_tokens_P']} tokens, "
                  f"mean excess {summary[fam].get(f'mean_cost_{c0}')}", "",
                  "| | cost | prompt |", "|---|---|---|"]
        for r in worst:
            lines.append(f"| least natural | {r['cost_vs_P'][c0]:+.2f} | {r['prompts'][c0]} |")
        for r in best:
            lines.append(f"| most natural | {r['cost_vs_P'][c0]:+.2f} | {r['prompts'][c0]} |")
        lines.append("")
    with open(os.path.join(OUT, "naturalness.md"), "w") as f:
        f.write("\n".join(lines))

    print(f"{'family':<22}{'NLL(P)':>9}{'tok(P)':>8}{'cost(S)':>10}{'tok(S)':>8}")
    for fam, s in summary.items():
        cost = [v for k, v in s.items() if k.startswith("mean_cost_S")]
        tk = [v for k, v in s.items() if k.startswith("mean_tokens_S")]
        print(f"{fam:<22}{s['mean_nll_P']:>9.2f}{s['mean_tokens_P']:>8.1f}"
              f"{(sum(cost)/len(cost) if cost else float('nan')):>10.2f}"
              f"{(sum(tk)/len(tk) if tk else float('nan')):>8.1f}")


if __name__ == "__main__":
    main()
