"""Build the target-removed prompts for the base rate b.

For every item in prompts.jsonl, a prompt that says the same scene without
mentioning the target. Existence-canceling prompts are rule-built from five
templates (build_family1_negbench.py), so the negation clause is stripped
by regex. The hand-written families are rewritten by Claude Opus 5 with an
instruction to delete the mention and change nothing else; the result is
checked for the target string and re-asked once with a firmer instruction
if it survives. Items where the check still fails are written with
"flag": true for hand review.

Two modes. --mode delete (default) removes the mention and, where needed,
its clause. --mode placeholder keeps the clause for attribution and
perspectival items and puts "something" / "someone" / "some place" in the
target's slot, so the licensing frame ("believing that there is ... at the
ridge", "inside the closed box is ...") survives; figurative and
existence-canceling items are copied from the delete output, since a
placeholder as the vehicle of a comparison reads badly.

Output: data/generation/notarget_prompts.jsonl (delete) or
        data/generation/notarget_placeholder_prompts.jsonl (placeholder)
  {item_id, family, target, prompt, text, method, model?, flag?}
Resumable: items already present are skipped.
"""
import argparse
import re
import sys

from common import GEN, NOTARGET, PROMPTS, append_jsonl, load_env, read_jsonl

PLACEHOLDER_OUT = GEN / "notarget_placeholder_prompts.jsonl"
PLACEHOLDER_INSTRUCTION = (
    "Rewrite the following text-to-image prompt by replacing the mention of \"{target}\" "
    "with the most neutral placeholder that keeps the sentence grammatical: \"something\" "
    "for a thing or an animal, \"someone\" only for a person, \"some place\" for a place, "
    "\"some words\" for a piece of text. Drop any adjectives that described only "
    "\"{target}\". Do not describe, hint at, or categorise what was removed, and change "
    "nothing else in the prompt. Reply with the rewritten prompt only.\n\nPrompt: {prompt}")

TEMPLATE_RE = re.compile(
    r",\s*(with no|without a single|no|and not a single|without any)\s+(?P<e>.+?)"
    r"(\s+in sight|\s+anywhere in the scene)?\s*\.?$", re.I)

INSTRUCTION = (
    "Rewrite the following text-to-image prompt so that it no longer mentions \"{target}\". "
    "Delete the mention outright: do not replace it with a synonym, a broader word, a "
    "description of it, or a placeholder such as \"something\" or \"it\". If the mention "
    "is the object of a clause such as \"believing that ...\", \"remembering the ...\", "
    "\"as ... as a ...\", \"inside is a ...\", drop that clause or shorten it to the "
    "smallest grammatical form that no longer refers to \"{target}\" at all. Keep every "
    "other detail and the original wording, and do not add anything new. Reply with the "
    "rewritten prompt only.\n\nPrompt: {prompt}")
FIRMER = ("Your previous rewrite still contained \"{target}\" or a stand-in for it. Delete the "
          "mention and any substitute word or placeholder, and reply with the rewritten "
          "prompt only.")


def norm(s):
    return re.sub(r"[^a-z0-9 ]", " ", s.lower()).split()


def mentions(text, target):
    """True if the target's content words (all of them) appear in the text."""
    words = [w for w in norm(target) if len(w) > 2]
    t = " " + " ".join(norm(text)) + " "
    return bool(words) and all(f" {w} " in t or f" {w}s " in t for w in words)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["delete", "placeholder"], default="delete")
    ap.add_argument("--provider", choices=["openai", "anthropic"], default="openai")
    ap.add_argument("--model", default="", help="override the model id")
    args = ap.parse_args()
    model = args.model or {"openai": "gpt-5.2-2025-12-11", "anthropic": "claude-opus-5"}[args.provider]
    out_path = NOTARGET if args.mode == "delete" else PLACEHOLDER_OUT
    instruction = INSTRUCTION if args.mode == "delete" else PLACEHOLDER_INSTRUCTION
    deleted = ({r["item_id"]: r for r in read_jsonl(NOTARGET)}
               if args.mode == "placeholder" and NOTARGET.exists() else {})
    load_env()
    if args.provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic(max_retries=4)

        def ask(msgs):
            rsp = client.messages.create(model=model, max_tokens=1024,
                                         output_config={"effort": "low"}, messages=msgs)
            return "".join(b.text for b in rsp.content if b.type == "text")
    else:
        from openai import OpenAI
        client = OpenAI()

        def ask(msgs):
            rsp = client.chat.completions.create(model=model, messages=msgs,
                                                 max_completion_tokens=2000)
            return rsp.choices[0].message.content or ""
    done = {r["item_id"] for r in read_jsonl(out_path)} if out_path.exists() else set()
    rows = [r for r in read_jsonl(PROMPTS) if r["item_id"] not in done]
    print(f"{len(rows)} items to build ({len(done)} done)", flush=True)
    n_flag = 0
    for n, r in enumerate(rows, 1):
        out = {"item_id": r["item_id"], "family": r["family"], "target": r["target"],
               "prompt": r["prompt"]}
        if args.mode == "placeholder" and r["family"] in ("figurative", "cancellation") \
                and r["item_id"] in deleted:
            out = dict(deleted[r["item_id"]])
        elif r["family"] == "cancellation":
            m = TEMPLATE_RE.search(r["prompt"])
            if not m:
                sys.exit(f"no template match: {r['prompt']}")
            out["text"] = r["prompt"][:m.start()].strip()
            out["method"] = "rule"
        else:
            msgs = [{"role": "user", "content": instruction.format(target=r["target"], prompt=r["prompt"])}]
            text = ""
            for attempt in range(2):
                text = ask(msgs).strip().strip('"')
                if not mentions(text, r["target"]):
                    break
                msgs += [{"role": "assistant", "content": text},
                         {"role": "user", "content": FIRMER.format(target=r["target"])}]
            out["text"] = text
            out["method"] = "llm-" + args.mode
            out["model"] = model
            if mentions(text, r["target"]):
                out["flag"] = True
                n_flag += 1
        append_jsonl(out_path, out)
        if n % 50 == 0 or n == len(rows):
            print(f"[{n}/{len(rows)}] flagged so far: {n_flag}", flush=True)


if __name__ == "__main__":
    main()
