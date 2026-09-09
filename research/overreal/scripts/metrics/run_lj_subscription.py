"""LLM-judge (LJ) via the Claude Code CLI, using the subscription.

Scores images with the verbatim VIEScore text-to-image semantic-consistency
protocol (vendored under viescore_prompts/), judged by Claude through
`claude -p`. Resumable; backs off and retries when the usage limit bites.

  python run_lj_subscription.py --pilot        # the ~500-image pilot slice
  python run_lj_subscription.py                # full eval sample

Output: data/overreal_v1/metrics/lj_subscription.jsonl
        {image_id, score, model, sec, ts}
"""
import argparse
import collections
import json
import random
import re
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
DS = ROOT / "data" / "overreal_v1"
OUT = DS / "metrics" / "lj_subscription.jsonl"
VP = Path(__file__).parent / "viescore_prompts"
CONTEXT = (VP / "context.txt").read_text()
RULE = (VP / "t2i_rule_SC.txt").read_text()
PILOT_PER_CELL = 7
SEED = 20260909


def pilot_rows(rows):
    cells = collections.defaultdict(list)
    for r in rows:
        cells[(r["generator"], r["family"], r["prompt_cond"])].append(r)
    out = []
    for key in sorted(cells):
        rng = random.Random(f"{SEED}/pilot/{key}")
        pool = sorted(cells[key], key=lambda r: r["image_id"])
        out += pool if len(pool) <= PILOT_PER_CELL else rng.sample(pool, PILOT_PER_CELL)
    return out


def judge_once(img_path, prompt):
    task = (f"First use the Read tool to view the image at: {img_path}\n"
            f"Then follow the instructions below.\n\n{CONTEXT}\n\n"
            + RULE.replace("<prompt>", prompt))
    p = subprocess.run(
        ["claude", "-p", task, "--model", "claude-opus-5",
         "--allowedTools", "Read", "--output-format", "json"],
        capture_output=True, text=True, timeout=600)
    if p.returncode != 0:
        return None, None, (p.stderr or p.stdout)[:300]
    try:
        rsp = json.loads(p.stdout)
        text = rsp.get("result", "")
        model = rsp.get("model") or rsp.get("modelUsage") or "claude-opus-5"
        if isinstance(model, dict):
            model = ",".join(model.keys())
    except json.JSONDecodeError:
        return None, None, p.stdout[:300]
    m = re.search(r'"score"\s*:\s*\[?\s*(\d+(?:\.\d+)?)', text)
    if not m:
        m = re.search(r"\[\s*(\d+(?:\.\d+)?)\s*\]", text)
    if not m:
        return None, model, text[:300]
    return float(m.group(1)), model, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    meta = {}
    for line in open(DS / "metadata.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if r.get("file_name"):
            meta[r["image_id"]] = (str(DS / r["file_name"]), r.get("prompt"))
    rows = [json.loads(l) for l in open(DS / "eval_sample.jsonl", encoding="utf-8")]
    if args.pilot:
        rows = pilot_rows(rows)
    done = set()
    if OUT.exists():
        done = {json.loads(l)["image_id"] for l in open(OUT, encoding="utf-8")}
    todo = [r for r in rows if r["image_id"] not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"LJ: {len(todo)} images to judge ({len(done)} done)", flush=True)

    OUT.parent.mkdir(exist_ok=True)
    fails = 0
    with open(OUT, "a", encoding="utf-8") as f:
        for n, r in enumerate(todo, 1):
            path, prompt = meta[r["image_id"]]
            t0 = time.time()
            score, model, err = judge_once(path, prompt)
            if score is None:
                fails += 1
                print(f"[FAIL] {r['image_id']}: {err}", flush=True)
                if err and ("limit" in str(err).lower() or "rate" in str(err).lower()):
                    print("usage limit suspected, sleeping 15 min", flush=True)
                    time.sleep(900)
                elif fails >= 8:
                    raise SystemExit("8 failures, aborting for inspection")
                continue
            fails = 0
            f.write(json.dumps({"image_id": r["image_id"], "score": score,
                                "model": model, "sec": round(time.time() - t0, 1),
                                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
                    + "\n")
            f.flush()
            if n % 20 == 0 or n == len(todo):
                print(f"[{n}/{len(todo)}]", flush=True)


if __name__ == "__main__":
    main()
