"""LLM-judge (LJ) via the OpenAI API, as the cross-judge for the Claude one.

Same verbatim VIEScore t2i protocol and the same pilot slice as
run_lj_subscription.py, judged by a pinned GPT snapshot.

  python run_lj_openai.py --pilot [--limit N]

Output: data/overreal_v1/metrics/lj_gpt.jsonl {image_id, score, model, sec, ts}
"""
import argparse
import base64
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gen"))
from common import load_env  # noqa: E402

from run_lj_subscription import CONTEXT, RULE, pilot_rows  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]
DS = ROOT / "data" / "overreal_v1"
OUT = DS / "metrics" / "lj_gpt.jsonl"
MODEL = "gpt-5.2-2025-12-11"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    load_env()
    from openai import OpenAI
    client = OpenAI()

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
    print(f"LJ-GPT: {len(todo)} images to judge ({len(done)} done)", flush=True)

    fails = 0
    with open(OUT, "a", encoding="utf-8") as f:
        for n, r in enumerate(todo, 1):
            path, prompt = meta[r["image_id"]]
            ext = "png" if path.endswith("png") else "jpeg"
            b64 = base64.b64encode(open(path, "rb").read()).decode()
            task = CONTEXT + "\n\n" + RULE.replace("<prompt>", prompt)
            t0 = time.time()
            try:
                rsp = client.chat.completions.create(
                    model=MODEL,
                    messages=[{"role": "user", "content": [
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/{ext};base64,{b64}"}},
                        {"type": "text", "text": task}]}],
                    max_completion_tokens=2000)
                text = rsp.choices[0].message.content or ""
            except Exception as e:  # noqa: BLE001
                fails += 1
                print(f"[FAIL] {r['image_id']}: {type(e).__name__}: {str(e)[:200]}",
                      flush=True)
                if "rate" in str(e).lower():
                    time.sleep(60)
                elif fails >= 8:
                    raise SystemExit("8 failures, aborting")
                continue
            m = (re.search(r'"score"\s*:\s*\[?\s*(\d+(?:\.\d+)?)', text)
                 or re.search(r"\[\s*(\d+(?:\.\d+)?)\s*\]", text))
            if not m:
                fails += 1
                print(f"[NOPARSE] {r['image_id']}: {text[:200]}", flush=True)
                continue
            fails = 0
            f.write(json.dumps({"image_id": r["image_id"], "score": float(m.group(1)),
                                "model": MODEL, "sec": round(time.time() - t0, 1),
                                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
                    + "\n")
            f.flush()
            if n % 25 == 0 or n == len(todo):
                print(f"[{n}/{len(todo)}]", flush=True)


if __name__ == "__main__":
    main()
