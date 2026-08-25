"""Nano Banana (gemini-2.5-flash-image) runner, deployed pipeline.

One image per prompt. LLM-native model: prompt expansion is internal and
unobservable, so this model contributes no expander text — main results only.
Caveat recorded in the paper: this is also the model that generated the
dataset's original images.

Usage: python run_google.py [--limit 1]
"""
import argparse
import datetime
import os
import time

from common import (append_jsonl, done_keys, image_path, load_env,
                    load_prompts, manifest_path)

MODEL_ID = "gemini-2.5-flash-image"
COND = "deployed"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--model-id", default=MODEL_ID)
    args = ap.parse_args()

    load_env()
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

    prompts = load_prompts("raw")
    done = done_keys("nanobanana", COND)
    jobs = [(i, f, p) for i, f, p in prompts if (i, 0) not in done]
    print(f"nanobanana ({args.model_id}): {len(jobs)} images to go ({len(done)} done)")

    n_done, consec_fail = 0, 0
    for item_id, family, prompt in jobs:
        t0 = time.time()
        try:
            rsp = client.models.generate_content(
                model=args.model_id,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_modalities=["TEXT", "IMAGE"]))
            blobs = [p.inline_data.data for p in rsp.candidates[0].content.parts
                     if getattr(p, "inline_data", None)]
            if not blobs:
                raise RuntimeError("no image part in response")
        except Exception as e:  # noqa: BLE001 — log and move on; rerun picks it up
            print(f"[FAIL] {item_id}: {type(e).__name__}: {str(e)[:200]}", flush=True)
            consec_fail += 1
            if consec_fail >= 5:
                raise SystemExit("5 consecutive failures — aborting run")
            continue
        consec_fail = 0
        out = image_path("nanobanana", COND, item_id, 0)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(blobs[0])
        append_jsonl(manifest_path("nanobanana", COND), {
            "item_id": item_id, "family": family, "model": "nanobanana",
            "api_model_id": args.model_id, "cond": COND, "seed": 0,
            "prompt": prompt,
            "file": str(out.relative_to(out.parents[4])),
            "sec": round(time.time() - t0, 1),
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        })
        n_done += 1
        if n_done % 10 == 0 or n_done == len(jobs):
            print(f"[{n_done}/{len(jobs)}] {item_id}", flush=True)
        if args.limit and n_done >= args.limit:
            print("limit reached, stopping")
            break


if __name__ == "__main__":
    main()
