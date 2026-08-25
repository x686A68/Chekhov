"""OpenAI image runners: GPT-Image and DALL·E 3, deployed pipeline (expansion on).

Both models: 2 images per prompt (samples 0/1; the API has no seed control, the
sample index stands in for the seed slot in the manifest). DALL·E 3 returns its
rewritten prompt per image; each sample's rewrite is stored in the manifest and
sample 0's also lands in expanded_prompts.jsonl as the "dalle3" expander text
for Phase 2. GPT-Image exposes no rewrite (LLM-native).

Sync mode. For the full GPT-Image run consider --batch (Batch API, 50% off,
24h window) — submit/poll implemented in batch_openai.py.

Usage:
  python run_openai.py --model gpt-image --limit 1     # smoke test
  python run_openai.py --model dalle3
"""
import argparse
import base64
import datetime
import time

from common import (EXPANDED, append_jsonl, done_keys, image_path,
                    load_env, load_prompts, manifest_path, read_jsonl)

MODEL_IDS = {"gpt-image": "gpt-image-1.5", "dalle3": "dall-e-3"}
N_SAMPLES = {"gpt-image": 2, "dalle3": 2}
COND = "deployed"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODEL_IDS))
    ap.add_argument("--model-id", default=None, help="override the API model id")
    ap.add_argument("--limit", type=int, default=0, help="stop after N images")
    args = ap.parse_args()

    load_env()
    from openai import OpenAI
    client = OpenAI()
    model_id = args.model_id or MODEL_IDS[args.model]

    prompts = load_prompts("raw")
    done = done_keys(args.model, COND)
    jobs = [(i, f, p, s) for i, f, p in prompts
            for s in range(N_SAMPLES[args.model]) if (i, s) not in done]
    print(f"{args.model} ({model_id}): {len(jobs)} images to go ({len(done)} done)")

    exp_done = ({r["item_id"] for r in read_jsonl(EXPANDED) if r["expander"] == "dalle3"}
                if EXPANDED.exists() else set())

    n_done = 0
    for item_id, family, prompt, sample in jobs:
        t0 = time.time()
        kwargs = dict(model=model_id, prompt=prompt, n=1, size="1024x1024")
        if args.model == "gpt-image":
            kwargs["quality"] = "medium"
        else:
            kwargs.update(quality="standard", response_format="b64_json")
        try:
            rsp = client.images.generate(**kwargs)
        except Exception as e:  # noqa: BLE001 — log and move on; rerun picks it up
            print(f"[FAIL] {item_id} s{sample}: {type(e).__name__}: {str(e)[:200]}",
                  flush=True)
            continue
        datum = rsp.data[0]
        out = image_path(args.model, COND, item_id, sample)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(base64.b64decode(datum.b64_json))
        revised = getattr(datum, "revised_prompt", None)
        append_jsonl(manifest_path(args.model, COND), {
            "item_id": item_id, "family": family, "model": args.model,
            "api_model_id": model_id, "cond": COND, "seed": sample,
            "size": "1024x1024",
            "quality": kwargs.get("quality"), "prompt": prompt,
            "revised_prompt": revised,
            "file": str(out.relative_to(out.parents[4])),
            "sec": round(time.time() - t0, 1),
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        })
        if args.model == "dalle3" and revised and sample == 0 and item_id not in exp_done:
            append_jsonl(EXPANDED, {
                "item_id": item_id, "family": family, "expander": "dalle3",
                "expander_model": model_id, "text": revised,
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            })
            exp_done.add(item_id)
        n_done += 1
        if n_done % 10 == 0 or n_done == len(jobs):
            print(f"[{n_done}/{len(jobs)}] {item_id} s{sample}", flush=True)
        if args.limit and n_done >= args.limit:
            print("limit reached, stopping")
            break


if __name__ == "__main__":
    main()
