"""Ideogram runner, deployed pipeline (Magic Prompt on). One image per prompt.

The response's final prompt text (Magic Prompt output) goes into the manifest
and into expanded_prompts.jsonl as the "ideogram" expander for Phase 2.
Image URLs expire, so files are downloaded immediately.

Usage: python run_ideogram.py [--limit 1]
"""
import argparse
import datetime
import os
import time

import requests

from common import (EXPANDED, append_jsonl, done_keys, image_path,
                    load_env, load_prompts, manifest_path, read_jsonl)

URL = "https://api.ideogram.ai/v1/ideogram-v3/generate"
COND = "deployed"
SPEED = "BALANCED"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    load_env()
    headers = {"Api-Key": os.environ["IDEOGRAM_API_KEY"]}

    prompts = load_prompts("raw")
    done = done_keys("ideogram", COND)
    jobs = [(i, f, p) for i, f, p in prompts if (i, 0) not in done]
    print(f"ideogram: {len(jobs)} images to go ({len(done)} done)")

    exp_done = ({r["item_id"] for r in read_jsonl(EXPANDED) if r["expander"] == "ideogram"}
                if EXPANDED.exists() else set())

    n_done = 0
    for item_id, family, prompt in jobs:
        t0 = time.time()
        body = {"prompt": prompt, "aspect_ratio": "1x1", "num_images": 1,
                "rendering_speed": SPEED, "magic_prompt": "ON"}
        try:
            rsp = requests.post(URL, headers=headers, json=body, timeout=120)
            rsp.raise_for_status()
            datum = rsp.json()["data"][0]
            img = requests.get(datum["url"], timeout=120)
            img.raise_for_status()
        except Exception as e:  # noqa: BLE001 — log and move on; rerun picks it up
            print(f"[FAIL] {item_id}: {type(e).__name__}: {str(e)[:200]}", flush=True)
            continue
        out = image_path("ideogram", COND, item_id, 0)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(img.content)
        magic = datum.get("prompt")
        append_jsonl(manifest_path("ideogram", COND), {
            "item_id": item_id, "family": family, "model": "ideogram",
            "cond": COND, "seed": 0, "rendering_speed": SPEED,
            "prompt": prompt, "magic_prompt": magic,
            "api_seed": datum.get("seed"), "safe": datum.get("is_image_safe"),
            "file": str(out.relative_to(out.parents[4])),
            "sec": round(time.time() - t0, 1),
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        })
        if magic and item_id not in exp_done:
            append_jsonl(EXPANDED, {
                "item_id": item_id, "family": family, "expander": "ideogram",
                "expander_model": f"ideogram-v3 magic_prompt ({SPEED})",
                "text": magic,
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            })
            exp_done.add(item_id)
        n_done += 1
        if n_done % 10 == 0 or n_done == len(jobs):
            print(f"[{n_done}/{len(jobs)}] {item_id}", flush=True)
        if args.limit and n_done >= args.limit:
            print("limit reached, stopping")
            break


if __name__ == "__main__":
    main()
