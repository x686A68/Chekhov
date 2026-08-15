"""Generate images for Family 1 S-candidates with Nano Banana (Gemini image API).

Pilot default: 20 stratified items (10 plausible / 10 implausible) x 2 samples,
gemini-3.1-flash-image, 1K, 1:1, thinking_level=minimal.

Design notes (mirror of the conversation that settled them):
- No app-side prompt rewriting: we call the image model directly.
- The API exposes a `seed` field but the docs do not promise the image model honors
  it; we pass a deterministic seed per (item, sample) and record it, so honoring can
  be checked empirically (same prompt+seed twice -> near-identical images?).
- Safety blocks and text-only responses are recorded in the manifest, never silently
  dropped (a blocked prompt is missing data, not a suppressed elephant).
- Resumable: existing (item, sample) outputs are skipped; the manifest is append-only.

Usage:
    GEMINI_API_KEY=... .venv/bin/python scripts/run_images_nanobanana.py            # pilot
    GEMINI_API_KEY=... .venv/bin/python scripts/run_images_nanobanana.py --n-items 0  # all items

Cost at defaults: 40 images x ~$0.067 (1K) ~= $2.7.
"""
import argparse
import json
import os
import random
import time

from google import genai
from google.genai import types
from google.genai.errors import APIError

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
ITEMS = os.path.join(BASE, "dataset", "family1", "f1_S_candidates_v1.jsonl")
SEED = 20260814


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default=ITEMS)
    ap.add_argument("--out-dir", default=os.path.join(BASE, "dataset", "family1", "images", "nb2_pilot"))
    ap.add_argument("--model", default="gemini-3.1-flash-image")
    ap.add_argument("--n-items", type=int, default=20, help="stratified subset size; 0 = all")
    ap.add_argument("--samples", type=int, default=2, help="independent samples per prompt")
    ap.add_argument("--image-size", default="1K")
    ap.add_argument("--aspect", default="1:1")
    ap.add_argument("--thinking", default="minimal", help="thinking_level; 'off' to omit")
    ap.add_argument("--max-retries", type=int, default=5)
    return ap.parse_args()


def select_items(path, n):
    items = [json.loads(l) for l in open(path)]
    if not n or n >= len(items):
        return items
    rng = random.Random(SEED)
    half = n // 2
    out = []
    for bin_name, k in (("plausible", half), ("implausible", n - half)):
        pool = [i for i in items if i["plausibility"] == bin_name]
        out.extend(rng.sample(pool, k))
    return sorted(out, key=lambda i: i["id"])


def build_config(args, seed):
    cfg = dict(
        response_modalities=["IMAGE"],
        image_config=types.ImageConfig(aspect_ratio=args.aspect, image_size=args.image_size),
        seed=seed,
        candidate_count=1,
    )
    if args.thinking != "off":
        cfg["thinking_config"] = types.ThinkingConfig(thinking_level=args.thinking)
    return types.GenerateContentConfig(**cfg)


def call_once(client, args, prompt, seed):
    """Returns (status, image_bytes, mime, text, block_reason, model_version)."""
    config = build_config(args, seed)
    for attempt in range(args.max_retries):
        try:
            resp = client.models.generate_content(model=args.model, contents=prompt, config=config)
        except APIError as e:
            transient = e.code in (429, 500, 502, 503, 504)
            modality_issue = e.code == 400 and "modalit" in str(e).lower()
            if modality_issue and config.response_modalities == ["IMAGE"]:
                config.response_modalities = ["TEXT", "IMAGE"]
                continue
            if transient and attempt < args.max_retries - 1:
                time.sleep(2 ** attempt * 2)
                continue
            return "error", None, None, f"{e.code}: {e.message}"[:500], None, None
        fb = getattr(resp, "prompt_feedback", None)
        if fb and getattr(fb, "block_reason", None):
            return "blocked", None, None, None, str(fb.block_reason), None
        text, img, mime = [], None, None
        for cand in resp.candidates or []:
            for part in (cand.content.parts or []) if cand.content else []:
                if getattr(part, "inline_data", None) and part.inline_data.data:
                    img, mime = part.inline_data.data, part.inline_data.mime_type
                elif getattr(part, "text", None):
                    text.append(part.text)
        version = getattr(resp, "model_version", None)
        if img is None:
            reason = str(resp.candidates[0].finish_reason) if resp.candidates else "no candidates"
            return "no_image", None, None, " ".join(text)[:500] or reason, reason, version
        return "ok", img, mime, " ".join(text)[:500] or None, None, version
    return "error", None, None, "retries exhausted", None, None


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    manifest_path = os.path.join(args.out_dir, "manifest.jsonl")
    done = set()
    if os.path.exists(manifest_path):
        for l in open(manifest_path):
            r = json.loads(l)
            if r["status"] == "ok":
                done.add((r["item_id"], r["sample"]))

    client = genai.Client()  # GEMINI_API_KEY from env
    items = select_items(args.items, args.n_items)
    total = len(items) * args.samples
    print(f"{len(items)} items x {args.samples} samples = {total} images -> {args.out_dir}")

    n_ok = n_bad = 0
    with open(manifest_path, "a") as mf:
        for it in items:
            for s in range(args.samples):
                if (it["id"], s) in done:
                    continue
                seed = SEED + hash((it["id"], s)) % 10**6
                t0 = time.time()
                status, img, mime, text, reason, version = call_once(client, args, it["prompt"], seed)
                path = None
                if status == "ok":
                    ext = "png" if mime and "png" in mime else "jpg"
                    path = os.path.join(args.out_dir, f"{it['id']}_s{s}.{ext}")
                    with open(path, "wb") as f:
                        f.write(img)
                    n_ok += 1
                else:
                    n_bad += 1
                mf.write(json.dumps(dict(
                    item_id=it["id"], sample=s, status=status, image_path=path,
                    prompt=it["prompt"], target=it["target"], plausibility=it["plausibility"],
                    model=args.model, model_version=version, seed=seed,
                    image_size=args.image_size, aspect=args.aspect, thinking=args.thinking,
                    text=text, block_reason=reason,
                    latency_s=round(time.time() - t0, 1), ts=int(time.time()),
                )) + "\n")
                mf.flush()
                print(f"{it['id']} s{s}: {status}" + (f" ({reason})" if reason else ""))
    print(f"done: {n_ok} ok, {n_bad} failed/blocked; manifest: {manifest_path}")


if __name__ == "__main__":
    main()
