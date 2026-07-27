"""Phase 2 — image pilot with FLUX.1-dev.

Every item is generated under all three conditions with the *same seed*, so S, P and A
differ only in the licensing device — the image-side counterpart of GOAL.md rule 1.
GOAL.md asks for S and P only (120 images); A is added because rule 3 (the coincidental
base rate is both a validity check and a filter) applies to the image modality too.

Images are written as they are produced, one PNG per (item, condition), with a
manifest line each. Per-image wall-clock and peak VRAM are recorded for the budget
projection.

Usage: CUDA_VISIBLE_DEVICES=5 python scripts/run_images.py [--conditions SPA] [--steps 50]
"""
import argparse
import json
import os
import time
import zlib

os.environ.setdefault("HF_HOME", "/data/users/jiahao_huang/hf")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ITEMS = os.path.join(ROOT, "pilot", "items")
OUT = os.path.join(ROOT, "pilot", "images")

MODEL = "black-forest-labs/FLUX.1-dev"


def load_items():
    rows = []
    for fn in sorted(os.listdir(ITEMS)):
        if fn.endswith(".jsonl"):
            with open(os.path.join(ITEMS, fn)) as f:
                rows += [json.loads(l) for l in f if l.strip()]
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", default="SPA")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--guidance", type=float, default=3.5)
    # A second worker on the other GPU walks the same list backwards and appends to its
    # own manifest; both re-read every manifest periodically, so they meet in the middle
    # with at most a couple of duplicated images (same seed, so byte-identical anyway).
    ap.add_argument("--manifest", default="manifest.jsonl")
    ap.add_argument("--reverse", action="store_true")
    args = ap.parse_args()

    import torch
    from diffusers import FluxPipeline

    items = load_items()
    todo = [(it, c) for it in items for c in args.conditions]
    print(f"{len(items)} items x {len(args.conditions)} conditions = {len(todo)} images", flush=True)

    t0 = time.time()
    pipe = FluxPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to("cuda")
    load_s = time.time() - t0
    print(f"pipeline loaded in {load_s:.0f}s", flush=True)

    manifest_path = os.path.join(OUT, args.manifest)
    os.makedirs(OUT, exist_ok=True)
    if args.reverse:
        todo = todo[::-1]

    def already_done():
        seen = set()
        for fn in os.listdir(OUT):
            if fn.startswith("manifest") and fn.endswith(".jsonl"):
                with open(os.path.join(OUT, fn)) as f:
                    for l in f:
                        if l.strip():
                            r = json.loads(l)
                            seen.add((r["id"], r["condition"]))
        return seen

    done = already_done()
    if done:
        print(f"resuming, {len(done)} images already present", flush=True)

    times = []
    with open(manifest_path, "a") as mf:
        for i, (it, cond) in enumerate(todo):
            if (it["id"], cond) in done:
                continue
            d = os.path.join(OUT, it["family"])
            os.makedirs(d, exist_ok=True)
            path = os.path.join(d, f"{it['id']}_{cond}.png")
            prompt = it["image_prompts"][cond]
            # stable across processes and resumed runs, and identical for S/P/A
            seed = zlib.crc32(it["id"].encode()) % (2**31)
            t1 = time.time()
            img = pipe(prompt, num_inference_steps=args.steps, guidance_scale=args.guidance,
                       height=args.size, width=args.size,
                       generator=torch.Generator("cuda").manual_seed(seed)).images[0]
            dt = time.time() - t1
            times.append(dt)
            img.save(path)
            mf.write(json.dumps({
                "id": it["id"], "family": it["family"], "entity": it["entity"],
                "scenario_id": it["scenario_id"], "scenario": it["scenario"],
                "device": it["device"], "condition": cond, "prompt": prompt,
                "image_target": it.get("image_target", "referent"),
                "path": os.path.relpath(path, ROOT), "seed": seed, "steps": args.steps,
                "size": args.size, "guidance": args.guidance, "gen_s": round(dt, 2),
            }, ensure_ascii=False) + "\n")
            mf.flush()
            if i % 10 == 0:
                done |= already_done()
            if i % 10 == 0:
                print(f"[{i+1}/{len(todo)}] {dt:.1f}s {it['id']}_{cond}", flush=True)

    cost = {
        "model": MODEL, "n_images": len(times), "load_s": round(load_s, 1),
        "mean_s_per_image": round(sum(times) / len(times), 2) if times else None,
        "total_s": round(sum(times), 1), "steps": args.steps, "size": args.size,
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1),
    }
    with open(os.path.join(OUT, "cost.json"), "w") as f:
        json.dump(cost, f, indent=2)
    print("COST", json.dumps(cost), flush=True)


if __name__ == "__main__":
    main()
