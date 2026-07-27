"""Second image generator: Qwen-Image.

Two reasons to add it. First, family 5 returned 0.00 under two different constructions
and the diagnosis was that FLUX.1-dev writes pseudo-text — it makes no lexical commitment
unless told exactly which word to print, so the use-mention choice never arises. Qwen-Image
renders text far more reliably, which is the property family 5 needs. Second, every
image-side result so far rests on one generator; a second one turns "FLUX does this" into
a claim with at least one point of comparison.

Same items, same conditions, same seeds as run_images.py, so the two generators are
directly comparable image for image.

Usage: CUDA_VISIBLE_DEVICES=5 python scripts/run_images_qwen.py --families 5_use_mention,...
"""
import argparse
import json
import os
import time
import zlib

os.environ.setdefault("HF_HOME", "/data/users/jiahao_huang/hf")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ITEMS = os.path.join(ROOT, "pilot", "items")
OUT = os.path.join(ROOT, "pilot", "images_qwen")

MODEL = "Qwen/Qwen-Image"


def load_items(families):
    rows = []
    for fn in sorted(os.listdir(ITEMS)):
        if fn.endswith(".jsonl"):
            with open(os.path.join(ITEMS, fn)) as f:
                rows += [json.loads(l) for l in f if l.strip()]
    if families:
        keep = set(families.split(","))
        rows = [r for r in rows if r["family"] in keep]
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", default="")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--guidance", type=float, default=4.0)
    ap.add_argument("--manifest", default="manifest.jsonl")
    ap.add_argument("--reverse", action="store_true")
    args = ap.parse_args()

    import torch
    from diffusers import DiffusionPipeline

    items = load_items(args.families)
    todo = [(it, c) for it in items for c in it["image_prompts"]]
    if args.reverse:
        todo = todo[::-1]
    print(f"{len(items)} items -> {len(todo)} images", flush=True)

    t0 = time.time()
    pipe = DiffusionPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to("cuda")
    print(f"pipeline loaded in {time.time()-t0:.0f}s", flush=True)

    os.makedirs(OUT, exist_ok=True)
    manifest_path = os.path.join(OUT, args.manifest)

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
        print(f"resuming, {len(done)} already present", flush=True)

    times = []
    with open(manifest_path, "a") as mf:
        for i, (it, cond) in enumerate(todo):
            if (it["id"], cond) in done:
                continue
            d = os.path.join(OUT, it["family"])
            os.makedirs(d, exist_ok=True)
            path = os.path.join(d, f"{it['id']}_{cond}.png")
            prompt = it["image_prompts"][cond]
            seed = zlib.crc32(it["id"].encode()) % (2**31)  # same seed as the FLUX run
            t1 = time.time()
            img = pipe(prompt=prompt, negative_prompt=" ", num_inference_steps=args.steps,
                       true_cfg_scale=args.guidance, height=args.size, width=args.size,
                       generator=torch.Generator("cuda").manual_seed(seed)).images[0]
            dt = time.time() - t1
            times.append(dt)
            img.save(path)
            mf.write(json.dumps({
                "id": it["id"], "family": it["family"], "entity": it["entity"],
                "scenario_id": it["scenario_id"], "scenario": it["scenario"],
                "device": it["device"], "condition": cond, "prompt": prompt,
                "generator": "Qwen-Image",
                "image_target": it.get("image_target", "referent"),
                "path": os.path.relpath(path, ROOT), "seed": seed, "steps": args.steps,
                "size": args.size, "guidance": args.guidance, "gen_s": round(dt, 2),
            }, ensure_ascii=False) + "\n")
            mf.flush()
            if i % 10 == 0:
                done |= already_done()
                print(f"[{i+1}/{len(todo)}] {dt:.1f}s {it['id']}_{cond}", flush=True)

    if times:
        cost = {"model": MODEL, "n_images": len(times),
                "mean_s_per_image": round(sum(times) / len(times), 2),
                "total_s": round(sum(times), 1), "steps": args.steps, "size": args.size,
                "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1)}
        with open(os.path.join(OUT, "cost.json"), "w") as f:
            json.dump(cost, f, indent=2)
        print("COST", json.dumps(cost), flush=True)


if __name__ == "__main__":
    main()
