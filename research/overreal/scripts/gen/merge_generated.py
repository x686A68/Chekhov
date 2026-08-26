"""Merge generated images (data/generation/) into overreal_v1.

Runs after build_overreal.py (+ add_cancellation_to_overreal.py). Idempotent:
rows already in metadata.jsonl are skipped, images are copied only if absent,
so it can be re-run as Phase 2 lanes finish.

Layout : images/<family>/prompt_NNNN/gen_<model>_<cond>_s<seed>.<ext>
Schema : same keys as the marathon rows. Generated rows have label_1/label_2
         [], agreement/included null, annotated false. Refusals (a lane that
         will never produce the image) get refused=true rows with no file.

--finalize-lane <model>_<cond> may be passed repeatedly: for those lanes the
missing (item, seed) pairs are recorded as refusals. Only finalize a lane once
its run is complete and remaining gaps are confirmed persistent.
"""
import argparse
import datetime
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import GEN, PROMPTS, read_jsonl  # noqa: E402

DS = GEN.parent / "overreal_v1"
META = DS / "metadata.jsonl"

CANONICAL = {
    "flux": "flux.1-dev",
    "qwen-image": "qwen-image",
    "omnigen2": "omnigen2",
    "sd35m": "sd3.5-medium",
    "sd35l": "sd3.5-large",
    "gpt-image": "gpt-image-1.5",
    "ideogram": "ideogram-v3",
    "nanobanana": "gemini-2.5-flash-image-api",
}
SAMPLES = {("gpt-image", "deployed"): 2, ("ideogram", "deployed"): 1,
           ("nanobanana", "deployed"): 1}          # local lanes: 2 (default)
REFUSAL_REASON = {"gpt-image": "safety_rejection",
                  "nanobanana": "text_only_reply"}


def gen_params(model, r):
    if model in ("gpt-image",):
        return {"api_model_id": r.get("api_model_id"), "quality": r.get("quality"),
                "size": r.get("size")}
    if model == "ideogram":
        return {"rendering_speed": r.get("rendering_speed"),
                "api_seed": r.get("api_seed")}
    if model == "nanobanana":
        return {"api_model_id": r.get("api_model_id")}
    return {"steps": r.get("steps"), "cfg": r.get("cfg"), "size": r.get("size")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--finalize-lane", action="append", default=[],
                    metavar="MODEL_COND", help="record missing pairs as refusals")
    args = ap.parse_args()

    prompts = {r["item_id"]: r for r in read_jsonl(PROMPTS)}
    existing = {json.loads(l)["image_id"] for l in open(META, encoding="utf-8")}
    template_keys = list(json.loads(open(META, encoding="utf-8").readline()).keys())

    n_new = n_skip = n_refused = 0
    out_rows = []
    for mf in sorted((GEN / "manifests").glob("*.jsonl")):
        model, cond = mf.stem.rsplit("_", 1)
        seen = set()
        for r in read_jsonl(mf):
            item_id, seed = r["item_id"], r["seed"]
            seen.add((item_id, seed))
            image_id = f"{item_id}/gen_{model}_{cond}_s{seed}"
            if image_id in existing:
                n_skip += 1
                continue
            src = GEN.parent / r["file"]
            ext = src.suffix.lower()
            fam, pid = item_id.split("/")
            rel = f"images/{fam}/{pid}/gen_{model}_{cond}_s{seed}{ext}"
            dst = DS / rel
            if not dst.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
            p = prompts[item_id]
            out_rows.append({
                "file_name": rel, "image_id": image_id, "family": fam,
                "item_id": item_id, "source_folder": None, "annotator": None,
                "prompt": p["prompt"], "target": p["target"],
                "label_1": [], "label_2": [], "agreement": None,
                "included": None, "generator": CANONICAL[model],
                "prompt_cond": cond, "input_prompt": r["prompt"],
                "seed": seed, "gen_params": gen_params(model, r),
                "annotated": False, "refused": False, "raw_path": r["file"],
            })
            existing.add(image_id)
            n_new += 1

        if mf.stem in args.finalize_lane:
            n_samples = SAMPLES.get((model, cond), 2)
            for item_id, p in prompts.items():
                for seed in range(n_samples):
                    if (item_id, seed) in seen:
                        continue
                    image_id = f"{item_id}/gen_{model}_{cond}_s{seed}"
                    if image_id in existing:
                        continue
                    fam = item_id.split("/")[0]
                    out_rows.append({
                        "file_name": None, "image_id": image_id, "family": fam,
                        "item_id": item_id, "source_folder": None, "annotator": None,
                        "prompt": p["prompt"], "target": p["target"],
                        "label_1": [], "label_2": [], "agreement": None,
                        "included": None, "generator": CANONICAL[model],
                        "prompt_cond": cond, "input_prompt": p["prompt"],
                        "seed": seed, "gen_params": None,
                        "annotated": False, "refused": True,
                        "refusal_reason": REFUSAL_REASON.get(model, "unknown"),
                        "raw_path": None,
                    })
                    existing.add(image_id)
                    n_refused += 1

    for row in out_rows:
        assert [k for k in row if k != "refusal_reason"] == template_keys, \
            (template_keys, list(row))
    with open(META, "a", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"appended {n_new} image rows, {n_refused} refusal rows "
          f"({n_skip} already present)")


if __name__ == "__main__":
    main()
