"""Score overreal_v1 images with conventional T2I metrics.

Scope: every metadata row with an image file (generated and marathon alike),
scored against the item's raw prompt. One metric per process:

  CUDA_VISIBLE_DEVICES=0 python run_conventional.py --metric clipscore
  CUDA_VISIBLE_DEVICES=1 python run_conventional.py --metric pickscore
  CUDA_VISIBLE_DEVICES=2 python run_conventional.py --metric hpsv2

Output: data/overreal_v1/metrics/<metric>.jsonl, one {image_id, score} line
per image; already-scored ids are skipped, so runs resume.

CLIPScore follows Hessel et al. 2021 (2.5 * max(cos, 0)) with CLIP ViT-L/14.
PickScore is the raw preference logit of PickScore_v1. HPSv2 uses the hpsv2
package's v2.1 checkpoint. (VQAScore runs from its own env; the LLM judge is
deferred.)
"""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("HF_HOME", "/data/users/jiahao_huang/hf")

ROOT = Path(__file__).resolve().parents[4]
DS = ROOT / "data" / "overreal_v1"
OUT_DIR = DS / "metrics"


def load_rows():
    rows = []
    for line in open(DS / "metadata.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if r.get("file_name") and not r.get("refused") and r.get("prompt"):
            rows.append((r["image_id"], str(DS / r["file_name"]), r["prompt"]))
    return rows


def run_clipscore(todo, emit):
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14",
                                      torch_dtype=torch.float16).to("cuda").eval()
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    B = 32
    with torch.no_grad():
        for i in range(0, len(todo), B):
            batch = todo[i:i + B]
            imgs = [Image.open(p).convert("RGB") for _, p, _ in batch]
            inp = proc(text=[t for _, _, t in batch], images=imgs,
                       return_tensors="pt", padding=True, truncation=True).to("cuda")
            out = model(**inp)
            ie = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
            te = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
            cos = (ie * te).sum(-1)
            for (iid, _, _), c in zip(batch, cos):
                emit(iid, 2.5 * max(float(c), 0.0))


def run_pickscore(todo, emit):
    import torch
    from PIL import Image
    from transformers import AutoModel, AutoProcessor
    model = AutoModel.from_pretrained("yuvalkirstain/PickScore_v1",
                                      torch_dtype=torch.float16).to("cuda").eval()
    proc = AutoProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
    B = 16
    with torch.no_grad():
        for i in range(0, len(todo), B):
            batch = todo[i:i + B]
            imgs = [Image.open(p).convert("RGB") for _, p, _ in batch]
            inp = proc(text=[t for _, _, t in batch], images=imgs,
                       return_tensors="pt", padding=True, truncation=True,
                       max_length=77).to("cuda")
            out = model(**inp)
            ie = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
            te = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
            score = model.logit_scale.exp() * (ie * te).sum(-1)
            for (iid, _, _), c in zip(batch, score):
                emit(iid, float(c))


def run_hpsv2(todo, emit):
    import hpsv2
    from PIL import Image
    for iid, path, prompt in todo:
        img = Image.open(path).convert("RGB")
        s = hpsv2.score(img, prompt, hps_version="v2.1")
        emit(iid, float(s[0]))


RUNNERS = {"clipscore": run_clipscore, "pickscore": run_pickscore,
           "hpsv2": run_hpsv2}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", required=True, choices=list(RUNNERS))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    OUT_DIR.mkdir(exist_ok=True)
    out_path = OUT_DIR / f"{args.metric}.jsonl"
    done = set()
    if out_path.exists():
        done = {json.loads(l)["image_id"] for l in open(out_path, encoding="utf-8")}
    rows = [r for r in load_rows() if r[0] not in done]
    if args.limit:
        rows = rows[:args.limit]
    print(f"{args.metric}: {len(rows)} images to score ({len(done)} done)", flush=True)

    f = open(out_path, "a", encoding="utf-8")
    n = 0

    def emit(iid, score):
        nonlocal n
        f.write(json.dumps({"image_id": iid, "score": score}) + "\n")
        n += 1
        if n % 500 == 0:
            f.flush()
            print(f"[{n}/{len(rows)}]", flush=True)

    RUNNERS[args.metric](rows, emit)
    f.close()
    print(f"{args.metric}: wrote {n} scores", flush=True)


if __name__ == "__main__":
    main()
