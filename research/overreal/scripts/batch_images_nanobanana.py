"""Full Family 1 image run via the Gemini Batch API (50% of standard price).

Workflow (each step is a subcommand; state lives next to the outputs):

    build   items JSONL -> batch request JSONL (500 items x 2 samples = 1000 requests,
            same per-(item,sample) seeds as the pilot runner; --exclude-manifest skips
            (item,sample) pairs already generated, e.g. the pilot's 31)
    submit  upload the JSONL, create the batch job, save its name to job.json
    status  print the job state (target turnaround < 24 h)
    fetch   download results, write images + manifest.jsonl (same schema as the pilot
            runner, plus batch job metadata), print a summary

Config matches the pilot exactly: gemini-3.1-flash-image, 1K, 1:1,
thinking_level=minimal, per-request deterministic seed.

Usage:
    V=.venv/bin/python
    GEMINI_API_KEY=... $V scripts/batch_images_nanobanana.py build --exclude-manifest \
        dataset/family1/images/nb2_pilot/manifest.jsonl
    GEMINI_API_KEY=... $V scripts/batch_images_nanobanana.py submit
    GEMINI_API_KEY=... $V scripts/batch_images_nanobanana.py status
    GEMINI_API_KEY=... $V scripts/batch_images_nanobanana.py fetch
"""
import argparse
import base64
import json
import os
import time
import zlib

from google import genai
from google.genai import types

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
ITEMS = os.path.join(BASE, "dataset", "family1", "f1_S_candidates_v1.jsonl")
OUT_DIR = os.path.join(BASE, "dataset", "family1", "images", "nb2_full")
REQUESTS = os.path.join(OUT_DIR, "batch_requests.jsonl")
JOB_FILE = os.path.join(OUT_DIR, "job.json")
MANIFEST = os.path.join(OUT_DIR, "manifest.jsonl")

MODEL = "gemini-3.1-flash-image"
SEED = 20260814
SAMPLES = 2


def sample_seed(item_id, s):
    # stable across processes (the pilot runner used Python's salted hash(), so its
    # seeds are reproducible only via its manifest; from here on crc32 is canonical)
    return SEED + zlib.crc32(f"{item_id}/{s}".encode()) % 10**6


def gen_config(seed):
    cfg = types.GenerateContentConfig(
        response_modalities=["IMAGE"],
        image_config=types.ImageConfig(aspect_ratio="1:1", image_size="1K"),
        thinking_config=types.ThinkingConfig(thinking_level="minimal"),
        seed=seed,
        candidate_count=1,
    )
    return cfg.model_dump(exclude_none=True, mode="json", by_alias=True)


def cmd_build(args):
    exclude = set()
    for mpath in args.exclude_manifest or []:
        for l in open(mpath):
            r = json.loads(l)
            if r["status"] == "ok":
                exclude.add((r["item_id"], r["sample"]))
    items = [json.loads(l) for l in open(args.items)]
    os.makedirs(OUT_DIR, exist_ok=True)
    n = 0
    with open(REQUESTS, "w") as f:
        for it in items:
            for s in range(SAMPLES):
                if (it["id"], s) in exclude:
                    continue
                f.write(json.dumps({
                    "key": f"{it['id']}_s{s}",
                    "request": {
                        "contents": [{"parts": [{"text": it["prompt"]}]}],
                        "generation_config": gen_config(sample_seed(it["id"], s)),
                    },
                }) + "\n")
                n += 1
    print(f"{n} requests -> {REQUESTS} (excluded {len(exclude)} already-generated)")


def cmd_submit(args):
    client = genai.Client()
    up = client.files.upload(
        file=REQUESTS,
        config=types.UploadFileConfig(display_name="f1_nb2_full", mime_type="jsonl"))
    job = client.batches.create(model=MODEL, src=up.name,
                                config={"display_name": "f1_nb2_full"})
    with open(JOB_FILE, "w") as f:
        json.dump({"job": job.name, "src": up.name, "ts": int(time.time())}, f)
    print(f"submitted: {job.name} (state {job.state.name}); saved to {JOB_FILE}")


def _get_job(client):
    return client.batches.get(name=json.load(open(JOB_FILE))["job"])


def cmd_status(args):
    job = _get_job(genai.Client())
    print(job.state.name, getattr(job, "error", None) or "")


def cmd_fetch(args):
    client = genai.Client()
    job = _get_job(client)
    if job.state.name != "JOB_STATE_SUCCEEDED":
        print(f"not ready: {job.state.name}")
        return
    items = {i["id"]: i for i in (json.loads(l) for l in open(ITEMS))}
    raw = client.files.download(file=job.dest.file_name)
    n_ok = n_bad = 0
    with open(MANIFEST, "w") as mf:
        for line in raw.decode("utf-8").splitlines():
            rec = json.loads(line)
            key = rec["key"]
            item_id, s = key.rsplit("_s", 1)
            it, s = items[item_id], int(s)
            status, path, text, reason = "no_image", None, None, None
            if "error" in rec:
                status, reason = "error", str(rec["error"])[:300]
            else:
                resp = rec.get("response", {})
                fb = resp.get("promptFeedback") or resp.get("prompt_feedback") or {}
                if fb.get("blockReason") or fb.get("block_reason"):
                    status = "blocked"
                    reason = str(fb.get("blockReason") or fb.get("block_reason"))
                texts = []
                for cand in resp.get("candidates", []):
                    for part in cand.get("content", {}).get("parts", []):
                        blob = part.get("inlineData") or part.get("inline_data")
                        if blob and blob.get("data"):
                            mime = blob.get("mimeType") or blob.get("mime_type") or ""
                            ext = "png" if "png" in mime else "jpg"
                            path = os.path.join(OUT_DIR, f"{key}.{ext}")
                            with open(path, "wb") as f:
                                f.write(base64.b64decode(blob["data"]))
                            status = "ok"
                        elif part.get("text"):
                            texts.append(part["text"])
                text = " ".join(texts)[:500] or None
            n_ok += status == "ok"
            n_bad += status != "ok"
            mf.write(json.dumps(dict(
                item_id=item_id, sample=s, status=status, image_path=path,
                prompt=it["prompt"], target=it["target"], plausibility=it["plausibility"],
                model=MODEL, seed=sample_seed(item_id, s), image_size="1K", aspect="1:1",
                thinking="minimal", text=text, block_reason=reason,
                batch_job=job.name, ts=int(time.time()),
            )) + "\n")
    print(f"fetched: {n_ok} ok, {n_bad} failed/blocked -> {MANIFEST}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--items", default=ITEMS)
    b.add_argument("--exclude-manifest", nargs="*")
    sub.add_parser("submit")
    sub.add_parser("status")
    sub.add_parser("fetch")
    args = ap.parse_args()
    dict(build=cmd_build, submit=cmd_submit, status=cmd_status, fetch=cmd_fetch)[args.cmd](args)


if __name__ == "__main__":
    main()
