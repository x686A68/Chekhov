"""GPT-Image via the OpenAI Batch API (50% off, 24h completion window).

  python batch_openai.py submit    # build request file for pending images, upload, start
  python batch_openai.py status    # poll the active batch
  python batch_openai.py fetch     # download results, write images + manifest lines

State lives in data/generation/batches/gpt-image.json (batch id + file ids).
custom_id encodes item and sample: "<item_id>|s<sample>". Failed lines are
reported by fetch; a fresh submit picks up whatever the manifest still lacks.
"""
import argparse
import base64
import datetime
import json
import sys

from common import (GEN, append_jsonl, done_keys, image_path, load_env,
                    load_prompts, manifest_path)

MODEL_ID = "gpt-image-1.5"
QUALITY = "medium"
N_SAMPLES = 2
COND = "deployed"
STATE = GEN / "batches" / "gpt-image.json"


def client():
    load_env()
    from openai import OpenAI
    return OpenAI()


def submit():
    c = client()
    done = done_keys("gpt-image", COND)
    jobs = [(i, p, s) for i, _, p in load_prompts("raw")
            for s in range(N_SAMPLES) if (i, s) not in done]
    if not jobs:
        print("nothing to submit"); return
    req = GEN / "batches" / "gpt-image_requests.jsonl"
    req.parent.mkdir(parents=True, exist_ok=True)
    with open(req, "w", encoding="utf-8") as f:
        for item_id, prompt, sample in jobs:
            f.write(json.dumps({
                "custom_id": f"{item_id}|s{sample}",
                "method": "POST",
                "url": "/v1/images/generations",
                "body": {"model": MODEL_ID, "prompt": prompt, "n": 1,
                         "size": "1024x1024", "quality": QUALITY},
            }, ensure_ascii=False) + "\n")
    up = c.files.create(file=open(req, "rb"), purpose="batch")
    batch = c.batches.create(input_file_id=up.id,
                             endpoint="/v1/images/generations",
                             completion_window="24h")
    STATE.write_text(json.dumps({"batch_id": batch.id, "input_file_id": up.id,
                                 "n_requests": len(jobs)}))
    print(f"submitted batch {batch.id} with {len(jobs)} requests")


def status():
    c = client()
    b = c.batches.retrieve(json.loads(STATE.read_text())["batch_id"])
    print(b.status, b.request_counts)


def fetch():
    c = client()
    st = json.loads(STATE.read_text())
    b = c.batches.retrieve(st["batch_id"])
    if b.status != "completed":
        sys.exit(f"batch is {b.status}, not completed")
    prompts = {i: (f, p) for i, f, p in load_prompts("raw")}
    done = done_keys("gpt-image", COND)
    n_ok = n_fail = 0
    for line in c.files.content(b.output_file_id).text.splitlines():
        r = json.loads(line)
        item_id, s = r["custom_id"].rsplit("|s", 1)
        sample = int(s)
        if (item_id, sample) in done:
            continue
        body = (r.get("response") or {}).get("body") or {}
        if r.get("error") or not body.get("data"):
            print(f"[FAIL] {r['custom_id']}: {r.get('error') or body.get('error')}")
            n_fail += 1
            continue
        family, prompt = prompts[item_id]
        out = image_path("gpt-image", COND, item_id, sample)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(base64.b64decode(body["data"][0]["b64_json"]))
        append_jsonl(manifest_path("gpt-image", COND), {
            "item_id": item_id, "family": family, "model": "gpt-image",
            "api_model_id": MODEL_ID, "cond": COND, "seed": sample,
            "size": "1024x1024", "quality": QUALITY, "prompt": prompt,
            "batch_id": st["batch_id"],
            "file": str(out.relative_to(out.parents[4])),
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        })
        n_ok += 1
    print(f"fetched {n_ok} images, {n_fail} failed"
          + (" — rerun submit for the failures" if n_fail else ""))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["submit", "status", "fetch"])
    {"submit": submit, "status": status, "fetch": fetch}[ap.parse_args().cmd]()
