"""OverReal annotation Space (Gradio).

Double-blind labeling: each image is labeled independently by two annotators;
nobody ever sees another person's labels. Annotators open a personal link
(?annotator=Name&key=secret); keys live in the ANNOTATOR_KEYS Space secret.

Dynamic assignment, at request time:
  1. never an image the requester already labeled, never one with 2 labels
  2. prefer images that already have exactly one label (by someone else)
  3. otherwise a fresh image; random tie-break; a soft 10-minute hold keeps
     two people off the same fresh image when alternatives exist

Results append to a per-session JSONL under ann_data/ and a CommitScheduler
pushes them to the private annotations dataset every 2 minutes. On startup
all previous shards are pulled, so restarts lose nothing.
"""
import json
import os
import random
import time
import uuid
from pathlib import Path

import urllib.parse

import gradio as gr
from PIL import Image
from huggingface_hub import CommitScheduler, hf_hub_download, snapshot_download

IMG_REPO = "huangjh16/overreal-annotation-images"
ANN_REPO = "huangjh16/overreal-annotations"
LABELS = ["disruptive", "silent", "integrated", "withheld", "other"]
TOKEN = os.environ["HF_TOKEN"]
KEYS = json.loads(os.environ["ANNOTATOR_KEYS"])  # {name: secret}

tasks_path = hf_hub_download(IMG_REPO, "tasks.jsonl", repo_type="dataset", token=TOKEN)
TASKS = {}
for line in open(tasks_path, encoding="utf-8"):
    r = json.loads(line)
    TASKS[r["image_id"]] = r

# rebuild state from every shard previously committed
state = {}   # image_id -> {annotator: [labels]}
try:
    snap = snapshot_download(ANN_REPO, repo_type="dataset", token=TOKEN,
                             allow_patterns=["*.jsonl"])
    for f in sorted(Path(snap).rglob("*.jsonl")):
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            state.setdefault(r["image_id"], {})[r["annotator"]] = r["labels"]
except Exception as e:  # noqa: BLE001 — empty repo on first boot
    print("no previous annotations:", e)

ann_dir = Path("ann_data")
ann_dir.mkdir(exist_ok=True)
out_file = ann_dir / f"ann-{uuid.uuid4().hex[:8]}.jsonl"
scheduler = CommitScheduler(repo_id=ANN_REPO, repo_type="dataset",
                            folder_path=ann_dir, every=2, token=TOKEN)

holds = {}  # image_id -> (annotator, ts)


GUIDE = """
### Label guide

- **disruptive** — the target appears, and its appearance causes at least one of:
  (1) the image breaks common sense or logic — e.g. a reader facing the *back* of
  the letter they are reading, a screen facing the camera instead of its user, a
  tiger on a sports field; (2) the image contradicts the prompt — e.g. the prompt
  says the target is absent, or sealed inside an opaque box, yet it is visible.
- **silent** — the target appears; unexpected, but not disruptive — typically as
  background or a small detail. E.g. *"her face is like a red apple"* → a
  red-faced child shown **eating an apple**; *"the roof is like a banana"* →
  banana **trees in the background**.
- **integrated** — the target appears, but in a natural way, within what a normal
  viewer would accept. E.g. *"the roof is like a banana"* → a banana-shaped roof
  (and no banana trees); a thought bubble depicting what a character imagines;
  contents shown inside a closed drawer as such; a poster style that merges a
  person with their attribute.
- **withheld** — the target does **not** appear, and the image does not
  contradict the prompt. E.g. something is in a closed bag → only the closed bag
  is shown; *"her face is like a red apple"* → just a red-faced child.
- **other** — the image is severely corrupted, or does not match the prompt at all.

You may select two labels, but **please prefer a single label** — pick two only
when the case is genuinely ambiguous.
"""

def auth(qs, request: gr.Request = None):
    q = {k: v[0] for k, v in urllib.parse.parse_qs((qs or "").lstrip("?")).items()}
    if not q and request is not None:
        q = dict(request.query_params)
    name = q.get("annotator")
    return name if name in KEYS and q.get("key") == KEYS[name] else None


def progress_text(name):
    mine = sum(1 for a in state.values() if name in a)
    filled = sum(min(len(a), 2) for a in state.values())
    return f"**{name}** — you: {mine} · overall: {filled}/{2 * len(TASKS)}"


def pick(name):
    now = time.time()
    fresh, half = [], []
    for iid in TASKS:
        anns = state.get(iid, {})
        if name in anns or len(anns) >= 2:
            continue
        h = holds.get(iid)
        if h and h[0] != name and now - h[1] < 600:
            continue
        (half if len(anns) == 1 else fresh).append(iid)
    pool = half or fresh
    if not pool:  # everything held or done: retry ignoring holds
        pool = [i for i in TASKS
                if name not in state.get(i, {}) and len(state.get(i, {})) < 2]
    random.shuffle(pool)
    for iid in pool[:20]:  # image may not be uploaded yet — skip and try next
        try:
            path = hf_hub_download(IMG_REPO, TASKS[iid]["file"],
                                   repo_type="dataset", token=TOKEN)
            img = Image.open(path)
            img.load()
            holds[iid] = (name, now)
            return iid, img
        except Exception:  # noqa: BLE001
            continue
    return None, None


def serve(qs, request: gr.Request):
    name = auth(qs, request)
    if not name:
        return (gr.update(visible=False), gr.update(visible=True),
                None, "", "", "", [], "")
    iid, path = pick(name)
    if iid is None:
        return (gr.update(visible=True), gr.update(visible=False),
                None, "", "", progress_text(name) + " — no tasks left, thank you!", [], "")
    t = TASKS[iid]
    return (gr.update(visible=True), gr.update(visible=False),
            path, f"**Prompt:** {t['prompt']}", f"**Target:** {t['target']}",
            progress_text(name), [], iid)


def cap_two(labels):
    return labels[-2:] if len(labels) > 2 else labels


def submit(labels, iid, qs, request: gr.Request):
    name = auth(qs, request)
    if not name or iid not in TASKS:
        return serve(qs, request)
    if not labels:
        gr.Warning("Pick at least one label.")
        return tuple(gr.skip() for _ in range(8))
    rec = {"image_id": iid, "annotator": name, "labels": labels,
           "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    with scheduler.lock:
        with open(out_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    state.setdefault(iid, {})[name] = labels
    holds.pop(iid, None)
    return serve(qs, request)


with gr.Blocks(title="OverReal annotation") as demo:
    with gr.Column(visible=False) as main:
        prog = gr.Markdown()
        with gr.Accordion("Label guide — read me first", open=True):
            gr.Markdown(GUIDE)
        img = gr.Image(type="pil", height=560, interactive=False,
                       show_download_button=False)
        prompt_md = gr.Markdown()
        target_md = gr.Markdown()
        labels_in = gr.CheckboxGroup(LABELS, label="Labels (pick 1–2)")
        submit_btn = gr.Button("Submit & next", variant="primary")
        skip_btn = gr.Button("Skip (come back later)")
    with gr.Column(visible=True) as denied:
        gr.Markdown("### Invalid or missing link\nPlease use your personal "
                    "annotation link, or contact Jiahao.")

    iid_state = gr.State("")
    qs_box = gr.Textbox(visible=False)
    outs = [main, denied, img, prompt_md, target_md, prog, labels_in, iid_state]
    demo.load(None, None, qs_box, js="() => window.location.search")
    qs_box.change(serve, qs_box, outs)
    labels_in.change(cap_two, labels_in, labels_in)
    submit_btn.click(submit, [labels_in, iid_state, qs_box], outs)
    skip_btn.click(serve, qs_box, outs)

demo.launch(ssr_mode=False, show_error=True)
