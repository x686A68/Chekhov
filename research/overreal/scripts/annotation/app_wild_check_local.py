"""OverReal in-the-wild human check (Gradio, local-only variant: no HF hosting).

Blind re-labeling of a stratified sample of judge-labeled in-the-wild images
(reviewer 3P6s, Q2). Same guide and labels as the OverReal-Det annotation;
the annotator never sees the judge's label, the generator or the prompt
condition. Each annotator labels every image once, in random order; results
are appended to local JSONL shards under data/wild_check_annotations/.
"""
import json
import os
import random
import time
import urllib.parse
import uuid
from pathlib import Path

import gradio as gr
from PIL import Image
import threading

ROOT = Path(__file__).resolve().parents[4]
STAGING = ROOT / "data" / "wild_check_staging"          # tasks.jsonl + images/
ANN_DIR = ROOT / "data" / "wild_check_annotations"      # local result shards
LABELS = ["disruptive", "silent", "integrated", "withheld", "other"]
KEYS = json.load(open(ROOT / "data" / "wild_check_keys.private.json"))  # {name: secret}

tasks_path = STAGING / "tasks.jsonl"
TASKS = {}
for line in open(tasks_path, encoding="utf-8"):
    r = json.loads(line)
    TASKS[r["image_id"]] = r
GOAL = len(TASKS)

# replay every previous local shard in timestamp order
state = {}     # image_id -> {annotator: [labels]}
history = {}   # annotator -> [image_id] in first-annotation order
records = []
ANN_DIR.mkdir(parents=True, exist_ok=True)
for f in sorted(ANN_DIR.glob("*.jsonl")):
    records += [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
for r in sorted(records, key=lambda r: r.get("ts", "")):
    state.setdefault(r["image_id"], {})[r["annotator"]] = r["labels"]
    h = history.setdefault(r["annotator"], [])
    if r["image_id"] not in h:
        h.append(r["image_id"])

out_file = ANN_DIR / f"ann-{uuid.uuid4().hex[:8]}.jsonl"


class _Sched:  # same interface as CommitScheduler, local only
    lock = threading.Lock()


scheduler = _Sched()

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

N_OUT = 10  # components updated by every navigation handler


def auth(qs, request: gr.Request = None):
    q = {k: v[0] for k, v in urllib.parse.parse_qs((qs or "").lstrip("?")).items()}
    if not q and request is not None:
        q = dict(request.query_params)
    name = q.get("annotator")
    return name if name in KEYS and q.get("key") == KEYS[name] else None


def progress_text(name):
    mine = len(history.get(name, []))
    goal = " 🎉 all done — thank you!" if mine >= GOAL else ""
    return f"**{name}** — you: **{mine} / {GOAL}**{goal}"


def hist_update(name):
    # entries show only the ordinal and the prompt: image ids would leak the
    # generator/condition and labels would prime the revisit
    h = history.get(name, [])
    choices = [f"#{n} · {TASKS[iid]['prompt'][:80]}"
               for n, iid in sorted(enumerate(h, 1), reverse=True)]
    return gr.update(choices=choices, value=None,
                     label=f"Your {len(h)} annotations — click to revisit, type to search")


def load_image(iid):
    img = Image.open(STAGING / TASKS[iid]["file"])
    img.load()
    return img


def pick(name):
    now = time.time()
    pool = []
    for iid in TASKS:
        if name in state.get(iid, {}):
            continue
        h = holds.get(iid)
        if h and h[0] != name and now - h[1] < 600:
            continue
        pool.append(iid)
    random.shuffle(pool)
    for iid in pool[:20]:  # image may not be uploaded yet — skip and try next
        try:
            img = load_image(iid)
            holds[iid] = (name, now)
            return iid, img
        except Exception:  # noqa: BLE001
            continue
    return None, None


def render(iid, img, name, pos, prefill):
    t = TASKS[iid]
    note = f"  ·  *reviewing {pos} back — submitting overwrites*" if pos else ""
    return (gr.update(visible=True), gr.update(visible=False),
            img, f"**Prompt:** {t['prompt']}", f"**Target:** {t['target']}",
            progress_text(name) + note, prefill, iid, pos, hist_update(name))


def serve(qs, request: gr.Request):
    name = auth(qs, request)
    if not name:
        return (gr.update(visible=False), gr.update(visible=True),
                None, "", "", "", [], "", 0, gr.update())
    iid, img = pick(name)
    if iid is None:
        return (gr.update(visible=True), gr.update(visible=False),
                None, "", "", progress_text(name) + " — no tasks left, thank you!",
                [], "", 0, hist_update(name))
    return render(iid, img, name, 0, [])


def revisit(name, pos):
    h = history.get(name, [])
    if not h or pos > len(h):
        gr.Warning("No earlier annotation.")
        return tuple(gr.skip() for _ in range(N_OUT))
    iid = h[-pos]
    return render(iid, load_image(iid), name, pos, state[iid][name])


def previous(pos, qs, request: gr.Request):
    name = auth(qs, request)
    if not name:
        return serve(qs, request)
    return revisit(name, pos + 1)


def goto(sel, qs, request: gr.Request):
    name = auth(qs, request)
    if not name or not sel:
        return tuple(gr.skip() for _ in range(N_OUT))
    h = history.get(name, [])
    try:
        n = int(sel.split(" · ")[0].lstrip("#"))
        assert 1 <= n <= len(h)
    except (ValueError, AssertionError):
        return tuple(gr.skip() for _ in range(N_OUT))
    return revisit(name, len(h) - n + 1)


def cap_two(labels):
    return labels[-2:] if len(labels) > 2 else labels


def submit(labels, iid, qs, request: gr.Request):
    name = auth(qs, request)
    if not name or iid not in TASKS:
        return serve(qs, request)
    if not labels:
        gr.Warning("Pick at least one label.")
        return tuple(gr.skip() for _ in range(N_OUT))
    rec = {"image_id": iid, "annotator": name, "labels": labels,
           "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    with scheduler.lock:
        with open(out_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    state.setdefault(iid, {})[name] = labels
    h = history.setdefault(name, [])
    if iid not in h:
        h.append(iid)
    holds.pop(iid, None)
    return serve(qs, request)


with gr.Blocks(title="OverReal wild check") as demo:
    with gr.Column(visible=False) as main:
        prog = gr.Markdown()
        hist_dd = gr.Dropdown(choices=[], visible=True, interactive=True,
                              filterable=True,
                              label="Your annotations — click to revisit, type to search")
        with gr.Accordion("Label guide — read me first", open=True):
            gr.Markdown(GUIDE)
        img = gr.Image(type="pil", height=560, interactive=False,
                       show_download_button=False)
        prompt_md = gr.Markdown()
        target_md = gr.Markdown()
        labels_in = gr.CheckboxGroup(LABELS, label="Labels (pick 1–2)")
        with gr.Row():
            prev_btn = gr.Button("← Previous")
            submit_btn = gr.Button("Submit & next", variant="primary")
            skip_btn = gr.Button("→ Resume / Skip")
    with gr.Column(visible=True) as denied:
        gr.Markdown("### Invalid or missing link\nPlease use your personal "
                    "annotation link, or contact Jiahao.")

    iid_state = gr.State("")
    pos_state = gr.State(0)
    qs_box = gr.Textbox(visible=False)
    outs = [main, denied, img, prompt_md, target_md, prog, labels_in,
            iid_state, pos_state, hist_dd]
    demo.load(None, None, qs_box, js="() => window.location.search")
    qs_box.change(serve, qs_box, outs)
    labels_in.change(cap_two, labels_in, labels_in)
    submit_btn.click(submit, [labels_in, iid_state, qs_box], outs)
    prev_btn.click(previous, [pos_state, qs_box], outs)
    skip_btn.click(serve, qs_box, outs)
    hist_dd.input(goto, [hist_dd, qs_box], outs)

demo.launch(ssr_mode=False, show_error=True, share=False,
            server_name="127.0.0.1", server_port=int(os.environ.get("PORT", "7861")))
