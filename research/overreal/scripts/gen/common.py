"""Shared paths and IO for the OverReal generation runs.

Layout under data/generation/:
  prompts.jsonl           one line per item: item_id, family, prompt, target
  expanded_prompts.jsonl  one line per (item, expander): text + provenance
  images/<model>/<cond>/<family>_<nnnn>__s<seed>.png
  manifests/<model>_<cond>.jsonl   one line per saved image
"""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
DATASET = ROOT / "data" / "overreal_v1"
GEN = ROOT / "data" / "generation"
PROMPTS = GEN / "prompts.jsonl"
EXPANDED = GEN / "expanded_prompts.jsonl"

SEEDS = [0, 1]          # two images per prompt
CONDS = ["raw", "qwen", "dalle3", "ideogram"]


def load_env():
    """Read ~/Chekhov/.env into os.environ (keys only set if absent)."""
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def append_jsonl(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_prompts(cond):
    """Return [(item_id, family, prompt_text)] for a condition.

    raw -> the item's own prompt; otherwise the expander's rewritten text.
    """
    if cond == "raw":
        return [(r["item_id"], r["family"], r["prompt"]) for r in read_jsonl(PROMPTS)]
    rows = [r for r in read_jsonl(EXPANDED) if r["expander"] == cond]
    fam = {r["item_id"]: r["family"] for r in read_jsonl(PROMPTS)}
    return [(r["item_id"], fam[r["item_id"]], r["text"]) for r in rows]


def image_path(model, cond, item_id, seed):
    fam, num = item_id.split("/")            # e.g. figurative/prompt_0012
    name = f"{fam}_{num.removeprefix('prompt_')}__s{seed}.png"
    return GEN / "images" / model / cond / name


def manifest_path(model, cond):
    return GEN / "manifests" / f"{model}_{cond}.jsonl"


def done_keys(model, cond):
    """(item_id, seed) pairs already recorded, for resumable runs."""
    p = manifest_path(model, cond)
    if not p.exists():
        return set()
    return {(r["item_id"], r["seed"]) for r in read_jsonl(p)}
