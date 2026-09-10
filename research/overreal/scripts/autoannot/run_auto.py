"""Run the cascade annotator over a set of images.

  python run_auto.py --backend vllm --model Qwen/Qwen2.5-VL-7B-Instruct --split dev
  python run_auto.py --backend vllm --model Qwen/Qwen2.5-VL-72B-Instruct --tp 2 --split dev
  python run_auto.py --backend claude --model claude-opus-5 --split dev --shard 0/6
  python run_auto.py --backend openai --model gpt-5.2-2025-12-11 --split dev --shard 0/4
  python run_auto.py --backend anthropic --model claude-opus-5 --split test --shard 0/3 \
      --out data/overreal_v1/auto/final__claude-opus-5-api.jsonl

--split: dev | test | amb (from auto_split.jsonl) | sample (eval_sample.jsonl)
         | all (sample plus every split row); comma-separated combinations ok.
--protocol: key of common.PROTOCOLS.

Output: data/overreal_v1/auto/<protocol>__<model>.jsonl, one line per image
  {image_id, protocol, model, questions: {Q: text}, raw: {Q: reply},
   answers: {Q: bool|null}, label, sec, ts}
Resumable: already-annotated image_ids are skipped. The vllm backend runs the
cascade stage by stage over the whole batch (Q1 for all, Q2 for the yes set,
Q3 for the Q2-no set); the claude backend runs image by image.
"""
import argparse
import json
import subprocess
import time
from pathlib import Path

from common import (OUT_DIR, YESNO, derive_label, load_eval_sample, load_meta,
                    load_split, parse_yesno, questions, run_path, to_tree)


def select_rows(which, meta):
    ids = []
    parts = which.split(",")
    if "all" in parts:
        parts = ["sample", "dev", "test", "amb"]
    seen = set()
    for p in parts:
        rows = load_eval_sample() if p == "sample" else [r for r in load_split() if r["split"] == p]
        for r in rows:
            if r["image_id"] not in seen and meta[r["image_id"]].get("file_name"):
                seen.add(r["image_id"])
                ids.append(r["image_id"])
    return ids


def image_path(meta, iid):
    return str((Path(__file__).resolve().parents[4] / "data" / "overreal_v1" / meta[iid]["file_name"]))


# ------------------------------------------------------------------ backends

class VllmBackend:
    def __init__(self, model, tp, max_pixels):
        from vllm import LLM, SamplingParams
        kw = {}
        if max_pixels:
            kw["mm_processor_kwargs"] = {"max_pixels": max_pixels}
        self.llm = LLM(model=model, tensor_parallel_size=tp, max_model_len=8192,
                       limit_mm_per_prompt={"image": 1}, gpu_memory_utilization=0.9,
                       trust_remote_code=True, **kw)
        self.sp = SamplingParams(max_tokens=4, temperature=0)

    def ask_batch(self, items):
        """items: list of (image_path, question) -> list of raw replies."""
        from PIL import Image
        convs = []
        for path, q in items:
            img = Image.open(path).convert("RGB")
            convs.append([{"role": "user", "content": [
                {"type": "image_pil", "image_pil": img},
                {"type": "text", "text": YESNO + q}]}])
        outs = self.llm.chat(convs, self.sp, use_tqdm=True)
        return [o.outputs[0].text.strip() for o in outs]


class ClaudeBackend:
    def __init__(self, model):
        self.model = model

    def ask(self, path, q):
        """One yes/no question; None if the CLI keeps failing (the caller then
        leaves the image for a later resume instead of recording a blank)."""
        task = (f"First use the Read tool to view the image at: {path}\n"
                f"Then answer the question about it.\n\n{YESNO}{q}")
        for attempt in range(6):
            p = subprocess.run(
                ["claude", "-p", task, "--model", self.model,
                 "--allowedTools", "Read", "--output-format", "json"],
                capture_output=True, text=True, timeout=600)
            try:
                rsp = json.loads(p.stdout)
            except json.JSONDecodeError:
                rsp = None
            if p.returncode == 0 and rsp and not rsp.get("is_error"):
                return rsp.get("result", "")
            err = (rsp.get("result") if rsp else None) or (p.stderr or p.stdout)[:400]
            print(f"[claude fail {attempt}] {str(err)[:400]}", flush=True)
            low = str(err).lower()
            if "limit" in low or "rate" in low or "overload" in low:
                print("usage limit suspected, sleeping 15 min", flush=True)
                time.sleep(900)
            else:
                time.sleep(30 * (attempt + 1))
        return None


class OpenAIBackend:
    """Pinned GPT snapshot through the OpenAI API, image sent as base64
    (same transport as metrics/run_lj_openai.py)."""

    def __init__(self, model):
        import os
        env = Path(__file__).resolve().parents[4] / ".env"   # same file gen/common.load_env reads
        if env.exists():
            for line in env.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, _, v = line.partition("=")
                    os.environ.setdefault(k.strip(), v.strip())
        from openai import OpenAI
        self.client = OpenAI()
        self.model = model
        self.tokens = {"in": 0, "out": 0, "calls": 0}

    def ask(self, path, q):
        import base64
        ext = "png" if path.endswith("png") else "jpeg"
        b64 = base64.b64encode(open(path, "rb").read()).decode()
        for attempt in range(4):
            try:
                rsp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": [
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/{ext};base64,{b64}"}},
                        {"type": "text", "text": YESNO + q}]}],
                    max_completion_tokens=1000)
                u = rsp.usage
                self.tokens["in"] += u.prompt_tokens
                self.tokens["out"] += u.completion_tokens
                self.tokens["calls"] += 1
                return rsp.choices[0].message.content or ""
            except Exception as e:  # noqa: BLE001
                print(f"[openai fail {attempt}] {type(e).__name__}: {str(e)[:200]}", flush=True)
                time.sleep(30 if "rate" in str(e).lower() else 10)
        return None


class AnthropicBackend:
    """Claude through the Anthropic API. The image block carries a cache
    breakpoint so the second and later questions about the same image read
    it from cache (5-minute TTL; the questions of one image run back to
    back). Effort low, adaptive thinking left on. Usage is accumulated per
    token class and priced at list rates so the driver can stop at a
    dollar cap."""
    PRICE = {"in": 5.0, "cache_write": 6.25, "cache_read": 0.5, "out": 25.0}  # $/MTok, Opus 5

    def __init__(self, model, max_usd=0.0):
        import os
        env = Path(__file__).resolve().parents[4] / ".env"
        if env.exists():
            for line in env.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, _, v = line.partition("=")
                    os.environ.setdefault(k.strip(), v.strip())
        import anthropic
        self.anthropic = anthropic
        self.client = anthropic.Anthropic(max_retries=4)
        self.model = model
        self.max_usd = max_usd
        self.tokens = {"in": 0, "cache_write": 0, "cache_read": 0, "out": 0, "calls": 0}

    def usd(self):
        return sum(self.tokens[k] * self.PRICE[k] / 1e6 for k in self.PRICE)

    @staticmethod
    def encode(path):
        """Base64 payload under the API's 10 MB cap. Files over 5 MB are
        resized to the 1568-px long edge the API applies anyway, as JPEG."""
        import base64
        import io
        import os
        if os.path.getsize(path) <= 5_000_000:
            ext = "png" if path.endswith("png") else "jpeg"
            return ext, base64.b64encode(open(path, "rb").read()).decode()
        from PIL import Image
        im = Image.open(path).convert("RGB")
        im.thumbnail((1568, 1568))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=92)
        return "jpeg", base64.b64encode(buf.getvalue()).decode()

    def ask(self, path, q):
        if self.max_usd and self.usd() > self.max_usd:
            print(f"[STOP] spend {self.usd():.2f} USD exceeds cap {self.max_usd}", flush=True)
            return None
        ext, b64 = self.encode(path)
        for attempt in range(6):
            try:
                rsp = self.client.messages.create(
                    model=self.model, max_tokens=1024,   # room for adaptive thinking
                    output_config={"effort": "low"},
                    messages=[{"role": "user", "content": [
                        {"type": "image",
                         "source": {"type": "base64", "media_type": f"image/{ext}", "data": b64},
                         "cache_control": {"type": "ephemeral"}},
                        {"type": "text", "text": YESNO + q}]}])
                u = rsp.usage
                self.tokens["in"] += u.input_tokens
                self.tokens["cache_write"] += getattr(u, "cache_creation_input_tokens", 0) or 0
                self.tokens["cache_read"] += getattr(u, "cache_read_input_tokens", 0) or 0
                self.tokens["out"] += u.output_tokens
                self.tokens["calls"] += 1
                if rsp.stop_reason == "refusal":
                    return "refusal"
                text = "".join(b.text for b in rsp.content if b.type == "text")
                if text.strip():
                    return text
                print(f"[anthropic empty {attempt}] stop={rsp.stop_reason}", flush=True)
            except self.anthropic.RateLimitError as e:
                print(f"[anthropic 429 {attempt}] {str(e)[:200]}", flush=True)
                time.sleep(60 * (attempt + 1))
            except self.anthropic.APIStatusError as e:
                if e.status_code >= 500 or e.status_code == 408:
                    print(f"[anthropic {e.status_code} {attempt}] retrying", flush=True)
                    time.sleep(30 * (attempt + 1))
                    continue
                print(f"[STOP] API error {e.status_code}: {str(e)[:300]}", flush=True)
                print(f"spend so far: {self.usd():.2f} USD", flush=True)
                raise SystemExit(3)
            except self.anthropic.APIConnectionError as e:
                print(f"[anthropic conn {attempt}] {str(e)[:200]}", flush=True)
                time.sleep(30 * (attempt + 1))
        return None


# ------------------------------------------------------------------ driver

def record(iid, protocol, model, qs, raw, ans, t0):
    return {"image_id": iid, "protocol": protocol, "model": model,
            "questions": qs, "raw": raw, "answers": ans, "label": derive_label(ans, qs),
            "sec": round(time.time() - t0, 1),
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["vllm", "claude", "openai", "anthropic"], required=True)
    ap.add_argument("--max-usd", type=float, default=0.0, help="anthropic: stop this worker past this spend")
    ap.add_argument("--model", required=True)
    ap.add_argument("--protocol", default="v1")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--max-pixels", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--out", default="", help="override output path")
    ap.add_argument("--families", default="", help="comma-separated family filter")
    args = ap.parse_args()

    meta = load_meta()
    ids = select_rows(args.split, meta)
    if args.families:
        keep = set(args.families.split(","))
        ids = [i for i in ids if meta[i]["family"] in keep]
    out = Path(args.out) if args.out else run_path(args.protocol, args.model)
    OUT_DIR.mkdir(exist_ok=True)
    done = set()
    if out.exists():
        done = {json.loads(l)["image_id"] for l in open(out, encoding="utf-8")}
    todo = [i for i in ids if i not in done]
    k, n = map(int, args.shard.split("/"))
    todo = [i for j, i in enumerate(todo) if j % n == k]
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(todo)} images to annotate ({len(done)} done) -> {out}", flush=True)
    if not todo:
        return

    qs_of = {}
    for iid in todo:
        m = meta[iid]
        qs_of[iid] = questions(args.protocol, m["family"], m["target"], m["prompt"])

    f = open(out, "a", encoding="utf-8")
    if args.backend == "vllm":
        be = VllmBackend(args.model, args.tp, args.max_pixels)
        t0 = time.time()
        raw = {iid: {} for iid in todo}
        ans = {iid: {} for iid in todo}
        tree = {iid: to_tree(qs_of[iid]) for iid in todo}
        cur = {iid: tree[iid]["start"] for iid in todo}
        for _round in range(10):
            items, keys = [], []
            for iid in todo:
                node = cur[iid]
                if node in tree[iid]["nodes"]:
                    items.append((image_path(meta, iid), tree[iid]["nodes"][node]["q"]))
                    keys.append(iid)
            if not items:
                break
            print(f"round {_round}: {len(items)} questions", flush=True)
            for iid, r in zip(keys, be.ask_batch(items)):
                node = cur[iid]
                raw[iid][node] = r
                a = parse_yesno(r)
                ans[iid][node] = a
                cur[iid] = None if a is None else tree[iid]["nodes"][node]["yes" if a else "no"]
        for iid in todo:
            f.write(json.dumps(record(iid, args.protocol, args.model, qs_of[iid],
                                      raw[iid], ans[iid], t0), ensure_ascii=False) + "\n")
        f.flush()
        print(f"done in {time.time() - t0:.0f}s", flush=True)
    else:
        be = {"claude": lambda: ClaudeBackend(args.model),
              "openai": lambda: OpenAIBackend(args.model),
              "anthropic": lambda: AnthropicBackend(args.model, args.max_usd)}[args.backend]()
        for n_, iid in enumerate(todo, 1):
            t0 = time.time()
            path = image_path(meta, iid)
            raw, ans = {}, {}
            tree = to_tree(qs_of[iid])
            node = tree["start"]
            failed = False
            while node in tree["nodes"]:
                reply = be.ask(path, tree["nodes"][node]["q"])
                if reply is None:
                    failed = True
                    break
                raw[node] = reply
                ans[node] = parse_yesno(reply)
                if ans[node] is None:
                    break
                node = tree["nodes"][node]["yes" if ans[node] else "no"]
            if failed:
                print(f"[SKIP] {iid}: backend failed, left for resume", flush=True)
                continue
            f.write(json.dumps(record(iid, args.protocol, args.model, qs_of[iid],
                                      raw, ans, t0), ensure_ascii=False) + "\n")
            f.flush()
            if n_ % 200 == 0 or n_ == len(todo):
                spend = f" spend={be.usd():.2f}USD" if hasattr(be, "usd") else ""
                print(f"[{n_}/{len(todo)}]{spend}", flush=True)
        if getattr(be, "tokens", None):
            print("tokens:", be.tokens, flush=True)
            if hasattr(be, "usd"):
                print(f"spend: {be.usd():.2f} USD", flush=True)


if __name__ == "__main__":
    main()
