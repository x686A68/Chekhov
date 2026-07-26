"""
Thrust A: real-data downstream harm (HotpotQA / MuSiQue).

We inject one salient IRRELEVANT passage (a Chekhov's gun) into the retrieved
context of a real multi-hop QA example and ask whether it (a) surfaces in the
answer (intrusion) and (b) degrades answer accuracy. The key analysis links the
two: are answers that intrude the distractor less accurate than those that don't?

  Control  : original context passages.
  Treatment: one non-supporting passage replaced by a salient off-topic passage.

Metrics: SQuAD-style EM / token-F1 vs gold (+ aliases); intrusion = injected
entity keyword appears in the model response.

Usage: python run_harm.py --dataset hotpot --n 200 --out harm/hotpot
"""
import json, os, re, argparse, time, string
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Salient irrelevant passages to inject (distinctive off-topic entities).
INJECT = [
    ("The 2010 Eruptions in Iceland",
     "The volcanoes of Iceland are among the most active on Earth. The 2010 eruption "
     "of Eyjafjallajokull sent ash across Europe and grounded thousands of flights.",
     [r"iceland", r"volcano", r"eyjafjallaj"]),
    ("Axolotl",
     "The axolotl is a neotenic salamander native to lakes near Mexico City. Unusually, "
     "it retains its larval features for life and can regenerate entire limbs.",
     [r"axolotl", r"salamander"]),
    ("Catan (board game)",
     "Catan is a multiplayer board game designed by Klaus Teuber. Players collect "
     "resources and build settlements on a hexagonal island, trading to reach ten points.",
     [r"catan", r"klaus teuber"]),
    ("Voyager 2",
     "Voyager 2 is a space probe launched by NASA in 1977. It is the only spacecraft "
     "to have visited both Uranus and Neptune, and has now entered interstellar space.",
     [r"voyager", r"space probe", r"interstellar"]),
    ("Vespa",
     "The Vespa is an Italian brand of scooter manufactured by Piaggio. Introduced in "
     "1946, its steel unibody and distinctive shape made it a postwar design icon.",
     [r"vespa", r"scooter", r"piaggio"]),
    ("Silk Road",
     "The Silk Road was a network of Eurasian trade routes active from the 2nd century "
     "BCE. It carried silk, spices, and ideas between China, India, and the Mediterranean.",
     [r"silk road"]),
    ("Trebuchet",
     "A trebuchet is a medieval siege engine that uses a counterweight to hurl "
     "projectiles. It could throw stones weighing over a hundred kilograms at castle walls.",
     [r"trebuchet", r"siege engine", r"counterweight"]),
    ("Mariana Trench",
     "The Mariana Trench is the deepest known part of the ocean, reaching nearly 11,000 "
     "metres. Its Challenger Deep has been visited by only a handful of crewed descents.",
     [r"mariana", r"challenger deep"]),
    ("Anglerfish",
     "Anglerfish are deep-sea predators that lure prey with a bioluminescent appendage. "
     "In some species the tiny male permanently fuses to the much larger female.",
     [r"anglerfish", r"bioluminescent"]),
    ("Dog agility",
     "Dog agility is a competitive sport in which handlers direct dogs through an "
     "obstacle course of jumps, tunnels, and weave poles against the clock.",
     [r"dog agility", r"weave poles"]),
]

# ---- SQuAD-style normalization / EM / F1 ----
def norm(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())

def em(pred, golds):
    p = norm(pred)
    return float(any(p == norm(g) for g in golds))

def f1(pred, golds):
    best = 0.0
    pt = norm(pred).split()
    for g in golds:
        gt = norm(g).split()
        if not pt or not gt:
            best = max(best, float(pt == gt)); continue
        common = {}
        for t in pt:
            if t in gt:
                common[t] = min(pt.count(t), gt.count(t))
        ns = sum(common.values())
        if ns == 0:
            continue
        prec = ns / len(pt); rec = ns / len(gt)
        best = max(best, 2 * prec * rec / (prec + rec))
    return best

# ---- dataset loaders -> unified (question, [gold aliases], [(title, text, is_support)]) ----
def load_examples(dataset, n):
    from datasets import load_dataset
    out = []
    if dataset == "hotpot":
        d = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
        for ex in d.select(range(min(n, len(d)))):
            titles = ex["context"]["title"]; sents = ex["context"]["sentences"]
            support = set(ex["supporting_facts"]["title"])
            paras = [(t, " ".join(s), t in support) for t, s in zip(titles, sents)]
            out.append((ex["question"], [ex["answer"]], paras))
    elif dataset == "musique":
        d = load_dataset("bdsaglam/musique", "answerable", split="validation")
        for ex in d.select(range(min(n, len(d)))):
            paras = [(p["title"], p["paragraph_text"], bool(p["is_supporting"]))
                     for p in ex["paragraphs"]]
            golds = [ex["answer"]] + list(ex.get("answer_aliases") or [])
            out.append((ex["question"], golds, paras))
    return out

def build_context(paras):
    return "\n\n".join(f"[{t}] {txt}" for t, txt, _ in paras)

def make_treatment(paras, inj):
    title, text, keys = inj
    # replace the first non-supporting passage; keep count/positions
    new = list(paras)
    for i, (t, txt, sup) in enumerate(new):
        if not sup:
            new[i] = (title, text, False)
            return new, keys
    new.append((title, text, False))   # fallback: append
    return new, keys

PROMPT_SHORT = ("Answer the question using the context. Respond with 'Answer: <short answer>' "
                "on the first line, then one sentence of justification.\n\nContext:\n{ctx}\n\n"
                "Question: {q}")
PROMPT_LONG = ("Using the context, answer the question in a short helpful paragraph "
               "(about 3-5 sentences).\n\nContext:\n{ctx}\n\nQuestion: {q}")
# Open real task over the same retrieved documents: summarize what is noteworthy.
PROMPT_SUMM = ("Read the following documents and write a short, engaging summary "
               "(about 4 sentences) of the most noteworthy things they mention.\n\n"
               "Documents:\n{ctx}\n\n(You do not need to mention everything; highlight "
               "what stands out.)")

def extract_answer(resp):
    m = re.search(r"answer\s*:\s*(.+)", resp, re.IGNORECASE)
    line = m.group(1) if m else resp.strip().split("\n")[0]
    return line.strip().strip(".").split("\n")[0][:120]

def covers(resp, golds):
    """Answer coverage: a gold alias appears (normalized) in the response.
    A faithfulness/harm proxy that works for long-form answers."""
    r = norm(resp)
    return float(any(norm(g) and norm(g) in r for g in golds))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["hotpot", "musique"])
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=96)
    ap.add_argument("--mode", choices=["short", "long", "summ"], default="short")
    args = ap.parse_args()
    PROMPT = {"short": PROMPT_SHORT, "long": PROMPT_LONG, "summ": PROMPT_SUMM}[args.mode]
    os.makedirs(args.out, exist_ok=True)

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to("cuda:0")
    model.eval()
    exs = load_examples(args.dataset, args.n)
    print(f"loaded {args.model} + {len(exs)} {args.dataset} in {time.time()-t0:.0f}s", flush=True)

    def prompt_text(ctx, q):
        msgs = [{"role": "user", "content": PROMPT.format(ctx=ctx, q=q)}]
        try:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False)
        except TypeError:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    @torch.no_grad()
    def gen(prompts):
        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True,
                  max_length=3500).to(model.device)
        out = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id)
        g = out[:, enc["input_ids"].shape[1]:]
        return [tok.decode(x, skip_special_tokens=True) for x in g]

    # build jobs
    jobs = []
    for i, (q, golds, paras) in enumerate(exs):
        inj = INJECT[i % len(INJECT)]
        tpar, keys = make_treatment(paras, inj)
        jobs.append({"q": q, "golds": golds, "keys": keys,
                     "ctrl_prompt": prompt_text(build_context(paras), q),
                     "treat_prompt": prompt_text(build_context(tpar), q)})

    rows = []
    for i in range(0, len(jobs), args.batch):
        chunk = jobs[i:i+args.batch]
        cout = gen([j["ctrl_prompt"] for j in chunk])
        tout = gen([j["treat_prompt"] for j in chunk])
        for j, cr, tr in zip(chunk, cout, tout):
            ca, ta = extract_answer(cr), extract_answer(tr)
            intr = any(re.search(k, tr.lower()) for k in j["keys"])
            intr_ctrl = any(re.search(k, cr.lower()) for k in j["keys"])
            first_sent = re.split(r"(?<=[.!?])\s", tr.strip())[0].lower() if tr.strip() else ""
            leads = any(re.search(k, first_sent) for k in j["keys"])
            rows.append({
                "q": j["q"], "golds": j["golds"],
                "ctrl_em": em(ca, j["golds"]), "ctrl_f1": f1(ca, j["golds"]),
                "treat_em": em(ta, j["golds"]), "treat_f1": f1(ta, j["golds"]),
                "ctrl_cov": covers(cr, j["golds"]), "treat_cov": covers(tr, j["golds"]),
                "intrusion": intr, "intrusion_ctrl": intr_ctrl, "leads": leads,
                "ctrl_ans": ca, "treat_ans": ta, "treat_resp": tr[:400]})
        print(f"  {min(i+args.batch,len(jobs))}/{len(jobs)} ({time.time()-t0:.0f}s)", flush=True)

    with open(os.path.join(args.out, "results.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    import statistics as st
    n = len(rows)
    intruded = [r for r in rows if r["intrusion"]]
    clean = [r for r in rows if not r["intrusion"]]
    summ = {
        "dataset": args.dataset, "model": args.model, "mode": args.mode, "n": n,
        "ctrl_em": st.mean(r["ctrl_em"] for r in rows),
        "treat_em": st.mean(r["treat_em"] for r in rows),
        "ctrl_f1": st.mean(r["ctrl_f1"] for r in rows),
        "treat_f1": st.mean(r["treat_f1"] for r in rows),
        "ctrl_cov": st.mean(r["ctrl_cov"] for r in rows),
        "treat_cov": st.mean(r["treat_cov"] for r in rows),
        "intrusion_rate": len(intruded) / n,
        "intrusion_rate_ctrl": sum(r["intrusion_ctrl"] for r in rows) / n,
        "leads_rate": sum(r["leads"] for r in rows) / n,
        # KEY LINK: coverage (answer correctness) among intruded vs clean treatment cases
        "treat_cov_intruded": (st.mean(r["treat_cov"] for r in intruded) if intruded else None),
        "treat_cov_clean": (st.mean(r["treat_cov"] for r in clean) if clean else None),
        "treat_em_intruded": (st.mean(r["treat_em"] for r in intruded) if intruded else None),
        "treat_em_clean": (st.mean(r["treat_em"] for r in clean) if clean else None),
        "n_intruded": len(intruded),
        "seconds": round(time.time() - t0, 1),
    }
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summ, f, indent=2)
    print("FINAL_METRIC:", summ["intrusion_rate"])
    print(json.dumps(summ, indent=2))

if __name__ == "__main__":
    main()
