"""
Thrust B2 (deployable mitigation): connectivity filtering.

Observation: a Chekhov's-gun distractor injected into a retrieved set is a TOPICAL
OUTLIER -- it is semantically disconnected from the genuinely related passages,
which share entities and jointly support the task. We therefore score each passage
by its connectivity (mean cosine similarity of its embedding to the other passages)
and drop the least-connected passage(s) before summarizing. This needs no oracle
knowledge of which passage is the distractor.

We report: intrusion rate before vs after filtering, relevance of the summary to
the genuine supporting passages, and detection precision (how often the dropped
passage is in fact the injected distractor).

Usage: python run_filter_mitigation.py --dataset hotpot --n 120 --drop 1
"""
import json, os, re, argparse, time
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from run_harm import INJECT, load_examples, make_treatment

INSTR = ("Read the following documents and write a short, engaging summary "
         "(about 4 sentences) of the most noteworthy things they mention.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="hotpot", choices=["hotpot", "musique"])
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--drop", type=int, default=1, help="how many least-connected passages to drop")
    ap.add_argument("--max_new_tokens", type=int, default=150)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to("cuda:0")
    model.eval()
    etok = AutoTokenizer.from_pretrained("BAAI/bge-large-en-v1.5")
    emodel = AutoModel.from_pretrained("BAAI/bge-large-en-v1.5", dtype=torch.float16).to("cuda:0")
    emodel.eval()
    exs = load_examples(args.dataset, args.n)
    print(f"loaded, {len(exs)} examples ({time.time()-t0:.0f}s)", flush=True)

    @torch.no_grad()
    def emb(texts):
        enc = etok([t if t.strip() else "." for t in texts], padding=True, truncation=True,
                   max_length=256, return_tensors="pt").to("cuda:0")
        e = emodel(**enc).last_hidden_state[:, 0]
        return F.normalize(e, p=2, dim=1).float().cpu()

    def chat(text):
        m = [{"role": "user", "content": text}]
        try:
            return tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False)
        except TypeError:
            return tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)

    @torch.no_grad()
    def summarize(paras):
        ctx = "\n\n".join(f"[{t}] {txt}" for t, txt, _ in paras)
        p = chat(f"{INSTR}\n\nDocuments:\n{ctx}")
        enc = tok(p, return_tensors="pt", truncation=True, max_length=3000).to("cuda:0")
        out = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id)
        return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)

    def connectivity_filter(paras, drop):
        texts = [f"{t}. {txt}" for t, txt, _ in paras]
        E = emb(texts)                      # [P, d]
        S = E @ E.T                          # cosine sims
        P = len(paras)
        conn = [(S[i].sum().item() - 1.0) / (P - 1) for i in range(P)]  # mean sim to others
        order = sorted(range(P), key=lambda i: conn[i])   # least connected first
        dropped = set(order[:drop])
        kept = [paras[i] for i in range(P) if i not in dropped]
        return kept, dropped

    rows = []
    for i, (q, golds, paras) in enumerate(exs):
        inj = INJECT[i % len(INJECT)]
        tpar, keys = make_treatment(paras, inj)          # injected distractor at some non-support slot
        inj_title = inj[0]
        # baseline summary (no filter)
        base = summarize(tpar)
        # filtered summary
        kept, dropped = connectivity_filter(tpar, args.drop)
        filt = summarize(kept)
        # was the injected distractor among the dropped?
        dropped_titles = [tpar[j][0] for j in dropped]
        caught = inj_title in dropped_titles
        support_text = " ".join(txt for _, txt, sup in tpar if sup)[:1000]
        rows.append({
            "base_intr": any(re.search(k, base.lower()) for k in keys),
            "filt_intr": any(re.search(k, filt.lower()) for k in keys),
            "caught_distractor": caught,
            "filt_summary": filt[:300], "support": support_text})
        if (i + 1) % 30 == 0:
            print(f"  {i+1}/{len(exs)} ({time.time()-t0:.0f}s)", flush=True)
        torch.cuda.empty_cache()

    # relevance of filtered summaries to genuine support
    se = emb([r["filt_summary"] for r in rows]); pe = emb([r["support"] for r in rows])
    rel_filt = float((se * pe).sum(1).mean())
    n = len(rows)
    summ = {
        "dataset": args.dataset, "model": args.model, "n": n, "drop": args.drop,
        "base_intrusion": sum(r["base_intr"] for r in rows) / n,
        "filtered_intrusion": sum(r["filt_intr"] for r in rows) / n,
        "distractor_caught_rate": sum(r["caught_distractor"] for r in rows) / n,
        "relevance_filtered_to_support": rel_filt,
        "seconds": round(time.time() - t0, 1),
    }
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summ, f, indent=2)
    print("FINAL_METRIC:", summ["filtered_intrusion"])
    print(json.dumps(summ, indent=2))

if __name__ == "__main__":
    main()
