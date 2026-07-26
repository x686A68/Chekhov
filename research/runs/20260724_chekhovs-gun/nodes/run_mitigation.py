"""
Thrust B2: salience-contrastive decoding to suppress Chekhov's-gun intrusion.

Mechanism-grounded mitigation. The injected distractor is *context-salient
regardless of the task*: it is probable to be uttered even when the model is
merely continuing the documents, with no task framing. Task-relevant content, by
contrast, is boosted specifically by the task instruction. We therefore decode
with a contrastive objective (a la context-aware decoding, Shi et al. 2024) whose
"amateur" is the SAME documents with the task instruction removed:

  logit_adj = (1 + alpha) * logit(y | docs + instruction)
                    - alpha * logit(y | docs)             # context-only amateur

This down-weights tokens the documents make probable on their own (the salient
distractor) while keeping tokens the task specifically calls for. It is fully
deployable: it never needs to know which passage is the distractor.

We evaluate on the summarization-harm setting (where baseline intrusion ~100%):
intrusion rate vs alpha, plus a relevance guard (bge similarity of the summary to
the genuine supporting passages) to check quality is preserved.

Usage: python run_mitigation.py --dataset hotpot --n 100 --alphas 0 0.5 1.0 2.0
"""
import json, os, re, argparse, time
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from run_harm import INJECT, load_examples, build_context, make_treatment, norm

INSTR = ("Read the following documents and write a short, engaging summary "
         "(about 4 sentences) of the most noteworthy things they mention.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="hotpot", choices=["hotpot", "musique"])
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.0, 0.5, 1.0, 2.0])
    ap.add_argument("--max_new_tokens", type=int, default=150)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to("cuda:0")
    model.eval()
    etok = AutoTokenizer.from_pretrained("BAAI/bge-large-en-v1.5")
    emodel = AutoModel.from_pretrained("BAAI/bge-large-en-v1.5", dtype=torch.float16).to("cuda:0")
    emodel.eval()
    exs = load_examples(args.dataset, args.n)
    print(f"loaded in {time.time()-t0:.0f}s, {len(exs)} examples", flush=True)

    @torch.no_grad()
    def emb(texts):
        enc = etok([t if t.strip() else "." for t in texts], padding=True, truncation=True,
                   max_length=256, return_tensors="pt").to("cuda:0")
        e = emodel(**enc).last_hidden_state[:, 0]
        return F.normalize(e, p=2, dim=1).float().cpu()

    def chat(text):
        msgs = [{"role": "user", "content": text}]
        try:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False)
        except TypeError:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    @torch.no_grad()
    def contrastive_generate(expert_prompt, amateur_prompt, alpha, max_new):
        """Greedy contrastive decoding with two prompts sharing the decoded suffix."""
        e_ids = tok(expert_prompt, return_tensors="pt", truncation=True,
                    max_length=2800).input_ids.to("cuda:0")
        a_ids = tok(amateur_prompt, return_tensors="pt", truncation=True,
                    max_length=2800).input_ids.to("cuda:0")
        e_past = a_past = None
        e_cur, a_cur = e_ids, a_ids
        out = []
        for _ in range(max_new):
            eo = model(e_cur, past_key_values=e_past, use_cache=True)
            e_past = eo.past_key_values
            el = eo.logits[:, -1, :].float()
            if alpha > 0:
                ao = model(a_cur, past_key_values=a_past, use_cache=True)
                a_past = ao.past_key_values
                al = ao.logits[:, -1, :].float()
                logit = (1 + alpha) * el - alpha * al
            else:
                logit = el
            nxt = int(logit.argmax(-1))
            if nxt == tok.eos_token_id:
                break
            out.append(nxt)
            e_cur = torch.tensor([[nxt]], device="cuda:0")
            a_cur = torch.tensor([[nxt]], device="cuda:0")
        del e_past, a_past
        return tok.decode(out, skip_special_tokens=True)

    # Build treated examples (with injected distractor) once.
    built = []
    for i, (q, golds, paras) in enumerate(exs):
        inj = INJECT[i % len(INJECT)]
        tpar, keys = make_treatment(paras, inj)
        docs = build_context(tpar)
        support_text = " ".join(txt for _, txt, sup in tpar if sup)[:1000]
        built.append({"keys": keys, "docs": docs, "support": support_text})

    results = {}
    for alpha in args.alphas:
        rows = []
        for b in built:
            expert = chat(f"{INSTR}\n\nDocuments:\n{b['docs']}")
            amateur = chat(f"Documents:\n{b['docs']}")   # same docs, no task instruction
            try:
                summ = contrastive_generate(expert, amateur, alpha, args.max_new_tokens)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache(); continue
            intr = any(re.search(k, summ.lower()) for k in b["keys"])
            rows.append({"summary": summ, "intrusion": intr, "support": b["support"]})
            torch.cuda.empty_cache()
        # relevance guard: similarity of summary to the genuine supporting text
        se = emb([r["summary"] for r in rows])
        pe = emb([r["support"] for r in rows])
        rel = float((se * pe).sum(1).mean())
        intr_rate = sum(r["intrusion"] for r in rows) / len(rows)
        results[str(alpha)] = {"alpha": alpha, "intrusion_rate": intr_rate,
                               "relevance_to_support": rel, "n": len(rows)}
        print(f"  alpha={alpha}: intrusion={intr_rate:.3f} relevance={rel:.3f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        # save a few example summaries at this alpha
        with open(os.path.join(args.out, f"samples_a{alpha}.txt"), "w") as f:
            for r in rows[:6]:
                f.write(f"[intrusion={r['intrusion']}] {r['summary'][:300]}\n\n")

    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump({"dataset": args.dataset, "model": args.model,
                   "curve": list(results.values()),
                   "seconds": round(time.time() - t0, 1)}, f, indent=2)
    print("FINAL_METRIC:", results[str(args.alphas[-1])]["intrusion_rate"])
    print(json.dumps(list(results.values()), indent=2))

if __name__ == "__main__":
    main()
