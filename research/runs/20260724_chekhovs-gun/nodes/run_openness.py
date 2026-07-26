"""
Quantify task openness correctly, as WITHIN-PROMPT answer diversity.

For each control prompt (no distractor) at each ladder level, sample K answers at
temperature 0.8 and measure their mutual semantic diversity (mean pairwise cosine
distance). Averaging over prompts gives an openness score per level that isolates
how many different good answers the SAME task admits -- independent of which
passage is used and independent of the distractor. We then correlate openness
with the intrusion rate measured separately.
"""
import json, os
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel

BASE = os.path.dirname(os.path.abspath(__file__))
LEVELS = ["L0_extractive", "L1_summary", "L2_takeaway", "L3_creative"]
GEN = "Qwen/Qwen3-8B"
EMB = "BAAI/bge-large-en-v1.5"
K = 5           # samples per prompt
N_ITEMS = 24    # prompts per level (subset for speed)

def main():
    gtok = AutoTokenizer.from_pretrained(GEN, padding_side="left")
    if gtok.pad_token is None:
        gtok.pad_token = gtok.eos_token
    gmodel = AutoModelForCausalLM.from_pretrained(GEN, dtype=torch.bfloat16).to("cuda:0")
    gmodel.eval()
    etok = AutoTokenizer.from_pretrained(EMB)
    emodel = AutoModel.from_pretrained(EMB, dtype=torch.float16).to("cuda:0"); emodel.eval()

    @torch.no_grad()
    def embed(texts):
        chunk = [t if t.strip() else "." for t in texts]
        enc = etok(chunk, padding=True, truncation=True, max_length=256,
                   return_tensors="pt").to("cuda:0")
        e = emodel(**enc).last_hidden_state[:, 0]
        return F.normalize(e, p=2, dim=1).float().cpu()

    def prompt_text(msgs):
        try:
            return gtok.apply_chat_template(msgs, tokenize=False,
                                            add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            return gtok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    import re
    def strip_think(t):
        return re.sub(r"<think>.*?</think>", "", t, flags=re.DOTALL).strip()

    results = []
    for L in LEVELS:
        items = json.load(open(os.path.join(BASE, f"data_ladder_{L}.json")))[:N_ITEMS]
        prompts = [prompt_text(it["control"]) for it in items]
        # K samples for each prompt
        per_prompt_samples = [[] for _ in prompts]
        for s in range(K):
            enc = gtok(prompts, return_tensors="pt", padding=True).to("cuda:0")
            torch.manual_seed(100 + s)
            with torch.no_grad():
                out = gmodel.generate(**enc, max_new_tokens=120, do_sample=True,
                                      temperature=0.8, top_p=0.95,
                                      pad_token_id=gtok.pad_token_id)
            gen = out[:, enc["input_ids"].shape[1]:]
            for i, g in enumerate(gen):
                per_prompt_samples[i].append(strip_think(gtok.decode(g, skip_special_tokens=True)))
        # within-prompt diversity
        divs = []
        for samples in per_prompt_samples:
            e = embed(samples)
            n = len(e)
            sim = e @ e.T
            iu = torch.triu_indices(n, n, offset=1)
            divs.append(float((1 - sim[iu[0], iu[1]]).mean()))
        openness = sum(divs) / len(divs)
        # intrusion from the earlier ladder run
        summ = json.load(open(os.path.join(BASE, "ladder", L, "summary.json")))
        results.append({"level": L, "openness_within_prompt": openness,
                        "intrusion": summ["treatment_intrusion_rate"]})
        print(f"{L}: openness={openness:.3f} intrusion={summ['treatment_intrusion_rate']:.3f}",
              flush=True)

    def spearman(xs, ys):
        def rank(v):
            order = sorted(range(len(v)), key=lambda i: v[i]); r = [0]*len(v)
            for rr, i in enumerate(order): r[i] = rr
            return r
        rx, ry = rank(xs), rank(ys); n = len(xs)
        d2 = sum((a-b)**2 for a, b in zip(rx, ry))
        return 1 - 6*d2/(n*(n*n-1))
    xs = [r["openness_within_prompt"] for r in results]
    ys = [r["intrusion"] for r in results]
    rho = spearman(xs, ys)
    json.dump({"levels": results, "spearman_rho": rho},
              open(os.path.join(BASE, "ladder", "openness_summary.json"), "w"), indent=2)
    print("spearman rho:", rho)

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    names = ["extractive", "summary", "takeaway", "creative"]
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    ax.plot(xs, ys, "-o", color="#c1432c")
    for x, y, nm in zip(xs, ys, names):
        ax.annotate(nm, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xlabel("Task openness  (within-prompt answer diversity)")
    ax.set_ylabel("Distractor intrusion rate")
    ax.set_title(f"Intrusion rises with task openness  (Spearman $\\rho$={rho:.2f})")
    ax.set_ylim(-0.03, 1)
    fig.tight_layout()
    fig.savefig(os.path.join(BASE, "..", "figures", "fig7_openness_ladder.png"), dpi=160)
    print("wrote fig7_openness_ladder.png")

if __name__ == "__main__":
    main()
