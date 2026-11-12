"""OverReal-Gen vs. in-the-wild: per-system over-realization rates, Spearman rank correlation, ranks, and pooled disruptive share of over-realization (reviewer KXAW, Q3).
Run from the repo root with the Fraud env python (needs scipy).
"""
import os, sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import sys, collections
sys.path.insert(0,'.')
import wild_rates as W
from scipy.stats import spearmanr, kendalltau
wild, items = W.wild_labels(); gen = W.gen_labels()
def r(labs):
    return W.rates(labs)
print("family | model | Gen OR | wild OR | Gen DO/OR | wild DO/OR")
allg, allw = [], []
for fam in W.FAMS:
    g_or, w_or, names = [], [], []
    print("##", W.FAM_NAME[fam])
    for m in W.ORDER:
        labs = wild.get((m,fam,'random'),[])+wild.get((m,fam,'targeted'),[])
        w = r(labs); g = r(gen.get((m,fam),[]))
        if not w or not g: 
            print(f"  {W.MODEL_NAME[m]}: missing (wild n={len(labs)}, gen n={len(gen.get((m,fam),[]))})"); continue
        names.append(W.MODEL_NAME[m]); g_or.append(g['or']); w_or.append(w['or'])
        gd = g['disruptive']/g['or'] if g['or'] else float('nan'); wd = w['disruptive']/w['or'] if w['or'] else float('nan')
        print(f"  {W.MODEL_NAME[m]:13s} Gen OR {g['or']:.2f} (n{g['n']})  wild OR {w['or']:.2f} (n{w['n']})  DO/OR gen {gd:.2f} wild {wd:.2f}")
    if len(names)>=3:
        rho,p = spearmanr(g_or,w_or); tau,_=kendalltau(g_or,w_or)
        rank = lambda v: [sorted(v,reverse=True).index(x)+1 for x in v]
        print(f"  Spearman rho={rho:.2f} (p={p:.2f}), Kendall tau={tau:.2f}, n={len(names)}")
        print("  ranks gen :", dict(zip(names, rank(g_or))))
        print("  ranks wild:", dict(zip(names, rank(w_or))))
        allg += g_or; allw += w_or
rho,p = spearmanr(allg,allw); print(f"\nPooled over (model,family) cells: Spearman rho={rho:.2f} p={p:.3f} n={len(allg)}")
# DO share pooled per family
for fam in W.FAMS:
    gl=[l for m in W.ORDER for l in gen.get((m,fam),[])]; wl=[l for m in W.ORDER for l in wild.get((m,fam,'random'),[])+wild.get((m,fam,'targeted'),[])]
    cg=collections.Counter(gl); cw=collections.Counter(wl)
    f=lambda c: (c['disruptive']/(c['disruptive']+c['silent']) if c['disruptive']+c['silent'] else float('nan'), (c['disruptive']+c['silent'])/sum(c.values()))
    print(f"{W.FAM_NAME[fam]:20s} pooled DO/OR gen {f(cg)[0]:.2f} wild {f(cw)[0]:.2f} | pooled OR gen {f(cg)[1]:.2f} wild {f(cw)[1]:.2f}")