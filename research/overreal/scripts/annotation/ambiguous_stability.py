"""OverReal-Det: over-realization rates and generator rankings under alternative treatments of the ambiguous images (reviewer 3P6s, Q3). Run from the repo root with the Fraud env python (scipy)."""
import os
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".."))
import json,collections
from scipy.stats import spearmanr
meta={}
for l in open('data/overreal_v1/metadata.jsonl'):
    r=json.loads(l); meta[r['image_id']]=r
rows=[json.loads(l) for l in open('data/overreal_v1/det_split.jsonl')]
rows=[r for r in rows if r['split'] in ('gold','ambiguous')]
print('images',len(rows), collections.Counter(r['split'] for r in rows))
print('multi-label annotator sets:', sum(len(r['labels_1'])>1 or len(r['labels_2'])>1 for r in rows))
print('prompt conds:', collections.Counter(meta[r['image_id']].get('prompt_cond') for r in rows))
OR={'disruptive','silent'}
def val(labs):  # label set -> 'or' / 'ok' / None(other-only)
    s=set(labs)-{'other'}
    if not s: return None
    if s<=OR: return 'or'
    if not (s&OR): return 'ok'
    return 'mixed'
schemes={
 'gold only': lambda r: val(r['gold_labels']) if r['split']=='gold' else None,
 'annotator 1': lambda r: val(r['labels_1']),
 'annotator 2': lambda r: val(r['labels_2']),
 'ambiguous -> OR': lambda r: val(r['gold_labels']) if r['split']=='gold' else 'or',
 'ambiguous -> correct': lambda r: val(r['gold_labels']) if r['split']=='gold' else 'ok',
}
fams=['cancellation','attribution','figurative','perspectival']
gens=sorted({meta[r['image_id']]['generator'] for r in rows})
print('generators',gens)
res={}
for name,f in schemes.items():
    cell=collections.defaultdict(collections.Counter)
    for r in rows:
        v=f(r)
        if v in ('or','ok'):
            m=meta[r['image_id']]; cell[(m['family'],m['generator'])][v]+=1
            cell[(m['family'],'ALL')][v]+=1
    res[name]={k:(c['or']/(c['or']+c['ok']), c['or']+c['ok']) for k,c in cell.items()}
print('\nPooled OR rate per family')
print(f"{'scheme':22s}"+''.join(f'{f[:12]:>14s}' for f in fams))
for name in schemes:
    print(f'{name:22s}'+''.join(f"{res[name].get((f,'ALL'),(float('nan'),0))[0]:14.2f}" for f in fams))
print('\nmixed (annotator gave both OR and correct labels):', sum(val(r['labels_1'])=='mixed' or val(r['labels_2'])=='mixed' for r in rows))
print('\nGenerator ranking agreement with gold-only (Spearman), per family; n gens with >=20 images')
for f in fams:
    gs=[g for g in gens if res['gold only'].get((f,g),(0,0))[1]>=20]
    base=[res['gold only'][(f,g)][0] for g in gs]
    line=f'{f:13s} n_gen={len(gs)} '
    for name in list(schemes)[1:]:
        alt=[res[name].get((f,g),(float("nan"),0))[0] for g in gs]
        rho=spearmanr(base,alt)[0] if len(gs)>=3 else float('nan')
        maxd=max(abs(a-b) for a,b in zip(base,alt))
        line+=f'| {name}: rho={rho:.2f} maxΔ={maxd:.2f} '
    print(line)
print('\nper-generator rates (gold only / ann1 / ann2 / amb->OR / amb->correct)')
for f in fams:
    for g in gens:
        if res['gold only'].get((f,g),(0,0))[1]>=20:
            print(f'  {f[:5]} {g:28s} n={res["gold only"][(f,g)][1]:4d} '+' / '.join(f"{res[n].get((f,g),(float('nan'),0))[0]:.2f}" for n in schemes))