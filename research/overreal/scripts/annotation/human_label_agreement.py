"""OverReal-Det: human-human agreement per label (binary kappa per label, integrated by family). Run from the repo root with the Fraud env python."""
import os
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".."))
import json, collections, itertools
rows=[json.loads(l) for l in open('data/overreal_v1/det_split.jsonl')]
meta={}
for l in open('data/overreal_v1/metadata.jsonl'):
    r=json.loads(l); meta[r['image_id']]=r
# doubly annotated, one label each, exclude self-pair (marathon cancellation)
pairs=[]
for r in rows:
    m=meta[r['image_id']]
    wave=r['wave']
    if wave=='marathon' and m['family']=='cancellation': continue
    if len(r['labels_1'])==1 and len(r['labels_2'])==1:
        pairs.append((m['family'],r['labels_1'][0],r['labels_2'][0]))
print('n pairs',len(pairs))
labs=['disruptive','silent','integrated','withheld','other']
print('label values seen',collections.Counter(a for _,a,b in pairs for a in (a,b)))
def kappa(ps):
    n=len(ps); po=sum(a==b for a,b in ps)/n
    c1=collections.Counter(a for a,b in ps); c2=collections.Counter(b for a,b in ps)
    pe=sum(c1[k]*c2[k] for k in set(c1)|set(c2))/n**2
    return po,(po-pe)/(1-pe)
# confusion
C=collections.Counter((a,b) for _,a,b in pairs)
L=sorted(set(a for _,a,b in pairs for a in (a,b)))
print('\t'+'\t'.join(l[:5] for l in L))
for a in L: print(a[:5]+'\t'+'\t'.join(str(C[(a,b)]) for b in L))
print('overall 4/5-label kappa',kappa([(a,b) for _,a,b in pairs]))
for lab in L:
    ps=[(a==lab,b==lab) for _,a,b in pairs]
    po,k=kappa(ps)
    both=sum(a and b for a,b in ps); either=sum(a or b for a,b in ps)
    n1=sum(a for a,b in ps); n2=sum(b for a,b in ps)
    print(f'{lab:11s} binary kappa={k:.2f} po={po:.3f} both={both} either={either} ann1={n1} ann2={n2} pos-agree(2*both/(n1+n2))={2*both/(n1+n2):.2f}')
print('--- integrated by family')
for fam in ['cancellation','attribution','figurative','perspectival']:
    ps=[(a=='integrated',b=='integrated') for f,a,b in pairs if f==fam]
    if not ps: continue
    po,k=kappa(ps); both=sum(a and b for a,b in ps); n1=sum(a for a,b in ps); n2=sum(b for a,b in ps)
    print(f'{fam:13s} n={len(ps)} kappa={k:.2f} both={both} ann1={n1} ann2={n2} pos-agree={2*both/max(1,n1+n2):.2f}')