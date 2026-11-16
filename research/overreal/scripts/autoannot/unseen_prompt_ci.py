"""Bootstrap confidence intervals (resampled over prompts) for the judge-vs-human agreement on the unseen-prompt test subset, its difference from the other test images, and the subset composition (reviewer 3P6s follow-up). Run with the Fraud env python."""
import os, sys, json, collections, random
sys.path.insert(0, "/data/users/jiahao_huang/Chekhov/research/overreal/scripts/autoannot")
os.chdir("/data/users/jiahao_huang/Chekhov/research/overreal/scripts/autoannot")
from common import derive_label, load_split
from eval_auto import kappa, gold_of, COARSE
auto={}
for l in open('/data/users/jiahao_huang/Chekhov/data/overreal_v1/auto/final__claude-opus-5-api.jsonl'):
    r=json.loads(l); r['label']=derive_label(r['answers'],r['questions']); auto[r['image_id']]=r
split=load_split()
item=lambda iid:'/'.join(iid.split('/')[:2])
dev_items={item(r['image_id']) for r in split if r['split']=='dev'}
test=[r for r in split if r['split']=='test' and r['image_id'] in auto]
for r in test:
    a=auto[r['image_id']]['label']; g=gold_of(r,a)
    r['_p']=(g,a); r['_c']=(COARSE.get(g,g),COARSE.get(a,a)); r['_item']=item(r['image_id']); r['_unseen']=r['_item'] not in dev_items
print('example id', test[0]['image_id'], 'keys', [k for k in test[0] if not k.startswith('_')])
def k(rows,key): return kappa([r[key] for r in rows])
def boot(rows, f, B=5000, cluster=True, seed=0):
    rng=random.Random(seed); out=[]
    if cluster:
        g=collections.defaultdict(list)
        for r in rows: g[r['_item']].append(r)
        keys=list(g)
        for _ in range(B):
            s=[x for kk in rng.choices(keys,k=len(keys)) for x in g[kk]]
            v=f(s)
            if v==v: out.append(v)
    else:
        for _ in range(B):
            v=f(rng.choices(rows,k=len(rows)))
            if v==v: out.append(v)
    out.sort(); return out[int(.025*len(out))], out[int(.975*len(out))-1]
un=[r for r in test if r['_unseen']]; se=[r for r in test if not r['_unseen']]
print('prompts: test',len({r['_item'] for r in test}),'unseen',len({r['_item'] for r in un}),'seen',len({r['_item'] for r in se}))
for name,rows in [('full test',test),('unseen',un),('seen',se)]:
    for key,lab in [('_p','k4'),('_c','k2')]:
        lo,hi=boot(rows,lambda s:k(s,key)); lo2,hi2=boot(rows,lambda s:k(s,key),cluster=False)
        print(f'{name:10s} n={len(rows):4d} {lab}={k(rows,key):.3f} cluster CI [{lo:.2f},{hi:.2f}]  image CI [{lo2:.2f},{hi2:.2f}]')
# difference unseen - seen: independent bootstrap
for key,lab in [('_p','k4'),('_c','k2')]:
    rng=random.Random(1); d=[]
    def cl(rows):
        g=collections.defaultdict(list)
        for r in rows: g[r['_item']].append(r)
        return g
    gu,gs=cl(un),cl(se); ku,ks=list(gu),list(gs)
    for _ in range(5000):
        a=[x for kk in rng.choices(ku,k=len(ku)) for x in gu[kk]]; b=[x for kk in rng.choices(ks,k=len(ks)) for x in gs[kk]]
        d.append(k(a,key)-k(b,key))
    d.sort(); print(f'diff unseen-seen {lab}: {k(un,key)-k(se,key):+.3f} CI [{d[125]:+.2f},{d[4874]:+.2f}]')
# composition
for field in ['family','generator']:
    cu=collections.Counter(r.get(field) for r in un); cs=collections.Counter(r.get(field) for r in test)
    print(field, {kk:(cu[kk], round(cu[kk]/len(un),2), round(cs[kk]/len(test),2)) for kk in sorted(cs, key=str)})
cu=collections.Counter(r['_p'][0] for r in un); cs=collections.Counter(r['_p'][0] for r in test)
print('gold', {kk:(cu[kk], round(cu[kk]/len(un),2), round(cs[kk]/len(test),2)) for kk in cs})
# per family unseen vs seen
for fam in ['cancellation','attribution','figurative','perspectival']:
    a=[r for r in un if r['family']==fam]; b=[r for r in se if r['family']==fam]; t=[r for r in test if r['family']==fam]
    line=f'{fam:13s}'
    for nm,rows in [('unseen',a),('seen',b),('all',t)]:
        if len(rows)>=10:
            lo,hi=boot(rows,lambda s:k(s,'_p'),B=2000); lo2,hi2=boot(rows,lambda s:k(s,'_c'),B=2000)
            line+=f' | {nm} n={len(rows)} k4={k(rows,"_p"):.2f}[{lo:.2f},{hi:.2f}] k2={k(rows,"_c"):.2f}[{lo2:.2f},{hi2:.2f}]'
        else: line+=f' | {nm} n={len(rows)}'
    print(line)
# family-standardised: full-test kappa if families weighted as in unseen subset (resample test to unseen family mix)
rng=random.Random(2); fam_n=collections.Counter(r['family'] for r in un); byfam=collections.defaultdict(list)
for r in se: byfam[r['family']].append(r)
vals4=[];vals2=[]
for _ in range(5000):
    s=[x for f,n in fam_n.items() for x in rng.choices(byfam[f],k=n)]
    vals4.append(k(s,'_p')); vals2.append(k(s,'_c'))
vals4.sort(); vals2.sort()
import statistics
print('seen images resampled to the unseen family mix (n=148): k4 mean %.2f [%.2f,%.2f]; k2 mean %.2f [%.2f,%.2f]'%(statistics.mean(vals4),vals4[125],vals4[4874],statistics.mean(vals2),vals2[125],vals2[4874]))
