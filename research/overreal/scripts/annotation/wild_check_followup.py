"""Wild human check, follow-up for reviewer 3P6s: bootstrap interval of the reweighted over-realization rate per family, the perspectival images where the automatic and human labels disagree, and the automatic vs. human over-realization rate on the OverReal-Det test split. Run with the Fraud env python."""
import os, sys, json, collections, random, glob
ROOT="/data/users/jiahao_huang/Chekhov"; os.chdir(ROOT)
sys.path.insert(0, "research/overreal/scripts/autoannot")
from common import derive_label, load_split
from eval_auto import kappa, gold_of, COARSE
rj=lambda p:[json.loads(l) for l in open(p)]
# ---- A2: mental-state unseen confusion
auto={}
for l in open('data/overreal_v1/auto/final__claude-opus-5-api.jsonl'):
    r=json.loads(l); r['label']=derive_label(r['answers'],r['questions']); auto[r['image_id']]=r
split=load_split(); item=lambda iid:'/'.join(iid.split('/')[:2])
dev_items={item(r['image_id']) for r in split if r['split']=='dev'}
test=[r for r in split if r['split']=='test' and r['image_id'] in auto]
un=[r for r in test if item(r['image_id']) not in dev_items]
ms=[r for r in un if r['family']=='attribution']
print('mental-state unseen: images',len(ms),'prompts',len({item(r['image_id']) for r in ms}))
print(collections.Counter((gold_of(r,auto[r['image_id']]['label']),auto[r['image_id']]['label']) for r in ms))
# ---- B: wild
samp={r["image_id"]:r for r in rj("research/overreal/wild/human_check_sample.jsonl")}
ann={}
for f in sorted(glob.glob("data/wild_check_annotations/ann-*.jsonl")):
    for r in rj(f): ann[r["image_id"]]=r
LABS=["disruptive","silent","integrated","withheld","other"]; OR=lambda l:l in("disruptive","silent")
rows=collections.defaultdict(list)
for iid,s in samp.items():
    a=ann[iid]; j=s["judge_label"]; h=j if j in a["labels"] else a["labels"][0]
    rows[s["family"]].append((j,h,s))
uni=collections.defaultdict(collections.Counter); seen=set()
for f in sorted(glob.glob("data/overreal_v1/auto/final__claude-opus-5-api__wild_*.jsonl")):
    for r in rj(f):
        if r["image_id"] in seen: continue
        seen.add(r["image_id"]); fam=r["image_id"].split("/")[-1].split("_wild_")[0]
        l=r.get("label") or derive_label(r["answers"],r["questions"])
        if l: uni[fam][l]+=1
def rate(fam, rws, drop_other=False):
    c=collections.Counter((j,h) for j,h,_ in rws); u=uni[fam]; num=den=0
    for j in LABS:
        if j=="other": continue
        tot=sum(c[(j,h)] for h in LABS)
        if not tot: 
            if u[j]: num+=u[j]*(1 if OR(j) else 0); den+=u[j]
            continue
        p_or=sum(c[(j,h)] for h in LABS if OR(h))/tot; p_oth=c[(j,"other")]/tot
        num+=u[j]*p_or; den+=u[j]*((1-p_oth) if drop_other else 1)
    return num/den
for fam in ["cancellation","attribution","figurative","perspectival"]:
    u=uni[fam]; N=sum(v for k,v in u.items() if k!="other")
    auto_or=sum(v for k,v in u.items() if OR(k))/N
    rng=random.Random(0); bs=[]; by=collections.defaultdict(list)
    for x in rows[fam]: by[x[0]].append(x)
    for _ in range(5000):
        s=[y for j,xs in by.items() for y in rng.choices(xs,k=len(xs))]
        bs.append(rate(fam,s))
    bs.sort()
    print(f'{fam:13s} universe {dict(u)} auto OR {auto_or:.3f} reweighted {rate(fam,rows[fam]):.3f} CI [{bs[125]:.3f},{bs[4874]:.3f}] (human-other dropped from denominator: {rate(fam,rows[fam],True):.3f})')
print('--- perspectival: automatic over-realized, human not')
for j,h,s in rows["perspectival"]:
    if OR(j) and not OR(h): print(f'{j:10s}->{h:10s} {s["generator"]:8s} target={s["target"]!r} | {s["prompt"][:150]}')
print('--- perspectival: automatic silent, human disruptive'); 
for j,h,s in rows["perspectival"]:
    if j=="silent" and h=="disruptive": print(f'{s["generator"]:8s} target={s["target"]!r} | {s["prompt"][:110]}')
# ---- C: OverReal-Det test: automatic vs human OR rate per family
print('--- OverReal-Det test: OR rate by automatic vs human gold (other excluded on both)')
for fam in ['cancellation','attribution','figurative','perspectival']:
    t=[r for r in test if r['family']==fam]
    ps=[(gold_of(r,auto[r['image_id']]['label']),auto[r['image_id']]['label']) for r in t]
    ps=[(g,a) for g,a in ps if g!='other' and a not in('other',None)]
    print(f'{fam:13s} n={len(ps)} human OR {sum(OR(g) for g,a in ps)/len(ps):.3f} auto OR {sum(OR(a) for g,a in ps)/len(ps):.3f}  auto-OR&human-correct {sum(OR(a) and not OR(g) for g,a in ps)} auto-correct&human-OR {sum(OR(g) and not OR(a) for g,a in ps)}')
