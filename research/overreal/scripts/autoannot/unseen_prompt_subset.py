"""Judge vs. human agreement on the test images whose prompt never appears in the development split (reviewer 3P6s, Q1). Run with the Fraud env python."""
import os, sys
os.chdir(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ".")
import json,collections
from common import derive_label, load_split
from eval_auto import kappa, gold_of, COARSE
auto={}
for l in open('/data/users/jiahao_huang/Chekhov/data/overreal_v1/auto/final__claude-opus-5-api.jsonl'):
    r=json.loads(l); r['label']=derive_label(r['answers'],r['questions']); auto[r['image_id']]=r
split=load_split()
item=lambda iid:'/'.join(iid.split('/')[:2])
dev_items={item(r['image_id']) for r in split if r['split']=='dev'}
test=[r for r in split if r['split']=='test' and r['image_id'] in auto]
def report(name,rows):
    ps=[(gold_of(r,auto[r['image_id']]['label']),auto[r['image_id']]['label']) for r in rows]
    c=[(COARSE.get(a,a),COARSE.get(b,b)) for a,b in ps]
    print(f'{name:22s} n={len(ps)} k4={kappa(ps):.2f} k2={kappa(c):.2f}')
report('full test',test)
sub=[r for r in test if item(r['image_id']) not in dev_items]
report('unseen-prompt',sub)
for fam in ['attribution','figurative','perspectival']:
    report('  '+fam,[r for r in sub if r['family']==fam])