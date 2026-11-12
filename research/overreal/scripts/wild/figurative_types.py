"""Figurative prompts: appearance/shape comparisons vs. quality/behavior comparisons, on OverReal-Gen and in the wild, with outcome shares per type (reviewer KXAW, Q3).
Run from the repo root with the Fraud env python.
"""
import os
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".."))
import json,collections,re,sys,glob
sys.path.insert(0,'research/overreal/scripts/wild'); sys.path.insert(0,'research/overreal/scripts/autoannot')
import wild_rates as W
from common import derive_label
APPEAR=re.compile(r'shaped?\b|shape\b|looks? (just |exactly )?like|looking like|resembl|alike\b|lookalike|\w+ ?-? ?like\b(?! a\b| an\b| the\b)|like appearance|similar to|esque\b|in the form of|made (out )?of|the size of|sized\b|as if it were|modeled after|features|face\b',re.I)
wild={r['item_id']:r for r in W.read_jsonl('data/generation/wild_prompts.jsonl') if r['family']=='figurative'}
gen={r['item_id']:r for r in W.read_jsonl('data/generation/prompts.jsonl') if r['family']=='figurative'}
isapp=lambda p: bool(APPEAR.search(p))
wa=sum(isapp(r['prompt']) for r in wild.values()); ga=sum(isapp(r['prompt']) for r in gen.values())
print(f'appearance-type: wild {wa}/{len(wild)}={wa/len(wild):.2f}  gen {ga}/{len(gen)}={ga/len(gen):.2f}')
print('wild NON-appearance:'); [print('  ',r['target'],'|',r['prompt'][:100]) for r in wild.values() if not isapp(r['prompt'])]
print('gen appearance-type sample:'); 
for r in list(r for r in gen.values() if isapp(r['prompt']))[:12]: print('  ',r['target'],'|',r['prompt'][:100])
# within-Gen: DO/OR by type, raw cond, eval sample, all generators
es={r['image_id']:r for r in W.read_jsonl('data/overreal_v1/eval_sample.jsonl')}
labs=collections.defaultdict(list)
for r in W.read_jsonl('data/overreal_v1/auto/final__claude-opus-5-api__sample.jsonl'):
    e=es.get(r['image_id'])
    if not e or e['family']!='figurative' or e['prompt_cond'] not in ('raw','deployed'): continue
    item='/'.join(r['image_id'].split('/')[:2]); l=derive_label(r['answers'],r['questions'])
    if l and l!='other' and item in gen: labs['appearance' if isapp(gen[item]['prompt']) else 'quality'].append(l)
for k,v in labs.items():
    c=collections.Counter(v); n=len(v)
    print(f'GEN {k:10s} n={n} DO {c["disruptive"]/n:.2f} SO {c["silent"]/n:.2f} I {c["integrated"]/n:.2f} W {c["withheld"]/n:.2f} DO/OR {c["disruptive"]/(c["disruptive"]+c["silent"]):.2f}')
# wild by type
wl=collections.defaultdict(list); seen=set()
for f in sorted(glob.glob('data/overreal_v1/auto/final__claude-opus-5-api__wild_*.jsonl')):
    for r in W.read_jsonl(f):
        if r['image_id'] in seen: continue
        seen.add(r['image_id']); _,m,_,name=r['image_id'].split('/')
        if not name.startswith('figurative'): continue
        fam,rest=name.split('_wild_'); item=f"{fam}/wild_{rest.split('__')[0]}"; l=derive_label(r['answers'],r['questions'])
        if l and l!='other' and item in wild: wl['appearance' if isapp(wild[item]['prompt']) else 'quality'].append(l)
for k,v in wl.items():
    c=collections.Counter(v); n=len(v)
    print(f'WILD {k:10s} n={n} DO {c["disruptive"]/n:.2f} SO {c["silent"]/n:.2f} I {c["integrated"]/n:.2f} W {c["withheld"]/n:.2f} DO/OR {c["disruptive"]/max(1,c["disruptive"]+c["silent"]):.2f}')