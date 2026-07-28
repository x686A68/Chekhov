"""Third round of constructions, both requested after reviewing round two.

**4a3 — occlusion with occlusion actually entailed.**
The v2 wording said an entity was "on the far side" of "a tall wooden fence". That does
not entail invisibility: an elephant is taller than a fence, so rendering it above the
fence is a faithful reading, not a failure. S_imp was therefore not a weaker *marking* of
the same fact, it was a weaker *fact*, which undercuts the explicit/implicit comparison
the cell exists to make. v3 fixes it two ways: every barrier is given an absolute height
that exceeds every entity, and the entity is placed "entirely" beyond it, so occlusion
follows from the geometry alone. The observer stays in frame, which v2 established works.
n also doubles, 4 entities x 6 barriers = 24 items, because this cell carries the
pilot's largest claimed effect on 12 items.

Full crossing is used here deliberately, unlike family 3: any animal behind any wall is a
natural combination, so the naturalness objection to exhaustive crossing does not apply.

**6d — family 6 v3 with an explicit verb in the request.**
6c's image prompt ends "Anyway — a school gymnasium after the game." with no verb. Adding
"draw" makes the request an instruction, which is what the failed transcript version did
too, so the direction is not obvious a priori: it may clarify the request or it may bring
back the comic-strip artefact. The text prompts already contain a verb, so only the image
prompts differ and the text side does not need rerunning.

Output: pilot/items/4a3_occlusion_v3.jsonl, pilot/items/6d_relevance_draw.jsonl
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_items import ENTITIES, TASK_SEE, user  # noqa: E402
from build_f6v2 import CONVERSATIONS, ENTITIES as F6_ENTITIES, REQUESTS, UNRELATED  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pilot", "items")

# Every barrier is solid, unbroken and taller than the tallest entity (an elephant, ~3 m),
# so "entirely on the far side" entails "not visible from this side".
BARRIERS = [
    ("f4a3b1", "Tomas", "a four-metre brick wall"),
    ("f4a3b2", "Iris", "a solid steel hoarding four metres high"),
    ("f4a3b3", "Delia", "the windowless brick side of a warehouse"),
    ("f4a3b4", "Rafael", "a five-metre concrete flood wall"),
    ("f4a3b5", "Nadia", "a solid timber fence four metres high"),
    ("f4a3b6", "Kwame", "the blank rear wall of a cinema"),
]


def items_4a3():
    out = []
    for e in ENTITIES:
        for bid, obs, barrier in BARRIERS:
            task = TASK_SEE.format(who=obs)
            view = f"the view from where {obs} stands"
            here = f"{obs} stands facing {barrier}."
            out.append(dict(
                family="4a3_occlusion_v3", entity=e["entity"], scenario_id=bid,
                scenario=barrier, device="entirely beyond a barrier taller than the entity",
                person=obs,
                prompts={
                    "S_exp": user(f"{here} {e['art'].capitalize()} {e['entity']} stands "
                                  f"entirely on the far side of it, where {obs} cannot see "
                                  f"it. {task}"),
                    "S_imp": user(f"{here} {e['art'].capitalize()} {e['entity']} stands "
                                  f"entirely on the far side of it. {task}"),
                    "P": user(f"{here} {e['art'].capitalize()} {e['entity']} stands on "
                              f"{obs}'s side of it. {task}"),
                    "A": user(f"{here} {task}"),
                },
                image_prompts={
                    "S_exp": f"{view}: {barrier}, with {e['art']} {e['entity']} entirely on "
                             f"the far side of it, which {obs} cannot see",
                    "S_imp": f"{view}: {barrier}, with {e['art']} {e['entity']} entirely on "
                             f"the far side of it",
                    "P": f"{view}: {e['art']} {e['entity']} standing on this side of {barrier}",
                    "A": f"{view}: {barrier}",
                },
                barrier=barrier,
            ))
    return out


def items_6d():
    """Identical to 6c except that the image request carries the verb 'draw'."""
    rows = []
    for i, (cid, tmpl_e, tmpl_n, _reply) in enumerate(CONVERSATIONS):
        for j, e in enumerate(F6_ENTITIES):
            rid, scene, _req = REQUESTS[(i + j) % len(REQUESTS)]
            u_e = tmpl_e.format(art=e["art"], E=e["entity"])
            u_n = tmpl_n

            def ask(remark, what):
                return f"{remark} Anyway — draw {what}."

            rows.append(dict(
                family="6d_relevance_draw", entity=e["entity"], scenario_id=f"{cid}_{rid}",
                scenario=scene, device="incidental prior remark, prose, verb in request",
                conversation_id=cid, request_id=rid,
                image_prompts={
                    "S_imp": ask(u_e, scene),
                    "S_exp": ask(u_e + " " + UNRELATED, scene),
                    "P": ask(u_n, f"{scene}, including {e['art']} {e['entity']}"),
                    "A": ask(u_n, scene),
                },
            ))
    return rows


def main():
    for build, name in ((items_4a3, "4a3_occlusion_v3"), (items_6d, "6d_relevance_draw")):
        rows = build()
        for k, r in enumerate(rows):
            r["id"] = f"{name}_{k:02d}"
            a = r.get("prompts", r["image_prompts"])["A"]
            a_text = a if isinstance(a, str) else " ".join(t["content"] for t in a)
            assert r["entity"] not in a_text.lower(), (r["id"], a_text)
        with open(os.path.join(OUT, f"{name}.jsonl"), "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        n_cond = len(rows[0]["image_prompts"])
        print(f"{name}: {len(rows)} items x {n_cond} = {len(rows)*n_cond} images")
        for c in ("S_imp", "P", "A"):
            print(f"   {c}: {rows[0]['image_prompts'][c]}")
        print()


if __name__ == "__main__":
    main()
