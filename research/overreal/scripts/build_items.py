"""Phase 0 — build the text pilot items.

Every item is a (scenario Sc, entity E, licensing device D) triple assembled three ways:

    S = Sc + D(E)   E should NOT be realized
    P = Sc + E      E SHOULD be realized
    A = Sc          coincidental base rate

Construction is templated so that S, P and A differ only in the licensing device
(GOAL.md rule 1), and two checks are enforced programmatically (rule 2): the scenario
text and the task instruction never contain the entity string.

Design: 4 candidate entities x 3 scenarios = 12 items per family cell, so that Q1
("which entity fails in all six families?") has a balanced 3 items per (entity, family).
Family 4 is split into 4a (occlusion, core entity pool) and 4b (diegetic legibility,
carrier pool, no core entity) because the two are scored differently — 7 cells total.

Output: pilot/items/<cell>.jsonl
"""
import json
import os

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pilot", "items")

# --- core entity pool -------------------------------------------------------------
# Four candidate entities for Q1. All four support a fixed simile frame (family 3),
# occlusion (4a) and a stencilled word (5), which is what makes them pool-compatible.
ENTITIES = [
    {"entity": "elephant", "art": "an", "adj": "heavy", "plural": "elephants"},
    {"entity": "snake", "art": "a", "adj": "slippery", "plural": "snakes"},
    {"entity": "tiger", "art": "a", "adj": "fierce", "plural": "tigers"},
    {"entity": "wolf", "art": "a", "adj": "hungry", "plural": "wolves"},
]

TASK3 = "Write a three-sentence description of {x}. Reply with the description only."
TASK_SEE = "Write three sentences describing what {who} can see. Reply with the description only."


def user(text):
    return [{"role": "user", "content": text}]


def items_f1():
    """Family 1 — existence-canceling. Device: 'no'."""
    scenarios = [
        ("f1s1", "a hotel lobby at midnight"),
        ("f1s2", "a school gymnasium after the game"),
        ("f1s3", "a riverside campsite at dawn"),
    ]
    out = []
    for e in ENTITIES:
        for sid, sc in scenarios:
            out.append(dict(
                family="1_existence", entity=e["entity"], scenario_id=sid, scenario=sc,
                device=f"no {e['entity']}",
                prompts={
                    "S": user(TASK3.format(x=f"{sc} with no {e['entity']} in it")),
                    "P": user(TASK3.format(x=f"{sc} with {e['art']} {e['entity']} in it")),
                    "A": user(TASK3.format(x=sc)),
                },
                image_prompts={
                    "S": f"{sc} with no {e['entity']} in it",
                    "P": f"{sc} with {e['art']} {e['entity']} in it",
                    "A": sc,
                },
            ))
    return out


def items_f2():
    """Family 2 — attribution. Device: 'believed'. Failure = realized *unmarked*."""
    scenarios = [
        ("f2s1", "the ridge", "Maya"),
        ("f2s2", "the loading dock", "Tomas"),
        ("f2s3", "the orchard", "Iris"),
    ]
    out = []
    for e in ENTITIES:
        for sid, place, person in scenarios:
            task = TASK3.format(x=place)
            out.append(dict(
                family="2_attribution", entity=e["entity"], scenario_id=sid, scenario=place,
                device=f"{person} believed", person=person,
                prompts={
                    "S": user(f"{person} believed there was {e['art']} {e['entity']} at {place}. {task}"),
                    "P": user(f"There was {e['art']} {e['entity']} at {place}. {task}"),
                    "A": user(task),
                },
                image_prompts={
                    "S": f"{person} at {place}, believing that there is {e['art']} {e['entity']} at {place}",
                    "P": f"{person} at {place}, and there is {e['art']} {e['entity']} at {place}",
                    "A": f"{person} at {place}",
                },
            ))
    return out


def items_f3():
    """Family 3 — figurative. Device: 'as ADJ as'. Failure = literal, unmarked."""
    scenarios = [
        ("f3s1", "The porter", "the porter arriving at the inn"),
        ("f3s2", "The coach", "the coach at halftime"),
        ("f3s3", "The auditor", "the auditor entering the office"),
    ]
    out = []
    for e in ENTITIES:
        for sid, subj, sc in scenarios:
            task = TASK3.format(x=sc)
            out.append(dict(
                family="3_figurative", entity=e["entity"], scenario_id=sid, scenario=sc,
                device=f"as {e['adj']} as {e['art']} {e['entity']}",
                prompts={
                    "S": user(f"{subj} was as {e['adj']} as {e['art']} {e['entity']}. {task}"),
                    "P": user(f"{subj} was standing beside {e['art']} {e['entity']}. {task}"),
                    "A": user(task),
                },
                image_prompts={
                    "S": f"{sc}, as {e['adj']} as {e['art']} {e['entity']}",
                    "P": f"{sc}, standing beside {e['art']} {e['entity']}",
                    "A": sc,
                },
            ))
    return out


def items_f4a():
    """Family 4a — perspectival occlusion. Failure = narrating what the focalizer cannot see."""
    scenarios = [
        ("f4as1", "Tomas", "the high stone wall"),
        ("f4as2", "Iris", "the parked lorry"),
        ("f4as3", "Delia", "the hedge at the end of the yard"),
    ]
    out = []
    for e in ENTITIES:
        for sid, obs, barrier in scenarios:
            task = TASK_SEE.format(who=obs)
            out.append(dict(
                family="4a_occlusion", entity=e["entity"], scenario_id=sid, scenario=barrier,
                device=f"completely hidden behind {barrier}", person=obs,
                prompts={
                    "S": user(f"From where {obs} stood, {e['art']} {e['entity']} was completely hidden behind {barrier}. {task}"),
                    "P": user(f"From where {obs} stood, {e['art']} {e['entity']} was in plain view beside {barrier}. {task}"),
                    "A": user(f"{obs} stood beside {barrier}. {task}"),
                },
                image_prompts={
                    "S": f"{e['art']} {e['entity']} completely hidden behind {barrier}, photographed from where {obs} stands",
                    "P": f"{e['art']} {e['entity']} in plain view beside {barrier}, photographed from where {obs} stands",
                    "A": f"{barrier}, photographed from where {obs} stands",
                },
            ))
    return out


def items_f4a2():
    """Family 4a, second construction — occlusion with the camera at the observer.

    Three problems with items_f4a, all of which make its result ambiguous:
    the viewpoint anchor was vacuous (the observer was never in frame and the camera was
    not his), a correct S image was indistinguishable from A, and a failure could not be
    told apart from the model simply ignoring the barrier. Here the prompt puts the
    camera *at* the observer, so the audience's access and the observer's access coincide
    (DECISIONS.md #11), and a validity question about the barrier separates "ignored the
    occlusion" from "never drew the wall".

    S comes in two strengths, so that explicitness of marking becomes a measured factor
    inside the family rather than only a stipulated ordering between families:
      S_exp  states the perceptual fact outright ("that Tomas cannot see")
      S_imp  gives only the spatial relation ("on the far side"), occlusion to be inferred
    """
    scenarios = [
        ("f4a2s1", "Tomas", "a high stone wall"),
        ("f4a2s2", "Iris", "a tall wooden fence"),
        ("f4a2s3", "Delia", "a parked lorry"),
    ]
    out = []
    for e in ENTITIES:
        for sid, obs, barrier in scenarios:
            task = TASK_SEE.format(who=obs)
            view = f"the view from where {obs} stands"
            out.append(dict(
                family="4a2_occlusion_v2", entity=e["entity"], scenario_id=sid,
                scenario=barrier, device="far side of the barrier", person=obs,
                prompts={
                    "S_exp": user(f"{obs} stands facing {barrier}. {e['art'].capitalize()} "
                                  f"{e['entity']} stands on the far side of it, where {obs} "
                                  f"cannot see it. {task}"),
                    "S_imp": user(f"{obs} stands facing {barrier}. {e['art'].capitalize()} "
                                  f"{e['entity']} stands on the far side of it. {task}"),
                    "P": user(f"{obs} stands facing {barrier}. {e['art'].capitalize()} "
                              f"{e['entity']} stands on {obs}'s side of it. {task}"),
                    "A": user(f"{obs} stands facing {barrier}. {task}"),
                },
                image_prompts={
                    "S_exp": f"{view}: {barrier}, with {e['art']} {e['entity']} behind it "
                             f"that {obs} cannot see",
                    "S_imp": f"{view}: {barrier}, with {e['art']} {e['entity']} on the far side",
                    "P": f"{view}: {e['art']} {e['entity']} standing on this side of {barrier}",
                    "A": f"{view}: {barrier}",
                },
                barrier=barrier,
            ))
    return out


CARRIERS = [
    ("f4bc1", "a letter"),
    ("f4bc2", "a phone screen"),
    ("f4bc3", "a laptop screen"),
    ("f4bc4", "a notebook page"),
]
MESSAGES = ["I'm sorry", "the meeting is cancelled", "flight 402 at nine"]


def items_f4b():
    """Family 4b — diegetic legibility. Carrier pool, not the core entity pool.

    The realized unit is the *information*, not an object: failure = the observer's
    description reports what the inscription says.
    """
    out = []
    for cid, carrier in CARRIERS:
        for i, msg in enumerate(MESSAGES):
            task = TASK_SEE.format(who="Ben")
            out.append(dict(
                family="4b_legibility", entity=msg, scenario_id=f"{cid}m{i+1}", scenario=carrier,
                device="reading, back to the doorway", person="Ben",
                prompts={
                    "S": user(f'Anna sits at the table reading {carrier} that says "{msg}". Ben stands in the doorway behind her. {task}'),
                    "P": user(f'Anna sits at the table holding up {carrier} that says "{msg}" toward the doorway. Ben stands in the doorway facing her. {task}'),
                    "A": user(f'Anna sits at the table reading {carrier}. Ben stands in the doorway behind her. {task}'),
                },
                image_prompts={
                    "S": f'Anna sits at the table reading {carrier} that says "{msg}"',
                    "P": f'Anna holds up {carrier} that says "{msg}", turned toward the camera',
                    "A": f'Anna sits at the table reading {carrier}',
                },
            ))
    return out


def items_f5():
    """Family 5 — use-mention. Device: the word, stencilled. Failure = the referent appears."""
    scenarios = [
        ("f5s1", "the warehouse"),
        ("f5s2", "the freight yard"),
        ("f5s3", "the loading bay"),
    ]
    out = []
    for e in ENTITIES:
        for sid, sc in scenarios:
            task = TASK3.format(x=sc)
            out.append(dict(
                family="5_use_mention", entity=e["entity"], scenario_id=sid, scenario=sc,
                device=f"crate stencilled with the word {e['entity'].upper()}",
                prompts={
                    "S": user(f"In {sc} stands a crate stencilled with the word {e['entity'].upper()}. {task}"),
                    "P": user(f"In {sc} stands a crate beside {e['art']} {e['entity']}. {task}"),
                    "A": user(f"In {sc} stands a crate. {task}"),
                },
                image_prompts={
                    # The image side is the dual of the text side, so the realized unit
                    # is the *word form*, not the referent: "a crate of elephants" must
                    # not come back with ELEPHANT stencilled on it. P licenses the word.
                    "S": f"{sc} with a crate of {e['plural']}",
                    "P": f"{sc} with a crate stencilled with the word {e['entity'].upper()}",
                    "A": f"{sc} with a crate",
                },
                image_target="word_form",
            ))
    return out


TEXT_BEARING = [
    {"entity": "book", "art": "a"},
    {"entity": "letter", "art": "a"},
    {"entity": "newspaper", "art": "a"},
    {"entity": "menu", "art": "a"},
]


def items_f5b():
    """Family 5, second construction — text-bearing carriers.

    The crate construction (items_f5) gave the model no occasion to render text at all,
    so its null result says nothing about the family. These entities *must* carry text,
    and the norm is that they carry their content, not their own category name: a book
    cover shows a title, not the word BOOK. That makes the use-mention choice live.

    Realized unit = the entity's own word form rendered as text in the image.
    """
    scenarios = [
        ("f5bs1", "a wooden desk by a window"),
        ("f5bs2", "a cafe table in the morning"),
        ("f5bs3", "a shop counter"),
    ]
    out = []
    for e in TEXT_BEARING:
        for sid, sc in scenarios:
            task = TASK3.format(x=f"{sc} with {e['art']} {e['entity']} on it")
            out.append(dict(
                family="5b_text_bearing", entity=e["entity"], scenario_id=sid, scenario=sc,
                device=f"plain use of '{e['entity']}'",
                prompts={
                    "S": user(task),
                    "P": user(TASK3.format(
                        x=f"{sc} with {e['art']} {e['entity']} on it, the word "
                          f"{e['entity'].upper()} printed across it")),
                    "A": user(TASK3.format(x=sc)),
                },
                image_prompts={
                    "S": f"{sc} with {e['art']} {e['entity']} on it",
                    "P": f"{sc} with {e['art']} {e['entity']} on it, the word "
                         f"{e['entity'].upper()} printed across it",
                    "A": sc,
                },
                image_target="word_form",
            ))
    return out


def items_f6():
    """Family 6 — relevance. No licensing device; the incidental prior mention is the manipulation.

    Turn structure is held identical across conditions: prior remark, acknowledgement,
    request. S carries the entity in the prior remark; P licenses it in the request; A
    has it nowhere.
    """
    requests = [
        ("f6s1", "Write a three-sentence description of a peaceful place.", "a peaceful place"),
        ("f6s2", "Write a three-sentence description of a busy market.", "a busy market"),
        ("f6s3", "Write a three-sentence opening for a bedtime story.", "a bedtime story illustration"),
    ]
    prior_neutral = "My cousin works at a wildlife park and looks after the ticket desk."
    ack = "Noted."
    out = []
    for e in ENTITIES:
        prior_entity = f"My cousin works at a wildlife park and looks after {e['art']} {e['entity']}."
        for sid, req, scene in requests:
            def turns(prior, request):
                return [
                    {"role": "user", "content": prior},
                    {"role": "assistant", "content": ack},
                    {"role": "user", "content": request + " Reply with the description only."},
                ]
            out.append(dict(
                family="6_relevance", entity=e["entity"], scenario_id=sid, scenario=req,
                device="incidental prior mention",
                prompts={
                    "S": turns(prior_entity, req),
                    "P": turns(prior_neutral, req + f" Include {e['art']} {e['entity']}."),
                    "A": turns(prior_neutral, req),
                },
                # T2I has no conversation, so the incidental mention rides along in the
                # prompt as an explicitly unrelated aside — the closest single-prompt analogue
                image_prompts={
                    "S": f"{scene}. Unrelated aside: my cousin looks after {e['art']} {e['entity']} at a wildlife park.",
                    "P": f"{scene} with {e['art']} {e['entity']}. Unrelated aside: my cousin looks after the ticket desk at a wildlife park.",
                    "A": f"{scene}. Unrelated aside: my cousin looks after the ticket desk at a wildlife park.",
                },
            ))
    return out


BUILDERS = [items_f1, items_f2, items_f3, items_f4a, items_f4a2, items_f4b, items_f5,
            items_f5b,
            items_f6]


def check_leakage(item):
    """Rule 2 — the entity string must not appear in the scenario or the instruction."""
    ent = item["entity"].lower()
    if item["family"] == "4b_legibility":
        return []  # 4b's 'entity' is the message; the carrier scenario cannot contain it
    problems = []
    if ent in item["scenario"].lower():
        problems.append(f"entity in scenario: {item['scenario']}")
    a_text = " ".join(t["content"] for t in item["prompts"]["A"] if t["role"] == "user").lower()
    if ent in a_text:
        problems.append(f"entity in A prompt: {a_text}")
    return problems


def main():
    os.makedirs(OUT, exist_ok=True)
    total, bad = 0, 0
    for build in BUILDERS:
        rows = build()
        cell = rows[0]["family"]
        for i, r in enumerate(rows):
            r["id"] = f"{cell}_{i:02d}"
            probs = check_leakage(r)
            if probs:
                bad += 1
                print(f"LEAKAGE {r['id']}: {probs}")
        with open(os.path.join(OUT, f"{cell}.jsonl"), "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        total += len(rows)
        print(f"{cell}: {len(rows)} items -> {len(rows)*3} prompts")
    print(f"TOTAL {total} items, {total*3} prompts per model; leakage failures: {bad}")


if __name__ == "__main__":
    main()
