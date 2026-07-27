"""Family 6 v3 — prose serialisation, after the transcript template failed on FLUX.

build_f6v2.py serialises the exchange with `User:` / `Assistant:` markers. FLUX reads
that as a description of a comic strip and draws one: speech bubbles, panel borders and
garbled lettering, in the A condition as much as in S, so the effect is the template and
not the family. The A condition is what caught it.

This variant keeps the same content and the same incidental-mention structure but drops
every cue that the input is a transcript: no speaker labels, no turn boundaries, just one
person talking. The suppression device is still purely pragmatic — the mention is
irrelevant to the request and nothing says so — which is what family 6 requires.

The v2 items remain in the repository; both are reported (GOAL.md rule 4).

Output: pilot/items/6c_relevance_v3.jsonl
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_f6v2 import CONVERSATIONS, ENTITIES, REQUESTS, UNRELATED  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pilot", "items")

TASK = "Write a three-sentence description of {scene}. Reply with the description only."


def prose(remark, scene):
    """One speaker, no labels: a passing remark, then the request."""
    return f"{remark} Anyway — {scene}."


def main():
    rows = []
    for i, (cid, tmpl_e, tmpl_n, _reply) in enumerate(CONVERSATIONS):
        for j, e in enumerate(ENTITIES):
            rid, scene, _req = REQUESTS[(i + j) % len(REQUESTS)]
            u_e = tmpl_e.format(art=e["art"], E=e["entity"])
            u_n = tmpl_n
            task = TASK.format(scene=scene)
            rows.append(dict(
                family="6c_relevance_v3", entity=e["entity"], scenario_id=f"{cid}_{rid}",
                scenario=scene, device="incidental prior remark, prose",
                conversation_id=cid, request_id=rid,
                prompts={
                    "S_imp": [{"role": "user", "content": f"{u_e} Anyway — {task}"}],
                    "S_exp": [{"role": "user", "content": f"{u_e} {UNRELATED} {task}"}],
                    "P": [{"role": "user", "content": f"{u_n} Anyway — " + TASK.format(
                        scene=f"{scene}, including {e['art']} {e['entity']}")}],
                    "A": [{"role": "user", "content": f"{u_n} Anyway — {task}"}],
                },
                image_prompts={
                    "S_imp": prose(u_e, scene),
                    "S_exp": prose(u_e + " " + UNRELATED, scene),
                    "P": prose(u_n, f"{scene}, including {e['art']} {e['entity']}"),
                    "A": prose(u_n, scene),
                },
            ))
    for k, r in enumerate(rows):
        r["id"] = f"6c_relevance_v3_{k:02d}"
        a_text = " ".join(t["content"] for t in r["prompts"]["A"]).lower()
        assert r["entity"] not in a_text, (r["id"], a_text)

    path = os.path.join(OUT, "6c_relevance_v3.jsonl")
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(rows)} items -> {len(rows)*4} prompts -> {path}")
    for c in ("S_imp", "P", "A"):
        print(f"\n{c}: {rows[0]['image_prompts'][c]}")


if __name__ == "__main__":
    main()
