"""Family 6 v2 — relevance, with a conversation template instead of a marker.

The pilot's family 6 flagged the irrelevant mention with the words "Unrelated aside:",
which turns a family *defined* by the absence of marking into a marked one. The fix is
to carry the irrelevance in the discourse structure: an ordinary stretch of conversation,
then a "by the way" request. A text-to-image model has no turn mechanism, but Gricean
relevance is a property of the input's structure, not of the architecture — if the
structure is in the input and the model realizes the entity anyway, that is
over-realization. Serialising the conversation into the prompt is therefore enough, and
it avoids introducing a second generator as a confound.

Conditions, all sharing the same conversational shell so the template itself is not the
manipulation:

  S_imp  turn 1 mentions E in passing; the request does not          (implicit noise)
  S_exp  the same, plus a clause saying the mention is unrelated     (explicit noise)
  P      turn 1 is neutral; the request asks for E                   (licensed)
  A      turn 1 is neutral; the request does not mention E           (base rate)

Twenty conversations x three entities = 60 items, following the family-specific ratio
in the paper's P3: for relevance the load-bearing factor is the conversation, not the
entity — whether a mention reads as incidental depends on the surrounding talk, not on
which animal was named. Each conversation is paired with a request drawn from a separate
pool so that entity, conversation and request are counterbalanced rather than fully
crossed (DECISIONS.md #16, #24).

Output: pilot/items/6b_relevance_v2.jsonl
"""
import json
import os

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pilot", "items")

ENTITIES = [
    {"entity": "elephant", "art": "an"},
    {"entity": "tiger", "art": "a"},
    {"entity": "wolf", "art": "a"},
]

# Twenty everyday exchanges. Each is (id, turn-1 template with {E} for the entity slot,
# the neutral turn-1 used in P and A, the assistant's reply). The neutral version is
# matched for length and topic and contains no animal and no zoo or park, so the A
# condition is not primed the way the pilot's "wildlife park" phrasing was.
CONVERSATIONS = [
    ("c01", "My cousin started a new job last month. She looks after {art} {E} — apparently it recognises her footsteps now.",
            "My cousin started a new job last month. She looks after the ticket desk — apparently she has memorised every price already.",
            "That sounds like a rewarding job."),
    ("c02", "I could not sleep last night. There was a documentary on about {art} {E} and I ended up watching the whole thing.",
            "I could not sleep last night. There was a documentary on about deep sea cables and I ended up watching the whole thing.",
            "Late-night documentaries have a way of doing that."),
    ("c03", "My daughter has decided that {art} {E} is her favourite animal. She draws nothing else at the moment.",
            "My daughter has decided that origami is her favourite hobby. She folds nothing else at the moment.",
            "Children get wonderfully single-minded about these things."),
    ("c04", "There was a strange story in the paper — {art} {E} got loose from a transport lorry on the motorway.",
            "There was a strange story in the paper — a shipping container got loose from a transport lorry on the motorway.",
            "That must have caused quite a delay."),
    ("c05", "A colleague of mine has {art} {E} tattooed across his forearm. He says it took four sittings.",
            "A colleague of mine has a compass tattooed across his forearm. He says it took four sittings.",
            "Four sittings is a serious commitment."),
    ("c06", "We went to a talk at the museum yesterday. Most of it was about how {art} {E} finds its way over long distances.",
            "We went to a talk at the museum yesterday. Most of it was about how early clocks kept time over long voyages.",
            "That sounds like an interesting evening."),
    ("c07", "My flatmate is reading a novel where the narrator is {art} {E}. She keeps reading passages out loud to me.",
            "My flatmate is reading a novel where the narrator is a retired postman. She keeps reading passages out loud to me.",
            "An unusual choice of narrator."),
    ("c08", "I lost an argument at dinner about whether {art} {E} can recognise itself in a mirror.",
            "I lost an argument at dinner about whether a kettle boils faster with the lid off.",
            "Dinner arguments are rarely settled."),
    ("c09", "The primary school down the road has {art} {E} painted on the wall by the gate. It has been there for years.",
            "The primary school down the road has a sundial painted on the wall by the gate. It has been there for years.",
            "Those murals do tend to outlast everything else."),
    ("c10", "A friend sent me a photograph of {art} {E} she saw on holiday. I still have not replied.",
            "A friend sent me a photograph of a lighthouse she saw on holiday. I still have not replied.",
            "You should probably reply before she asks."),
    ("c11", "There is a charity near us that raises money for {art} {E}. They had a stall at the market on Saturday.",
            "There is a charity near us that raises money for the lifeboat station. They had a stall at the market on Saturday.",
            "Market stalls are good for that sort of thing."),
    ("c12", "My father tells a story about seeing {art} {E} when he was seven. The details change every time.",
            "My father tells a story about seeing a solar eclipse when he was seven. The details change every time.",
            "Stories do drift over the years."),
    ("c13", "Someone at work brought in a calendar with {art} {E} on every month. Nobody has the heart to take it down.",
            "Someone at work brought in a calendar with a different bridge on every month. Nobody has the heart to take it down.",
            "Office decorations have surprising staying power."),
    ("c14", "I keep meaning to finish the book I started, the one about {art} {E}. It has been on the table for weeks.",
            "I keep meaning to finish the book I started, the one about the salt trade. It has been on the table for weeks.",
            "Books on tables have a habit of staying there."),
    ("c15", "My neighbour's ringtone is the sound {art} {E} makes. I hear it through the wall about six times a day.",
            "My neighbour's ringtone is an old telephone bell. I hear it through the wall about six times a day.",
            "That would wear thin quickly."),
    ("c16", "The pub quiz last night had an entire round on {art} {E}. We got two out of ten.",
            "The pub quiz last night had an entire round on rivers of Europe. We got two out of ten.",
            "Two out of ten is at least memorable."),
    ("c17", "A student of mine wrote her whole dissertation on {art} {E}. It was ninety pages.",
            "A student of mine wrote her whole dissertation on tram networks. It was ninety pages.",
            "Ninety pages is a lot of dedication."),
    ("c18", "I saw a mural on the way in this morning, {art} {E} across the whole side of a building.",
            "I saw a mural on the way in this morning, a harvest scene across the whole side of a building.",
            "Those large murals really change a street."),
    ("c19", "My brother swears he heard {art} {E} while camping last summer. Nobody else did.",
            "My brother swears he heard a church bell while camping last summer. Nobody else did.",
            "Camping does strange things to the ears."),
    ("c20", "There is a soft toy shaped like {art} {E} that my son will not travel without.",
            "There is a soft toy shaped like a rocket that my son will not travel without.",
            "Those become non-negotiable very fast."),
]

# Requests are drawn from a separate pool and rotated, so entity, conversation and request
# are counterbalanced rather than fully crossed.
REQUESTS = [
    ("r1", "a school gymnasium after the game", "Write a three-sentence description of a school gymnasium after the game."),
    ("r2", "a harbour at first light", "Write a three-sentence description of a harbour at first light."),
    ("r3", "a launderette on a rainy evening", "Write a three-sentence description of a launderette on a rainy evening."),
    ("r4", "a hospital waiting room", "Write a three-sentence description of a hospital waiting room."),
    ("r5", "a bakery just before closing", "Write a three-sentence description of a bakery just before closing."),
]

UNRELATED = "This has nothing to do with what I am about to ask."
REPLY_ONLY = " Reply with the description only."


def turns(user1, reply, user2):
    return [{"role": "user", "content": user1},
            {"role": "assistant", "content": reply},
            {"role": "user", "content": user2}]


def as_image_prompt(user1, reply, user2_scene):
    """Serialise the exchange for a text-to-image model, which has no turn mechanism."""
    return (f"User: {user1}\n"
            f"Assistant: {reply}\n"
            f"User: By the way, please draw {user2_scene}.")


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for i, (cid, tmpl_e, tmpl_n, reply) in enumerate(CONVERSATIONS):
        for j, e in enumerate(ENTITIES):
            rid, scene, req = REQUESTS[(i + j) % len(REQUESTS)]
            u1_e = tmpl_e.format(art=e["art"], E=e["entity"])
            u1_n = tmpl_n
            ask = "By the way, " + req[0].lower() + req[1:] + REPLY_ONLY
            ask_p = ("By the way, " + req[0].lower() + req[1:].rstrip(".")
                     + f", including {e['art']} {e['entity']}." + REPLY_ONLY)
            rows.append(dict(
                family="6b_relevance_v2", entity=e["entity"], scenario_id=f"{cid}_{rid}",
                scenario=scene, device="incidental prior mention in conversation",
                conversation_id=cid, request_id=rid,
                prompts={
                    "S_imp": turns(u1_e, reply, ask),
                    "S_exp": turns(u1_e + " " + UNRELATED, reply, ask),
                    "P": turns(u1_n, reply, ask_p),
                    "A": turns(u1_n, reply, ask),
                },
                image_prompts={
                    "S_imp": as_image_prompt(u1_e, reply, scene),
                    "S_exp": as_image_prompt(u1_e + " " + UNRELATED, reply, scene),
                    "P": as_image_prompt(u1_n, reply, f"{scene}, including {e['art']} {e['entity']}"),
                    "A": as_image_prompt(u1_n, reply, scene),
                },
            ))
    for k, r in enumerate(rows):
        r["id"] = f"6b_relevance_v2_{k:02d}"
        # rule 2: the entity must not appear in the A prompt anywhere
        a_text = " ".join(t["content"] for t in r["prompts"]["A"]).lower()
        assert r["entity"] not in a_text, (r["id"], a_text)

    path = os.path.join(OUT, "6b_relevance_v2.jsonl")
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(rows)} items -> {len(rows)*4} prompts -> {path}")
    print("\nexample image prompt (S_imp):\n" + rows[0]["image_prompts"]["S_imp"])


if __name__ == "__main__":
    main()
