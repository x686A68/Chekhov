"""
Build paired (Treatment/Control) datasets for the Chekhov's Gun intrusion study.

Two paradigms:
  1) multi-turn dialogue  -> data_dialogue.json
  2) reading comprehension -> data_reading.json

Each item is a paired minimal-difference example:
  - treatment: context contains a salient but IRRELEVANT distractor D ("the gun")
  - control:   identical context with D removed / replaced by inert filler
  - the final question Q is identical in both, and its correct answer does NOT involve D.
We later measure whether the model's answer to Q mentions D (lexical intrusion),
and (mechanistically) how much D's probability is elevated.
"""
import json, os, itertools, re

OUT = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Distractors D: (planting_phrase, [surface keyword regexes to detect intrusion])
# Chosen to be DISTINCTIVE (not generic words) so control-group intrusion ~ 0,
# and topically UNRELATED to the target questions below.
# ---------------------------------------------------------------------------
# NOTE: distractors are chosen to be DISTINCTIVE proper-noun / rare-entity topics
# that cannot plausibly appear as a natural answer to the lifestyle/advice questions
# used below. This keeps the control-group intrusion rate at the true floor (~0),
# so any treatment-group hit is a genuine priming-induced intrusion rather than a
# coincidental on-topic mention. (Overlap-prone topics such as succulents, sourdough,
# honey, or lavender were deliberately excluded after a pilot showed they surface
# naturally in gift/breakfast/sleep advice.)
DISTRACTORS = [
    ("my sister just got back from a trip to see the volcanoes in Iceland",
        [r"iceland", r"volcano"]),
    ("we adopted a pet axolotl for the kids last month",
        [r"axolotl"]),
    ("I got completely hooked on the board game Catan over the holidays",
        [r"catan"]),
    ("my roommate keeps talking about the Voyager 2 space probe",
        [r"voyager", r"space probe"]),
    ("I've been trying to identify constellations with a stargazing app",
        [r"constellation", r"stargaz"]),
    ("my cousin competes in competitive dog agility trials",
        [r"dog agility"]),
    ("I'm fascinated by the deep-sea anglerfish documentaries",
        [r"anglerfish"]),
    ("my neighbor restores vintage Vespa scooters",
        [r"vespa"]),
    ("I've been reading about the history of the Silk Road",
        [r"silk road"]),
    ("my brother is really into medieval trebuchets and siege engines",
        [r"trebuchet", r"siege engine"]),
    ("a friend showed me her collection of antique typewriters",
        [r"typewriter"]),
    ("I keep seeing documentaries about the Mariana Trench",
        [r"mariana"]),
]

# Inert control filler: same conversational slot, introduces NO reusable topic.
CONTROL_FILLERS = [
    "honestly my day has been pretty ordinary, nothing much worth mentioning",
    "anyway, things have been fairly quiet on my end lately",
    "not much new to report really, just the usual routine",
    "it's been a calm and uneventful stretch for me recently",
    "nothing too exciting going on with me these days",
]

# ---------------------------------------------------------------------------
# DIALOGUE paradigm
# Structure: [user plants D] -> [k neutral filler exchanges] -> [final Q]
# Q is open-ended (room for intrusion) and unrelated to every D above.
# distance = number of turns between the planting turn and the question.
# ---------------------------------------------------------------------------
FILLER_EXCHANGES = [
    ("Can you remind me what the boiling point of water is at sea level?",
     "At sea level, water boils at 100 degrees Celsius (212 degrees Fahrenheit)."),
    ("What's the difference between weather and climate?",
     "Weather is the short-term state of the atmosphere; climate is the long-term average over decades."),
    ("How many time zones does the world have?",
     "There are 24 standard time zones, though some regions use half-hour or quarter-hour offsets."),
    ("What's a quick way to convert miles to kilometers?",
     "Multiply miles by about 1.61 to get kilometers."),
    ("Can you spell 'necessary' for me?",
     "It's spelled n-e-c-e-s-s-a-r-y."),
]

# Final open questions Q (answer has nothing to do with any distractor).
FINAL_QUESTIONS = [
    "Anyway, can you suggest a simple and relaxing weekend morning routine?",
    "By the way, what are a couple of good tips for staying focused while studying?",
    "Anyway, could you recommend a few ways to save money on groceries?",
    "On a different note, what's a good beginner strategy for getting better at running?",
    "Anyway, do you have advice for someone trying to sleep better at night?",
    "By the way, what are some thoughtful low-cost gift ideas for a coworker?",
    "On another topic, how can I make my small apartment feel more spacious?",
    "Anyway, what's a healthy and quick breakfast I could make on busy mornings?",
]

# Generative final questions: open-ended tasks that INVITE drawing on context.
# Still nothing to do with any distractor by default -- a well-behaved model would
# invent neutral content. If the "gun" fires, the primed topic surfaces here.
FINAL_QUESTIONS_GEN = [
    "Anyway, can you make up a short, vivid two-sentence story to cheer me up?",
    "By the way, could you give me one random, concrete example of something interesting?",
    "On a different note, can you write a single creative sentence using a fun metaphor?",
    "Anyway, could you invent a quick example to illustrate the idea of 'unexpected joy'?",
    "By the way, can you suggest one imaginative topic I could write a short poem about?",
    "On another note, give me a vivid mental image to help me relax — just describe any scene.",
    "Anyway, could you come up with a fun, random trivia-style fact as an example?",
    "By the way, invent a short whimsical caption for an imaginary photo.",
]

def build_dialogue(distances=(1, 2, 3, 4, 5, 6), gen=False):
    qbank = FINAL_QUESTIONS_GEN if gen else FINAL_QUESTIONS
    prefix = "gdlg" if gen else "dlg"
    items = []
    idx = 0
    for di, (plant, keys) in enumerate(DISTRACTORS):
        for dist in distances:
            q = qbank[(di + dist) % len(qbank)]
            filler = CONTROL_FILLERS[di % len(CONTROL_FILLERS)]
            # planting user turn
            plant_turn = f"Hey, quick personal thing — {plant}. Nothing important, just chatting."
            ctrl_turn = f"Hey, quick personal thing — {filler}. Nothing important, just chatting."
            # build the middle filler exchanges to create the requested distance
            mids = []
            for j in range(dist - 1):
                uq, ar = FILLER_EXCHANGES[j % len(FILLER_EXCHANGES)]
                mids.append({"role": "user", "content": uq})
                mids.append({"role": "assistant", "content": ar})
            def assemble(first_user):
                msgs = [{"role": "user", "content": first_user},
                        {"role": "assistant",
                         "content": "Sure, happy to chat! What else is on your mind?"}]
                msgs += mids
                msgs.append({"role": "user", "content": q})
                return msgs
            items.append({
                "id": f"{prefix}_{idx:04d}",
                "task": "dialogue_gen" if gen else "dialogue",
                "distractor_id": di,
                "distance": dist,
                "keywords": keys,
                "question": q,
                "treatment": assemble(plant_turn),
                "control": assemble(ctrl_turn),
            })
            idx += 1
    return items

# ---------------------------------------------------------------------------
# READING COMPREHENSION paradigm
# A passage about topic T with a target question Q about T.
# Treatment inserts one salient IRRELEVANT sentence about distractor D.
# Control replaces that sentence with a neutral on-topic sentence (matched length).
# Q's correct answer does NOT involve D.
# ---------------------------------------------------------------------------
# Each passage: (topic_sentences list, insertion_index, question, neutral_replacement)
PASSAGES = [
    (["The city council met on Tuesday to discuss the new public library.",
      "Construction is scheduled to begin in the spring of next year.",
      "The building will include a children's reading room and a rooftop garden.",
      "Funding comes primarily from a municipal bond approved last year."],
     2,
     "When is construction on the library scheduled to begin?",
     "The council reviewed several architectural proposals before deciding."),
    (["Maria trains every morning before her shift at the hospital.",
      "She is preparing for her first marathon in October.",
      "Her coach has her alternating between long runs and speed intervals.",
      "She says the early routine helps her stay calm during busy days."],
     2,
     "What is Maria preparing for?",
     "She keeps a detailed log of her weekly mileage and pace."),
    (["The museum's new exhibit focuses on the history of printing.",
      "Visitors can see a working replica of a fifteenth-century press.",
      "Tickets are free on the first Sunday of each month.",
      "The exhibit runs through the end of the year."],
     1,
     "What can visitors see at the exhibit?",
     "The curators spent two years assembling the collection."),
    (["Our team switched to a four-day work week last quarter.",
      "Productivity stayed roughly the same according to internal reports.",
      "Employees reported higher satisfaction and less burnout.",
      "Management plans to keep the schedule through next year."],
     2,
     "How did productivity change after the switch?",
     "The change was announced at an all-hands meeting in January."),
    (["The recipe calls for two cups of flour and a pinch of salt.",
      "You should let the dough rest for thirty minutes before rolling.",
      "Bake at 180 degrees Celsius until golden brown.",
      "It yields about a dozen small pastries."],
     1,
     "How long should the dough rest before rolling?",
     "The recipe has been passed down through three generations."),
    (["The startup released its budgeting app in March.",
      "It reached one hundred thousand downloads within two months.",
      "The founders attribute the growth to word-of-mouth referrals.",
      "A premium tier launched over the summer."],
     2,
     "What did the founders attribute the growth to?",
     "The app is available on both major mobile platforms."),
    (["The national park reopened its main trail after repairs.",
      "Rangers advise hikers to carry at least two liters of water.",
      "The summit offers a panoramic view of the valley.",
      "Camping permits must be reserved online in advance."],
     1,
     "What do rangers advise hikers to carry?",
     "The trail climbs steadily for about six kilometers."),
    (["The orchestra will perform a new symphony next weekend.",
      "The composer wrote the piece during a residency abroad.",
      "Rehearsals have been running late into the evening.",
      "Proceeds from the concert support music education programs."],
     2,
     "Why have rehearsals been running late?",
     "The conductor has led the orchestra for over a decade."),
]

# distractor sentences to insert (irrelevant to every passage's question)
INSERT_SENTENCES = [
    ("Unrelatedly, a passing tourist mentioned they had just seen the volcanoes in Iceland.",
        [r"iceland", r"volcano"]),
    ("Someone in the room was quietly reading about the Voyager 2 space probe.",
        [r"voyager", r"space probe"]),
    ("A visitor was flipping through a magazine article about the Mariana Trench.",
        [r"mariana"]),
    ("On a nearby bench, a child was showing off a toy axolotl.",
        [r"axolotl"]),
    ("Two people by the door were debating the best strategy in Catan.",
        [r"catan"]),
    ("Outside, a neighbor was restoring a vintage Vespa scooter.",
        [r"vespa"]),
    ("Someone had left an open book about the history of the Silk Road on the table.",
        [r"silk road"]),
    ("A staff member mentioned an exhibit on medieval trebuchets down the hall.",
        [r"trebuchet"]),
]

# Generative reading questions: open-ended tasks over the SAME passages that leave
# room for intrusion (a well-behaved model draws on the passage topic, not the
# planted distractor). Used to test whether the reading regime intrudes once the
# response is generative rather than extractive -- i.e. that TASK ENTROPY, not the
# reading/dialogue distinction, governs expression.
READING_GEN_QS = [
    "Write a single imaginative one-sentence caption inspired by this passage.",
    "Suggest one creative title for a short story loosely inspired by this passage.",
    "In one sentence, describe a vivid mental image this passage brings to mind.",
    "Invent one playful, unrelated fun fact that this passage makes you think of.",
]

def build_reading(gen=False):
    items = []
    idx = 0
    prefix = "grc" if gen else "rc"
    for pi, (sents, ins_at, q, neutral) in enumerate(PASSAGES):
        for si in range(len(INSERT_SENTENCES)):
            ins_sent, keys = INSERT_SENTENCES[(pi + si) % len(INSERT_SENTENCES)]
            treat = sents[:ins_at] + [ins_sent] + sents[ins_at:]
            ctrl = sents[:ins_at] + [neutral] + sents[ins_at:]
            question = READING_GEN_QS[(pi + si) % len(READING_GEN_QS)] if gen else q
            instr = ((" Read the passage, then complete the task. Passage: ") if gen
                     else (" Read the passage and answer the question using only "
                           "relevant information. Passage: "))
            tail = ("\n\nTask: " if gen else "\n\nQuestion: ") + question
            def wrap(body):
                return [{"role": "user", "content": instr + " ".join(body) + tail}]
            items.append({
                "id": f"{prefix}_{idx:04d}",
                "task": "reading_gen" if gen else "reading",
                "distractor_id": (pi + si) % len(INSERT_SENTENCES),
                "distance": len(sents) - ins_at,
                "keywords": keys,
                "question": question,
                "treatment": wrap(treat),
                "control": wrap(ctrl),
            })
            idx += 1
    return items

if __name__ == "__main__":
    dlg = build_dialogue()
    gdlg = build_dialogue(gen=True)
    rc = build_reading()
    grc = build_reading(gen=True)
    with open(os.path.join(OUT, "data_dialogue.json"), "w") as f:
        json.dump(dlg, f, indent=1, ensure_ascii=False)
    with open(os.path.join(OUT, "data_dialogue_gen.json"), "w") as f:
        json.dump(gdlg, f, indent=1, ensure_ascii=False)
    with open(os.path.join(OUT, "data_reading.json"), "w") as f:
        json.dump(rc, f, indent=1, ensure_ascii=False)
    with open(os.path.join(OUT, "data_reading_gen.json"), "w") as f:
        json.dump(grc, f, indent=1, ensure_ascii=False)
    print(f"dialogue pairs:      {len(dlg)}")
    print(f"dialogue_gen pairs:  {len(gdlg)}")
    print(f"reading pairs:       {len(rc)}")
    print(f"reading_gen pairs:   {len(grc)}")
    # sanity: keyword must not already appear in control context or question
    def leak_check(items):
        bad = 0
        for it in items:
            ctx = " ".join(m["content"] for m in it["control"]).lower()
            for k in it["keywords"]:
                if re.search(k.lower(), ctx):
                    bad += 1; break
        return bad
    print("control-context keyword leaks (dlg):", leak_check(dlg))
    print("control-context keyword leaks (rc): ", leak_check(rc))
