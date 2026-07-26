"""
Openness-ladder stimuli: the SAME reading passages (with the same planted
distractor) asked at graded levels of task openness, from fully extractive to
fully creative. Used to show intrusion rises monotonically with task openness,
turning the binary extractive/generative contrast into a graded relationship.

Openness is later quantified independently as the semantic diversity of the
CONTROL-group answers at each level (more open task -> more varied answers).
"""
import json, os
from build_dataset import PASSAGES, INSERT_SENTENCES

OUT = os.path.dirname(os.path.abspath(__file__))

# Four graded question templates, applied to every passage. L0 is the passage's
# own extractive question; L1-L3 increase task openness.
LADDER = {
    "L0_extractive": None,  # use each passage's native extractive question
    "L1_summary": "Summarize this passage in one sentence.",
    "L2_takeaway": "What is one interesting takeaway or reflection prompted by this passage?",
    "L3_creative": "Write a one-sentence imaginative caption inspired by this passage.",
}

def build_level(level_key, question):
    items = []
    idx = 0
    for pi, (sents, ins_at, native_q, neutral) in enumerate(PASSAGES):
        for si in range(len(INSERT_SENTENCES)):
            ins_sent, keys = INSERT_SENTENCES[(pi + si) % len(INSERT_SENTENCES)]
            treat = sents[:ins_at] + [ins_sent] + sents[ins_at:]
            ctrl = sents[:ins_at] + [neutral] + sents[ins_at:]
            q = native_q if question is None else question
            gen = question is not None
            instr = ((" Read the passage, then complete the task. Passage: ") if gen
                     else (" Read the passage and answer the question using only "
                           "relevant information. Passage: "))
            tail = ("\n\nTask: " if gen else "\n\nQuestion: ") + q
            def wrap(body):
                return [{"role": "user", "content": instr + " ".join(body) + tail}]
            items.append({
                "id": f"{level_key}_{idx:04d}", "task": level_key,
                "distractor_id": (pi + si) % len(INSERT_SENTENCES),
                "distance": len(sents) - ins_at, "keywords": keys, "question": q,
                "treatment": wrap(treat), "control": wrap(ctrl),
            })
            idx += 1
    return items

if __name__ == "__main__":
    for key, q in LADDER.items():
        items = build_level(key, q)
        with open(os.path.join(OUT, f"data_ladder_{key}.json"), "w") as f:
            json.dump(items, f, indent=1, ensure_ascii=False)
        print(f"{key}: {len(items)} pairs")
