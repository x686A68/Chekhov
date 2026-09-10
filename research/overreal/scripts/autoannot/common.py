"""Shared pieces of the automated annotator.

Paths, metadata loading, the target-type heuristic, and the versioned
question protocols. A protocol is a cascade of single positive yes/no
questions whose answers map onto the four outcome labels of the paper
(disruptive / silent / integrated / withheld):

    Q1 present?   no  -> withheld
                  yes -> Q2 disruptive?   yes -> disruptive
                                          no  -> Q3 integrated?  yes -> integrated
                                                                 no  -> silent

Q1 is asked without the prompt (a content question about the picture, as in
the pilot); Q2 and Q3 see the raw prompt, because the disruptive / integrated
criteria are relations between the picture and the prompt. A family may set
Q2 to None, meaning presence alone is disruptive (existence-canceling).
A protocol may also relabel the branches with "yes2" / "yes3" / "else"
keys (v3 asks the integrated question before the disruptive one for
perspectival items, because a cutaway also shows the target "in its place").
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
DS = ROOT / "data" / "overreal_v1"
OUT_DIR = DS / "auto"
SPLIT = DS / "auto_split.jsonl"

LABELS = ["disruptive", "silent", "integrated", "withheld"]
ALL_LABELS = LABELS + ["other"]
FAMILIES = ["cancellation", "attribution", "figurative", "perspectival"]

YESNO = "Answer with one word, Yes or No, and nothing else.\n\n"


def load_meta():
    meta = {}
    for line in open(DS / "metadata.jsonl", encoding="utf-8"):
        r = json.loads(line)
        meta[r["image_id"]] = r
    return meta


def load_split():
    return [json.loads(l) for l in open(SPLIT, encoding="utf-8")]


def load_eval_sample():
    return [json.loads(l) for l in open(DS / "eval_sample.jsonl", encoding="utf-8")]


def article(phrase):
    p = phrase.strip()
    if re.match(r"^(a|an|the)\s", p, flags=re.I):
        return ""
    return "an " if p[0].lower() in "aeiou" else "a "


def clean_target(t):
    return t.strip().rstrip(".").strip()


_TEXT_WORDS = re.compile(r"\b(the words?|says|reads|written|writing|text saying|"
                         r"engraved with)\b", re.I)


def is_text_target(target, prompt):
    """Perspectival items come in two kinds: inscribed information that an
    in-scene observer reads, and objects hidden behind or inside something.
    A target is treated as text when the prompt introduces it after a colon
    or quotation mark, or talks about words / reading / writing."""
    t = re.escape(clean_target(target))
    if re.search(r'(:|"|“|”|‘|’|\')\s*' + t, prompt):
        return True
    return bool(_TEXT_WORDS.search(prompt))


def with_prompt(prompt, q):
    return f'This image was generated from the prompt:\n"{prompt}"\n\n{q}'


# ---------------------------------------------------------------- protocols

def _v1(family, target, prompt):
    """Protocol v1: first draft of the family-specific cascade (2026-09-09)."""
    t = clean_target(target)
    a = article(t)
    if family == "cancellation":
        return {
            "Q1": f"Is there {a}{t} anywhere in this image?",
            "Q2": None,
            "Q3": None,
        }
    if family == "attribution":
        return {
            "Q1": f"Is there {a}{t} anywhere in this image?",
            "Q2": with_prompt(prompt,
                  f"In this image, is {a}{t} physically present in the scene, "
                  f"as an actual thing that is there with the person?"),
            "Q3": with_prompt(prompt,
                  f"In this image, is {a}{t} shown as something the person imagines, "
                  f"remembers or believes, for example inside a thought bubble, a dream "
                  f"haze, a separate inset or a ghostly overlay?"),
        }
    if family == "figurative":
        return {
            "Q1": f"Is there {a}{t} anywhere in this image?",
            "Q2": with_prompt(prompt,
                  f"In this image, does the presence of {a}{t} make the scene absurd, "
                  f"impossible or nonsensical as a picture of what the prompt describes?"),
            "Q3": with_prompt(prompt,
                  f"In this image, is {a}{t} merged with the person or thing it describes, "
                  f"as a deliberate visual metaphor in a stylized or poster-like composition?"),
        }
    if family == "perspectival":
        if is_text_target(target, prompt):
            return {
                "Q1": f'Can you, the viewer of this image, read the words "{t}" anywhere in it?',
                "Q2": with_prompt(prompt,
                      f'In this image, is the text "{t}" turned toward the camera and readable, '
                      f"although the prompt places it where the viewer could not see it?"),
                "Q3": with_prompt(prompt,
                      f'In this image, is the text "{t}" revealed through a deliberate device '
                      f"such as an inset, a cutaway, a reflection or a thought bubble?"),
            }
        return {
            "Q1": f"Is any part of {a}{t} visible in this image?",
            "Q2": with_prompt(prompt,
                  f"In this image, is {a}{t} visible in the place where the prompt says it is "
                  f"hidden, covered, closed away or facing away from the viewer?"),
            "Q3": with_prompt(prompt,
                  f"In this image, is {a}{t} revealed through a deliberate device such as a "
                  f"cutaway view, an inset, a reflection or a thought bubble?"),
        }
    raise ValueError(family)


def _v2(family, target, prompt):
    """Protocol v2 (2026-09-09), after inspecting v1 errors on dev.

    Attribution: v1 asked whether the target is "physically present in the
    scene", which is true of silent cases too (a remembered trumpet lying on
    the stage); disruptive is a contradiction, so Q2 now asks that directly.
    Perspectival: v1's Q2 "visible in the place where the prompt says it is
    hidden" failed on prints on the wrong side (an eagle on the front instead
    of the back); Q2 now asks whether the viewer sees the target directly
    although the prompt hides it, and Q3 asks about indirect devices
    (reflection, cutaway, a part peeking out), which is what humans labelled
    integrated. Q1 for hidden objects is the plain presence question, since
    "any part of ... visible" missed stickers and prints.
    Figurative: Q3 no longer says "merged ... as a visual metaphor", which
    the 7B answered yes for background animals; it asks whether the target is
    rendered only as a stylized attribute or as words."""
    t = clean_target(target)
    a = article(t)
    if family == "cancellation":
        return _v1(family, target, prompt)
    if family == "attribution":
        return {
            "Q1": f"Is there {a}{t} anywhere in this image?",
            "Q2": with_prompt(prompt,
                  f"In this image, does the presence of {a}{t} contradict the prompt, "
                  f"or make the scene implausible or illogical?"),
            "Q3": with_prompt(prompt,
                  f"In this image, is {a}{t} shown as something the person imagines, "
                  f"remembers or believes, for example inside a thought bubble, a dream "
                  f"haze, a separate inset or a ghostly overlay?"),
        }
    if family == "figurative":
        return {
            "Q1": f"Is there {a}{t} anywhere in this image?",
            "Q2": with_prompt(prompt,
                  f"In this image, does the presence of {a}{t} make the scene absurd, "
                  f"impossible or nonsensical as a picture of what the prompt describes?"),
            "Q3": with_prompt(prompt,
                  f"In this image, is {a}{t} rendered only figuratively, as a stylized "
                  f"attribute of the person or thing it describes or as written words, "
                  f"with no literal {t} placed in the scene?"),
        }
    if family == "perspectival":
        if is_text_target(target, prompt):
            return {
                "Q1": f'Can you, the viewer of this image, read the words "{t}" anywhere in it?',
                "Q2": with_prompt(prompt,
                      f'According to the prompt, the viewer should not be able to read "{t}". '
                      f'In this image, is the text "{t}" nevertheless directly readable to the '
                      f"viewer, on the very item the prompt describes?"),
                "Q3": with_prompt(prompt,
                      f'In this image, is the text "{t}" shown only indirectly, through a device '
                      f"such as a reflection, a cutaway or see-through view, an inset, a thought "
                      f"bubble, or a small part peeking out of the hiding place?"),
            }
        return {
            "Q1": f"Is there {a}{t} anywhere in this image?",
            "Q2": with_prompt(prompt,
                  f"According to the prompt, the viewer should not be able to see the {t}. "
                  f"In this image, is the {t} nevertheless directly visible to the viewer, "
                  f"in the very place the prompt describes?"),
            "Q3": with_prompt(prompt,
                  f"In this image, is the {t} shown only indirectly, through a device such as "
                  f"a reflection, a cutaway or see-through view, an inset, a thought bubble, "
                  f"or a small part peeking out of the hiding place?"),
        }
    raise ValueError(family)


def _v3(family, target, prompt):
    """Protocol v3 (2026-09-09), from Claude Opus 5 errors on dev under v1.

    Attribution: silent->disruptive (14/22) came from Q2 "physically present
    in the scene", which is true of a remembered trumpet lying on the stage;
    Q2 now asks for a contradiction or an implausible scene.
    Figurative: disruptive->silent came from images in which the target
    replaces the scene outright (a tiger, no coach), which is not "absurd as a
    picture"; Q2 now covers replacement and out-of-place literal objects. Q1
    adds "in any form, including ... written words" because integrated cases
    render the idiom as text, and Q3 asks for figurative-only rendering.
    Perspectival: disruptive->withheld (56/208) were prints, stickers and
    pictures that Q1 "any part of an eagle visible" rejected; Q1 now lists
    those forms. Integrated->disruptive came from cutaways, which do show the
    target "in its place", so the device question is asked first and sends
    to integrated; the direct-exposure question follows and sends to
    disruptive; the remainder (a second copy elsewhere) is silent. For text
    targets Q1 asks whether the written side faces the viewer, legible or
    not, since garbled but exposed text is still labelled disruptive, and the
    reader's own point of view counts as a device."""
    t = clean_target(target)
    a = article(t)
    if family == "cancellation":
        return _v1(family, target, prompt)
    if family == "attribution":
        return {
            "Q1": f"Is there {a}{t} anywhere in this image?",
            "Q2": with_prompt(prompt,
                  f"In this image, does the presence of {a}{t} contradict the prompt, "
                  f"or make the scene implausible or illogical?"),
            "Q3": with_prompt(prompt,
                  f"In this image, is {a}{t} shown as something the person imagines, "
                  f"remembers or believes, for example inside a thought bubble, a dream "
                  f"haze, a separate inset or a ghostly overlay?"),
        }
    if family == "figurative":
        return {
            "Q1": f"Is there {a}{t} anywhere in this image, in any form, including as a "
                  f"drawing, a logo, a costume, a shape, or written words?",
            "Q2": with_prompt(prompt,
                  f"In this image, is there a literal {t} that either replaces the scene "
                  f"the prompt describes or would be out of place in that scene in reality?"),
            "Q3": with_prompt(prompt,
                  f"In this image, is {a}{t} rendered only figuratively, as a stylized "
                  f"attribute of the person or thing it describes or as written words, "
                  f"with no literal {t} placed in the scene?"),
        }
    if family == "perspectival":
        device = ("through a device such as a reflection, a cutaway or see-through view, "
                  "an inset, a thought bubble, a small part peeking out of the hiding "
                  "place, or a picture taken from the in-scene person's own point of view")
        if is_text_target(target, prompt):
            return {
                "Q1": with_prompt(prompt,
                      f'In this image, is the written side of the item that carries "{t}" '
                      f"(the letter, note, page, screen, card or sign the prompt describes) "
                      f"turned toward the viewer, so that its writing can be seen, whether "
                      f"or not it is legible?"),
                "Q2": with_prompt(prompt,
                      f'In this image, is the writing "{t}" shown to the viewer only {device}?'),
                "Q3": with_prompt(prompt,
                      f'According to the prompt, the writing "{t}" should be out of the '
                      f"viewer's sight. In this image, is it nevertheless in plain view, on "
                      f"the very item the prompt describes?"),
                "yes2": "integrated", "yes3": "disruptive", "else": "silent",
            }
        return {
            "Q1": f"Is there {a}{t} anywhere in this image, in any form, including as a "
                  f"print, picture, sticker, logo or drawing?",
            "Q2": with_prompt(prompt,
                  f"In this image, is the {t} shown to the viewer only {device}?"),
            "Q3": with_prompt(prompt,
                  f"According to the prompt, the {t} should be out of the viewer's sight. "
                  f"In this image, is the {t} nevertheless in plain view, on or in the very "
                  f"item the prompt describes?"),
            "yes2": "integrated", "yes3": "disruptive", "else": "silent",
        }
    raise ValueError(family)


OTHER_GATE = ("Is this image severely corrupted, or unrelated to the scene the prompt "
              "describes?")


def _v4(family, target, prompt):
    """Protocol v4 (2026-09-10). Tree format. Adds an "other" gate (Q0) for
    every family, and for attribution splits the disruptive decision in two:
    Q2 physically present as an actual thing (v1 wording, which separated
    present from imagined well), then Q2b whether a real target could
    plausibly be there given what the prompt says about the surroundings.
    Gold: a cake on the desk the prompt calls empty, a tiger in the bare
    bedroom, a kite already on the desk the child is only expecting it on
    are disruptive; a trumpet on the stage the man remembers it from, a
    lighthouse in view of the sailor, a beach ball on the beach are silent.
    Other families keep their v3 questions."""
    t = clean_target(target)
    a = article(t)
    gate = {"q": with_prompt(prompt, OTHER_GATE), "yes": "other", "no": "Q1"}
    if family != "attribution":
        tree = to_tree(_v3(family, target, prompt))
        nodes = {"Q0": gate, **tree["nodes"]}
        return {"start": "Q0", "nodes": nodes}
    return {"start": "Q0", "nodes": {
        "Q0": gate,
        "Q1": {"q": f"Is there {a}{t} anywhere in this image?",
               "yes": "Q2", "no": "withheld"},
        "Q2": {"q": with_prompt(prompt,
                   f"In this image, is {a}{t} physically present in the scene, "
                   f"as an actual thing that is there with the person?"),
               "yes": "Q2b", "no": "Q3"},
        "Q2b": {"q": with_prompt(prompt,
                    f"Given what the prompt says about the surroundings, could a real {t} "
                    f"plausibly be there as an ordinary part of the actual scene?"),
                "yes": "silent", "no": "disruptive"},
        "Q3": {"q": with_prompt(prompt,
                   f"In this image, is {a}{t} shown as something the person imagines, "
                   f"remembers or believes, for example inside a thought bubble, a dream "
                   f"haze, a separate inset or a ghostly overlay?"),
               "yes": "integrated", "no": "silent"},
    }}


def _v5(family, target, prompt):
    """Protocol v5 (2026-09-10). Attribution: v1's presence question, then a
    prominence question. Dev gold has the remembered trumpet on the stage,
    the lighthouse in the distance, the balloon in the sky, the book on the
    train as silent, and the cake on the desk beside the worker, the tiger in
    the bedroom, the bug in the child's hand as disruptive; v3's
    "contradicts the prompt" and v4's "could plausibly be there" both sent
    the disruptive cases to silent. Figurative: Q2 asks whether the literal
    target dominates or replaces the scene, since v3's "out of place in
    reality" sent 55 silent images (a bomb as a table centrepiece, swans
    behind the bride) to disruptive. Other families as v4."""
    t = clean_target(target)
    a = article(t)
    gate = {"q": with_prompt(prompt, OTHER_GATE), "yes": "other", "no": "Q1"}
    if family == "attribution":
        return {"start": "Q0", "nodes": {
            "Q0": gate,
            "Q1": {"q": f"Is there {a}{t} anywhere in this image?",
                   "yes": "Q2", "no": "withheld"},
            "Q2": {"q": with_prompt(prompt,
                       f"In this image, is {a}{t} physically present in the scene, "
                       f"as an actual thing that is there with the person?"),
                   "yes": "Q2b", "no": "Q3"},
            "Q2b": {"q": with_prompt(prompt,
                        f"In this image, is the {t} right beside the person and prominent "
                        f"in the picture, as opposed to distant, small or in the background?"),
                    "yes": "disruptive", "no": "silent"},
            "Q3": {"q": with_prompt(prompt,
                       f"In this image, is {a}{t} shown as something the person imagines, "
                       f"remembers or believes, for example inside a thought bubble, a dream "
                       f"haze, a separate inset or a ghostly overlay?"),
                   "yes": "integrated", "no": "silent"},
        }}
    if family == "figurative":
        return {"start": "Q0", "nodes": {
            "Q0": gate,
            "Q1": {"q": f"Is there {a}{t} anywhere in this image, in any form, including as a "
                        f"drawing, a logo, a costume, a shape, or written words?",
                   "yes": "Q2", "no": "withheld"},
            "Q2": {"q": with_prompt(prompt,
                       f"In this image, is a literal {t} the main subject or a dominant "
                       f"element of the picture, or does it replace the scene the prompt "
                       f"describes?"),
                   "yes": "disruptive", "no": "Q3"},
            "Q3": {"q": with_prompt(prompt,
                       f"In this image, is {a}{t} rendered only figuratively, as a stylized "
                       f"attribute of the person or thing it describes or as written words, "
                       f"with no literal {t} placed in the scene?"),
                   "yes": "integrated", "no": "silent"},
        }}
    return _v4(family, target, prompt)


def _v6(family, target, prompt):
    """Protocol v6 (2026-09-10). Perspectival only; other families as v5.
    v3 sent 40 disruptive images to integrated because its device list had
    "cutaway or see-through view" and "a small part peeking out": a gift box
    drawn transparent or a lid left ajar matched them, and by the author's
    rule an opened or transparent container is disruptive. v3 also sent 31
    disruptive images to silent because "on or in the very item" does not
    cover a dog seen through the open door or a horse over the fence. The
    exposure question now comes first and names the ways a hiding place can
    fail; the device list keeps reflection, inset, thought bubble and the
    in-scene person's point of view."""
    if family != "perspectival":
        return _v5(family, target, prompt)
    t = clean_target(target)
    a = article(t)
    gate = {"q": with_prompt(prompt, OTHER_GATE), "yes": "other", "no": "Q1"}
    device = ("a reflection, an inset, a thought bubble, or a picture taken from the "
              "in-scene person's own point of view")
    if is_text_target(target, prompt):
        return {"start": "Q0", "nodes": {
            "Q0": gate,
            "Q1": {"q": with_prompt(prompt,
                       f'In this image, is the written side of the item that carries "{t}" '
                       f"(the letter, note, page, screen, card or sign the prompt describes) "
                       f"turned toward the viewer, so that its writing can be seen, whether "
                       f"or not it is legible?"),
                   "yes": "Q2", "no": "withheld"},
            "Q2": {"q": with_prompt(prompt,
                       f'In this image, is the writing "{t}" shown to the viewer only through '
                       f"{device}?"),
                   "yes": "integrated", "no": "Q3"},
            "Q3": {"q": with_prompt(prompt,
                       f'According to the prompt, the writing "{t}" should be out of the '
                       f"viewer's sight. In this image, is it nevertheless in plain view, on "
                       f"the very item the prompt describes?"),
                   "yes": "disruptive", "no": "silent"},
        }}
    return {"start": "Q0", "nodes": {
        "Q0": gate,
        "Q1": {"q": f"Is there {a}{t} anywhere in this image, in any form, including as a "
                    f"print, picture, sticker, logo or drawing?",
               "yes": "Q2", "no": "withheld"},
        "Q2": {"q": with_prompt(prompt,
                   f"According to the prompt, the {t} should be hidden from the viewer: inside, "
                   f"behind, underneath, or on the far side of something. In this image, is the "
                   f"{t} nevertheless exposed to the viewer at that very spot, for example "
                   f"because the container, door, curtain, barrier or surface is shown open, "
                   f"transparent, removed, or turned the wrong way?"),
               "yes": "disruptive", "no": "Q3"},
        "Q3": {"q": with_prompt(prompt,
                   f"In this image, is the {t} shown only indirectly, through {device}?"),
               "yes": "integrated", "no": "silent"},
    }}


def _v7(family, target, prompt):
    """Protocol v7 (2026-09-10), the last tuning round. Attribution: the
    disruptive / silent split in the gold follows the kind of mental state,
    not the picture: a target the person believes or expects to be in the
    present scene (a tiger at the ridge, a kite on the desk, a robot in the
    hallway) is disruptive when shown, a target remembered from the past or
    elsewhere (the trumpet, the radio, the bakery) is silent when shown. Q2b
    now asks that, about the prompt. Perspectival: v6's exposure question
    missed prints that moved to the visible side (an apple on the front of
    the shirt when the prompt puts it on the back) and objects set beside
    their closed container; both are named. Other families as v5 / v6."""
    t = clean_target(target)
    a = article(t)
    gate = {"q": with_prompt(prompt, OTHER_GATE), "yes": "other", "no": "Q1"}
    if family == "attribution":
        tree = _v5(family, target, prompt)
        tree["nodes"]["Q2b"] = {"q": with_prompt(prompt,
            f"According to the prompt, is the {t} something the person believes, expects "
            f"or imagines to be in the present scene right now, as opposed to something "
            f"remembered from the past or from another place?"),
            "yes": "disruptive", "no": "silent"}
        return tree
    if family == "perspectival" and not is_text_target(target, prompt):
        tree = _v6(family, target, prompt)
        tree["nodes"]["Q2"]["q"] = with_prompt(prompt,
            f"According to the prompt, the {t} should be hidden from the viewer: inside, "
            f"behind, underneath, or on the far side of something. In this image, is the "
            f"{t} nevertheless exposed to the viewer, for example because the container, "
            f"door, curtain or barrier is shown open, transparent or removed, because the "
            f"{t} appears on the visible side instead of the hidden side, or because it "
            f"sits outside its container right beside it?")
        return tree
    return _v6(family, target, prompt)


def _final(family, target, prompt):
    """The protocol used for the release run (frozen 2026-09-10). Per family
    the best dev version, each with the Q0 "other" gate:
      cancellation  v1  (dev kappa 0.88)
      attribution   v5  (0.59; v1 0.56, v3 0.43, v4 0.43, v7 0.57)
      figurative    v1  (0.66; v3 0.63, v5 0.63)
      perspectival  v6  (0.59; v1 0.39, v3 0.55, v7 0.49)
    Nothing in this function may change after the test-split run."""
    if family in ("cancellation", "figurative"):
        tree = to_tree(_v1(family, target, prompt))
        gate = {"q": with_prompt(prompt, OTHER_GATE), "yes": "other", "no": "Q1"}
        return {"start": "Q0", "nodes": {"Q0": gate, **tree["nodes"]}}
    if family == "attribution":
        return _v5(family, target, prompt)
    if family == "perspectival":
        return _v6(family, target, prompt)
    raise ValueError(family)


PROTOCOLS = {"v1": _v1, "v2": _v2, "v3": _v3, "v4": _v4, "v5": _v5, "v6": _v6, "v7": _v7,
             "final": _final}


def questions(protocol, family, target, prompt):
    return PROTOCOLS[protocol](family, target, prompt)


def to_tree(qs):
    """Normalise a protocol dict to {"start", "nodes": {key: {"q", "yes", "no"}}}.
    v1-v3 dicts (Q1/Q2/Q3 with optional yes2/yes3/else) are converted; a dict
    that already has "nodes" is returned as is. Leaves are label strings."""
    if "nodes" in qs:
        return qs
    yes2 = qs.get("yes2", "disruptive")
    yes3 = qs.get("yes3", "integrated")
    rest = qs.get("else", "silent")
    nodes = {"Q1": {"q": qs["Q1"], "yes": "Q2" if qs.get("Q2") else yes2, "no": "withheld"}}
    if qs.get("Q2"):
        nodes["Q2"] = {"q": qs["Q2"], "yes": yes2, "no": "Q3" if qs.get("Q3") else rest}
    if qs.get("Q3"):
        nodes["Q3"] = {"q": qs["Q3"], "yes": yes3, "no": rest}
    return {"start": "Q1", "nodes": nodes}


def derive_label(answers, qs=None):
    """Walk the tree with the recorded yes/no answers; None if a needed
    answer is missing or unparsable."""
    tree = to_tree(qs or {})
    cur = tree["start"]
    for _ in range(10):
        if cur not in tree["nodes"]:
            return cur
        a = answers.get(cur)
        if a is None:
            return None
        cur = tree["nodes"][cur]["yes" if a else "no"]
    return None


def parse_yesno(raw):
    m = re.search(r"\b(yes|no)\b", raw or "", flags=re.I)
    return (m.group(1).lower() == "yes") if m else None


def model_slug(model):
    return model.split("/")[-1].lower().replace(".", "")


def run_path(protocol, model):
    return OUT_DIR / f"{protocol}__{model_slug(model)}.jsonl"
