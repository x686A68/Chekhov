"""Replace annotator names with stable pseudonymous ids across data/overreal_v1.

Names leak from three metadata fields and one audit column, so all four are rewritten
together; changing `annotator` alone would leave the names recoverable from
`source_folder` ("Jiahao_1_check") and `raw_path`.

  annotator     David            -> annotator_01
  source_folder David_100_check  -> annotator_01_100_check
  raw_path      OverReal/Attribution/Jiahao_1_check/x.png
                                 -> OverReal/Attribution/annotator_03_1_check/x.png
  audit_log.csv folder column    -> same substitution

Ids are assigned in alphabetical order of the real names so the mapping is
reproducible. The map is written OUTSIDE the dataset directory
(data/annotator_map.private.json) so that publishing data/overreal_v1 does not ship
the key; keep it private, it is the only way back to the real names.

Idempotent: values already of the form annotator_NN are left alone.
"""
import csv
import json
import os
import re

ROOT = "/home/jiahao_huang/Chekhov"
DS = os.path.join(ROOT, "data", "overreal_v1")
META = os.path.join(DS, "metadata.jsonl")
AUDIT = os.path.join(DS, "audit_log.csv")
MAP = os.path.join(ROOT, "data", "annotator_map.private.json")

# the cancellation labels were drafted by a model pass and then checked by this person,
# who is therefore the annotator of record for that split
CANCELLATION_ANNOTATOR = "Jiahao"
MODEL_PLACEHOLDER = "claude-subagent"
DONE = re.compile(r"^annotator_\d{2}")


def load_map(names):
    if os.path.exists(MAP):
        m = json.load(open(MAP))["name_to_id"]
    else:
        m = {}
    for n in sorted(names):
        m.setdefault(n, f"annotator_{len(m) + 1:02d}")
    return m


def main():
    rows = [json.loads(l) for l in open(META)]
    names = {r["annotator"] for r in rows
             if r["annotator"] != MODEL_PLACEHOLDER and not DONE.match(r["annotator"])}
    names.add(CANCELLATION_ANNOTATOR)
    name_map = load_map(names)

    def sub_folder(v):
        """David_100_check -> annotator_01_100_check"""
        if not v or DONE.match(v):
            return v
        low = v.lower()
        for n, i in name_map.items():          # folder casing is inconsistent in the
            nl = n.lower()                     # raw dump (e.g. "xanh_10_check")
            if low == nl or low.startswith(nl + "_"):
                return i + v[len(n):]
        return v

    def sub_path(v):
        if not v:
            return v
        parts = v.split("/")
        return "/".join(sub_folder(p) for p in parts)

    n_anon = 0
    for r in rows:
        if r["annotator"] == MODEL_PLACEHOLDER:
            r["annotator"] = CANCELLATION_ANNOTATOR
        before = (r["annotator"], r["source_folder"], r["raw_path"])
        r["annotator"] = name_map.get(r["annotator"], r["annotator"])
        r["source_folder"] = sub_folder(r["source_folder"])
        r["raw_path"] = sub_path(r["raw_path"])
        n_anon += (r["annotator"], r["source_folder"], r["raw_path"]) != before

    with open(META, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(AUDIT) as f:
        audit = list(csv.reader(f))
    header, body = audit[0], audit[1:]
    fi = header.index("folder")
    for row in body:
        if len(row) > fi:
            row[fi] = sub_folder(row[fi])
    with open(AUDIT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(body)

    with open(MAP, "w") as f:
        json.dump({
            "note": "PRIVATE. Maps real annotator names to the pseudonymous ids used in "
                    "data/overreal_v1. Kept outside the dataset directory so it is not "
                    "published with it.",
            "name_to_id": name_map,
        }, f, indent=2, ensure_ascii=False)

    print(f"rewrote {n_anon}/{len(rows)} metadata rows, {len(body)} audit rows")
    print(f"ids: {len(name_map)} -> {MAP}")


if __name__ == "__main__":
    main()
