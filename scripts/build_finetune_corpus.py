"""Generate a synthetic training corpus for CNI entity extraction.

CRITICAL DESIGN CONSTRAINT
--------------------------
Training uses ONLY entities present in CNI_GAZETTEER. The held-out CNI
entities used by the evaluation set (NPCIL, BARC, Sriharikota, Kudankulam,
Trombay, Tarapur, Ennore, HAL) NEVER appear here.

That makes the experiment meaningful: if the fine-tuned model scores on the
held_out slice, it learned "a CNI organisation appears in this syntactic and
script context" rather than memorising surface strings -- which is precisely
the generalisation a gazetteer cannot provide. If it scores ~0 there, it
memorised, and ADR-001 stands with much stronger evidence.

Output: data/finetune/train.jsonl, dev.jsonl  (text + char-span entities)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "data" / "finetune"

# --- Entity inventory: IN-GAZETTEER ONLY -------------------------------------
LOC_LATIN = ["Avadi", "Chennai", "Kalpakkam", "Vizag"]
LOC_TAMIL = ["ஆவடி", "சென்னை", "கல்பாக்கம்", "விசாகப்பட்டினம்"]
ORG_LATIN = ["DRDO", "ISRO", "BHEL"]

# Distractor entities: real place/org names that are NOT CNI, so the model
# must learn context rather than "any capitalised token is an entity".
DISTRACTOR_LOC = ["Madurai", "Coimbatore", "Salem", "Trichy", "Erode"]
DISTRACTOR_LOC_TA = ["மதுரை", "கோயம்புத்தூர்", "சேலம்", "திருச்சி"]
DISTRACTOR_ORG = ["Infosys", "Wipro", "Reliance", "Airtel"]

# --- Templates. {L} location slot, {O} organisation slot ---------------------
T_EN_THREAT = [
    "We will shut down the {O} servers tomorrow.",
    "Coordinated attack planned against {O} infrastructure.",
    "Our team has gained access to internal {O} systems.",
    "Reconnaissance complete on the {L} facility perimeter.",
    "The {L} site is our next target. Coordinates shared.",
    "Phishing campaign against {O} staff is now live.",
    "Power grid near {L} will go dark tonight.",
    "{O} internal documents will be released next week.",
    "We have mapped the {L} approaches and gate positions.",
    "Credentials for {O} employees have been harvested.",
    "Operations against {L} begin at dawn.",
    "{L} and {O} are both on the list.",
]
T_EN_BENIGN = [
    "Travelling to {L} next week for a family wedding.",
    "The weather in {L} has been pleasant this month.",
    "{O} announced a new scholarship programme today.",
    "Applying for a graduate role at {O} this year.",
    "Traffic near {L} station was heavy this morning.",
    "My cousin works at {O} in the research division.",
    "Visited {L} over the weekend, lovely beach there.",
    "{O} reported strong quarterly results.",
    "Took the train from {L} yesterday evening.",
    "Anyone know a good restaurant near {L}?",
]
T_TA_THREAT = [
    "{L} பகுதியை குறிவைக்க திட்டம் உள்ளது.",
    "நாளை {O} சர்வர்களை முடக்குவோம்.",
    "{L} அருகில் உள்ள அமைப்புகள் நமது இலக்கு.",
    "{O} நிறுவனத்தின் தகவல்கள் சேகரிக்கப்பட்டன.",
    "{L} துறைமுகத்தை முடக்க தயாராகுங்கள்.",
    "{O} ஊழியர்களின் கடவுச்சொற்கள் கிடைத்துவிட்டன.",
    "{L} மற்றும் {O} இரண்டும் நமது பட்டியலில் உள்ளன.",
]
T_TA_BENIGN = [
    "{L} நகரில் இன்று மழை பெய்தது.",
    "{L} பகுதியில் புதிய பள்ளி கட்டப்படுகிறது.",
    "{O} நிறுவனம் புதிய திட்டத்தை அறிவித்தது.",
    "நாளை {L} செல்ல உள்ளேன்.",
    "{L} அருகில் நல்ல உணவகம் உள்ளது.",
    "{O} இன்று ஒரு அறிவிப்பு வெளியிட்டது.",
]
T_MIX = [
    "நாளை {O} அமைப்புகளை முடக்குவோம்.",
    "{L} மற்றும் {O} நமது கவனத்தில் உள்ளன.",
    "நமது இலக்கு {L}. {O} சர்வர்களை தாக்க வேண்டும்.",
    "{O} கண்காட்சி {L} நகரில் நடைபெறுகிறது.",
    "{L} la irukura {O} office ah check pannunga.",
]
T_ROMAN = [
    "{O} servers ah nalaikku down panna poren.",
    "{L} la periya problem varum, wait pannunga.",
    "{L} la oru nalla hotel recommend pannunga.",
    "{O} oda new project romba interesting ah irukku.",
    "Naan {L} ku poren next week.",
]
# No-entity sentences, so the model learns to abstain.
T_EMPTY = [
    "Good morning everyone, have a blessed day.",
    "Looking for a new laptop, any recommendations?",
    "The match was rained out and rescheduled.",
    "இன்று மாலை திரைப்படம் பார்க்க செல்கிறோம்.",
    "நாளை பள்ளிக்கு விடுமுறை அறிவிக்கப்பட்டுள்ளது.",
    "Vanakkam nanba, eppadi irukinga?",
    "Ami bazar e jacchi sabji kinte.",
    "Buy cheap crypto tokens now, huge gains!",
    "Click here to claim your free prize today.",
    "We are going to bomb this presentation if we don't prepare.",
    "Khela hobe kal amader para te cricket match.",
    "Our strategy is to target younger customers.",
]


# Contextual affixes multiply the combination space; offsets are computed
# after composition so they stay correct.
PREFIX = ["", "", "Update: ", "Alert: ", "Breaking: ", "FYI - ", "Note: ",
          "நண்பர்களே, ", "கவனிக்க: ", "தகவல்: ", "Guys, ", "Team, "]
SUFFIX = ["", "", " Please confirm.", " Share widely.", " More to follow.",
          " தகவல் பகிரவும்.", " உறுதி செய்யவும்.", " Stay alert.",
          " Details in the next message.", " பின்னர் தொடர்வோம்."]


def lcg(seed=20260821):
    s = seed
    while True:
        s = (1103515245 * s + 12345) % (2 ** 31)
        yield s


def build(n_target=4000):
    rnd = lcg()

    def pick(seq):
        return seq[next(rnd) % len(seq)]

    groups = [
        (T_EN_THREAT, LOC_LATIN, ORG_LATIN, "latin"),
        (T_EN_BENIGN, LOC_LATIN, ORG_LATIN, "latin"),
        (T_TA_THREAT, LOC_TAMIL, ORG_LATIN, "native_tamil"),
        (T_TA_BENIGN, LOC_TAMIL, ORG_LATIN, "native_tamil"),
        (T_MIX, LOC_TAMIL + LOC_LATIN, ORG_LATIN, "mixed"),
        (T_ROMAN, LOC_LATIN, ORG_LATIN, "romanized"),
        # distractors: same templates, non-CNI entities, labelled as entities
        # of the same type so the model learns the TYPE, not the CNI list.
        (T_EN_BENIGN, DISTRACTOR_LOC, DISTRACTOR_ORG, "latin"),
        (T_TA_BENIGN, DISTRACTOR_LOC_TA, DISTRACTOR_ORG, "native_tamil"),
        (T_EN_THREAT, DISTRACTOR_LOC, DISTRACTOR_ORG, "latin"),
    ]

    rows, seen = [], set()
    guard = 0
    while len(rows) < n_target and guard < n_target * 60:
        guard += 1
        if next(rnd) % 100 < 12:                     # ~12% entity-free
            text, ents = pick(T_EMPTY), []
        else:
            tmpl, locs, orgs, script = pick(groups)
            t = pick(tmpl)
            text, ents = t, []
            if "{L}" in text:
                text = text.replace("{L}", pick(locs), 1)
            if "{O}" in text:
                text = text.replace("{O}", pick(orgs), 1)
            for pool, etype in ((locs, "LOC"), (orgs, "ORG")):
                for v in pool:
                    idx = text.find(v)
                    if idx >= 0 and not any(
                        idx < e["end"] and e["start"] < idx + len(v) for e in ents
                    ):
                        ents.append({"start": idx, "end": idx + len(v),
                                     "label": etype, "value": v})
        # Compose with affixes, then recompute offsets against the final text.
        text = pick(PREFIX) + text + pick(SUFFIX)
        if text in seen:
            continue
        seen.add(text)
        final = []
        for e in ents:
            idx = text.find(e["value"])
            if idx >= 0:
                final.append({"start": idx, "end": idx + len(e["value"]),
                              "label": e["label"], "value": e["value"]})
        rows.append({"text": text, "entities": sorted(final, key=lambda e: e["start"])})
    return rows


def main() -> int:
    rows = build()
    split = int(len(rows) * 0.9)
    OUT.mkdir(parents=True, exist_ok=True)
    for name, part in (("train.jsonl", rows[:split]), ("dev.jsonl", rows[split:])):
        with (OUT / name).open("w", encoding="utf-8") as fh:
            for r in part:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Leakage guard: assert no held-out eval entity appears in training text.
    banned = ["npcil", "barc", "sriharikota", "kudankulam", "trombay",
              "tarapur", "ennore", "hal ", "கூடங்குளம்", "ஸ்ரீஹரிகோட்டா"]
    leaks = [r["text"] for r in rows if any(b in r["text"].lower() for b in banned)]
    print(f"total {len(rows)}  train {split}  dev {len(rows)-split}")
    n_ent = sum(len(r['entities']) for r in rows)
    n_empty = sum(1 for r in rows if not r['entities'])
    print(f"  entity spans {n_ent}   entity-free sentences {n_empty}")
    print(f"  HELD-OUT LEAKAGE CHECK: {len(leaks)} (must be 0)")
    if leaks:
        print("  !!! LEAK !!!", leaks[:3])
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
