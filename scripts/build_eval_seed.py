"""Build the SAINTEL evaluation seed set.

Authored (synthetic) records only. Real harvested records are appended later
by the annotation workflow; this script produces the baseline the benchmark
can run against today.

Design constraints enforced here (see tests/data/annotation_guidelines.md):
  * held_out records use CNI entities deliberately ABSENT from CNI_GAZETTEER,
    so gazetteer recall is 0 there by construction. That slice is the actual
    experiment.
  * adversarial negatives use in-gazetteer terms in benign contexts. These
    drive the false-positive rate.
  * entity offsets are computed, never hand-written.
  * split is deterministic and stratified; no RNG seeding surprises.

All Tamil here is authored by a non-fluent contributor and carries
tamil_verified=false. Do not report numbers over unverified Tamil.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.layer3_native_nlp.gazetteer import CNI_GAZETTEER

OUT_DIR = ROOT / "tests" / "data"

HELD_OUT_TERMS = {
    "hal", "npcil", "barc", "sriharikota", "kudankulam", "trombay",
    "tarapur", "ennore", "manali refinery", "ordnance factory medak",
}

# (text, script, language, triage, intent, [(entity_type, value)], notes)
# script: native_tamil | romanized | mixed | latin
N = "noise"
S = "signal"
B = "benign"

RECORDS: list[tuple] = [
    # ---------- A. ADVERSARIAL NEGATIVES: in-gazetteer terms, benign context ----------
    ("Travelling to Chennai next week for my cousin's wedding.", "latin", "en", N, B,
     [("location", "Chennai")], "adversarial: gazetteer location, benign"),
    ("The Chennai weather has been unbearable this summer.", "latin", "en", N, B,
     [("location", "Chennai")], "adversarial"),
    ("Anyone know a good biryani place near Avadi?", "latin", "en", N, B,
     [("location", "Avadi")], "adversarial: gazetteer location, food chat"),
    ("Traffic near Avadi railway station is terrible today.", "latin", "en", N, B,
     [("location", "Avadi")], "adversarial"),
    ("ISRO successfully launched another communication satellite today.", "latin", "en", N, B,
     [("organization", "ISRO")], "adversarial: public news about CNI org"),
    ("My uncle retired from ISRO after thirty years of service.", "latin", "en", N, B,
     [("organization", "ISRO")], "adversarial"),
    ("DRDO announced a new scholarship programme for engineering students.", "latin", "en", N, B,
     [("organization", "DRDO")], "adversarial: benign org news"),
    ("Applying for a graduate trainee role at DRDO this year.", "latin", "en", N, B,
     [("organization", "DRDO")], "adversarial: job seeking"),
    ("BHEL reported strong quarterly earnings according to the business press.", "latin", "en", N, B,
     [("organization", "BHEL")], "adversarial: financial news"),
    ("Took the train from Vizag to Chennai, beautiful coastal route.", "latin", "en", N, B,
     [("location", "Vizag"), ("location", "Chennai")], "adversarial: two gazetteer locations, travel"),
    ("Kalpakkam has a lovely beach if you avoid the weekend crowd.", "latin", "en", N, B,
     [("location", "Kalpakkam")], "adversarial: sensitive site, tourism context"),
    ("Visiting the Vizag steel plant museum with the kids on Sunday.", "latin", "en", N, B,
     [("location", "Vizag")], "adversarial"),
    ("சென்னையில் இன்று நல்ல மழை பெய்தது.", "native_tamil", "ta", N, B,
     [("location", "சென்னை")], "adversarial: Tamil, weather"),
    ("ஆவடி பேருந்து நிலையம் அருகில் புதிய கடை திறந்துள்ளது.", "native_tamil", "ta", N, B,
     [("location", "ஆவடி")], "adversarial: Tamil, new shop"),
    ("சென்னை மெட்ரோ ரயில் சேவை நாளை தாமதமாகும்.", "native_tamil", "ta", N, B,
     [("location", "சென்னை")], "adversarial: Tamil, transit notice"),
    ("கல்பாக்கம் பகுதியில் மீன்பிடி தொழில் நன்றாக உள்ளது.", "native_tamil", "ta", N, B,
     [("location", "கல்பாக்கம்")], "adversarial: Tamil, fishing industry"),
    ("Chennai la oru nalla hotel recommend pannunga.", "romanized", "code_switched", N, B,
     [("location", "Chennai")], "adversarial: Tanglish, restaurant request"),
    ("Avadi la rent evlo irukum theriyuma?", "romanized", "code_switched", N, B,
     [("location", "Avadi")], "adversarial: Tanglish, rent enquiry"),
    ("ISRO oda new mission romba interesting ah irukku.", "romanized", "code_switched", N, B,
     [("organization", "ISRO")], "adversarial: Tanglish, space enthusiasm"),
    ("நாளை சென்னை வருகிறேன், DRDO கண்காட்சி பார்க்க.", "mixed", "code_switched", N, B,
     [("location", "சென்னை"), ("organization", "DRDO")], "adversarial: mixed script, public exhibition"),

    # ---------- B. BENIGN NEGATIVES: no CNI entity, threat-adjacent vocabulary ----------
    ("Khela hobe kal amader para te cricket match.", "romanized", "bn", N, B,
     [], "adversarial: 'khela hobe' cricket sense; currently scores 0.701 signal"),
    ("Khela hobe! Our team will destroy them in the final tomorrow.", "romanized", "bn", N, B,
     [], "adversarial: sports trash talk uses attack vocabulary"),
    ("We are going to absolutely bomb this presentation if we don't prepare.", "latin", "en", N, B,
     [], "adversarial: 'bomb' idiomatic"),
    ("That new phone launch completely killed the competition.", "latin", "en", N, B,
     [], "adversarial: 'killed' idiomatic"),
    ("Our strategy is to target the younger demographic this quarter.", "latin", "en", N, B,
     [], "adversarial: 'target' business sense"),
    ("The power went out in our building again, third time this week.", "latin", "en", N, B,
     [], "adversarial: outage, but domestic"),
    ("Network is down at the office, IT says it's a router problem.", "latin", "en", N, B,
     [], "adversarial: 'network down' benign IT"),
    ("Good morning everyone, have a blessed day.", "latin", "en", N, B, [], "plain benign"),
    ("Looking for a new laptop, any recommendations under 60k?", "latin", "en", N, B, [], "plain benign"),
    ("Buy cheap crypto tokens now 100x gains guaranteed!", "latin", "en", N, B, [], "spam"),
    ("Click here to win free prizes, limited time offer.", "latin", "en", N, B, [], "spam"),
    ("இன்று மாலை குடும்பத்துடன் திரைப்படம் பார்க்க செல்கிறோம்.", "native_tamil", "ta", N, B,
     [], "plain benign Tamil"),
    ("நாளை பள்ளிக்கு விடுமுறை என்று அறிவித்துள்ளனர்.", "native_tamil", "ta", N, B,
     [], "plain benign Tamil: school holiday"),
    ("Vanakkam nanba, eppadi irukinga? Saapteengala?", "romanized", "code_switched", N, B,
     [], "plain benign Tanglish greeting"),
    ("Naan kadai ku poren, kaaikari vaanga.", "romanized", "code_switched", N, B,
     [], "plain benign Tanglish"),
    ("Ami bazar e jacchi sabji kinte.", "romanized", "bn", N, B, [], "plain benign Bengali"),
    ("The match was rained out, rescheduled for next Sunday.", "latin", "en", N, B, [], "plain benign"),
    ("Happy new year to everyone celebrating today!", "latin", "en", N, B, [], "plain benign"),

    # ---------- C. SIGNAL: in-gazetteer entities ----------
    ("We will shut down the DRDO servers tomorrow at dawn.", "latin", "en", S, "cyberattack",
     [("organization", "DRDO")], "explicit cyberattack, in-gazetteer org"),
    ("Coordinated DDoS planned against DRDO web infrastructure this weekend.", "latin", "en", S, "cyberattack",
     [("organization", "DRDO"), ("tactic", "DDoS")], "explicit"),
    ("Our team has gained access to internal ISRO systems.", "latin", "en", S, "cyberattack",
     [("organization", "ISRO")], "breach claim"),
    ("Reconnaissance complete on the Avadi facility perimeter.", "latin", "en", S, "surveillance",
     [("location", "Avadi"), ("tactic", "Reconnaissance")], "surveillance of CNI site"),
    ("Comrades, the Kalpakkam site is the next target. Coordinates shared.", "latin", "en", S, "physical_threat",
     [("location", "Kalpakkam")], "physical threat to nuclear site"),
    ("Phishing campaign against BHEL staff is live, credentials incoming.", "latin", "en", S, "social_engineering",
     [("organization", "BHEL"), ("tactic", "Phishing")], "social engineering"),
    ("Power grid near Vizag will go dark tonight. Stay tuned.", "latin", "en", S, "infrastructure_disruption",
     [("location", "Vizag")], "infrastructure disruption threat"),
    ("நாளை DRDO சர்வர்களை முடக்குவோம்.", "mixed", "code_switched", S, "cyberattack",
     [("organization", "DRDO")], "Tamil + Latin org, mixed script"),
    ("நமது இலக்கு Avadi மற்றும் சென்னை. DRDO சர்வர்களை முடக்க வேண்டும்.", "mixed", "code_switched", S, "cyberattack",
     [("location", "Avadi"), ("location", "சென்னை"), ("organization", "DRDO")],
     "canonical showcase sentence; IndicNER fragments all three"),
    ("கல்பாக்கம் அணுமின் நிலையத்தை குறிவைக்க திட்டம் உள்ளது.", "native_tamil", "ta", S, "physical_threat",
     [("location", "கல்பாக்கம்")], "Tamil threat to nuclear plant"),
    ("சென்னை துறைமுக அமைப்புகளை முடக்க தயாராகுங்கள்.", "native_tamil", "ta", S, "infrastructure_disruption",
     [("location", "சென்னை")], "Tamil infrastructure threat"),
    ("ஆவடி ஆயுத தொழிற்சாலை பற்றிய தகவல் சேகரிக்கப்பட்டது.", "native_tamil", "ta", S, "surveillance",
     [("location", "ஆவடி")], "Tamil surveillance of ordnance factory"),
    ("DRDO servers ah nalaikku down panna poren da.", "romanized", "code_switched", S, "cyberattack",
     [("organization", "DRDO")], "Tanglish cyberattack claim"),
    ("Vizag port la periya problem varum, wait pannunga.", "romanized", "code_switched", S, "infrastructure_disruption",
     [("location", "Vizag")], "Tanglish veiled threat"),

    # ---------- D. SIGNAL + NOISE: HELD-OUT CNI entities (gazetteer recall = 0 here) ----------
    ("Targeting the NPCIL control network next. Access confirmed.", "latin", "en", S, "cyberattack",
     [("organization", "NPCIL")], "HELD OUT: NPCIL absent from gazetteer"),
    ("Kudankulam plant systems are our priority objective this month.", "latin", "en", S, "physical_threat",
     [("location", "Kudankulam")], "HELD OUT"),
    ("We have mapped the Sriharikota launch complex approaches.", "latin", "en", S, "surveillance",
     [("location", "Sriharikota")], "HELD OUT"),
    ("HAL production line schedules have been exfiltrated successfully.", "latin", "en", S, "cyberattack",
     [("organization", "HAL"), ("tactic", "exfiltrated")], "HELD OUT"),
    ("BARC internal documents will be released next week.", "latin", "en", S, "cyberattack",
     [("organization", "BARC")], "HELD OUT"),
    ("Trombay facility perimeter has weak points near the eastern gate.", "latin", "en", S, "surveillance",
     [("location", "Trombay")], "HELD OUT"),
    ("The Ennore port terminal will be disrupted during the strike.", "latin", "en", S, "infrastructure_disruption",
     [("location", "Ennore")], "HELD OUT"),
    ("Tarapur station staff credentials have been harvested.", "latin", "en", S, "social_engineering",
     [("location", "Tarapur")], "HELD OUT"),
    ("Manali refinery pipeline routes are documented and shared.", "latin", "en", S, "infrastructure_disruption",
     [("location", "Manali refinery")], "HELD OUT"),
    ("கூடங்குளம் அணுமின் நிலையம் குறித்த தகவல்கள் சேகரிக்கப்பட்டன.", "native_tamil", "ta", S, "surveillance",
     [("location", "கூடங்குளம்")], "HELD OUT: Kudankulam in Tamil"),
    ("ஸ்ரீஹரிகோட்டா ஏவுதளம் நமது கண்காணிப்பில் உள்ளது.", "native_tamil", "ta", S, "surveillance",
     [("location", "ஸ்ரீஹரிகோட்டா")], "HELD OUT: Sriharikota in Tamil"),
    ("NPCIL network ah target panna plan pannitom.", "romanized", "code_switched", S, "cyberattack",
     [("organization", "NPCIL")], "HELD OUT, Tanglish"),
    # held-out, benign (adversarial for any future expanded gazetteer)
    ("Visited the Sriharikota visitor centre, the launch viewing was amazing.", "latin", "en", N, B,
     [("location", "Sriharikota")], "HELD OUT adversarial: benign tourism"),
    ("HAL is hiring apprentices this year, applications close Friday.", "latin", "en", N, B,
     [("organization", "HAL")], "HELD OUT adversarial: benign recruitment"),
    ("Took a day trip to Ennore beach, very quiet compared to Marina.", "latin", "en", N, B,
     [("location", "Ennore")], "HELD OUT adversarial: benign travel"),
    ("BARC published a paper on isotope research last month.", "latin", "en", N, B,
     [("organization", "BARC")], "HELD OUT adversarial: benign science news"),
    ("கூடங்குளம் பகுதியில் புதிய பள்ளி கட்டப்படுகிறது.", "native_tamil", "ta", N, B,
     [("location", "கூடங்குளம்")], "HELD OUT adversarial: Tamil, school construction"),

    # ---------- E. FRAGMENTATION / VIRAMA TRAPS ----------
    ("DRDO மற்றும் ISRO அமைப்புகள் இலக்கு வைக்கப்படும்.", "mixed", "code_switched", S, "cyberattack",
     [("organization", "DRDO"), ("organization", "ISRO")], "two Latin orgs inside Tamil; both fragment"),
    ("BHEL ஊழியர்களின் கடவுச்சொற்கள் கிடைத்துவிட்டன.", "mixed", "code_switched", S, "social_engineering",
     [("organization", "BHEL")], "mixed script credential claim"),
    ("Vizag மற்றும் Chennai துறைமுகங்கள் நமது கவனத்தில்.", "mixed", "code_switched", S, "surveillance",
     [("location", "Vizag"), ("location", "Chennai")], "two Latin locations inside Tamil"),
    ("சென்னை நகரில் இன்று விழா நடைபெறுகிறது.", "native_tamil", "ta", N, B,
     [("location", "சென்னை")], "virama-bearing term, benign context"),
    ("விசாகப்பட்டினம் துறைமுகம் வழியாக சரக்கு அனுப்பப்படுகிறது.", "native_tamil", "ta", N, B,
     [("location", "விசாகப்பட்டினம்")], "long Tamil gazetteer term, benign shipping"),
    ("ஆவடி மற்றும் கல்பாக்கம் இரண்டும் நமது பட்டியலில் உள்ளன.", "native_tamil", "ta", S, "physical_threat",
     [("location", "ஆவடி"), ("location", "கல்பாக்கம்")], "two Tamil gazetteer locations, threat"),

    # ---------- F. SIGNAL: no CNI entity (generic threat) ----------
    ("Ransomware deployed across the victim network, payment demanded.", "latin", "en", S, "cyberattack",
     [("tactic", "Ransomware")], "generic threat, no CNI entity"),
    ("Emergency broadcast: restricted access protocol activated.", "latin", "en", S, "security_alert",
     [], "generic alert"),
    ("Unverified reports of infrastructure disruption in sector 4.", "latin", "en", S, "infrastructure_disruption",
     [], "generic, from original training data"),
    ("Threat actor communications intercepted on node B.", "latin", "en", S, "security_alert",
     [("actor", "Threat actor")], "generic"),
    ("Credential harvesting campaign is now active against the finance team.", "latin", "en", S, "social_engineering",
     [("tactic", "Credential harvesting")], "generic"),
    ("Protest scheduled tomorrow near the city central hub.", "latin", "en", S, "physical_threat",
     [], "generic; note 'city'/'hub' are NOT entities"),
]


def compute_entities(text: str, specs: list[tuple[str, str]], rec_id: str) -> list[dict]:
    out = []
    for etype, value in specs:
        idx = text.find(value)
        if idx < 0:
            raise SystemExit(f"[{rec_id}] entity {value!r} not found in text: {text!r}")
        out.append({"type": etype, "value": value, "start": idx, "end": idx + len(value)})
    return out


def classify_slice(entities: list[dict]) -> str:
    if not entities:
        return "none"
    vals = {e["value"].casefold() for e in entities}
    if any(v in HELD_OUT_TERMS for v in vals):
        return "held_out"
    if any(v in HELD_OUT_TERMS for v in vals) or any(
        any(h in v for h in HELD_OUT_TERMS) for v in vals
    ):
        return "held_out"
    gaz = set(CNI_GAZETTEER.keys())
    if any(v in gaz for v in vals):
        return "in_gazetteer"
    return "none"


def build() -> list[dict]:
    records = []
    for i, (text, script, lang, triage, intent, ent_specs, notes) in enumerate(RECORDS, 1):
        rec_id = f"syn-{i:04d}"
        entities = compute_entities(text, ent_specs, rec_id)
        slice_ = classify_slice(entities)
        # Tamil held-out terms are not in HELD_OUT_TERMS (which is Latin);
        # trust the authored note for those.
        if "HELD OUT" in notes:
            slice_ = "held_out"
        records.append({
            "id": rec_id,
            "text": text,
            "script": script,
            "language": lang,
            "source": "synthetic",
            "provenance": "authored-adversarial",
            "gazetteer_slice": slice_,
            "triage_label": triage,
            "intent_label": intent,
            "entities": entities,
            "tamil_verified": False if script in ("native_tamil", "mixed") else None,
            "notes": notes,
        })
    return records


def stratified_split(records: list[dict], dev_frac: float = 0.4):
    """Deterministic round-robin within each stratum -- no RNG."""
    buckets: dict[tuple, list[dict]] = {}
    for r in records:
        buckets.setdefault((r["triage_label"], r["gazetteer_slice"], r["script"]), []).append(r)
    dev, test = [], []
    for key in sorted(buckets):
        for i, r in enumerate(sorted(buckets[key], key=lambda x: x["id"])):
            (dev if (i % 5) < (dev_frac * 5) else test).append(r)
    return sorted(dev, key=lambda x: x["id"]), sorted(test, key=lambda x: x["id"])


def main() -> int:
    records = build()
    dev, test = stratified_split(records)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, rows in (("eval_set.dev.json", dev), ("eval_set.test.json", test)):
        (OUT_DIR / name).write_text(
            json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    print(f"total {len(records)}   dev {len(dev)}   test {len(test)}")
    for field in ("triage_label", "gazetteer_slice", "script"):
        counts: dict[str, int] = {}
        for r in records:
            counts[r[field]] = counts.get(r[field], 0) + 1
        print(f"  {field:16} " + "  ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    ents = sum(len(r["entities"]) for r in records)
    print(f"  entity spans     {ents}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
