# Ethics and Data Handling Statement

Drafted for inclusion in any publication arising from this work.
Last updated: 2026-08-25.

---

## 1. Nature of the research

SAINTEL is **defensive security research**. It extracts named critical-
infrastructure entities from public open-source text to support human analysts.
It is not an offensive tool, produces no exploits, and takes no action against
any system.

## 2. Data sources and consent

All collection is from **public** channels: content published openly by its
authors to an unrestricted audience. No private groups, no invite-only content,
no accounts created to obtain access, and no interaction with any monitored
channel. Channels requiring membership to read history were **excluded rather
than joined**, because joining is participation rather than observation.

Informed consent was not sought. This follows established practice for research
on publicly available online data, where consent is impractical and the content
was published without expectation of privacy. Precedents in the Telegram
research literature take the same position, some under explicit ethics-board
review and some under a determination that review was not required.

**An institutional determination has not yet been obtained for this project.**
This is recorded as outstanding in `collection_protocol.md` §8 and must be
resolved before publication, not after.

## 3. Personal data

The collected corpus is Tamil news and security-feed content. It was not
selected for, and does not target, any private individual.

Safeguards in place:

- The raw archive is **gitignored and dockerignored**; it is never published.
- `scripts/sample_for_annotation.py` **redacts** email addresses, URLs, phone
  numbers, and ID-like numeric strings before any human sees a sample.
- Only **aggregate statistics** appear in results; no message text is
  reproduced in any output artifact.
- No attempt is made, at any stage, to identify or profile individual authors.

**Limitation, stated plainly:** redaction happens at *sampling* time, not at
*capture* time. The raw archive holds whatever was collected. This is why
retention is bounded and why the archive is deleted once the annotated corpus
is frozen.

## 4. Ransomware leak-site collection

The Tor collection path is implemented and verified but **has not been run to
build a corpus**. Leak sites publish real victim data. The standing rule for
this project:

> Record the claiming group and the fact of a claim. **Never** store individual
> names, addresses, or documents.

Channels and markets distributing stolen credentials or payment-card data are
**excluded entirely**. They contribute nothing to a Tamil entity-extraction
corpus, and collecting them would mean holding stolen personal data — a
materially different legal position from observing public propaganda.

## 5. Dual-use and potential for harm

A system that processes regional-language communications carries real potential
for misuse, and the harm is asymmetric: a false positive is not a wasted CPU
cycle, it is a person wrongly flagged, in a language most reviewers of the
output cannot read.

Concrete mitigations, already in the code rather than aspirational:

- **The threat classifier is disconnected from the data path.** It measures
  recall 0.250; deploying it would both miss most real threats and generate
  false positives against benign speech. It is wired out, and the reason is
  documented at the point of the decision.
- **The intent classifier's output is not treated as a prediction**, because
  `bart-large-mnli` is English-only and produces a near-uniform distribution on
  Tamil.
- **The system is assistive, never autonomous.** Nothing in it makes or should
  make an automated decision about a person.
- **Measured limitations are published alongside results**, so no downstream
  user can mistake an upper-bound number for field performance.

## 6. Reproducibility and honesty

- All synthetic evaluation data is labelled `"source": "synthetic"` and
  describes no real event, plan, or capability.
- Negative and null results are reported: two of four repair tiers change zero
  records; the headline advantage is not statistically significant at n=37; the
  first real-data run produced 22% structurally malformed entities and zero
  CNI mentions.
- An architecture decision was reversed by its own experiment, and **both the
  original and the reversal are kept in the repository** (`adr-001`, `adr-002`).

## 7. Compliance

Indian critical infrastructure is the subject domain, so India's **Digital
Personal Data Protection Act** is the governing framework, alongside GDPR
should any EU-resident data be incidentally collected. The safeguards in §3
are designed against those requirements. A formal compliance review has **not**
been conducted and is outstanding alongside the ethics determination.
