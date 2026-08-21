"""Tests for the raw capture archive and the annotation sampler."""

import importlib.util
import json
from pathlib import Path

import pytest

from src.layer1_ingestion.capture_sink import (
    append_record, archive_path, build_record, load_seen_ids,
)

_spec = importlib.util.spec_from_file_location(
    "sampler", Path(__file__).resolve().parents[1] / "scripts" / "sample_for_annotation.py")
sampler = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sampler)


def rec(raw: bytes, **kw):
    kw.setdefault("topic", "raw_threat_stream")
    kw.setdefault("partition", 0)
    kw.setdefault("offset", 1)
    return build_record(raw, **kw)


class TestBuildRecord:
    def test_parses_json_payload(self):
        r = rec(b'{"text": "hello", "platform": "Telegram"}')
        assert r["payload"]["text"] == "hello"
        assert r["raw"] == '{"text": "hello", "platform": "Telegram"}'

    def test_non_json_still_archived_verbatim(self):
        # Archive must never lose a message just because it is malformed.
        r = rec(b"not json at all")
        assert r["payload"] is None
        assert r["raw"] == "not json at all"

    def test_json_array_is_not_treated_as_payload(self):
        r = rec(b'[1, 2, 3]')
        assert r["payload"] is None
        assert r["raw"] == "[1, 2, 3]"

    def test_undecodable_bytes_do_not_crash(self):
        r = rec(b"\xff\xfe bad bytes")
        assert r["payload"] is None and r["raw"]

    def test_native_script_preserved(self):
        r = rec(json.dumps({"text": "சென்னை"}).encode("utf-8"))
        assert r["payload"]["text"] == "சென்னை"

    def test_capture_id_is_topic_partition_offset(self):
        assert rec(b"{}", topic="t", partition=3, offset=99)["capture_id"] == "t:3:99"

    def test_kafka_timestamp_converted(self):
        r = rec(b"{}", kafka_timestamp_ms=1700000000000)
        assert r["kafka_timestamp"].startswith("2023-11-")

    def test_missing_timestamp_is_none(self):
        assert rec(b"{}")["kafka_timestamp"] is None


class TestArchiveIO:
    def test_daily_sharding(self):
        p = archive_path("2026-08-21T13:00:00+00:00", Path("/tmp/x"))
        assert p.name == "2026-08-21.jsonl"

    def test_append_then_reload_ids(self, tmp_path):
        for off in (1, 2, 3):
            append_record(rec(b'{"text":"m"}', offset=off), base=tmp_path)
        seen = load_seen_ids(tmp_path)
        assert seen == {f"raw_threat_stream:0:{o}" for o in (1, 2, 3)}

    def test_idempotency_key_survives_restart(self, tmp_path):
        r = rec(b'{"text":"m"}', offset=7)
        append_record(r, base=tmp_path)
        assert r["capture_id"] in load_seen_ids(tmp_path)

    def test_corrupt_line_does_not_break_reload(self, tmp_path):
        append_record(rec(b'{"text":"ok"}', offset=1), base=tmp_path)
        shard = next(tmp_path.glob("*.jsonl"))
        with shard.open("a", encoding="utf-8") as fh:
            fh.write("{ this is not json\n\n")
        assert len(load_seen_ids(tmp_path)) == 1

    def test_missing_dir_returns_empty(self, tmp_path):
        assert load_seen_ids(tmp_path / "nope") == set()


class TestScriptDetection:
    @pytest.mark.parametrize("text,expected", [
        ("Hello there", "latin"),
        ("சென்னை நகரம்", "native_tamil"),
        ("नमस्ते दुनिया", "native_devanagari"),
        ("আমার সোনার", "native_bengali"),
        ("DRDO சர்வர்", "mixed"),
    ])
    def test_detect(self, text, expected):
        assert sampler.detect_script(text) == expected


class TestRedaction:
    def test_no_pii_substring_survives(self):
        """Asserts the property that matters -- no raw PII leaves the archive.

        Deliberately does NOT assert which token replaced what: the patterns
        overlap (a 12-digit id is matched by the phone pattern first, so it is
        tagged [PHONE] not [ID]). Removal is the security property; the label
        is cosmetic.
        """
        original = "mail a@b.com or +91 98765 43210 see https://x.com/y id 123456789012"
        out, applied = sampler.redact(original)
        for secret in ("a@b.com", "98765", "43210", "x.com", "123456789012"):
            assert secret not in out, f"{secret!r} leaked through redaction"
        assert applied, "redaction should report what it applied"

    def test_each_pattern_fires_in_isolation(self):
        for text, token in [("mail a@b.com now", "[EMAIL]"),
                            ("see https://x.com/y", "[URL]"),
                            ("call +91 98765 43210", "[PHONE]"),
                            ("id 123456789012", "[PHONE]")]:
            out, _ = sampler.redact(text)
            assert token in out, f"{text!r} -> {out!r}"

    def test_clean_text_untouched(self):
        out, applied = sampler.redact("DRDO facility near Avadi")
        assert out == "DRDO facility near Avadi" and applied == []

    def test_native_script_not_mangled(self):
        out, _ = sampler.redact("சென்னை தாக்குதல்")
        assert out == "சென்னை தாக்குதல்"


class TestSampling:
    def _rows(self, n=40):
        out = []
        for i in range(n):
            t = ["hello world", "சென்னை நகரம்", "DRDO சர்வர்"][i % 3]
            out.append(build_record(json.dumps({"text": f"{t} {i}"}).encode("utf-8"),
                                    topic="t", partition=0, offset=i))
        return out

    def test_deterministic(self):
        rows = self._rows()
        a = [r["capture_id"] for r in sampler.sample(rows, 9, 42)]
        b = [r["capture_id"] for r in sampler.sample(rows, 9, 42)]
        assert a == b

    def test_stratifies_across_scripts(self):
        picked = sampler.sample(self._rows(), 9, 42)
        scripts = {sampler.detect_script(sampler.extract_text(r)) for r in picked}
        assert len(scripts) >= 2

    def test_deduplicates_identical_text(self):
        dupes = [build_record(json.dumps({"text": "same"}).encode("utf-8"),
                              topic="t", partition=0, offset=i) for i in range(10)]
        assert len(sampler.sample(dupes, 10, 1)) == 1

    def test_extract_text_falls_back_to_raw(self):
        assert sampler.extract_text({"payload": None, "raw": "plain"}) == "plain"

    @pytest.mark.parametrize("key", ["text", "content", "raw_text", "message"])
    def test_extract_text_key_variants(self, key):
        # Producers disagree on field names; the sampler must handle all of them.
        assert sampler.extract_text({"payload": {key: "v"}, "raw": "x"}) == "v"
