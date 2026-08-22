from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from stix2 import AttackPattern, Bundle, Identity, Location, Relationship, ThreatActor

from src.core.config import get_settings
from src.core.logger import get_logger
from src.core.schemas import ThreatEntity, ThreatSignal

settings = get_settings()
logger = get_logger(__name__, level=settings.log_level)


class STIXBundleFormatter:
	"""Convert validated SAINTEL signals into STIX 2.1 domain objects."""

	def __init__(self, producer_name: str = "SAINTEL") -> None:
		self.producer_name = producer_name.strip() or "SAINTEL"

	@staticmethod
	def _unique_entities(signal: ThreatSignal) -> list[ThreatEntity]:
		unique: dict[tuple[str, str], ThreatEntity] = {}
		for entity in signal.entities:
			key = (entity.entity_type, entity.value.casefold())
			current = unique.get(key)
			if current is None or entity.confidence > current.confidence:
				unique[key] = entity
		return list(unique.values())

	@staticmethod
	def _description(signal: ThreatSignal) -> str:
		return (
			f"SAINTEL signal {signal.signal_id} from {signal.source_type} "
			f"source {signal.source_id}; intent={signal.intent}; "
			f"confidence={signal.confidence:.4f}."
		)

	def build_bundle(self, signal: ThreatSignal) -> Bundle | None:
		"""Build a STIX 2.1 Bundle, returning None for malformed input."""
		try:
			validated_signal = ThreatSignal.model_validate(signal)
			entities = self._unique_entities(validated_signal)
			description = self._description(validated_signal)

			producer = Identity(
				name=self.producer_name,
				identity_class="organization",
				description="SAINTEL — South Asian Infrastructure, Native Threat Entity Labelling. Extracts CNI entities from code-switched South Asian text.",
				created=validated_signal.created_at,
				modified=validated_signal.created_at,
			)
			objects: list[Any] = [producer]
			actor_refs: list[str] = []
			organization_refs: list[str] = []
			location_refs: list[str] = []
			tactic_refs: list[str] = []

			for entity in entities:
				if entity.entity_type == "actor":
					actor = ThreatActor(
						name=entity.value,
						description=description,
						confidence=round(entity.confidence * 100),
						created_by_ref=producer.id,
						created=validated_signal.created_at,
						modified=validated_signal.created_at,
					)
					objects.append(actor)
					actor_refs.append(actor.id)
				elif entity.entity_type == "organization":
					organization = Identity(
						name=entity.value,
						identity_class="organization",
						description=description,
						confidence=round(entity.confidence * 100),
						created_by_ref=producer.id,
						created=validated_signal.created_at,
						modified=validated_signal.created_at,
					)
					objects.append(organization)
					organization_refs.append(organization.id)
				elif entity.entity_type == "location":
					location = Location(
						name=entity.value,
						region=entity.value,
						description=description,
						confidence=round(entity.confidence * 100),
						created_by_ref=producer.id,
						created=validated_signal.created_at,
						modified=validated_signal.created_at,
					)
					objects.append(location)
					location_refs.append(location.id)
				elif entity.entity_type == "tactic":
					tactic = AttackPattern(
						name=entity.value,
						description=description,
						confidence=round(entity.confidence * 100),
						created_by_ref=producer.id,
						created=validated_signal.created_at,
						modified=validated_signal.created_at,
					)
					objects.append(tactic)
					tactic_refs.append(tactic.id)

			if validated_signal.intent and validated_signal.intent != "unknown":
				intent_pattern = AttackPattern(
					name=validated_signal.intent,
					description=description,
					confidence=round(validated_signal.confidence * 100),
					created_by_ref=producer.id,
					created=validated_signal.created_at,
					modified=validated_signal.created_at,
				)
				objects.append(intent_pattern)
				tactic_refs.append(intent_pattern.id)

			for actor_ref in actor_refs:
				for target_ref in organization_refs:
					objects.append(
						Relationship(
							source_ref=actor_ref,
							target_ref=target_ref,
							relationship_type="targets",
							description=description,
							created_by_ref=producer.id,
							created=validated_signal.created_at,
							modified=validated_signal.created_at,
						)
					)
				for target_ref in location_refs:
					objects.append(
						Relationship(
							source_ref=actor_ref,
							target_ref=target_ref,
							relationship_type="located-at",
							description=description,
							created_by_ref=producer.id,
							created=validated_signal.created_at,
							modified=validated_signal.created_at,
						)
					)
				for target_ref in tactic_refs:
					objects.append(
						Relationship(
							source_ref=actor_ref,
							target_ref=target_ref,
							relationship_type="uses",
							description=description,
							created_by_ref=producer.id,
							created=validated_signal.created_at,
							modified=validated_signal.created_at,
						)
					)

			return Bundle(*objects, allow_custom=False)
		except Exception as exc:
			logger.exception(
				"STIX bundle formatting failed",
				extra={"signal_id": getattr(signal, "signal_id", "unknown"), "error": str(exc)},
			)
			return None

	def to_json(self, signal: ThreatSignal) -> str | None:
		"""Serialize a validated signal as STIX 2.1 JSON."""
		bundle = self.build_bundle(signal)
		if bundle is None:
			return None
		try:
			# ensure_ascii=False keeps native script literal in the exported
			# bundle. Without it stix2 emits \u0b9a.. escapes, which violates
			# the project's native-UTF-8 directive and makes the flagship CTI
			# artefact unreadable to a human analyst.
			return bundle.serialize(pretty=False, ensure_ascii=False)
		except Exception as exc:
			logger.exception("STIX bundle serialization failed", extra={"error": str(exc)})
			return None
