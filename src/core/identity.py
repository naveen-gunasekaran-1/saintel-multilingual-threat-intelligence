"""Canonical entity identity.

Single source of truth for how a (type, value) pair becomes a graph key.

This existed twice: `neo4j_connector._entity_id` wrote the ids, and
`synthesis_agent._entity_id` built the seed ids used to look them back up,
carrying the comment "must stay byte-identical to it". Two copies of a rule
that must never diverge is a latent bug: any change to casefolding or
separator in one place silently breaks GraphRAG retrieval, and nothing fails
loudly -- the multi-hop query just returns zero neighbours.
"""

from __future__ import annotations

__all__ = ["entity_id"]


def entity_id(entity_type: str, value: str) -> str:
    """Stable graph key for an entity.

    Casefolded so `DRDO` and `drdo` collapse to one node, and stripped so
    whitespace variation from tokenizer output does not fragment the graph.
    """
    return f"{entity_type}:{value.strip().casefold()}"
