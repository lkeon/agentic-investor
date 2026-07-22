"""Store versioned system prompts used for canonicalisation and hierarchy classification."""

CANONICAL_PROMPT_VERSION = "canonical_mvp_v1"
HIERARCHY_PROMPT_VERSION = "hierarchy_mvp_v1"


CANONICAL_SYSTEM_PROMPT = """
Consolidate clusters of mental-model fragments from one investor.

For each cluster:
- assign every fragment to exactly one returned canonical model;
- merge only fragments with substantially the same proposition, mechanism,
  conditions and decision implication;
- split fragments that merely discuss a similar topic or consequence;
- preserve important conditions and failure conditions;
- use no external knowledge;
- preserve the supplied fragment codes exactly;
- return concise fields and no commentary.

A cluster may produce one or several canonical models.
""".strip()


CONSTITUTION_SYSTEM_PROMPT = """
Assign each canonical investment model to the fixed investment constitution.

Return one assignment per canonical_code.
Choose one primary domain, no more than three secondary domains, and one short,
neutral concept family that can be shared across investors.
Do not rewrite or evaluate the canonical model.
""".strip()


RELATIONSHIP_SYSTEM_PROMPT = """
Classify meaningful relationships for each supplied pair of canonical models.

Return zero to three relationships per pair.
Use parent_of only when the source is genuinely broader than the target.
Use directional causal and conditional relations only when direction is clear.
relation_strength is the strength of the connection.
relation_confidence is confidence that the relation type and direction are right.
Use no external knowledge and preserve canonical codes exactly.
""".strip()
