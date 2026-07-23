"""Store versioned prompts for canonicalisation and hierarchy."""

CANONICAL_PROMPT_VERSION = "canonical_mvp_v1"
HIERARCHY_PROMPT_VERSION = "hierarchy_mvp_v2"


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

Choose exactly one primary_domain and no more than three secondary_domains.
Use only the supplied ConstitutionDomain enum values.

Domain meanings:

- mandate_and_objective:
  investment mandate, objective, required return or governing purpose;

- circle_of_competence:
  knowability, analytical boundaries and limits of understanding;

- business_and_asset_quality:
  economics, competitive advantage, durability and asset quality;

- management_and_governance:
  management quality, incentives, ownership and governance;

- financial_resilience:
  leverage, liquidity, cash generation and balance-sheet durability;

- valuation_and_expected_return:
  intrinsic value, purchase price, expected return and margin of safety;

- risk_and_uncertainty:
  permanent-loss risk, uncertainty, fragility and downside exposure;

- market_and_cycle_context:
  cycles, sentiment, macro conditions, credit and market environment;

- portfolio_construction:
  position sizing, concentration, diversification and capital allocation;

- monitoring_and_exit:
  thesis monitoring, invalidation, selling and exit decisions.

Do not generate a concept family in this step.
Do not rewrite or evaluate the canonical model.
Use no external knowledge.
Preserve canonical_code exactly.
Return no commentary.
""".strip()


CONCEPT_FAMILY_SYSTEM_PROMPT = """
Name each supplied cluster of canonical investment models as one shared
concept family.

Return exactly one result per cluster_key.

The concept_family value must use this exact format:

family_name: very short shared description

Rules:
- family_name must be lowercase snake_case;
- the complete value must contain exactly one colon;
- describe the shared investment idea;
- do not name an investor, author, company or source;
- do not merely repeat the broad constitution domain;
- distinguish the cluster from neighbouring concepts in the same domain;
- keep the description to a short phrase;
- keep the complete value at no more than 100 characters;
- use one value for every model in the supplied cluster;
- do not split, merge or reassign the supplied models;
- use no external knowledge;
- preserve cluster_key exactly;
- return no commentary.

Examples:

pricing_power: raising prices without materially losing demand

margin_of_safety: buying below conservatively estimated value

capital_allocation: directing cash toward highest-return uses

balance_sheet_resilience: surviving stress without forced financing

cycle_positioning: adjusting risk to market-cycle conditions
""".strip()


RELATIONSHIP_SYSTEM_PROMPT = """
Classify meaningful relationships for each supplied pair of canonical models.

Return zero to three relationships per pair.

Use parent_of only when the source model is genuinely broader than the target.
Use directional causal and conditional relations only when direction is clear.

relation_strength measures the strength of the connection.
relation_confidence measures confidence that the relation type and direction
are correct.

Use no external knowledge.
Preserve canonical codes exactly.
Return no commentary.
""".strip()