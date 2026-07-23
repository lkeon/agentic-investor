# Canonical mental-model pipeline — MVP

This folder converts document-level mental-model fragments into investor-specific
canonical mental models and then connects those models into a shared hierarchy.

The MVP deliberately uses only two new database tables:

```text
canonical_mental_models
canonical_model_edges
```

It does **not** use:

```text
canonical_pipeline_runs
canonical_model_fragments
review queues or review flags
```

Supporting evidence is stored directly on each canonical model as an array of
fragment codes.

## Conceptual flow

### Pass 1 — investor-specific canonicalisation

```text
embedded fragments for one investor
        ↓
local cosine-similarity clustering
        ↓
singletons promoted directly without an LLM call
        ↓
multi-fragment clusters sent to GPT-5.6 Luna
        ↓
Luna may merge or split each cluster
        ↓
Python validates fragment assignments and calculates weights
        ↓
canonical embeddings generated
        ↓
investor's canonical rows replaced transactionally
        ↓
JSONL export
```

Canonical models are never merged across investors in Pass 1.

### Pass 2 — constitution and hierarchy

```text
all canonical models
        ↓
Luna assigns shared constitution domains and concept families
        ↓
embeddings select nearby within-investor and cross-investor pairs
        ↓
Luna classifies meaningful relationships
        ↓
weighted graph edges replace the previous hierarchy
        ↓
enriched JSONL export
```

## Investment constitution

Every canonical model is placed in one primary top-level domain and optionally
several secondary domains:

```text
mandate_and_objective
circle_of_competence
business_and_asset_quality
management_and_governance
financial_resilience
valuation_and_expected_return
risk_and_uncertainty
market_and_cycle_context
portfolio_construction
monitoring_and_exit
```

The constitution is shared across investors. The canonical models themselves
remain investor-specific.

## Canonical-model edges

Each edge connects two canonical mental models. The MVP relation vocabulary is:

```text
parent_of
related_to
similar_to
overlaps_with
supports
contradicts
causes
increases
reduces
protects_against
requires
applies_when
fails_when
```

`child_of` is not stored separately. It is inferred from an incoming
`parent_of` edge.

Each edge stores:

```text
relation_strength    # how strong the relationship is
relation_confidence  # confidence that type and direction are correct
candidate_similarity # cosine similarity used to propose the pair
scope                 # within_investor or cross_investor
```

## Canonical weights

Only three static scores are stored:

```text
evidence_confidence
investor_importance
base_weight
```

They are deterministic Python heuristics, not probabilities and not LLM
self-assessments.

```text
evidence_confidence =
    40% evidence directness
  + 30% evidence breadth
  + 30% cluster coherence

investor_importance =
    70% evidence breadth
  + 30% decision-stage breadth

base_weight =
    70% evidence confidence
  + 30% investor importance
```

Evidence-directness class scores are currently:

```text
directly_stated  = 1.00
strongly_implied = 0.70
weakly_inferred  = 0.35
```

These coefficients should be treated as configurable ranking heuristics.

## Cost controls

- GPT-5.6 Luna is the default for both passes.
- Single fragments become canonical models without an LLM call.
- Clustering and graph candidate search run locally.
- Source quotations and document text are omitted from prompts.
- Several clusters or model pairs share each API request.
- Luna evaluates only embedding-selected candidate pairs, not every possible
  pair.
- Raw embeddings are excluded from JSONL unless explicitly requested.
- The model name is configurable, so the same pipeline can be reprocessed with
  another OpenAI model.

## Environment variables

```dotenv
CANONICAL_MODEL=gpt-5.6-luna
CANONICAL_REASONING_EFFORT=low

HIERARCHY_MODEL=gpt-5.6-luna
HIERARCHY_REASONING_EFFORT=low

# Defaults to the fragment embedding model when omitted.
CANONICAL_EMBEDDING_MODEL=text-embedding-3-large
```

CLI `--model` and `--reasoning-effort` arguments override the environment.

## Database setup

For a new database or when no earlier canonical schema exists:

```bash
PYTHONPATH=code python -m \
mental_model_pipeline.canonical.setup_database
```

## Pass 1 dry run

No database writes or OpenAI calls:

```bash
PYTHONPATH=code python -m \
mental_model_pipeline.canonical.canonicalise_all \
--investor-id buffett \
--dry-run
```

The output includes fragment count, cluster count, singleton count and estimated
paid API calls.

## Run Pass 1 for one investor

```bash
PYTHONPATH=code python -m \
mental_model_pipeline.canonical.canonicalise_all \
--investor-id buffett
```

If the investor already has canonical models, the command skips them. Rebuild
transactionally with:

```bash
PYTHONPATH=code python -m \
mental_model_pipeline.canonical.canonicalise_all \
--investor-id buffett \
--replace
```

The complete replacement set and its embeddings are generated before the
database transaction begins. A database failure rolls back to the previous set.

For clearer failure boundaries during the MVP, process one investor per command.

## Reprocess with a different OpenAI model

```bash
PYTHONPATH=code python -m \
mental_model_pipeline.canonical.canonicalise_all \
--investor-id buffett \
--model gpt-5.6-terra \
--replace
```

The database stores the model name and prompt version on every canonical row.
The MVP replaces old database rows rather than retaining model-run history.
Keep old JSONL files separately when comparisons are needed.

## Pass 2 dry run

This performs embedding-only pair selection and makes no OpenAI calls:

```bash
PYTHONPATH=code python -m \
mental_model_pipeline.canonical.hierarchy_pass \
--dry-run
```

The full pass first assigns constitution domains, then adds a small number of
same-concept-family candidates before relationship classification.

## Run Pass 2

```bash
PYTHONPATH=code python -m \
mental_model_pipeline.canonical.hierarchy_pass
```

Pass 2 replaces all existing graph edges and updates constitution fields in one
transaction.

## JSONL output

Both passes export to:

```text
data/processed/canonical/canonical_mental_models.jsonl
```

The export contains:

```text
canonical content
supporting fragment codes
three static weights
constitution placement
incoming and outgoing weighted edges
model and embedding provenance
```

Raw embedding arrays are omitted by default. Add:

```text
--include-embeddings-in-export
```

when they are specifically needed.

## Deterministic repair behaviour

The application checks Luna's fragment assignments:

```text
unknown fragment code   → removed
duplicate assignment    → first assignment retained
omitted fragment        → promoted to a singleton canonical model
omitted cluster         → all fragments promoted individually
invalid Pydantic output → API batch fails
```

Repair counts are printed to the console. No review-state columns or workflows
are created in this MVP.

## Suggested first test

1. Run database setup.
2. Dry-run Buffett clustering.
3. Process a small investor corpus with Luna.
4. Inspect the JSONL manually for over-merging.
5. Adjust `--similarity-threshold` before processing every investor.
6. Run the hierarchy dry run, then the full hierarchy pass.
