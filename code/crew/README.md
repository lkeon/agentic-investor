# Value-investing committee MVP

The `crew` package implements a deliberately small, structured workflow:

1. normalize the investment question and horizon;
2. build an evidence-only `MicroView`;
3. build a company-relevant `MacroView`;
4. translate evidence into several `MentalModelBridge` searches;
5. retrieve canonical models separately for each investor;
6. run an isolated investor round;
7. run peer review through each investor's original models;
8. produce a compact structured CIO decision;
9. write one JSON artifact.

Python controls the sequence directly. The MVP does not use CrewAI Flows,
memory, delegation, new result tables, or parallel execution.

## Runtime

CrewAI 1.15 requires Python 3.10 through 3.13. Create this project's virtual
environment with Python 3.13:

```bash
python3.13 -m venv .venvinvest
source .venvinvest/bin/activate
python -m pip install -r requirements.txt
```

The canonical database must already contain embedded models and hierarchy
edges.

## Research boundary

The MVP researcher structures only:

- facts present in the user's question; and
- optional evidence supplied with `--research-context`.

It does not have a live web or filings tool. Unsupported current facts become
explicit `unknown` claims. A future research tool can be attached to the
structured research stage without changing downstream schemas.

Supply research as UTF-8 text or JSON:

```bash
PYTHONPATH=code python -m crew.run_crew \
  "Should I invest in Example Company for at least two years?" \
  --research-context data/local/example_research.json \
  --investor buffett \
  --investor marks \
  --investor flatt
```

Without a research file:

```bash
PYTHONPATH=code python -m crew.run_crew \
  "Should I invest in Example Company?" \
  --investor buffett \
  --investor marks
```

## Safe retrieval check

This still makes structured normalization and research calls plus an embedding
request. It skips investor and CIO calls:

```bash
PYTHONPATH=code python -m crew.run_crew \
  "Should I invest in Example Company?" \
  --investor buffett \
  --investor marks \
  --dry-run
```

Default output:

```text
data/processed/crew/committee_result.json
```

The write is atomic and contains the question, evidence views, investor
bridges, both committee rounds, model configuration, embedding identity, and
CIO decision.

## Citation integrity and resume

Investor prompts receive an authoritative catalogue of evidence claim IDs and
mental-model codes. If an investor returns an identifier outside that
catalogue, the stage makes one correction attempt and validates the complete
output again. Invalid identifiers are never guessed or silently rewritten.

Before investor reasoning, and after every successfully validated investor,
the CLI writes a separate checkpoint beside the result:

```text
data/processed/crew/committee_result.checkpoint.json
```

After a failed reasoning stage, repeat the exact command with `--resume`. The
question, research-context contents, investor selection, retrieval options,
round-two setting, and model configuration must match the checkpoint:

```bash
PYTHONPATH=code python -m crew.run_crew \
  "Should I invest in Example Company?" \
  --investor buffett \
  --investor marks \
  --resume
```

The resumed run skips normalization, research, bridge construction, retrieval,
and any investor outputs already stored in the checkpoint.

## Reasoning model configuration

Models are full CrewAI provider-qualified names. Every stage falls back to one
default:

```dotenv
DEFAULT_REASONING_MODEL=openai/gpt-5-nano-2025-08-07

QUESTION_MODEL=
RESEARCH_MODEL=
BRIDGE_MODEL=
INVESTOR_ROUND_ONE_MODEL=
INVESTOR_ROUND_TWO_MODEL=
CIO_MODEL=
```

The same settings can be overridden per command:

```text
--question-model MODEL
--research-model MODEL
--bridge-model MODEL
--investor-round-one-model MODEL
--investor-round-two-model MODEL
--cio-model MODEL
```

## Embeddings

Only the OpenAI adapter is implemented in the MVP:

```dotenv
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-large
EMBEDDING_DIMENSIONS=1024
```

New canonical rows store the full identity:

```text
openai/text-embedding-3-large/1024
```

Retrieval selects only vectors created by the active embedding configuration.
Legacy rows containing only `text-embedding-3-large` remain readable during
the transition. Changing provider or model requires canonical re-embedding.
Changing dimensions also requires a database migration because the pgvector
column is fixed at 1,024 dimensions.

## Tests

The unit tests make no paid calls and do not require CrewAI to be imported:

```bash
PYTHONPATH=code python -m unittest crew.test_crew -v
```

They cover schema invariants, bridge evidence hydration, directional graph
edges, zero-vector rejection, model qualification, citation repair, checkpoint
compatibility, and investor output guardrails.
