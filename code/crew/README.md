# Minimal investor crew

This package lives directly under:

```text
code/crew/
```

It performs:

1. question normalisation with a structured LLM call;
2. query embedding using the canonical embedding model;
3. separate top-K MMC retrieval for every investor;
4. optional same-investor graph-neighbour expansion;
5. one isolated CrewAI Round 1 task per investor;
6. one CrewAI Round 2 peer-review task per investor;
7. JSON output without CIO synthesis or new database tables.

The package still imports canonical models, embeddings and the database
connection from `mental_model`. Only the Crew orchestration code has
moved to the top-level `crew` package.

## Folder structure

```text
code/
├── crew/
│   ├── __init__.py
│   ├── schemas.py
│   ├── retrieval.py
│   ├── agents.py
│   ├── run_crew.py
│   └── README.md
└── mental_model/
    ├── canonical/
    ├── database/
    └── fragments/
```

## Install

```bash
pip install "crewai>=1.15,<2"
```

The existing OpenAI, SQLAlchemy, NumPy and pgvector dependencies are reused.

## Environment

```dotenv
QUESTION_NORMALISER_MODEL=gpt-5.6-luna
INVESTOR_ANALYSIS_MODEL=gpt-5.6-luna
PEER_REVIEW_MODEL=gpt-5.6-luna
```

The three models are independent and can be replaced without changing
retrieval or database code.

## Safe retrieval check

This makes one normalisation call and one embedding request, but no CrewAI
investor-analysis calls:

```bash
PYTHONPATH=code python -m crew.run_crew \
  "Should I invest in Brookfield at the current valuation?" \
  --investor buffett \
  --investor marks \
  --investor flatt \
  --dry-run
```

## Full two-round run

```bash
PYTHONPATH=code python -m crew.run_crew \
  "Should I invest in Brookfield at the current valuation?" \
  --investor buffett \
  --investor marks \
  --investor flatt
```

Omit `--investor` to include every investor in the canonical table.

## Useful options

```text
--top-k 5
--neighbours 3
--skip-round-two
--analysis-model MODEL
--peer-review-model MODEL
--verbose
--output-path PATH
```

Default output:

```text
data/processed/crew/committee_result.json
```

Retrieved MMCs are dynamic task context. The CrewAI agent backstory stays
short and stable. Round 1 agents are isolated. Round 2 receives all peer views
but retains only the reviewing investor's MMC packet as its analytical
guardrail.
