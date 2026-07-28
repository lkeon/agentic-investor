# Agentic Investor

## The Diligence Room

The Diligence Room is the product experience built in this `agentic-investor`
repository: a value-investing decision-support system built around explicit
mental models. It helps answer a practical question:

> Given the evidence available today, how would several long-term investors
> frame this investment, and what decision follows?

The project separates facts from interpretation. Company and macro evidence
are structured first, relevant mental models are retrieved from an
investor-specific knowledge base, and each investor perspective reasons
independently. A CIO then compares those views and produces one concise,
auditable decision.

## What the product does

```text
Investment question + supplied research
                    ↓
       Company view + macro view
                    ↓
       Material analytical questions
                    ↓
 Investor-specific mental-model retrieval
                    ↓
      Independent investor reasoning
                    ↓
 Structured CIO decision and conditions
```

The result explains:

- the stance and confidence of each investor perspective;
- which mental models were retrieved and which were actually applied;
- how evidence and counterevidence support each inference;
- risks of permanent capital loss and thesis-break conditions;
- the CIO decision, holding approach, conditions, and missing information;
- the evidence and sources behind the analysis.

The holding approach has no fixed end date: ownership continues only while the
company thesis remains valid. Macro conditions are monitored as secondary
inputs and matter only when they have a direct, material effect on that thesis.

## Mental-model knowledge base

The repository also builds the knowledge base used by the committee. Source
documents from investors such as Warren Buffett, Charlie Munger, Howard Marks,
Bruce Flatt, Mohnish Pabrai, and Guy Spier are converted into structured,
attributed mental models.

Canonical models retain their source provenance, applicability conditions,
failure conditions, investment-stage relevance, and embedding identity. They
are stored in PostgreSQL with `pgvector` and can be explored through the
included 3D visualisation.

## Current MVP boundary

The committee structures only the investment question and research supplied by
the user. It does not yet browse the web or fetch live filings, prices, or
market data. Unsupported decision-relevant facts remain explicit unknowns
rather than being filled from model memory.

This is research and decision support, not personalised financial advice.

## Run the committee

The project uses Python 3.13 because the current CrewAI runtime requires Python
below 3.14.

```bash
python3.13 -m venv .venvinv
source .venvinv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Add valid `OPENAI_API_KEY` and `DATABASE_URL` values to `.env`.
`OPENAI_API_KEY` remains required for canonical-model retrieval embeddings.
To route any crew reasoning stage through OpenRouter, also set
`OPENROUTER_API_KEY` and give that stage an
`openrouter/<provider>/<model>` identifier. The committee expects a populated
canonical mental-model database; create its tables with:

```bash
PYTHONPATH=code python -m mental_model_pipeline.database.setup_database
```

Start The Diligence Room interface:

```bash
source .venvinv/bin/activate
python -m streamlit run code/frontend/app.py
```

Or run the committee directly:

```bash
PYTHONPATH=code python -m crew.run_crew \
  "Should I invest in Brookfield at the current price?" \
  --investor buffett \
  --investor marks \
  --investor flatt
```

## Project guide

- [Committee workflow and CLI](code/crew/README.md)
- [Streamlit product interface](code/frontend/README.md)
- [Source ingestion](code/data_ingestion/README.md)
- [Canonical mental-model construction](code/mental_model_pipeline/canonical/README.md)
- [Mental-model visualisation](code/vis/README.md)

## Validation

The committee contract tests are local and make no paid model calls:

```bash
PYTHONPATH=code python -m unittest crew.test_crew -v
```

## Deploy The Diligence Room

The `diligence-room-deploy` branch is prepared for Streamlit Community Cloud.
Create the app with:

- entrypoint: `code/frontend/app.py`;
- Python: 3.13;
- dependency file: `code/frontend/requirements.txt`.

Copy [`.streamlit/secrets.toml.example`](.streamlit/secrets.toml.example) into
the deployment's **Advanced settings → Secrets** field and replace every
placeholder. The three sensitive values are:

- `DATABASE_URL`: an externally reachable PostgreSQL database with `pgvector`
  enabled and the canonical mental-model tables populated;
- `OPENAI_API_KEY`: used by the fixed OpenAI embedding and retrieval layer;
- `OPENROUTER_API_KEY`: used by the reasoning stages.

The deployment template routes all reasoning stages through
`openrouter/deepseek/deepseek-v4-flash`. OpenAI remains available for embeddings
and for any reasoning stage later changed to an `openai/<model>` value.

Streamlit Community Cloud cannot access or start a PostgreSQL service on the
local machine. The populated mental-model database must therefore be migrated
to a managed PostgreSQL/pgvector service before the deployed committee can run.
Hosted runs use isolated temporary result files and never load another browser
session's latest local result.
