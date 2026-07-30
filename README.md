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
Investment question + supplied or bounded external research
                    ↓
 Company MicroView + shared daily US MacroView
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
company thesis remains valid. One compact daily US `MacroView` gives every
company the same rates, credit, Buffett Indicator proxy, and Shiller CAPE
backdrop. Company-specific transmission is assessed later and remains
secondary to business fundamentals and valuation.

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

External research is optional and disabled by default. When enabled, a bounded
collector may use SEC filings and XBRL, Exa structured research grounded in
official company investor-relations documents, an Alpaca IEX reference price,
FRED, and Yale's official Shiller dataset. Market capitalisation is calculated
from that price and the latest SEC-reported shares outstanding. The interface
exposes hard limits of two filings and four cited IR source documents from one
structured Exa search. Ungrounded Exa fields are discarded, and reasoning
agents never receive browser tools.

The MVP deliberately excludes general news, competitors, forecasts,
transcripts, social media, technical indicators, and autonomous browsing.
Unavailable facts retain explicit typed reasons instead of being filled from
model memory.

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

Optional external research also uses `SEC_USER_AGENT`,
`ALPACA_API_KEY_ID`, `ALPACA_API_SECRET_KEY`, `FRED_API_KEY`, and
`EXA_API_KEY`. These are required only for their selected sources; the
unconfigured source is reported as unavailable rather than silently replaced.

```bash
PYTHONPATH=code python -m mental_model_pipeline.database.setup_database
```

Start The Diligence Room interface:

```bash
source .venvinv/bin/activate
cd code/frontend
python -m streamlit run app.py
```

Or run the committee directly:

```bash
PYTHONPATH=code python -m crew.run_crew \
  "Should I invest in Brookfield at the current price?" \
  --external-research \
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
