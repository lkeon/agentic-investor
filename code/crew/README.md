# Value-investing committee MVP

The `crew` package turns an investment question and supplied research into
independent, mental-model-based investor views and one structured CIO decision.

## Workflow

1. Normalise the question and thesis-led holding policy.
2. Optionally collect a bounded `MicroResearchData` record.
3. Build an evidence-only company `MicroView`, hydrating collector-owned facts
   locally so an LLM cannot rewrite their values or citations.
4. Load or construct one company-independent daily US `MacroView` from
   `MacroResearchData`.
5. Create 1–6 focused `MentalModelBridge` searches, including any material
   company transmission from the shared macro environment.
6. Retrieve canonical models separately for each selected investor.
7. Produce one independent `InvestorReasoningOutput` per investor with a
   self-contained investment view of at least 80 words and 3–6 distinct,
   evidence-backed mental-model inferences.
8. Compare the investor outputs and produce an `InvestmentCommitteeOutput`.

Python controls the sequence directly. The MVP does not use CrewAI Flows,
memory, delegation, direct interaction between investors, or parallel
execution.

There is no fixed holding period. The system evaluates whether the business
thesis supports continued ownership; macro evidence is secondary and is used
only when it has a direct, material transmission to that thesis.

## Research boundary

Without `--external-research`, behavior is unchanged: the researchers use only
the question and optional UTF-8 `--research-context`.

With external research enabled, dedicated collectors—not reasoning agents—may
use:

- SEC filings and Company Facts XBRL;
- one Exa structured search grounded in official company investor-relations
  documents;
- one Alpaca IEX price snapshot, combined with SEC-reported shares
  outstanding to calculate market capitalisation;
- FRED rates, credit, nominal GDP, and corporate-equity data;
- Yale's official Shiller CAPE workbook.

The hard defaults are two filings and one cited IR source document. The UI/CLI
allow up to two filings and four cited IR source documents from one structured
Exa search. Exa returns only the requested company
description, qualitative observations, and optional disclosed credit rating;
every accepted field must be grounded in a validated official-company URL.
The daily macro collector makes at most five HTTP attempts and contains only
four indicator groups: 5-year Treasuries, broad investment-grade spreads, a
labelled Buffett proxy, and Shiller CAPE. No raw filing, XBRL history, search
snippet, or macro series is passed to investors.

## Run

Use Python 3.13 and the root environment:

```bash
source .venvinv/bin/activate
```

Run with optional supplied research:

```bash
PYTHONPATH=code python -m crew.run_crew \
  "Should I invest in Example Company at the current price?" \
  --research-context data/local/example_research.json \
  --investor buffett \
  --investor marks \
  --investor flatt
```

Enable the bounded sources:

```bash
PYTHONPATH=code python -m crew.run_crew \
  "Is Berkshire Hathaway still attractively valued?" \
  --external-research \
  --max-filings 2 \
  --max-ir-documents 1 \
  --investor buffett \
  --investor munger
```

Individual source flags support `--source` and `--no-source` forms:

```text
--[no-]sec-filings
--[no-]investor-relations
--[no-]market-data
--[no-]rates-and-credit
--[no-]aggregate-valuation
--[no-]credit-rating
```

`--credit-rating` is experimental and accepts a rating only from a bounded
official company document. If none is found, the rating remains unknown and
the shared broad investment-grade spread stays explicitly labelled as a proxy.

## Shared daily MacroView

The validated daily macro file is stored atomically as JSON under
`data/processed/crew/macro_views` by default. This is a versioned daily input,
not a general source cache. If today's compatible object is absent or invalid,
the first committee run constructs it within the five-request budget.

An end-of-day scheduler can prepare the next weekday in advance:

```bash
PYTHONPATH=code python -m crew.research.prepare_macro
```

An explicit date and rebuild are also supported:

```bash
PYTHONPATH=code python -m crew.research.prepare_macro \
  --applicable-date 2026-07-30 \
  --force
```

The path can be changed with `MACRO_VIEW_STORE_PATH`. Runtime-generated local
files on Streamlit Community Cloud are not guaranteed to survive a container
restart; lazy reconstruction keeps the MVP functional when that occurs.

Run without a research file:

```bash
PYTHONPATH=code python -m crew.run_crew \
  "Should I invest in Example Company?" \
  --investor buffett \
  --investor marks
```

The completed structured artifact is written atomically to:

```text
data/processed/crew/committee_result.json
```

## Safe retrieval check

`--dry-run` performs question normalisation, evidence structuring, bridge
construction, and mental-model retrieval, but skips investor reasoning and CIO
synthesis:

```bash
PYTHONPATH=code python -m crew.run_crew \
  "Should I invest in Example Company?" \
  --investor buffett \
  --investor marks \
  --dry-run
```

Structured model and embedding calls made before the reasoning stage may still
be billable.

## Citation integrity

Every investor receives an authoritative catalogue of evidence claim IDs and
mental-model codes. Unknown identifiers are rejected. The investor receives
one explicit correction attempt, after which an invalid output fails the run.

The CIO may cite only:

- evidence claims present in the `MicroView` or `MacroView`; and
- mental models actually applied in an investor inference.

## Model configuration

Reasoning models use CrewAI provider-qualified names and fall back to the
global default:

```dotenv
DEFAULT_REASONING_PROVIDER=openai
DEFAULT_REASONING_MODEL=openai/gpt-5-nano-2025-08-07
QUESTION_MODEL=openai/gpt-5-nano-2025-08-07
RESEARCH_MODEL=openai/gpt-5-nano-2025-08-07
BRIDGE_MODEL=openai/gpt-5-nano-2025-08-07
INVESTOR_MODEL=openai/gpt-5-nano-2025-08-07
CIO_MODEL=openai/gpt-5-nano-2025-08-07
```

Direct provider names continue to work. To route one or every reasoning stage
through OpenRouter, set its model to
`openrouter/<OpenRouter provider>/<model>`:

```dotenv
OPENROUTER_API_KEY=your-openrouter-api-key
OPENROUTER_API_BASE=https://openrouter.ai/api/v1
OR_SITE_URL=
OR_APP_NAME="The Diligence Room"

DEFAULT_REASONING_MODEL=openrouter/anthropic/your-model
QUESTION_MODEL=openrouter/openai/your-model
RESEARCH_MODEL=openrouter/google/your-model
BRIDGE_MODEL=openrouter/anthropic/your-model
INVESTOR_MODEL=openrouter/anthropic/your-model
CIO_MODEL=openrouter/openai/your-model
```

Only stages carrying the `openrouter/` prefix use the OpenRouter key and
endpoint. Other stages may remain on direct providers in the same run. The
adapter uses CrewAI's native OpenAI-compatible client, preserves the complete
OpenRouter model slug, and optionally sends `OR_SITE_URL` and `OR_APP_NAME` as
OpenRouter attribution headers. It also requires OpenRouter to select a route
that supports the request parameters. Because every reasoning stage produces
a Pydantic result, choose OpenRouter models that support structured outputs.

Command-line overrides:

```text
--question-model MODEL
--research-model MODEL
--bridge-model MODEL
--investor-model MODEL
--cio-model MODEL
```

Embedding configuration remains independently swappable:

```dotenv
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-large
EMBEDDING_DIMENSIONS=1024
```

External source configuration:

```dotenv
SEC_USER_AGENT="The Diligence Room your-email@example.com"
ALPACA_API_KEY_ID=your-alpaca-key-id
ALPACA_API_SECRET_KEY=your-alpaca-secret-key
ALPACA_DATA_API_BASE=https://data.alpaca.markets
FRED_API_KEY=your-fred-api-key
EXA_API_KEY=your-exa-api-key
MACRO_VIEW_STORE_PATH=data/processed/crew/macro_views
```

OpenRouter reasoning settings do not change embeddings. Changing embedding
identity requires compatible canonical vectors. Changing dimensions also
requires a database migration because the current pgvector column uses 1,024
dimensions.

## Tests

The contract tests make no paid model calls:

```bash
PYTHONPATH=code python -m unittest crew.test_crew -v
```
