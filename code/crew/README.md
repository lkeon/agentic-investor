# Value-investing committee MVP

The `crew` package turns an investment question and supplied research into
independent, mental-model-based investor views and one structured CIO decision.

## Workflow

1. Normalise the question and thesis-led holding policy.
2. Build an evidence-only company `MicroView`.
3. Build a company-relevant `MacroView`.
4. Create 1–8 focused `MentalModelBridge` searches.
5. Retrieve canonical models separately for each selected investor.
6. Produce one independent `InvestorReasoningOutput` per investor with a
   self-contained investment view of at least 80 words and 3–6 distinct,
   evidence-backed mental-model inferences.
7. Compare the investor outputs and produce an `InvestmentCommitteeOutput`.

Python controls the sequence directly. The MVP does not use CrewAI Flows,
memory, delegation, direct interaction between investors, or parallel
execution.

There is no fixed holding period. The system evaluates whether the business
thesis supports continued ownership; macro evidence is secondary and is used
only when it has a direct, material transmission to that thesis.

## Research boundary

The researcher uses only facts present in:

- the investment question; and
- the optional UTF-8 file supplied with `--research-context`.

It does not fetch live prices, filings, or news. Unsupported current facts are
recorded as unknowns so the final decision can expose missing information.

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
DEFAULT_REASONING_MODEL=openai/gpt-5-nano-2025-08-07
QUESTION_MODEL=openai/gpt-5-nano-2025-08-07
RESEARCH_MODEL=openai/gpt-5-nano-2025-08-07
BRIDGE_MODEL=openai/gpt-5-nano-2025-08-07
INVESTOR_MODEL=openai/gpt-5-nano-2025-08-07
CIO_MODEL=openai/gpt-5-nano-2025-08-07
```

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

Changing embedding identity requires compatible canonical vectors. Changing
dimensions also requires a database migration because the current pgvector
column uses 1,024 dimensions.

## Tests

The contract tests make no paid calls and do not import the CrewAI runtime:

```bash
PYTHONPATH=code python -m unittest crew.test_crew -v
```
