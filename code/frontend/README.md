# Investment Committee frontend

This Streamlit application is the product interface for the structured
value-investing committee in `code/crew`.

## Product flow

The interface presents two decision stages:

1. independent investor perspectives with a self-contained investment view of
   at least 80 words, retrieved models, applied models, 3–6 material
   inferences, risks, and evidence joined directly to the conclusion it
   supports;
2. a structured CIO decision comparing those views and identifying the
   decisive evidence, mental models, conditions, and information gaps.

It also provides:

- investment-question and optional research input;
- opt-in bounded external research with per-source controls;
- explicit filing, official-IR-document, and discovery-search limits;
- selection of every available investor perspective;
- compact mental-model retrieval controls;
- an always-visible, plain-language diligence timeline beneath the run controls;
- an optional scrollable technical stream retaining raw CrewAI, model,
  validation, database, and Python output;
- automatic display of the latest valid completed result;
- Markdown and validated JSON downloads;
- an on-demand, stage-by-stage viewer for retained Pydantic-compatible inputs
  and outputs;
- a Stop control that terminates the active local crew process;
- cited evidence and source references embedded inside investor and CIO
  reasoning, with complete records retained in structured output and downloads.

Selecting **Show technical execution log** also runs CrewAI in verbose mode.
The interface retains up to 8,000 technical lines without removing CrewAI
trace panels. When a run fails, the technical window is displayed
automatically even if verbose logging was not selected.

External research is disabled by default. When enabled, source-specific
collectors gather bounded official evidence before the reasoning stages;
investor and CIO agents never browse. It treats the output as decision support
rather than personalised financial advice. It has no fixed holding period:
the product supports continued ownership only while the core business thesis
remains valid. One shared daily US MacroView is secondary and matters only when
its company-specific transmission is material.

Reasoning stages may use direct providers or OpenRouter independently through
the root `.env`; OpenAI embedding configuration remains separate. See
[`code/crew/README.md`](../crew/README.md#model-configuration) for the model
syntax and required OpenRouter variables.

## Run

From the repository root:

```bash
source .venvinv/bin/activate
cd code/frontend
python -m streamlit run app.py
```

Open the local URL printed by Streamlit, normally
`http://localhost:8501`.

The frontend writes completed results to:

```text
data/processed/crew/committee_result.json
```

## Runtime requirements

- Python 3.13 with the root `requirements.txt` installed;
- a root `.env` containing valid `OPENAI_API_KEY` and `DATABASE_URL`;
- for optional external sources: `SEC_USER_AGENT`, `ALPACA_API_KEY_ID`,
  `ALPACA_API_SECRET_KEY`, `FRED_API_KEY`, and `EXA_API_KEY`;
- PostgreSQL containing the canonical mental-model data;
- permission to run `systemctl start postgresql` when the configured database
  is unavailable.

## Validation

From the repository root, run the committee contracts:

```bash
PYTHONPATH=code python -m unittest crew.test_crew -v
```

Render the frontend without starting a paid committee run:

```bash
cd code/frontend
python - <<'PY'
from streamlit.testing.v1 import AppTest

app = AppTest.from_file("app.py")
app.run()
assert not app.exception
print("Frontend render check passed")
PY
```
