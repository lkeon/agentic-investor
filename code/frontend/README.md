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

The MVP does not perform live web research. It treats the output as decision
support rather than personalised financial advice. It has no fixed holding
period: the product supports continued ownership only while the core business
thesis remains valid. Macro is shown as a secondary condition and matters only
when it has a direct, material effect on that thesis.

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
