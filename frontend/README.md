# Investment Committee frontend

This Streamlit app is the product interface for the structured value-investing
committee in `code/crew`. It runs the real crew workflow rather than a mock.

## What it provides

- investment-question and optional research input;
- investor selection and compact retrieval controls;
- live progress across research, retrieval, both investor rounds, and CIO
  synthesis;
- checkpoint-aware resume after a failed reasoning stage;
- head-icon investor buttons showing each independent thesis, its retrieved and
  applied mental models, reasoning, and cited evidence;
- peer agreements, disagreements, and view changes directly beneath the
  investor round;
- a structured CIO decision with confidence, horizon, conditions, and risks
  beneath peer review, making the reasoning sequence explicit;
- complete supporting evidence and source views;
- automatic display of the latest valid completed result;
- a Markdown report download, plus validated JSON for audit or further
  processing;
- a Stop control that terminates the active local crew process.

The UI treats the output as decision support, not personalised financial
advice. The underlying MVP structures only the question and research supplied
by the user; it does not perform live web research.

## Run

Use the same Python 3.13 environment as the crew. From the repository root:

```bash
source .venvinv/bin/activate
cd frontend
python -m streamlit run app.py
```

Open the local URL printed by Streamlit, normally
`http://localhost:8501`.

Running from `frontend/` lets Streamlit load the included theme from
`.streamlit/config.toml`. The service automatically sets `PYTHONPATH` for the
crew subprocess and writes completed results to:

```text
data/processed/crew/committee_result.json
```

Validated partial progress is stored separately in:

```text
data/processed/crew/committee_result.checkpoint.json
```

When a matching checkpoint exists, **Resume the matching checkpoint** becomes
available under **Committee settings**. Resume is accepted only when the
question, supplied research, investors, retrieval settings, round setting, and
models match the saved run.

## Runtime requirements

The root `requirements.txt` already includes Streamlit. The crew also requires:

- a valid `.env` containing `OPENAI_API_KEY` and `DATABASE_URL`;
- PostgreSQL with the canonical mental-model data;
- permission to run `systemctl start postgresql` if the configured database is
  unavailable.

## Validation

Run the crew contract tests:

```bash
PYTHONPATH=code python -m unittest crew.test_crew -v
```

Check that the Streamlit app renders without starting a paid committee run:

```bash
python - <<'PY'
from streamlit.testing.v1 import AppTest

app = AppTest.from_file("app.py")
app.run()
assert not app.exception
print("Frontend render check passed")
PY
```
