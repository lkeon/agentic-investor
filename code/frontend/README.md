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
python -m streamlit run code/frontend/app.py
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

## Streamlit Community Cloud deployment

Deploy this branch with:

```text
Entrypoint: code/frontend/app.py
Python:     3.13
```

Community Cloud runs from the repository root, so the deployment theme is in
the root `.streamlit/config.toml`. It finds the deployment-specific, reduced
dependency set beside the entrypoint in `code/frontend/requirements.txt`.

In **Advanced settings → Secrets**, paste the contents of the root
`.streamlit/secrets.toml.example` and replace the placeholders. Keep all
entries at the TOML root level so Streamlit exposes them as environment
variables to both the frontend process and its committee subprocess.

Required sensitive values:

- `DATABASE_URL` — a remotely reachable PostgreSQL/pgvector connection string
  containing the populated canonical mental-model database. Use the SQLAlchemy
  `postgresql+psycopg://` scheme and the managed provider's required TLS
  parameters, commonly `?sslmode=require`.
- `OPENAI_API_KEY` — required for the OpenAI embedding provider used during
  mental-model retrieval even when no reasoning stage uses OpenAI.
- `OPENROUTER_API_KEY` — required for the currently configured DeepSeek
  reasoning model.

The non-secret deployment settings route every reasoning stage through:

```toml
DEFAULT_REASONING_MODEL = "openrouter/deepseek/deepseek-v4-flash"
```

Individual stage variables remain optional. Add `QUESTION_MODEL`,
`RESEARCH_MODEL`, `BRIDGE_MODEL`, `INVESTOR_MODEL`, or `CIO_MODEL` only when a
stage should override the default. A value such as
`openai/gpt-5-nano-2025-08-07` routes that stage directly to OpenAI; a value
beginning `openrouter/` routes it through OpenRouter.

`DILIGENCE_DEPLOYMENT = "streamlit_cloud"` enables hosted-session safeguards:
completed artifacts use per-run temporary files and the app does not load a
result created by a different browser session.

The deployment cannot run `systemctl` or connect to PostgreSQL on your local
computer. Populate the managed database before deployment; the application
will report an actionable connection error if that external service is
unavailable.

## Validation

From the repository root, run the committee contracts:

```bash
PYTHONPATH=code python -m unittest crew.test_crew -v
```

Render the frontend without starting a paid committee run:

```bash
python - <<'PY'
from streamlit.testing.v1 import AppTest

app = AppTest.from_file("code/frontend/app.py")
app.run()
assert not app.exception
print("Frontend render check passed")
PY
```
