# Investment Committee Chainlit frontend

A minimal local frontend for testing the investment-committee user experience.
It runs in mock mode and requires no database, CrewAI connection, or LLM key.

## Run locally on Fedora/Linux

From this folder:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
chainlit run app.py -w
```

Open `http://localhost:8000` if the browser does not open automatically.

## What the mock demonstrates

- One-question input
- Mental-model retrieval step
- Independent Buffett, Marks, and Flatt assessments
- Committee debate
- Final synthesis with references
- A stacked layout that remains usable on mobile-width browsers

## Connect the existing CrewAI workflow

Keep `app.py` unchanged and replace the body of `run_committee()` in
`committee_service.py`.

Return a `CommitteeResult` containing:

```python
CommitteeResult(
    mental_models=["..."],
    perspectives=[
        InvestorPerspective(investor="Warren Buffett", assessment="..."),
    ],
    debate=[
        DebatePoint(speaker="Howard Marks", argument="..."),
    ],
    conclusion="...",
    sources=["..."],
)
```

If your existing `run_crew(question)` function is synchronous, call it without
blocking Chainlit's event loop:

```python
raw_result = await asyncio.to_thread(run_crew, question)
```

Then map `raw_result` into the `CommitteeResult` structure above.

## Suggested repository location

```text
code/
├── crew/
├── frontend/
│   ├── .chainlit/config.toml
│   ├── public/stylesheet.css
│   ├── app.py
│   ├── committee_service.py
│   ├── requirements.txt
│   └── README.md
└── vis/
```
