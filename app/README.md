# Streamlit decision-support app

Interactive tool for the Retention team. Adjust intervention costs, uplifts, and budget to see how the recommended policy changes.

**Live demo:** [https://janeruxi1-streamflix-churn-retention.streamlit.app/](https://janeruxi1-streamflix-churn-retention.streamlit.app/)

## Run locally

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

The app auto-generates `data/subscribers.csv` and trains `models/churn_model_v1.pkl` on first boot if they're missing — one-time setup takes about a minute, cached thereafter. To pre-generate them (faster first boot):

```bash
python src/data/simulate.py                  # generate the dataset
python notebooks/04_modeling.py              # train the model
```

## What's inside

Three tabs:
- **💰 Targeted vs blanket** — the monthly business case: net value and ROI of the targeted policy against the month-11 blanket credit, spend vs revenue kept, who gets which offer, and net value across budgets from $25k to $500k (budget slider on this tab)
- **👤 Look up a subscriber** — example subscribers (highest risk, one per offer, over budget, low risk) or any ID: churn risk, plan, lifetime value, the action and why (budget cut-off vs no offer pays), the value of each offer, and a SHAP chart of the reasons behind the score
- **⚙️ Assumptions** — cost and uplift of each offer and the blanket campaign's month and credit; every tab updates

## How the math stays honest

The app imports directly from `src/` — the same code path the analysis notebooks use, including the shadow-price budget allocation in `src/decisions/policy.py`. Changes there flow through both the notebooks and the demo without duplication. `requirements.txt` keeps scikit-learn below 1.9: version 1.9 changed how HistGradientBoosting fits, and a model retrained under it moves the headline by about $2k, so the cap keeps the live app on the reported numbers. `tests/test_app_smoke.py` checks the app shows +$174k / 1.87×.

## Deploy publicly (Streamlit Community Cloud — free)

1. Push the repo to GitHub (the data CSVs are committed; model files are gitignored and the app trains them on first boot).
2. Sign in at [share.streamlit.io](https://share.streamlit.io) with GitHub.
3. Click **New app** → select the `StreamFlix-Churn-Retention` repo → main branch → `app/streamlit_app.py`.
4. First boot takes ~30–45 seconds (data generation + model training). Every subsequent visit uses the cached artifacts.

No secrets or environment variables needed. The default deployment URL follows the pattern `https://<username>-<repo>.streamlit.app/`.
