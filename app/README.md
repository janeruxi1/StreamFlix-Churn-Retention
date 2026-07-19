# Streamlit decision-support app

Interactive tool for the Retention team. Adjust intervention costs, uplifts, and budget to see how the recommended policy changes.

**Live demo:** [https://janeruxi1-streamflix-churn-retention.streamlit.app/](https://janeruxi1-streamflix-churn-retention.streamlit.app/)

## Run locally

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

The app auto-generates `data/subscribers.csv` and trains `models/churn_model_v1.pkl` on first boot if they're missing — one-time setup takes ~30 seconds, cached thereafter. To pre-generate them (faster first boot):

```bash
python src/data/simulate.py                  # generate the dataset
python notebooks/04_modeling.py              # train the model
```

## What's inside

Two tabs:

- **📊 Policy overview** — KPI row (users targeted, total cost, net EV, ROI) plus head-to-head against the current blanket-m11 baseline. Lever mix shown below.
- **🔍 Per-user lookup** — pick a subscriber ID, see the model's risk score, top diagnostic features, and the recommended lever with expected value.

Sidebar controls:
- Monthly budget cap
- Cost + uplift per intervention (`curated_playlist`, `credit_5`, `premium_upgrade`)
- Blanket baseline settings (tenure month + cost per user)

## How the math stays honest

The app imports directly from `src/` — the same code path the analysis notebooks use. Changes to `src/decisions/policy.py` or `src/models/train.py` flow through both the notebooks and the demo without duplication.

## Deploy publicly (Streamlit Community Cloud — free)

1. Push the repo to GitHub (data and model files are gitignored; app auto-generates them on first boot).
2. Sign in at [share.streamlit.io](https://share.streamlit.io) with GitHub.
3. Click **New app** → select the `StreamFlix-Churn-Retention` repo → main branch → `app/streamlit_app.py`.
4. First boot takes ~30–45 seconds (data generation + model training). Every subsequent visit uses the cached artifacts.

No secrets or environment variables needed. The default deployment URL follows the pattern `https://<username>-<repo>.streamlit.app/`.
