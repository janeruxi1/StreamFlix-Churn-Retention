# Streamlit decision-support app

Interactive tool for the Retention team. Adjust intervention costs, uplifts, and budget to see how the recommended policy changes.

## Run locally

```bash
pip install -r requirements.txt
python src/data/simulate.py                  # generate the dataset
python notebooks/04_modeling.py              # train the model
streamlit run app/streamlit_app.py
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

- Streamlit will need `data/subscribers.csv` and `models/churn_model_v1.pkl` present at deploy time. Either commit small snapshots, or add a startup script that regenerates them.
