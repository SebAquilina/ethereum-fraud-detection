---
title: Ethereum Fraud Detection
emoji: 🛡️
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Ethereum Fraud Detection

A web app that scores Ethereum addresses for fraud risk. Give it an address and it
pulls the address's transaction history from Etherscan, builds a feature vector, and
runs it through a three-model ensemble (XGBoost + Random Forest + Isolation Forest).
There's also a live mode that scores addresses off new blocks as they're mined, and a
chat advisor (Google Gemini) that explains a result in plain language.

> **Live demo:** scan the QR code from the presentation, or visit the Hugging Face
> Space linked above.

## How the scoring works

Each address gets three independent scores:

- **XGBoost** and **Random Forest** are the supervised models, trained on a labelled
  fraud dataset. They carry most of the weight.
- **Isolation Forest** is unsupervised. It flags structurally unusual addresses and
  gets a small share of the final score.

The three are combined with a weighted average (the "fixed" mode). There's also an
"adaptive" mode that adjusts the Isolation Forest's weight per address - it leans on
the anomaly signal when the supervised models are uncertain, and backs off when they
aren't. You can switch modes from the header toggle at runtime.

The final number maps to a risk band: MINIMAL / LOW / MEDIUM / HIGH / CRITICAL.

## Running it locally

```bash
pip install -r requirements.txt
python src/app.py
```

Then open http://127.0.0.1:5001 and search an address.

`--port` and `--host` are available if you need them:

```bash
python src/app.py --port 8080 --host 0.0.0.0
```

## Configuration

The app reads a couple of optional environment variables:

| Variable | What it's for |
|---|---|
| `ETHERSCAN_API_KEY` | Your Etherscan key. Get one at https://etherscan.io/apis |
| `GEMINI_API_KEY` | Needed for the chat advisor. Get one at https://aistudio.google.com/ |
| `ETHEREUM_WS_URL` / `ETHEREUM_HTTP_URL` | An Ethereum node for the live block stream (e.g. Alchemy or Infura). Optional. |

## Deployment (Hugging Face Spaces)

This repo ships with a `Dockerfile` and the HF Space metadata at the top of this
README. To deploy:

1. Create a new Space at https://huggingface.co/new-space
2. Pick **Docker** as the SDK and link this GitHub repo (or push to the Space's git
   remote directly).
3. Under **Settings → Variables and secrets**, add `ETHERSCAN_API_KEY` and
   `GEMINI_API_KEY` as secrets. (Optional: `ETHEREUM_WS_URL` for the live block feed.)
4. The Space builds the Docker image and exposes the app on port 7860.

## Project layout

```
src/
  app.py              Flask app, the single-page UI, and the chat endpoint
  live_detector.py    loads the models and scores one address
  feature_engine.py   turns raw Etherscan data into the feature vector
  etherscan_client.py rate-limited Etherscan wrapper
  if_preprocessing.py Isolation-Forest-specific feature handling + calibration
  adaptive_ensemble.py the adaptive weighting rules
  stream_monitor.py   live block listener (optional)
models/baseline_models/
  the trained model files the app loads at startup
Dockerfile            build instructions for HF Spaces / any container host
```
