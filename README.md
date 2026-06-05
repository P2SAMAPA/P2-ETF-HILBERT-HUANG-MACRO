# Hilbert‑Huang Transform (HHT) with Macro‑Driven Mode Selection

Applies Empirical Mode Decomposition (EMD) to ETF returns to extract intrinsic mode functions (IMFs). A macro variable (e.g., VIX) determines which IMFs (high‑frequency vs low‑frequency) are selected. The score is the instantaneous amplitude of the selected IMFs at the last time point – a novel signal for regime‑aware momentum.

## Features
- Three ETF universes (FI/Commodities, Equity Sectors, Combined)
- Seven rolling windows (63–4536 days)
- EMD via `EMD-signal` library
- Hilbert transform for instantaneous amplitude
- Macro‑driven mode selection: high VIX → high‑frequency IMFs, low VIX → low‑frequency IMFs
- Score = weighted amplitude at last time point
- Two‑tab Streamlit dashboard (auto best, manual)
- Results stored on Hugging Face: `P2SAMAPA/p2-etf-hilbert-huang-macro-results`

## Usage

1. Set `HF_TOKEN` environment variable.
2. Install dependencies: `pip install -r requirements.txt` (note: `EMD-signal` may require C compiler; if issues, use `pip install EMD-signal`)
3. Run training: `python train.py` (EMD is O(n²) but fast for daily windows)
4. Launch dashboard: `streamlit run streamlit_app.py`

## Interpretation

- High score → the selected IMFs (based on macro) have large amplitude, suggesting the dominant oscillation (fast/slow) is active and may continue.

## Requirements

See `requirements.txt`.
