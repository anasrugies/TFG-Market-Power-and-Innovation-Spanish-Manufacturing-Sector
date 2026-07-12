# Market Power and Innovation in Spanish Manufacturing Firms

Replication code for my bachelor's thesis: *"Market Power and Innovation in
Spanish Manufacturing Firms: Does Distance to the Technological Frontier
Matter?"* — UC3M, Double Degree in International Studies and Economics.
Supervisor: Adelheid Holl.

## ⚠️ Data license

`Export_XX.xlsx` are proprietary ORBIS (Bureau van Dijk) extracts, licensed
to UC3M for academic use only. Do not redistribute. Contact me if you need
access without your own ORBIS license.

## What's here

- `0630_RegressionAnalysis_Final.py` — full pipeline: cleans the ORBIS
  panel, constructs TFP (Levinsohn–Petrin, 2003), market power (PCM/OPM),
  and frontier distance, then runs all six regression specifications
  (baseline, main, and four robustness checks).
- `Export_01–06.xlsx` — raw ORBIS exports.

## Sample

Spanish manufacturing firms (NACE Rev.2, Section C), 2015–2023, ≥10
employees, ≥3 consecutive years. Final panel: 15,711 firms, 107,228
firm-year observations.

## Method

Panel OLS (within estimator), firm + industry×year fixed effects,
firm-clustered SEs, all regressors lagged one period.

## Requirements

`pip install pandas numpy scipy linearmodels python-calamine`

## Citation

Srugies García, A. (2026). *Market Power and Innovation in Spanish
Manufacturing Firms: Does Distance to the Technological Frontier Matter?*
Bachelor Thesis, UC3M.
