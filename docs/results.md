# Results Provenance

This repository now distinguishes between two kinds of runs:

## 1. Benchmark-Oriented Runs

These are the runs that matter for scientific comparison and for the GSoC proposal itself.

- use the real ML4SCI / Kaggle / CERNBox datasets
- should ideally run on a GPU runtime
- are the only runs that should be quoted as model-performance evidence in a proposal or mentor discussion

Current benchmark references captured from the original notebook work:

- Task 1 reference test AUC: `0.7178`
- Task 2 paper baseline: `0.8076` AUC and `1/FPR @ TPR=70% = 4.47`

## 2. Fallback Validation Runs

These are developer-experience and reproducibility runs.

- use synthetic fallback data when external datasets are unavailable
- shrink training settings on CPU so `Run All` completes locally
- validate notebook execution, code paths, plots, loaders, and model wiring
- are not intended to be used as final scientific claims

## Recommended Reporting Practice

When presenting this project to GSoC mentors:

- quote benchmark results only from real-data runs
- mention fallback runs as a reproducibility and onboarding feature
- frame the fallback path as engineering support, not as evidence of model quality

That separation makes the repository look more rigorous and more trustworthy.
