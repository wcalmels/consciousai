# Contributing to ConsciousAI

## What we need most

1. **Validation on real datasets** — UCR archive, MIMIC-III, financial crisis data
2. **Domain adapters** — new connectivity methods for specific fields
3. **GPU kernel** — CUDA implementation of batch Φ for N > 500
4. **Comparison vs PyPhi** — rigorous side-by-side on small systems

## Setup

```bash
git clone https://github.com/YOUR_USER/consciousai.git
cd consciousai
pip install -e ".[dev]"
pytest tests/ -q
```

## Guidelines

- Every PR must pass `pytest tests/ -q` (32 tests)
- New features need at least one unit test
- Benchmark claims need reproducible code in `tests/benchmarks/`
- Be honest about what the spectral Φ approximation is and isn't

## Scope

This library is for **engineering applications** (anomaly detection, monitoring,
system health). It is NOT a replacement for PyPhi for neuroscience research.
New contributions should respect this scope.
