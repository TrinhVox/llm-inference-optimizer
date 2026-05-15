# LLM Inference Optimizer

Benchmarking and optimization toolkit for LLM inference — covering quantization (GPTQ, AWQ, BitsAndBytes), batching strategies, and KV-cache tuning.

## Requirements

**Python 3.10+** is required. The `.venv` in this repo uses Python 3.8 (EOL); create a new one:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

To also install quantization backends:

```bash
pip install -e ".[gptq]"   # GPTQ via auto-gptq
pip install -e ".[awq]"    # AWQ  via autoawq
```

## Project layout

```
src/llm_optimizer/   # main package
benchmarks/          # benchmark scripts
notebooks/           # Jupyter notebooks for exploration
results/             # output CSVs, plots (gitignored by default)
```

## Quick start

```python
from llm_optimizer import __version__
print(__version__)
```

## Notebooks

```bash
jupyter lab
```
