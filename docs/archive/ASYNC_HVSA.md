# Async HVSA

Rule-driven semi-async HVSA is the current recommended second-stage implementation path.

## What It Does

- Keeps the original hierarchical FL structure: client -> edge -> cloud.
- Preserves dual-TA masking and edge-side verifiability.
- Moves from round-based synchronous aggregation to window-based semi-async aggregation.
- Uses deterministic rules for:
  - edge flush timing
  - stale batch rejection
  - risk-aware batch acceptance

LLM advice is optional and is no longer required for the async baseline.

## Recommended Entry Point

```bash
python exp_async_hvsa.py
```

Useful options:

```bash
python exp_async_hvsa.py --dataset MNIST --target-batch-commits 12
python exp_async_hvsa.py --dataset MNIST --noniid --alpha 0.5
python exp_async_hvsa.py --dataset CIFAR10 --min-batch-size 3 --flush-timeout 4
python exp_async_hvsa.py --disable-verification
```

## Design Notes

- `agent_async_secure.py` still contains optional LLM hooks, but the async baseline is now rule-first.
- Flush decisions now always go through a deterministic fallback policy, and `RuleEngine` can constrain them even when no LLM is present.
- If later experiments show that adaptive learned orchestration is useful, it can be added on top of this baseline rather than replacing it.
