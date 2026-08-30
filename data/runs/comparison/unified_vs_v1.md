# Unified vs V1 benchmark

All values are test-set metrics. Delta = Unified − V1. Positive F1 deltas are improvements.

## Full taxonomy

| Split | Model | V1 classes | Unified classes | V1 macro F1 | Unified macro F1 | Δ macro | V1 micro F1 | Unified micro F1 | Δ micro |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| regular | linear | 150 | 145 | 0.4228 | 0.4010 | -0.0218 | 0.5178 | 0.4833 | -0.0344 |
| regular | mlp | 150 | 145 | 0.4784 | 0.4274 | -0.0510 | 0.5412 | 0.5086 | -0.0325 |
| regular | hierarchical_mlp | 150 | 145 | 0.4821 | 0.4255 | -0.0566 | 0.5559 | 0.5018 | -0.0541 |
| artist | linear | 150 | 145 | 0.3212 | 0.3549 | 0.0337 | 0.4511 | 0.4899 | 0.0387 |
| artist | mlp | 150 | 145 | 0.3646 | 0.3741 | 0.0094 | 0.5170 | 0.5063 | -0.0107 |
| artist | hierarchical_mlp | 150 | 145 | 0.3771 | 0.3774 | 0.0002 | 0.4963 | 0.5175 | 0.0212 |

## Parent-only

| Split | Model | V1 macro F1 | Unified macro F1 | Δ macro | V1 micro F1 | Unified micro F1 | Δ micro |
|---|---|---:|---:|---:|---:|---:|---:|
| regular | mlp | 0.6389 | 0.5743 | -0.0645 | 0.6916 | 0.6419 | -0.0497 |
| artist | mlp | 0.5877 | 0.5790 | -0.0088 | 0.6508 | 0.6623 | 0.0115 |

## Notes

- Macro F1 uses supported classes when the evaluator exposes that distinction.
- Raw inference is the default because it matches the historical V1 headline metrics.
- The unified and V1 full-taxonomy runs may have different active-class counts. Therefore this table measures each model on its own selected taxonomy/test split; it is not a strict same-class benchmark.
- The frozen V1 artist test should be used separately for a strict historical benchmark after training a leakage-quarantined unified model.
