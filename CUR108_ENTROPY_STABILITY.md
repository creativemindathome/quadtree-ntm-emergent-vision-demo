# Entropy-coded topology stabilization

This change keeps the recursive RGB innovation-bit objective as the only source
of structural supervision. It does not add target masks, oracle split labels, a
node-budget penalty, or a permanently forced depth.

## Mechanism

Three controls make early failure visible and less abrupt:

1. RGB bit logits used by the coding loss pass through a smooth `tanh` bound.
   The model parameters and diagnostic decoder remain unclamped.
2. The local bits-per-pixel softmin temperature can anneal linearly from
   `--structure-temperature-bpp` to
   `--structure-temperature-final-bpp`.
3. Every horizon logs reachable posterior entropy, saturation, split
   probability, evidence-gap quantiles, flag costs, and predictive-logit scale.

The operative posterior is still determined by compression evidence:

```text
evidence_gap_bpp = stop_cost_bpp - split_cost_bpp
posterior_split = sigmoid(evidence_gap_bpp / temperature_bpp)
```

Positive evidence gaps favor finer structure. Temperature controls how sharply
the posterior commits; it does not select a preferred depth.

## Deterministic smoke evidence

Two 12-update CPU runs used seed 108, the same causal-pinch-three-step-wild
environment, learned frontiers, depth-1 initialization, one depth-8 exploration
path, and an 85-node emergency execution ceiling.

| Configuration | Loss first → last | Mean expected nodes | Posterior entropy | Saturated posterior mass |
|---|---:|---:|---:|---:|
| Fixed `T=0.02`, no logit bound | 2.913 → 2.770 | 16.57 | 0.809 bits | 17.45% |
| `T: 0.05 → 0.02`, smooth bound 12 | 2.838 → 2.706 | 17.96 | 0.820 bits | 16.71% |

This smoke test establishes numerical health and confirms that the new
instrumentation detects partial posterior saturation. It is too short to claim
better convergence or generalization. A multi-seed 3k-update ablation remains
required before promoting the schedule as a training result.

## Recommended experiment

Compare at least three seeds under an equal update and candidate-evaluation
budget:

```text
A  fixed T=0.02, soft clip disabled
B  T=0.05→0.02, soft clip 12
C  T=0.05→0.01, soft clip 12
```

Reject a configuration if expected nodes collapse to the root/floor or full
support, posterior saturation exceeds 90% for a sustained window, gradients
become non-finite, or held-out RGB bits improve only by allocating more nodes.
