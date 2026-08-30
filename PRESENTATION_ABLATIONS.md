# Presentation ablations

These are compact, claim-bound experiments for a five-minute presentation.
They use held-out causal-pinch families and the trained 10,000-update model.

## 1. Episode state matters; selective slot attention is unresolved

| Condition | Loss | H1 IoU | H4 IoU | H8 IoU |
|---|---:|---:|---:|---:|
| Learned episode read | 0.977 | 0.0285 | 0.0224 | 0.0158 |
| No episode read | 1.004 | 0 | 0 | 0 |
| Uniform slot read | **0.972** | 0.0257 | 0.0212 | 0.0158 |
| Zero episode state before prediction | 1.021 | 0.0128 | 0.0099 | 0.0087 |

**Claim:** persistent episode state carries predictive information.

**Boundary:** learned selective addressing is not convincingly better than a
uniform slot average. This model may use its slots as gated global context
rather than clean object-indexed memory.

## 2. The model uses state, but barely uses temporal order

| Condition | Loss | H1 IoU | H1 mass ratio |
|---|---:|---:|---:|
| Full history | 0.972 | 0.0410 | 39.3× |
| Reversed history | 0.971 | 0.0406 | 39.9× |
| Shuffled history | 0.971 | 0.0416 | 39.8× |
| Reset episode state | 0.999 | 0.0192 | 90.8× |

**Claim:** erasing recurrent episode state damages prediction, but reversing or
shuffling observations barely changes the result.

**Boundary:** this is recurrent storage without strong evidence of learned
ordered dynamics. It should not be presented as robust object tracking.

## 3. More sensory entropy is not automatically more useful

| Input | Loss | H1 IoU | Target depth lift | Expected nodes |
|---|---:|---:|---:|---:|
| Full RGB | 0.972 | 0.0410 | +0.021 | 23.2 |
| Grayscale | **0.936** | **0.0624** | **+0.254** | 25.2 |

Removing color improves loss by 0.035 and H1 IoU by roughly 52%. The model also
places substantially more relative depth near the evaluator-defined object.

**Claim:** extra sensory variation can consume representational capacity without
improving the behaviorally relevant prediction.

**Boundary:** this does not prove that color is generally irrelevant. It shows
that color heterogeneity is not useful for this trained model on this synthetic
task distribution.

## Optional Nilsson bridge

The sensory-rung probe gives another useful result: 12×12 luminance outperforms
full 192×192 luminance on this task (`0.831` versus `0.903` loss). The lower
resolution also achieves higher H1 IoU (`0.116` versus `0.064`). This is a
synthetic engineering proxy—not a biological measurement—but it cleanly
illustrates that a richer eye is not automatically a better contract for a
particular behavior.

## Thirty-second narration

> The ablations reveal the actual boundary of the result. Remove episode state
> and prediction degrades sharply, so recurrence is carrying information. But
> shuffle or reverse the history and almost nothing changes: the model stores
> context without yet learning strong ordered dynamics. More surprisingly,
> grayscale beats RGB. Removing sensory entropy improves both prediction and
> localization. That is the central point: uncertainty has a bit cost, but the
> task determines whether the bit is worth preserving.

