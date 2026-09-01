# Uncapped self-stabilizing frontier experiment

CUR-109 tests whether compression-posterior distillation can learn its own
conditional execution frontier without a candidate-node ceiling.

The pure condition uses exact STOP/SPLIT marginalization, no structural
temperature, no RGB-logit bound, no forced prediction depth, and no candidate
cap. One complete-sibling exploration path keeps depth eight discoverable.
The stabilized condition changes only temperature and the loss-only logit
bound; the no-distillation condition removes the posterior-to-proposal bridge.

## Profile-guided implementation

At 48,517 candidates, the original dynamic implementation spent 11.57 seconds
in recursive compression and 6.16 seconds in backward. The optimized path:

- batches the recursive dynamic program by depth;
- resolves child rows with sorted heap-address search;
- detaches posterior diagnostics from the loss graph;
- skips the per-node RGB reconstruction raster during training;
- retains full reconstruction during explicit evaluation.

At the deterministic seed-108 update-25 state with 13,765 candidates:

| Implementation | Compression | Backward | Full update |
|---|---:|---:|---:|
| Dynamic rows + training raster | 1.380 s | 0.995 s | 2.929 s |
| Vectorized depth recursion | 0.144 s | 0.097 s | 0.681 s |

CPU is used for the long run. On this dynamic workload, the five-update MPS
smoke averaged 1.760 seconds per update versus 0.055 seconds on CPU because
frontier construction and many small indexed operations dominate.

## Run

```bash
python experiment_self_stabilizing_frontier.py \
  --updates 3000 \
  --seeds 108 \
  --variants stabilized \
  --device cpu \
  --checkpoint-every 250 \
  --output-dir runs/cur109-stabilized-3k
```

The long run skips the expensive seven-condition final evaluation and saves
checkpoints every 250 updates. Evaluation is performed separately from a
selected checkpoint so throughput and evaluation cost remain distinguishable.
