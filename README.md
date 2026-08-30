# Active Quadtree Recurrent World Model — 10K Presentation Snapshot

This repository is a self-contained, reproducible snapshot of the completed
10,000-update causal-pinch checkpoint used in the accompanying presentation.
It packages the exact model weights, environment generator, quadtree memory,
visual renderer, and episode-memory ablation.

![Soft attention and hard quadtree](artifacts/soft-attention-quadtree.gif)

## What is being demonstrated

A classical RNN compresses a scene into one recurrent vector. This model keeps
sparse recurrent payloads at exact quadtree addresses and supplements them with
16 global episode slots:

```text
RGB regions + quadtree address + time
                  |
                  v
       addressed recurrent memory
                  |
        episode-slot attention
                  |
                  v
       H1 / H4 / H8 predictions
```

The checkpoint uses a depth-8 virtual address space and the historical
variance-budgeted execution support of at most 341 candidates. The soft
posterior can allocate fewer effective nodes than the hard candidate tree.
Later uncapped-executor experiments are intentionally excluded from this
presentation snapshot.

## Run the demo

Python 3.11–3.14 is supported.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
pytest -q
python render_demo.py \
  --checkpoint checkpoints/checkpoint-010000.pt \
  --output generated/quadtree-demo.gif \
  --seed 701 --scale 2 --highlight-target --show-depth
```

The resulting animation contains raw environment input, evaluator-only target
annotation, hard leaves, soft effective depth, and reconstruction.

## Reproduce the attention ablation

```bash
python probe_attention.py \
  --checkpoint checkpoints/checkpoint-010000.pt \
  --families 16 --seed 2603 \
  --output generated/attention-ablation.json
```

The recorded reference result is in
[`artifacts/attention_ablation_16.json`](artifacts/attention_ablation_16.json).
It supports two different claims:

- Episode memory carries useful predictive information.
- Selective slot addressing is only weakly better than a uniform slot read in
  this checkpoint.

## Recurrence distinction

The presentation thought experiment and its reconstructed answer are in
[`RECURRENCE_THOUGHT_EXPERIMENT.md`](RECURRENCE_THOUGHT_EXPERIMENT.md).

## Artifact provenance

- Training update: 10,000
- Parameters: 600,632
- Maximum quadtree depth: 8
- Episode slots: 16
- Prediction slots: 4
- Candidate execution support: 341 nodes
- Environment: `causal_pinch_three_step_diverse`
- Original run: `causal-pinch-v3-depth8-diverse-long-10k`

This is a research demonstration, not a claim that conditional-compute scaling
has been solved. In this checkpoint, candidate support remains externally
bounded; the learned soft allocation is the object of inspection.

