# The missing test: same bits, different clocks

**Status: proposed test. No selective-renewal result has been measured.**

The current experiments establish learned spatial allocation, useful recurrent
state, and sensitivity to sensory content. They do not establish that the model
can schedule temporal renewal better than uniform frame acquisition.

## Controlled comparison

Hold fixed:

- scene and prediction task;
- quadtree geometry;
- total refresh/readout count;
- photon or acquisition budget;
- model capacity and evaluation families.

Compare three clocks:

| Clock | Schedule |
|---|---|
| **Uniform** | Every active leaf refreshes at equal cadence. Leaf ages remain approximately equal. |
| **Task-conditioned renewal** | Spend the same refresh count on leaves whose current model state predicts that new evidence could change H1/H4/H8 loss. Some leaves renew repeatedly while others age. |
| **Spatially shuffled null** | Preserve the learned schedule's refresh counts and cadence, but permute leaf identities. |

An optional stale condition performs no renewal after the initial observation.

The renewal policy may use model state and forecast utility. It must not use
evaluator target masks or future observations.

## Measurements

- held-out prediction loss;
- H1/H4/H8 IoU;
- refreshes per episode;
- leaf-age distribution by depth;
- prediction improvement per refresh;
- shuffled-null delta at exactly matched cost.

The decisive result is not merely that renewal beats staleness. At equal cost,
task-conditioned renewal must beat both uniform refresh and the spatially
shuffled null. The null distinguishes useful scheduling from simply performing
more refreshes.

## Presentation treatment

Use one two-column editorial card, not metric bars:

```text
SAME SCENE · SAME GEOMETRY · SAME REFRESH BUDGET

UNIFORM CLOCK                 RENEWAL CLOCK
all leaves refresh equally   selected leaves renew repeatedly
leaf ages stay equal         unselected evidence is allowed to age

SHUFFLED CLOCK / NULL
same counts and cadence; leaf identities permuted
```

Mark **PROPOSED TEST** in restrained red. Existing grayscale measurements may
motivate the question—target-depth lift `+0.254` versus `+0.021`, and expected
nodes `25.21` versus `23.17`—but those measurements concern spatial allocation,
not temporal renewal.

## Twenty-five-second narration

> The experiment we still need is a clock ablation. Hold the scene, leaf
> geometry, and readout budget fixed. Refresh every leaf uniformly; then spend
> the same budget renewing only leaves whose evidence could change the
> prediction. Finally shuffle those renewal assignments as a null. If selective
> renewal wins at equal cost—and the shuffled schedule loses—the camera has
> become a spatiotemporal sampler: not more pixels, but newer evidence where the
> task can change. Our current results motivate this test; they do not yet
> measure it.

