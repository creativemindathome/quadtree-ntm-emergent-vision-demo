# Recurrence thought experiment

A target moves from the left side of the canvas to the right and then becomes
occluded. What should each memory substrate retain?

## Exact spatial memory

Before occlusion, addressed rows should preserve observations tied to physical
regions: the target's local appearance, boundary evidence, recent occupied
locations, and interactions with the occluder. Old rows should retain an
explicit temporal age rather than pretending that stale evidence is current.

Spatial memory answers: **what was observed here, and when?**

## Episode memory

Episode memory should carry facts that remain attached to the entity while its
coordinates change: which object is the target, its motion trend, material or
causal state, and the hypothesis that the left-side and right-side sightings
belong to one continuing object.

Episode memory answers: **which entity or rule connects these locations?**

## Why one global RNN state struggles

A classical recurrent update

```text
h_t = F(frame_t, h_{t-1})
```

forces exact local evidence, object identity, motion, occlusion state, and
background changes to compete inside one vector. It must learn both an implicit
coordinate system and a protection mechanism that prevents new pixels from
overwriting old identity evidence.

The quadtree model separates these interference domains:

```text
addressed spatial rows  -> where and when
episode slots           -> which object, event, or rule
```

The distinction is useful when the environment is local, persistent, sparse,
and multiscale. It is an inductive bias, not a claim of universal superiority.

