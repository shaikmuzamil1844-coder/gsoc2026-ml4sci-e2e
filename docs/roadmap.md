# GSoC-Oriented Roadmap

## Current Baseline

This repository establishes two dense end-to-end baselines:

- electron vs photon calorimeter-image classification
- quark vs gluon jet-image classification

Both are relevant to the ML4SCI end-to-end project direction because they make detector-image modeling concrete and measurable before introducing sparse methods.

## Why Sparse Methods Matter

The quark/gluon detector images are overwhelmingly sparse. In practice that means:

- dense convolutions spend most FLOPs on zeros
- memory bandwidth is wasted on inactive pixels
- sparse convolution methods can target the active support directly

## Proposed Next Steps

1. Rebuild the dense baseline as reusable Python modules rather than notebook-only code.
2. Profile the dense pipeline to identify the true compute hotspots.
3. Introduce sparse representations and sparse convolution layers.
4. Compare dense vs sparse models on:
   - AUC
   - throughput
   - memory footprint
   - FLOPs or approximate compute cost
5. Study where sparsity helps most:
   - early layers vs deeper layers
   - hybrid dense/sparse backbones
   - different sparsity thresholds or event representations

## What Would Strengthen The Proposal Further

- one clean benchmark run on the real datasets with saved metrics
- training-time and inference-time profiling tables
- side-by-side dense vs sparse experiment plan
- ablations on channel usage, preprocessing, and model depth

## Mentor-Facing Positioning

This repo should be presented as:

- a credible dense baseline
- a reproducible engineering starting point
- a bridge toward sparse-ML research rather than the final sparse solution itself
