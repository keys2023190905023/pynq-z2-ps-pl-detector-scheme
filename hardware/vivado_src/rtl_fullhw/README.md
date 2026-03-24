# Tiny Full-Hardware Network

This directory contains a realistic next step beyond the current `PL conv + PS schedule`
design: a small multi-layer network whose layer sequencing stays inside PL.

What it demonstrates:

- full-PL layer chaining
- on-chip ping-pong feature-map storage
- no PS involvement between layers
- synthesizable Verilog that reuses the existing 3x3 convolution engine
- a folded multi-channel detect head with real channel accumulation
- detector-oriented record output formatting (`DET1`)

What it is not:

- not a full YOLOv8 detector
- not multi-channel/multi-scale detection yet
- not a final deployable helmet detector

Current structure:

- `FeatureMapDualPortRam.v`
  - On-chip feature-map buffer used between layers.
- `DetectionHeadAxisPacketizer.v`
  - Prepends a detector-head header and serializes record-major detection payloads.
- `TensorAxisPacketizer.v`
  - Prepends a compact tensor header so the final PL head tensor can be consumed by PS or later hardware post-processing.
- `TinyFullHwMultiChannelHead.v`
  - Tiny multi-channel 3x3 head that folds input channels through a shared convolution operator and accumulates partial sums on chip.
- `TinyFullHwNetworkTop.v`
  - Three-layer all-hardware network top.
- `TinyFullHwDetectorTop.v`
  - Wraps the multi-channel head and emits record-major detector output bytes with metadata.
- `TinyFullHwMultiChannelFeatureStage.v`
  - Generic multi-channel feature-map stage that keeps channel folding, on-chip accumulation and output streaming inside PL.
- `TinyFullHwPlOnlyDemoDetectorTop.v`
  - Two-stage PL-only demo detector: RGB stem stage in PL followed by a PL detect head, with PS only responsible for frame I/O and display.

The present network uses center-only kernels so simulation is easy to verify:
after 3 valid convolutions, the output equals the input cropped by 3 pixels
on each border.

The detector top now emits one record per output cell:

1. `x`
2. `y`
3. `w`
4. `h`
5. `obj`
6. class logits / scores

This is meant to be the hardware foundation for a later true detector:

1. replace fixed kernels with ROM-loaded layer weights
2. scale from this folded multi-channel head to wider tiled channel groups
3. add hardware neck/head tensor emitters for larger heads
4. optionally add box decode/NMS blocks

The newest PL-only demo path is:

1. RGB frame enters PL as channel-major bytes
2. `TinyFullHwMultiChannelFeatureStage` builds a 4-channel stem feature map on chip
3. `TinyFullHwDetectorTop` consumes that PL-produced tensor directly
4. PS only starts the pipeline, supplies camera bytes and displays decoded results
