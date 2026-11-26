# Ventilation Regime Segmentation During Anesthesia Recovery

This repository contains a PyTorch notebook that trains, evaluates, and compares three neural network architectures for automatic multi-class segmentation of ventilation regimes on a **synthetic** 128 Hz anesthesia waveform dataset (`CO2`, `AWP`, and `Cdyn` channels).

> **Note:** All experiments use synthetic data (`anesthesia_synth_dataset`) and are intended for research and prototyping rather than clinical decision-making.

## Models Compared

1. **1D U-Net (`UNet1D`)**: A 4-level 1D encoder-decoder baseline with skip connections operating on normalized 60-second sliding windows (`7,680` samples at 128 Hz) with Hann-weighted overlap averaging at inference.
2. **Breath-Tokenized BiGRU (`BreathBiGRU`)**: A 2-layer bidirectional GRU trained over breath and pause boundary tokens (18 extracted physiological morphology, timing, correlation, and rolling variability features).
3. **Breath-Tokenized Temporal Transformer (`BreathTransformer`)**: A 3-layer Transformer encoder with sinusoidal positional encoding attending globally across all breath tokens in a recording.
