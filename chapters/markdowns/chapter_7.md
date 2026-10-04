# Chapter 7 — Limitations

## 7.1 Overview

This chapter discusses the limitations of the proposed approach from several perspectives, including dataset properties, methodological choices, and deployment considerations.

## 7.2 Data Imbalance

The most prominent limitation is the naturally inherited imbalance of the data. The dataset is not rich in high-magnitude examples, so the model has seen only a small fraction of large-magnitude events. We did not observe a significant divergence on the test set, but the discrepancy between predictions and true magnitudes may grow substantially for very large events ($M > 7.0$). Although such events are exceedingly rare, they are the ones that cause the most destruction.

One remedy is carefully composed synthetic data that imitates large-magnitude events. This requires deep expertise in both seismology and regional characteristics to minimize the risk of data corruption.

## 7.3 Single-Station Analysis

The proposed model operates on observations from a single station. Modern operational EEW systems typically aggregate observations from multiple stations to improve both accuracy and robustness. A single-station model is more vulnerable to local site effects and isolated noise events. Extending the architecture to a multi-station setting is a direction for future work.

## 7.4 Dependence on P-wave Picking

The model relies on precise P-wave detection to extract the input window. While modern systems detect P-waves efficiently, small picking errors are inevitable. In a real deployment, even small offsets in the assumed P-wave arrival shift the input window and add noise to the input, which increases uncertainty in the magnitude estimate.

A robust deployment would either pair the model with a high-quality phase picker (such as PhaseNet) or expose the model during training to imperfectly picked arrival times, so that it becomes more resilient to small timing errors.

## 7.5 Regional Variability

All training data come from the INSTANCE dataset, which is geographically concentrated in Italy. Seismic waveforms depend on local geological conditions, so a model trained on Italian seismicity is unlikely to perform well in a different region (for example, Kyrgyzstan) without fine-tuning or full retraining on local data.

## 7.6 Summary

This chapter discussed the main limitations of the proposed approach: dataset imbalance, single-station design, dependence on P-wave picking, and regional variability of seismic signals. The results indicate that early seismic signals contain enough information to make an approximate estimate of event magnitude, but several practical concerns must be addressed before such a system can be considered operationally mature.
