# Chapter 6 — Experiments and Evaluation

## 6.1 Overview

This chapter describes the procedure for evaluating the proposed method, including the experimental setup, hyperparameters, results, and a discussion of their implications. The central question is: can early observations of P-waves provide sufficient information for magnitude estimation?

## 6.2 Experimental Environment

### 6.2.1 Software Framework

All experiments are conducted in Python. PyTorch provides the core deep learning environment. Other libraries include NumPy for numerical operations, Pandas for data handling, Matplotlib for visualization, and Scikit-learn for evaluation metrics.

### 6.2.2 Hardware Configuration

Deep learning models perform a substantial amount of matrix multiplication, which modern GPUs handle well via CUDA kernels. The following configuration was used for all experiments:

| Component | Specification |
|-----------|---------------|
| CPU | Intel Core i5-13400F |
| GPU | NVIDIA GeForce RTX 3070 |
| RAM | 16 GB |

*Table 6.1. Hardware configuration used for all experiments.*

This configuration allows a meaningfully large batch size during training, which positively affects training stability and model convergence.

## 6.3 Training Configuration

### 6.3.1 Hyperparameter Tuning

A trained model is a function of its input $x$ with parameters $\theta$. Training additionally requires a set of **hyperparameters**: parameters that are not learned via gradient descent but set manually.

Two main approaches exist for finding them: **Grid Search**, which exhaustively tests all combinations in a predefined grid, and **Random Search**, which samples values from allowed ranges. Bergstra and Bengio demonstrated that random search performs better than grid search in practice [26]. Random search is therefore used here. The search space is implemented in `_sample_hparams` in `src/training_on_stratified_sample.py`:

| Hyperparameter | Distribution |
|----------------|--------------|
| Learning rate $\eta$ | log-uniform, $10^{-3.3}$ to $10^{-2.3}$ |
| Weight decay $\lambda$ | log-uniform, $10^{-3}$ to $10^{-2}$ |
| Batch size | uniform over $\{128, 256\}$ |
| Huber $\delta$ | uniform, 0.15 to 0.6 |
| Cosine minimum LR $\eta_{\min}$ | log-uniform, $10^{-7}$ to $10^{-5}$ |

Sampling from a log-uniform distribution is appropriate because both the learning rate and the weight decay act on a logarithmic scale.

Subsequent analysis revealed that all of the best-performing experiments had $\delta \approx 0.3$, so this value was fixed for the remaining experiments. Early stopping is also used: training is halted if the validation MAE does not improve for 150 consecutive epochs.

### 6.3.2 Optimization Algorithm

Model parameters are optimized with AdamW [25]. The update equations are:

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$

$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \qquad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

$$\theta_t = \theta_{t-1} - \eta \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_{t-1} \right)$$

where $\eta$ is the learning rate and $\lambda$ is the weight decay coefficient, both sampled at the start of each trial as described above. $m_t$ is an exponential moving average of the gradients and $\hat{m}_t$ its bias-corrected version; $v_t$ is an exponential moving average of the squared gradients and $\hat{v}_t$ its bias-corrected version. The remaining hyperparameters are fixed:

| Hyperparameter | Value |
|----------------|-------|
| $\beta_1$ | 0.9 |
| $\beta_2$ | 0.999 |

*Table 6.2. Fixed hyperparameters of the AdamW optimizer.*

### 6.3.3 Learning Rate Scheduling

Exponential and cosine schedulers were considered initially, but early experiments showed the clear superiority of the cosine schedule, which was used throughout. The minimum learning rate $\eta_{\min}$ of the cosine schedule is itself a hyperparameter, searched in $[10^{-7}, 10^{-5}]$.

## 6.4 Experimental Design

### 6.4.1 Experiment 1: Best Hyperparameter Configuration

To find the best hyperparameters, 100 random-search trials were run on the 10-second window, giving the model as much information as possible. Each trial was given 500 epochs, with 150 epochs without improvement as the early-stopping threshold. The following configuration performed best and is used in all remaining experiments:

| Parameter | Value |
|-----------|-------|
| Learning rate | 0.001368 |
| Weight decay | 0.005068 |
| Batch size | 128 |
| Cosine $\eta_{\min}$ | 0.0000086534 |

*Table 6.3. Best hyperparameter configuration found by random search.*

### 6.4.2 Experiment 2: Dependence on Window Length

The model is trained on different P-wave observation windows:

$$W \in \{3, 5, 8, 10\} \text{ seconds}$$

The objective is to determine how the length of the early waveform window affects magnitude estimation. Longer windows are presumed to carry more information but reduce the available warning time. All results are evaluated on a held-out test set.

| Window Length | Bias | MAE | Spearman $r$ | $R^2$ |
|---------------|------|-----|--------------|-------|
| 3 s | −0.040 | 0.3699 | 0.7755 | 0.6483 |
| 5 s | −0.071 | 0.3453 | 0.8028 | 0.6851 |
| 8 s | −0.027 | 0.3217 | 0.8290 | 0.7225 |
| 10 s | −0.032 | 0.3152 | 0.8392 | 0.7478 |

*Table 6.4. Performance of the proposed model as a function of the P-wave observation window length, evaluated on the test set.*

Increasing the window length improves every metric, though the difference is not large. This is expected because longer windows contain more information about the rupture. However, the improvement comes at the cost of increased warning latency, so a trade-off exists between prediction accuracy and response time, which is critical for any EEW system.

## 6.5 Discussion

Several observations can be drawn from the results above.

1. **Diminishing returns.** The gain in MAE from extending the window from 5 to 10 seconds is approximately 0.03 magnitude units, and the gain from 3 to 5 seconds is roughly 0.025. Most of the magnitude-relevant information accessible to a single-station model is already present in the first few seconds of the P-wave, so operationally useful estimates can be obtained from very short windows at the cost of a modest accuracy penalty.
2. **Unexplained variance.** The coefficient of determination plateaus at about 0.75 for the longest window, so roughly 25% of the variance in magnitude remains unexplained by the single-station model.
3. **Small negative bias.** The bias is small in absolute value (between −0.07 and −0.03). The tendency to slightly underestimate is likely caused by the prevalence of low-to-moderate magnitude events in the training data, which motivates either oversampling of high-magnitude events or synthetic data generation.
4. **Ranking quality.** Spearman's rank correlation reaches 0.84 for the 10-second window. Because this metric is sensitive to the ordering of predictions rather than their absolute values, it indicates that the model preserves the relative ranking of events well even when individual point predictions are imperfect. For an alerting system that operates on a threshold rather than on point estimates, ranking quality can matter more than absolute MAE.

## 6.6 Visualization of Results

The predicted-versus-true scatter plot (Figure 6.1, left) gives an intuitive view of prediction quality on the test set; the red dashed line marks perfect prediction. The residual histogram (Figure 6.1, right) shows that the model neither significantly overestimates nor underestimates magnitude, with a mean residual (predicted − ground truth) of −0.032. Both plots use the test set of 10-second waveforms and the best configuration from Table 6.3.

Figure 6.2 shows the training and validation loss alongside the MAE over the course of training. Both curves decrease smoothly and stabilize at similar levels, indicating stable convergence without significant overfitting.

The figures are in the final thesis PDF (`docs/final/Papakov_Aidarbek_Senior_Thesis.pdf`); the underlying run outputs are under `training_results/`.

## 6.7 Summary

This chapter described the experimental setting used to evaluate the proposed CNN–LSTM–MLP model for earthquake magnitude estimation, presented quantitative results across multiple observation windows, and discussed the implications of these results for earthquake early warning systems.
