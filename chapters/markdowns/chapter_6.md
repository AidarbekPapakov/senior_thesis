# Chapter 6 — Experiments and Evaluation Methodology

## 6.1 Overview

In this chapter, we describe the procedure proposed method evaluation. This includes: the experiments, hyperparameters, metrics and results.

We mostly focus on the following question:

Can early observations of P-waves provide sufficient information for magnitude estimation? To address it, a series of experiments are done using different training configurations, dataset combinations, and model variants.

Because the experimental evaluation has not yet been performed at the time of writing, this chapter describes the experimental design and the metrics that will be used to evaluate model performance. Numerical results will be incorporated once the experiments are completed.

## 6.2 Experimental Environment

### 6.2.1 Software Framework

All experiments are conducted using Python programming language along the PyTorch deep learning library, which provides all the essential tools for training deep learning models.

Other libraries used in the experiments include NumPy for numerical operations, Pandas for data handling, Matplotlib for visualization, and Scikit-learn for evaluation metrics and statistical analysis.

### 6.2.2 Hardware Configuration

Deep learning models are quite notorious for performing multiples computations, more specifically, addition and multiplication, which modern GPUs excet at due to the CUDA kernels. Thus, all the experiments were completed on GPU equipped machine.

The configuration is the following:

| Component | Specification |
|-----------|---------------|
| CPU | [Intel Core i5-13400F] |
| GPU | [GeForce RTX 3070] |
| RAM | [16 GB] |

This configuration allowed to have a significant size of batch during model training, which positevely affects the convergence of the model.

## 6.3 Training Configuration

### 6.3.1 Optimization Algorithm

By training a model we always imply the change of learnable parameters. These parameters are optimized using the AdamW optimizer, which was introduced in Chapter 3. Here is the formal definition of AdamW optimization algorithm:

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$

$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \qquad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

$$\theta_t = \theta_{t-1} - \eta \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_{t-1} \right)$$

where $\eta$ is the learning rate and  $\lambda$ is the weight decay coefficient

Initial training hyperparameters are defined as placeholders:

| Hyperparameter | Value |
|----------------|-------|
| Initial learning rate | [ ] |
| $\beta_1$ | 0.9 |
| $\beta_2$ | 0.999 |
| $\epsilon$ | 1e-8 |

The learning rate may be adjusted during training using a learning rate scheduler.

### 6.3.2 Learning Rate Scheduling

To improve convergence, a learning rate scheduling strategy will be employed. Possible scheduling methods include reduce-on-plateau scheduling, where the learning rate is reduced if the validation loss stops improving, and cosine annealing, where the learning rate gradually decreases following a cosine curve.

Placeholder:

$$\text{Learning rate scheduler} = \text{[method to be selected]}$$

### 6.3.3 Batch Size and Epochs

The batch size controls the number of samples processed during each gradient update.

Placeholder values:

| Parameter | Value |
|-----------|-------|
| Batch size | [ ] |
| Number of epochs | [ ] |

Larger batch sizes can improve GPU utilization but may require more memory.

### 6.3.4 Early Stopping

To prevent overfitting, an early stopping criterion will be used. Training will terminate if the validation loss does not improve for a specified number of epochs.

Placeholder:

$$\text{Patience} = \text{[number of epochs]}$$

## 6.4 Experimental Design

A sequence of experiments will be conducted to evaluate different aspects of the proposed model.

### 6.4.1 Experiment 1 — Baseline Training on STEAD

The first experiment establishes a baseline by training the CNN–LSTM model exclusively on the STEAD dataset.

**Objective:** evaluate whether the model can learn meaningful magnitude predictions from global seismic data.

**Evaluation metrics:** RMSE, MAE, $R^2$, Pearson correlation coefficient.

Placeholder results:

| Metric | Value |
|--------|-------|
| RMSE | [ ] |
| MAE | [ ] |
| $R^2$ | [ ] |
| Pearson $r$ | [ ] |

*[Figure 6.1: Predicted vs true magnitude scatter plot for STEAD test set]*

### 6.4.2 Experiment 2 — Multi-Dataset Training

The second experiment investigates whether combining multiple datasets improves model generalization.

**Training data:** STEAD and INSTANCE.

**Hypothesis:** training on multiple datasets may allow the model to learn more general seismic signal representations.

Placeholder results:

| Metric | STEAD Only | STEAD + INSTANCE |
|--------|------------|------------------|
| RMSE | [ ] | [ ] |
| MAE | [ ] | [ ] |
| $R^2$ | [ ] | [ ] |

### 6.4.3 Experiment 3 — Regional Fine-Tuning

In this experiment, the model trained on global datasets is fine-tuned using the CAIAG regional dataset.

**Objective:** evaluate whether regional fine-tuning improves performance on Central Asian seismic data.

**Evaluation dataset:** CAIAG test set.

Placeholder results:

| Metric | Before Fine-Tuning | After Fine-Tuning |
|--------|-------------------|------------------|
| RMSE | [ ] | [ ] |
| MAE | [ ] | [ ] |

### 6.4.4 Experiment 4 — Architecture Ablation

To evaluate the contribution of the LSTM module, an ablation study will be performed comparing two model variants: a CNN-only architecture and a CNN–LSTM architecture.

**Hypothesis:** the LSTM component should improve performance by modeling temporal dependencies.

Placeholder results:

| Model | RMSE | MAE |
|-------|------|-----|
| CNN only | [ ] | [ ] |
| CNN + LSTM | [ ] | [ ] |

### 6.4.5 Experiment 5 — Window Length Sensitivity

The model will be trained using different P-wave observation windows:

$$W \in \{3, 5, 8\} \text{ seconds}$$

**Objective:** determine how the length of the early waveform window affects magnitude estimation accuracy.

**Hypothesis:** longer windows should provide more information but may reduce the warning time.

Placeholder results:

| Window Length | RMSE | MAE |
|---------------|------|-----|
| 3 s | [ ] | [ ] |
| 5 s | [ ] | [ ] |
| 8 s | [ ] | [ ] |

*[Figure 6.2: Model error vs window length]*

### 6.4.6 Experiment 6 — High-Magnitude Performance

Because large earthquakes are the most critical for early warning systems, model performance will be evaluated separately for events with magnitude $M \geq 5.0$.

This analysis will reveal whether the model can generalize to high-magnitude events despite dataset imbalance.

Placeholder results:

| Metric | Value |
|--------|-------|
| RMSE | [ ] |
| MAE | [ ] |

## 6.5 Statistical Robustness

Deep learning training results may vary depending on random initialization and dataset splits. To account for this variability, each experiment will be repeated multiple times.

Placeholder:

$$K = \text{[number of independent runs]}$$

Final metrics will be reported as:

$$\text{metric} = \text{mean} \pm \text{standard deviation}$$

This approach provides more reliable estimates of model performance.

## 6.6 Visualization of Results

Several visualization techniques will be used to interpret the model's predictions.

**Predicted vs True Magnitude** — scatter plots comparing predicted magnitudes to ground truth values provide an intuitive view of model performance.

*[Figure 6.3: Predicted vs actual earthquake magnitudes]*

**Error Distribution** — histograms of prediction errors will reveal whether the model systematically underestimates or overestimates earthquake magnitudes.

*[Figure 6.4: Distribution of magnitude prediction errors]*

**Learning Curves** — training and validation loss curves will be used to analyze model convergence.

*[Figure 6.5: Training and validation loss curves]*

## 6.7 Expected Outcomes

Based on previous research in seismic deep learning, several outcomes are anticipated.

First, it is expected that the CNN–LSTM architecture will outperform purely convolutional models by capturing temporal dependencies in seismic signals. Second, training on multiple datasets is expected to improve generalization performance across regions. Third, increasing the observation window length should improve prediction accuracy, though at the cost of reduced early warning time. Finally, the model may exhibit reduced accuracy for high-magnitude events due to the limited number of such events in the training dataset.

## 6.8 Summary

This chapter outlined the experimental framework used to evaluate the proposed CNN–LSTM model for earthquake magnitude estimation.

The experiments are designed to assess the impact of dataset selection, architecture design, and observation window length on model performance. Because the experiments have not yet been conducted, numerical results are presented as placeholders.

The next chapter will analyze the experimental results and discuss the implications of the findings for earthquake early warning systems.