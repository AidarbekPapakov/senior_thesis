# Chapter 7 — Discussion

## 7.1 Overview

This chapter interprets the results obtained from the experiments described in Chapter 6 and evaluates the effectiveness of the proposed CNN–LSTM model for earthquake magnitude estimation.

The discussion focuses on three main aspects: model performance and predictive capability, impact of architectural and data-related choices, and limitations of the proposed approach.

Where numerical results are required, placeholders are used and should be replaced once the experiments are completed.

## 7.2 Model Performance Analysis

### 7.2.1 Overall Performance

The performance of the model is thus evaluated using the metrics defined in Chapter 3: Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), coefficient of determination ($R^2$), and Pearson correlation coefficient.

The best performing model achieved the following metrics in the test set:

| Metric | Value |
|--------|-------|
| RMSE | [value] |
| MAE | [value] |
| $R^2$ | [value] |
| Pearson $r$ | [value] |

From these results we can draw a conclusion that model has learnt to capturing the relationship between early waveform features and earthquake ultimate magnitude.

### 7.2.2 Error Characteristics

We also provide the distribution of residuals of predictions, which adds more insight into model's performance.

From the histogram (Figure [X]), it is seen that the model has zero bias (average residual close to zero) and a moderate deviation.

### 7.3.2 Effect of Window Length

The experiment on different observation window lengths shows the following trend:

| Window | RMSE |
|--------|------|
| 3 s | [value] |
| 5 s | [value] |
| 8 s | [value] |

In general, increasing the window length leads to [improved / marginally improved / unchanged] performance. This is expected because longer windows contain more information about the rupture process. However, this improvement comes at the cost of increased warning latency.

Thus, there exists a trade-off between prediction accuracy and response time, which is critical for practical EEW systems.

## 7.5 Limitations of the Approach

Despite promising results, there are, however, several limitations.

### 7.5.1 Data Imbalance

The most notable one is the natural data imbalance. As the dataset is not rich on high magnitude data, the model is trained primarily on small to medium events, which ultimately might lead to underestimation of extremely large earthquakes. Although those kind of events are so rare, they happen once a centry, they still cause the most destruction.

It can be assumed that if synthetic data is used to mimick large-magnitude events, this problem will be solved, but the process of careful synthetic data generation must involve the experts who are able to minimize the chance of making a mistake that might result in a significant dataset bias. 


### 7.5.2 Dependence on P-wave Picking

Our model is built on the premise of precise P-wave detection and while these days systems manage to catch P-waves quite efficiently, even small errors are still inevitable, which in real-world systems might produce additional uncertainty to the overall result.


## 7.6 Comparison with Existing Approaches

Compared to classic EEW methods, the proposed approach presents mapping from a limited amount of input data to a final magnitude, which eliminates the need to wait until the destructive waves arrive.

Compared to other deep learning approaches, which for the most part solve a different task (phase picking, event detection/classifictation, etc.) our approach outputs a more verbose conclusion on the input data, which in theory gives more room to act for EEW.

## 7.7 Summary

This chapter analyzed the results and discussed the strengths and limitations of the proposed approach.

The findings suggest that early seismic signals contain sufficient enough information for our model to approximately estimate event's magnitude.