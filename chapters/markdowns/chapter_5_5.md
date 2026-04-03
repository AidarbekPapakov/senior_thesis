# Chapter 5.5 (Draft) Learnt Features

## Common misconception

For a long time, machine learning models, and more specifically deep learning models which contain millions if not billions of parameters, have been notoriosly viewed as a "black box". The part of this sentiment is justified if we take a look at classical statistical models that pose strong mathematical foundation behind the models, e.g. least square method, ARIMA, etc.

<!-- For example, in the tasks of computer vision, saliency maps (TODO: needs reference ig) are the technique to understand what pixel region affect the model's decision the most. -->


Nevertheless, over the past decade, researchers have tried to understand what these large models actually learn. Consequently, the term **Latent Space** (also called **Latent Feature Space** and **Embedding Space**) has emerged. The term denotes the high-dimensional vector space, where the mapped input data resides at different stages of forward pass. It is then presumed that if the model outputs reasonable results it must be at least moderately successful at mapping the input data into the corresponding vector space and thus data instance with akin charachteristics must be closer to each other than to the ones with dissimilar set of features.

Ultimately, to interpret the vector representation of input data in latent space different mathematical techniques have been borrowed from statistics. Among those are **Principal Component Analysis** (PCA), **T-distributed Stochastic Neighbor Embedding** (t-SNE), **Uniform Manifold Approximation and Projection** (UMAP), and others.

To abstract, in the proposed approach, the combination of CNN and RNN modules serve as "feature extraction" part, while the MLP serves as a regression head that makes a final decision based on the extracted features.

