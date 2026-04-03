# Chapter 5.5 (Draft) Learnt Features

## Common misconception

For a long time, machine learning models, and more specifically deep learning models which contain millions if not billions of parameters, have been notoriously viewed as a "black box". Part of this sentiment is justified if we take a look at classical statistical models that pose strong mathematical foundations behind the models, e.g. least square method, ARIMA, etc.

Nevertheless, over the past decade, researchers have tried to understand what these large models actually learn. Consequently, the term **Latent Space** (also called **Latent Feature Space** and **Embedding Space**) has emerged. The term denotes the high-dimensional vector space, where the mapped input data resides at different stages of the forward pass. It is then presumed that if the model outputs reasonable results it must be at least moderately successful at mapping the input data into the corresponding vector space and thus data instances with akin characteristics must be closer to each other than to the ones with a dissimilar set of features.

Ultimately, to interpret the vector representation of input data in latent space, different mathematical techniques have been borrowed from statistics. Among those are **Principal Component Analysis** (PCA), **T-distributed Stochastic Neighbor Embedding** (t-SNE), **Uniform Manifold Approximation and Projection** (UMAP), and others.

To abstract, in the proposed approach, the combination of CNN and RNN modules serve as the "feature extraction" part, while the MLP serves as a regression head that makes a final decision based on the extracted features.

## Dimensionality Reduction: The Case for PCA

To verify that the CNN–LSTM backbone is extracting meaningful embeddings, we extract the high-dimensional vectors produced by the LSTM block (i.e., the vectors that are passed to the MLP). As these vectors exist in a high-dimensional space, dimensionality reduction technique is required to map them into an interpretable for us coordinate system.

While non-linear manifold learning algorithms like t-SNE and UMAP are highly prefered in machine learning for mapping high dimensional data into a lower dimension, they struggle to be actually informative as whithin-cluster density and the distance between the clusters can be easily misinterpreted based on the parameters used for these algorithms. 

For this reason, Principal Component Analysis (PCA) was selected to map the data from the latent space. PCA is a linear, deterministic dimensionality reduction algorithm that transforms the original high-dimensional features into a new set of orthogonal variables, called principal components, which capture the maximum variance in the data.

Given a centered dataset of LSTM output vectors, $X$, PCA computes the covariance matrix $C$:

$$C = \frac{1}{n-1} X^T X$$

By performing an eigendecomposition on this covariance matrix, a set of eigenvectors ($v_i$) and their corresponding eigenvalues ($\lambda_i$) are obtained:

$$C v_i = \lambda_i v_i$$

By projecting the data onto the top three eigenvectors corresponding to the largest eigenvalues, the high-dimensional latent space is thus reduced. Unlike non-linear methods, PCA's linear projection guarantees that the Euclidean distances between the projected points remain physically meaningful and proportional to their original high-dimensional space.

## Visualizing the Extracted Features

To evaluate the interpretability of the proposed architecture, the latent vectors were extracted for a subset of the validation dataset and projected into a 3D space using the first three principal components derived.

*[Placeholder: Insert Figure 5.X - 3D PCA projection scatter plot of the LSTM latent vectors. Points should be color-coded using a continuous colormap corresponding to the true earthquake magnitude.]*

As illustrated in Figure 5.X, the first two principal components capture the primary axes of variation within the model's latent space. The projection reveals...

*[Placeholder: Describe the visual results here once generated. Expected observations:
- State the percentage of total variance explained by PC1, PC2, and PC3 (e.g., "The first three principal components account for 78% of the total variance...").
- Mention if there is a distinct, linear or curved gradient corresponding to the continuous magnitude scale.
- Note if the model successfully separates background noise or micro-earthquakes from larger events along the primary axis.]*

This dimensionality reduction technique verifies that the CNN with LSTM layers successfully manage to map the raw spectrograms into meaningful vector representations before the final step of regression is performed by the MLP.