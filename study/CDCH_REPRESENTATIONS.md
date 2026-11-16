# CDCH representations

## Canonical input

All four representations use the same unmodified Muzyka JavaScript in `pilot/cdch/ip-1.js`. Its SHA-256 is `110732b48bc6e8b2d5167eec30f7277245913bed98b9baaa3c65c0c22aab44d8`. The pilot oracle supplies the canvas calls. The new persistent Node adapter times the oracle and serializes its float64 matrix without rounding. Python reads that matrix and computes descriptors. Four probes reproduced the archived oracle matrices exactly, and all 560 archived compact descriptors reproduced within an absolute and relative tolerance of 10⁻¹².

For each RGB 128 × 128 crop the canonical algorithm starts at (64, 64). It samples one ray per degree in order, 0° pointing up. At integer radii starting at zero, the JavaScript rounds the two coordinates and samples the rounded pixel until it encounters the boundary or the radius reaches the image width. The luminance bin is the JavaScript round of 0.2125 R + 0.7154 G + 0.0721 B. A ray histogram accumulates integer bin counts with their canonical multiplicities, including duplicated coordinates from rounding. Each CDF F(θ,b) is the cumulative count through intensity b divided by that ray's sample count. Raw shape is 360 × 256, or 737,280 bytes in float64. Luma is the only canonical output under this protocol. Channel-specific distances would require an additional experiment with separately transformed inputs.

| ID | Model input | Float64 cache bytes per image |
|---|---|---:|
| A | 56 compact summary features | 448 |
| B | 98 features including A and 42 additional statistics | 784 |
| C | 768 angular-sector CDF values | 6,144 |
| D | Same values as C with an explicit 24 × 32 tensor layout | 6,144 |

D converts the normalized tensor to float32 for training and inference. Its model input therefore has 3,072 bytes per image prior to activation allocation.

## A. Compact pilot descriptor

Sample the CDF at b = 7, 23, …, 247. For each of the 16 columns, compute the angular mean, angular population standard deviation, and mean magnitude of the one-degree first difference with circular wrap. Transform the demeaned sampled CDF along angle using the real discrete Fourier transform. Retain harmonic magnitudes at k ∈ {1, 2, 4, 8, 16, 32, 64, 128}, divide by 360, and average across the 16 intensity indices for each k. Concatenate the 16 means, 16 standard deviations, 16 mean differences and eight harmonic values in that order.

## B. Enriched statistics

Let P(θ, b) = F(θ, b) − F(θ, b − 1), with F(θ, −1) = 0, and let x = b/255. Form the following 21 per-ray quantities in order. For each quantity append its mean and population standard deviation across the 360 rays to A.

1. Mean Σ P x, standard deviation √v, skewness Σ P(x − μ)³ / v³ᐟ², and kurtosis Σ P(x − μ)⁴ / v². Here v is Σ P(x − μ)². Set the two standardized higher moments to zero for v ≤ 10⁻¹⁰.
2. Normalized entropy −Σ P log₂(max(P, 10⁻³⁰⁰))/8, concentration Σ P², dark tail F(θ,15), light tail 1 − F(θ,239), maximum mass max P, and mean absolute mass difference across adjacent intensity bins.
3. Five inverse-CDF intensities min{b | F(θ,b) ≥ q}/255 at q = 0.10, 0.25, 0.50, 0.75, 0.90, followed by the interquartile and interdecile ranges.
4. Mean over intensity of the absolute CDF difference between angles θ and θ − k, with circular wrap at lags k = 1, 15, 90, and 180 degrees.

These summaries capture tonal shifts, concentration and smoothing that can alter luma diversity, tails or local histogram shape. The angular CDF lags distinguish abrupt angular changes from coarse directional variation. Scene content and conventional resampling can also move each of these quantities.

## C. Angular-sector CDF

For sector s = 0, …, 23 and intensity index j = 0, …, 31, average F(θ,8j+7) over the 15 consecutive degrees beginning at θ = 15s. Flatten in sector-major order. This preserves coarse absolute orientation and intensity structure. A finer or raw matrix may encode information lost in pooling. The 60-source training cohort motivated keeping the fixed design at 768 values.

## D. Structured angular CNN

Interpret the C vector as 24 circularly ordered angle positions and 32 channels of CDF ordinates. Standardize each ordinate using its mean and population standard deviation across training samples and sectors, clamping scale below 10⁻⁶. Use circular 1D angular convolutions with kernel size three and widths 32→12→8, SiLU after each convolution, global angular average pooling and one linear logit. This yields 1,469 trainable parameters. The order control uses a fixed random permutation of angular sectors with seed 197 and the same fitting procedure. Pooling confers invariance to cyclic integer sector shifts except for floating-point effects. Angular locality describes neighboring rays. Intensity ordinates are channels and receive no convolutional weight sharing across bins.

## Fitting

A, B and C use a train-only StandardScaler and six fixed-capacity classifiers. The linear SVM uses C = 0.1. The RBF SVM uses C = 1 and gamma = scale. The forest uses 128 trees of depth at most six with at least four samples per leaf. Histogram boosting uses at most 80 iterations, seven leaves, rate 0.07, and L2 regularization 1. The MLP uses one 16-unit hidden layer, L2 penalty 0.1, learning rate 0.003 and at most 200 iterations. Logistic regression matches the pilot. D uses AdamW, 25 epochs, learning rate 0.002, weight decay 0.02, mini-batches of 24 and binary logit loss. Nonconvergence for fixed iteration budgets is recorded in the model inventory. No target media informed features, scales, hyperparameters, model selection or thresholds.
