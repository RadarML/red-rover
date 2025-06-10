# Time Series Performance Statistics

!!! abstract "TL;DR"

    Use a effective sample size-corrected paired z-test to compare methods which have been evaluated on moderately-sized time series data.

You can install `tss` from github:
```sh
pip install "tss@git+ssh://git@github.com/WiseLabCMU/red-rover.git#subdirectory=stats"
```

## Usage

!!! abstract "In practice"

    This methodology has so far been used by:

    - [DART: Implicit Doppler Tomography for Radar Novel View Synthesis](https://wiselabcmu.github.io/dart/)

TODO: some usage guides and examples

## Why Time Series Statistics?

!!! failure "Time-Correlated Samples"

    Unlike scraped internet data, collected physical sensor data generally take the form of long time series, which have significant temporal correlations and cannot be viewed as independent samples[^1].

!!! failure "Finite Sample Size"

    Due to this temporal correlation, we find that in practice, datasets almost never have a large enough test split to be considered an "infinite sample size," even when their test set consists of thousands or tens of thousands of frames. This necessitates *statistical testing* to quantify the uncertainty in our evaluation.

## Procedure

**Assumptions**: While our goal is really to estimate the underlying effective sample size (ESS) of the underlying time series, we are not aware of any currently methods which can do so for extremely high-dimensional-spaces with low-dimensional structure[^2]. As such, we apply a univariate analysis on model performance metrics by roughly assuming that metrics change if and only if the data changes. Equivalently, and more verbosely, we assume that:

1. Changing metrics imply changing data,
2. Constant metrics imply constant data, and
3. The degree to which the first assumption is violated (i.e., different metrics result from random noise outside of the underlying data "signal") is roughly cancelled out by the degree to which the second is violated (i.e., the data changes, but the method performs the same).

**Effective Sample Size**: To estimate the effective sample size given these assumptions, we use an autocorrelation-based metric:
```
N_eff = N / (1 + 2 * (rho_1 + rho_2 + ...))
```
where `rho_t` is the `t`-lag autocorrelation. For details about how we calculate this, see [`effective_sample_size`][tss.effective_sample_size].

**Paired Test**: Paired tests on the difference between two models when applied to the same data control for "constant" sample variability which is unrelated to the performance of the underlying model. This allows for statistical tests on the *relative* performance of ablations with respect to their baselines.

[^1]: Intuitively, sampling the same signal (e.g., radar-lidar-camera tuples) with a greater frequency yields diminishing information: sampling an infinitesimally short video at an infinite frame rate clearly does not yield an infinite sample size.

[^2]: This concept is best explained via the "natural image manifold:" images have a lot of dimensions (`HxWxC`), but take a `np.random.random((h, w, c))` image, and you'll almost surely not end up with a "natural" image that you might actually encounter. The space of all such *natural images* can be thought of as a low-dimensional manifold, embedded in the high-dimensional image space.
