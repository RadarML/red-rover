# Time Series Performance Statistics

!!! abstract "TL;DR"

    Use a effective sample size-corrected paired z-test to compare methods which have been evaluated on moderately-sized time series data.

You can install `tss` from github:
```sh
pip install "tss@git+ssh://git@github.com/WiseLabCMU/red-rover.git#subdirectory=stats"
```

## Why Time Series Statistics?

!!! warning "Time-Correlated Samples"

    Unlike scraped internet data, collected physical sensor data generally take the form of long time series, which have significant temporal correlations and cannot be viewed as independent samples[^2].

!!! warning "Finite Sample Size"

    Due to this temporal correlation, we find that in practice, datasets almost never have a large enough test split to be considered an "infinite sample size," even when their test set consists of thousands or tens of thousands of frames. This necessitates *statistical testing* to quantify the uncertainty in our evaluation.

## Usage

!!! abstract "In practice"

    This methodology has so far been used by:

    - [DART: Implicit Doppler Tomography for Radar Novel View Synthesis](https://wiselabcmu.github.io/dart/)

In addition to using the [low level API][tss.stats], we provide a [high level API][tss] which can be used to index results and evaluations, then load the evaluations and calculate statistics.

### File Format

Time series evaluations are expected to be stored in `.npz` files, which contain multiple key-value pairs of metrics and/or timestamps.

- All metrics which might pass through this library must be 1D arrays with a time/sample index axis.
- Timestamps can have additional axes (e.g., for models which operate on a sequence of data); in this case, the last timestamp[^1] is used.
- The metrics and timestamps within each evaluation are expected to be synchronized.

!!! warning

    We assume that data in different evaluations of the same experiment (e.g., traces with different names - `bike/ptbreeze.out.npz` and `bike/ptbreeze.back.npz`) are temporally correlated. Evaluations on data traces which are recorded back-to-back must be combined into the same file!

[^1]: If more than one extra axis is provided, the last axis when the array is flattened in C-order is used.

???+ example

    <div class="grid" markdown>

    ```yaml title="With Timestamps"
    metrics/loss: float32[439890]
    metrics/map/bce: float32[439890]
    metrics/map/depth: float32[439890]
    timestamps/lidar: float64[439890]
    timestamps/radar: float64[439890,8]
    ```

    ```yaml title="Without Timestamps"
    loss: float32[17606]
    map_acc: float32[17606]
    map_chamfer: float32[17606]
    map_depth: float32[17606]
    map_f1: float32[17606]
    ```

    </div>

### Naming Convention

The file path to each evaluation, relative to some base path, should contain information about the experiment name and the sequence/trace name.

- These should be extractable using a regex.
- The regex should have two groups (`experiment` and `trace`), which are used to organize evaluations by `experiment` name and compare evaluations with the same `trace` name.

???+ example

    ```python title="With Experiment and Trace Name"
    pattern = r'^(?P<experiment>(.*))/eval/(?P<trace>(.*))\.npz$'
    relpath = 'small/patch.rdae/eval/bike/ptbreeze.out.npz'
    #          └──experiment──┘      └─────trace─────┘ 
    ```

    ```python title="With Experiment Name Only"
    pattern = r'^(?P<experiment>(.*))\.npz$'
    relpath = 'reduced2x/t8_p100.npz'
    #          └──experiment───┘      (trace=None)
    ```

### Using the High Level API

**Index evaluations**: using [`index`][tss.index], provide a base path where the evaluations are stored, and a regex pattern for finding evaluation files and extracting their `experiment` and `trace` names.

```python
import tss

path = "/shiraz/grt/results"  # path to where the evaluations are stored
pattern = r"^(?P<experiment>(.*))/eval/(?P<trace>(.*))\.npz$"

index = tss.index_results(path, pattern)
```

!!! tip

    This can take quite a long time if you have many evaluation files and
    are loading from a network drive (~3 seconds for ~3000 evaluations in
    a directory with 20k total files on a SMB share). You may want to
    cache the index or save them to disk somewhere!

**Compute Statistics**: we provide a all-inclusive [`dataframe_from_index`][tss.dataframe_from_index] function which returns a dataframe containing summary statistics for the specified index, given a key of interest and baseline method.

```python
experiments = ["small/p10", "small/p20", "small/p50", "small/base"]
df = tss.dataframe_from_index(
    index, "loss", baseline="small/base", experiments=experiments)
df
```

```title="output"
            abs/mean   abs/std  abs/stderr   abs/n     abs/ess  rel/mean   rel/std  rel/stderr   rel/n      rel/ess   pct/mean  pct/stderr  p0.05
name                                                                                                                                             
small/base  0.125371  0.070062    0.002442  162931  823.034877  0.000000  0.000000         NaN  162931     0.000000   0.000000         NaN  False
small/p10   0.161236  0.088207    0.002991  162931  869.479694  0.035865  0.039024    0.000769  162931  2577.969172  28.607590    0.613055   True
small/p20   0.152850  0.097548    0.003209  162931  924.222609  0.027480  0.045835    0.000945  162931  2353.289155  21.918760    0.753636   True
small/p50   0.134158  0.076811    0.002594  162931  877.094752  0.008787  0.027099    0.000406  162931  4453.599831   7.009018    0.323892   True
```

## Procedure

!!! abstract "Assumptions"

    While our goal is really to estimate the underlying effective sample size (ESS) of the underlying time series, we are not aware of any currently methods which can do so for extremely high-dimensional-spaces with low-dimensional structure[^3]. As such, we apply a univariate analysis on model performance metrics by roughly assuming that metrics change if and only if the data changes. Equivalently, and more verbosely, we assume that:

    1. Changing metrics imply changing data,
    2. Constant metrics imply constant data, and
    3. The degree to which the first assumption is violated (i.e., different metrics result from random noise outside of the underlying data "signal") is roughly cancelled out by the degree to which the second is violated (i.e., the data changes, but the method performs the same).


1. **Run a Paired Test**: Paired tests on the difference between two models when applied to the same data control for "constant" sample variability which is unrelated to the performance of the underlying model. This allows for statistical tests on the *relative* performance of ablations with respect to their baselines.

    - Evaluate the baseline method, and each alternative method, on the same samples.
    - Take the difference in performance metrics between each alternative and the baseline.
    - Perform a statistical test on these differences, where the null hypothesis is that the alternative methods are equivalent in performance to the baseline, and the alternative hypothesis is that a given alternative is different (2-sided test) or better (1-sided test).

2. **Calculate the Effective Sample Size**: To estimate the effective sample size given these assumptions, we use an autocorrelation-based metric:
    ```
    N_eff = N / (1 + 2 * (rho_1 + rho_2 + ...))
    ```
    where `rho_t` is the `t`-lag autocorrelation. For details about how we calculate this, see [`effective_sample_size`][tss.stats.effective_sample_size].

3. **Calculate the Standard Error**: Using the effective sample size, we can calculate the standard error
    ```
    SE = std / sqrt(N_eff)
    ```
    and perform a one or two-sided Z-test (assuming that `N_eff` is relatively large).

    !!! warning "Correct for Multiple Inference"

        In the case that multiple alternatives are compared against the baseline, it may be necessary to correct for multiple inference (i.e., the increased chances of getting a result with a low p-value if you evaluate many alternatives at once). Since different methods which tackle the same problem are highly correlated, this requires using a [Bonferroni correction](https://www.statsig.com/glossary/bonferroni-test).


[^2]: Intuitively, sampling the same signal (e.g., radar-lidar-camera tuples) with a greater frequency yields diminishing information: sampling an infinitesimally short video at an infinite frame rate clearly does not yield an infinite sample size.
[^3]: This concept is best explained via the "natural image manifold:" images have a lot of dimensions (`HxWxC`), but take a `np.random.random((h, w, c))` image, and you'll almost surely not end up with a "natural" image that you might actually encounter. The space of all such *natural images* can be thought of as a low-dimensional manifold, embedded in the high-dimensional image space.
