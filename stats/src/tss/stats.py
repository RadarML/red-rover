"""Time series analysis."""

from dataclasses import dataclass
from multiprocessing.pool import ThreadPool
from typing import TYPE_CHECKING, Any, Sequence, TypeVar

import numpy as np
import pandas as pd
from jaxtyping import Float, Integer, Num, Shaped
from scipy.signal import correlate


def __pmean(x: Shaped[np.ndarray, "N"], n: int = 0) -> Float[np.ndarray, "N2"]:
    """Calculate the partial mean for the first `n` items.

    Includes the remaining elements:

    ```
    E[x[0:]], E[x[1:]], E[x[2:]], ... E[x[n:]]
    ```
    """
    nc = np.arange(x.shape[0], x.shape[0] - n, -1)
    left = np.cumsum(x[:n][::-1])[::-1]
    right = np.sum(x[n:])
    return (left + right) / nc


def autocorrelation(
    x: Num[np.ndarray, "N"]
) -> Float[np.ndarray, "N2"] | None:
    """Calculate autocorrelation for time delays up to N/2.

    !!! info

        This method is `O(n log n)`: first and second moments are calculated
        with running means (`O(n)`), while cross correlations are calculated
        using the fft-based `scipy.signal.correlate` (`O(n log n)`).

    Args:
        x: time series data.

    Returns:
        Autocorrelation for time delays up to `N/2`; if the data is ever
            identical across an autocorrelation window (this should never
            happen), `None` is returned instead.
    """
    half = x.shape[0] // 2

    # First and second moments
    m1_left = __pmean(x, half)[1:]
    m1_right = __pmean(x[::-1], half)[1:]
    m2_left = __pmean(x**2, half)[1:]
    m2_right = __pmean(x[::-1]**2, half)[1:]

    # NOTE: adjust for empirical estimate of variance, covariance
    n = np.arange(x.shape[0] - 1, x.shape[0] - half, -1)
    std_left = np.sqrt((m2_left - m1_left**2) * n / (n - 1))
    std_right = np.sqrt((m2_right - m1_right**2) * n / (n - 1))

    # Use scipy.signal.correlate -- fft based -- to accelerate this step.
    # You can verify this line matches the naive code:
    # mcross = np.array([
    #     np.sum(x[i:] * x[:-i]) for i in range(1, half)]) / (n - 1)
    mcross = correlate(x, x, mode='same')[-half + 1:] / (n - 1)

    cov = (mcross - m1_left * m1_right) * n / (n - 1)

    # Estimated autocorrelation
    std = std_left * std_right
    if np.any(std == 0):
        return None
    else:
        return cov / std


def effective_sample_size(x: Num[np.ndarray, "t"]) -> float:
    """Calculate effective sample size (ESS) for a univariate time series.

    Let `x` have `N` samples. For autocorrelation `rho_t`, where `t` is the
    delay, in samples, we use the estimate:
    ```
    N_eff = N / (1 + 2 * (rho_1 + rho_2 + ...))
    ```

    !!! info

        This estimate is [commonly used](https://arxiv.org/pdf/1403.5536v1) to
        estimate effective sample sizes for Markov Chain Monte Carlo (MCMC)
        techniques, though it is readily adaptable to other time series
        analysis tasks.

    !!! note

        In our estimate, we sum up to `t = N/2` (the maximum value for which
        `rho_t` is empirically estimatable), as long as `rho_t` is positive.
        A simplified implementation is as follows:
        ```
        rho = np.array([
            np.cov(x[i:], x[:-i])[0, 1] / np.std(x[i:]) / np.std(x[:-i])
            for i in range(1, x.shape[0] // 2)])
        rho_valid = rho[:np.argmax(rho < 0)]
        return x.shape[0] / (1 + 2 * np.sum(rho)
        ```

        Our implementation is `O(n log n)`, and is optimized to reuse moment
        calculations and partial sums within moment calculations, as well as
        use scipy's FFT-based
        `correlate` for calculating `x[i:] * x[-i:]` for `i = 1, 2, ... N / 2`.

    Args:
        x: time series data.

    Returns:
        ESS estimate. In the edge case that `x` is constant, the ESS is
            reported as 0.0.
    """
    rho = autocorrelation(x)
    if rho is None:
        return 0.0
    else:
        if np.any(rho < 0):
            rho = rho[:np.argmax(rho < 0)]

        rho_sum = np.sum(rho).item()
        return x.shape[0] / (1 + 2 * rho_sum)


LeafType = TypeVar("LeafType", bound=np.ndarray)

if TYPE_CHECKING:
    NestedValues = Sequence["NestedValues"] | LeafType
else:
    NestedValues = Sequence[Any] | LeafType


@dataclass
class NDStats:
    """Mean, variance, and ESS tracking for a n-dimensional stack of values.

    Usage:
        - Initialize `NDStats` by either providing pre-computed `n`, `m1`,
            `m2`, and `ess` values, or by using [`from_values`][.] to compute
            these for you.
        - Multiple `NDStats` can be stacked using [`stack`][.]; stacking can
            be performed multiple times.
        - If multiple sequences are provided to `from_values`, statistic
            computation (parallelized using a thread pool) and stacking is
            performed automatically.

    Attributes:
        n: number of samples.
        m1: sum of values, i.e. accumulated first moment.
        m2: sum of squares, i.e. accumulated second moment.
        ess: effective sample size estimate.
    """

    n: Integer[np.ndarray, "*shape"] | np.integer
    m1: Float[np.ndarray, "*shape"] | np.floating
    m2: Float[np.ndarray, "*shape"] | np.floating
    ess: Float[np.ndarray, "*shape"] | np.floating

    @classmethod
    def from_values(
        cls, values: NestedValues[Num[np.ndarray, "_N"]], workers: int = -1
    ) -> "NDStats":
        """Initialize from 1-dimensional time series of values.

        !!! info

            If multiple time series are provided, the statistics are computed
            for each sequence, and the results are stacked.

        Args:
            values: input time series; the first two moments and effective
                sample size are computed.
            workers: number of parallel workers to use (only used in the case
                of multiple time series). If `<0`, all jobs are run in
                parallel; if `0`, the jobs are run in the main thread.

        Returns:
            Computed `NDStats` for the time series or set of time series.
        """
        if isinstance(values, np.ndarray) and values.ndim == 1:
            return cls(
                n=np.array(values.shape[0]),
                m1=np.sum(values).astype(np.float64),
                m2=np.sum(np.square(values)).astype(np.float64),
                ess=np.array(effective_sample_size(values)))
        else:
            if workers < 0:
                workers = len(values)

            if workers == 0:
                return cls.stack(*[cls.from_values(v) for v in values])

            with ThreadPool(workers) as p:
                stats = p.map(cls.from_values, values)
            return cls.stack(*stats)

    @property
    def shape(self) -> tuple[int, ...]:
        """ND shape."""
        return self.n.shape

    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return self.n.ndim

    @property
    def _n(self) -> Integer[np.ndarray, "..."] | np.integer:
        """Raw sample size, with extra dimensions to match the data."""
        n = self.n
        while len(n.shape) < len(self.m1.shape):
            n = np.expand_dims(n, -1)
        return n

    @property
    def mean(self) -> Float[np.ndarray, "*shape"] | np.floating:
        """Sample mean."""
        return self.m1 / self._n

    @property
    def std(self) -> Float[np.ndarray, "*shape"] | np.floating:
        """Unbiased estimate of the sample standard deviation."""
        return np.sqrt(
            (self.m2 / self._n - self.mean**2) * self._n / (self._n - 1))

    @property
    def stderr(self) -> Float[np.ndarray, "*shape"] | np.floating:
        """Sample standard error, with effective sample size correction."""
        with np.errstate(invalid='ignore'):
            return self.std / np.sqrt(self.ess)

    @property
    def zscore(self) -> Float[np.ndarray, "*shape"] | np.floating:
        """Z-score, assuming a zero null hypothesis."""
        return self.mean / self.stderr

    @staticmethod
    def stack(*stats) -> "NDStats":
        """Stack multiple NDStats containers."""
        return NDStats(
            n=np.stack([s.n for s in stats], axis=0),
            m1=np.stack([s.m1 for s in stats], axis=0),
            m2=np.stack([s.m2 for s in stats], axis=0),
            ess=np.stack([s.ess for s in stats], axis=0))


    def sum(self, axis: int = 0) -> "NDStats":
        """Get aggregate values.

        If this `NDStats` is not a vector, this simply returns the identity.

        Args:
            axis: axis to sum across.

        Returns:
            A new `NDStats` object reduced across the specified axis.
        """
        if len(self.n.shape) == 0:
            return self
        return NDStats(
            n=np.sum(self.n, axis=axis), m1=np.sum(self.m1, axis=axis),
            m2=np.sum(self.m2, axis=axis), ess=np.sum(self.ess, axis=axis))

    def reshape(self, *shape: int | np.integer) -> "NDStats":
        """Reshape all statistics."""
        return NDStats(
            n=self.n.reshape(*shape), m1=self.m1.reshape(*shape),
            m2=self.m2.reshape(*shape), ess=self.ess.reshape(*shape))

    def as_df(self, names: Sequence[str], prefix: str = "") -> pd.DataFrame:
        """Convert to a pandas dataframe.

        Creates these columns:

        - Data fields: `n`, `ess`
        - Computed values: [`mean`][^.], [`std`][^.], [`stderr`][^.]

        !!! warning

            This `NDStats` must be a 1-dimensional vector of statistics.

        Args:
            names: name of each entry in the `NDStats` vector.
            prefix: optional prefix to add to each column.

        Returns:
            A dataframe, where each entry is a row.
        """
        if not isinstance(self.n, np.ndarray) or len(self.n.shape) != 1:
            raise ValueError(
                "Cannot convert NDStats to DataFrame: expected a 1D vector, "
                f"got {self.n.shape}.")
        if len(names) != self.n.shape[0]:
            raise ValueError(
                f"Expected {self.n.shape[0]} names, got {len(names)}.")

        rows = zip(
            names, self.n, self.ess,  # type: ignore
            self.mean, self.std, self.stderr)  # type: ignore

        df = pd.DataFrame([{
            "name": name, f"{prefix}mean": mean, f"{prefix}std": std,
            f"{prefix}stderr": stderr, f"{prefix}n": n, f"{prefix}ess": ess
        } for name, n, ess, mean, std, stderr in rows])
        df.set_index("name", inplace=True)
        return df
