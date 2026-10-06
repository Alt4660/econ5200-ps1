"""Audit a transaction table and compute a robust average basket value."""

import numpy as np
import pandas as pd
from scipy import stats


def audit_report(df):
    """Check the quality of a table, one row per column. For every column it reports the data type and the share of values that are missing.
    For numeric columns it also reports the skew and the number of outliers (true/false columns don't count as numeric here). An outlier
    is any value more than 1.5 IQRs below the first quartile or above the third quartile, which is Tukey's rule. Text and true/false columns get
    NaN for the skew and outlier count.


    Args: df: Any DataFrame, for example the transaction table.

    Returns: A DataFrame indexed by column name with the columns ``dtype``,
        ``missing_share``, ``skew`` and ``n_outliers``. """
    rows = []
    for col in df.columns:
        s = df[col]
        row = {
            "column": col,
            "dtype": str(s.dtype),
            "missing_share": float(s.isna().mean()),
            "skew": np.nan,
            "n_outliers": np.nan,
        }
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            x = s.dropna()
            q1, q3 = x.quantile([0.25, 0.75])
            iqr = q3 - q1
            row["skew"] = float(x.skew())
            row["n_outliers"] = int(((x < q1 - 1.5 * iqr) | (x > q3 + 1.5 * iqr)).sum())
        rows.append(row)
    return pd.DataFrame(rows).set_index("column")


def robust_mean(x, method: str = "median") -> float:
    """Return a robust centre of ``x`` that resists extreme values.

    Args:
        x: Numbers to summarise. Missing values are dropped.
        Methods:
        One of ``"median"``: the middle value.
        ``"trimmed"``: the mean after dropping the lowest and highest 10% of values.
        ``"exclude"``: the mean after dropping values above an upper fence computed on the log scale, ``exp(Q3 + 1.5 * IQR)`` of the logs.
        This is the B2B exclusion rule. It needs strictly positive values.

    Returns: The robust centre as a float.

    Raises: ValueError: If ``x`` has no valid values, ``method`` is unknown, or
            ``method="exclude"`` is used with non-positive values. """

    values = pd.Series(x, dtype=float).dropna()   # drop missing values first
    if values.empty:
        raise ValueError("x has no valid values")

    if method == "median":
        return float(values.median())
    if method == "trimmed":
        return float(stats.trim_mean(values, 0.1))
    if method == "exclude":
        if (values <= 0).any():                   # logs need positive values
            raise ValueError("method='exclude' needs strictly positive values")
        logs = np.log(values)
        q1, q3 = logs.quantile([0.25, 0.75])
        fence = np.exp(q3 + 1.5 * (q3 - q1))      # upper fence, back in dollars
        return float(values[values <= fence].mean())
    raise ValueError(f"unknown method {method!r}; use 'median', 'trimmed' or 'exclude'")


def main() -> None:
    """Demonstrate both functions on a small simulated, contaminated table."""
    rng = np.random.default_rng(0)
    clean = rng.lognormal(mean=4, sigma=0.8, size=2_000)
    b2b = rng.lognormal(mean=8, sigma=0.5, size=10)
    df = pd.DataFrame({"basket_value": np.concatenate([clean, b2b])})

    print("Audit report:")
    print(audit_report(df).round(2))
    print("\nPlain mean:", round(df["basket_value"].mean(), 2))
    for method in ("median", "trimmed", "exclude"):
        print(f"robust_mean ({method}):", round(robust_mean(df["basket_value"], method), 2))
    print("True clean mean:", round(clean.mean(), 2))


if __name__ == "__main__":
    main()
