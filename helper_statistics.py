import marimo

__generated_with = "0.17.7"
app = marimo.App(width="medium")

with app.setup:
    import os
    import numpy as np
    import pandas as pd
    from scipy import stats
    from scipy.stats import mannwhitneyu, t


@app.function
# Define the modified function to handle NaN values
def calculate_sem(data):
    """Calculate the Standard Error of the Mean (SEM) along the specified axis, ignoring NaNs."""
    return np.nanstd(data, axis=1) / np.sqrt(np.sum(~np.isnan(data), axis=1))


@app.function
def wilcoxon_summary(bin1, bin2, alpha=0.05, alternative="two-sided"):
    def describe(x):
        n = len(x)
        if n == 0:
            return n, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        mean = np.mean(x)
        median = np.median(x)
        sd = np.std(x, ddof=1)
        q25, q75 = np.percentile(x, [25, 75])
        iqr = q75 - q25
        sem = sd / np.sqrt(n)
        moe = t.ppf(1 - alpha / 2, df=n - 1) * sem
        return n, mean, median, sd, iqr, sem, moe

    n1, mean1, median1, sd1, iqr1, sem1, moe1 = describe(bin1)
    n2, mean2, median2, sd2, iqr2, sem2, moe2 = describe(bin2)

    # If either bin is empty, skip test
    if n1 == 0 or n2 == 0:
        U = z = p = r_rb = d = np.nan
    else:
        U, p = mannwhitneyu(bin1, bin2, alternative=alternative)
        mu_U = n1 * n2 / 2
        sigma_U = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
        z = (U - mu_U) / sigma_U if sigma_U > 0 else np.nan
        r_rb = 1 - 2 * (U / (n1 * n2))
        sp = (
            np.sqrt(((n1 - 1) * sd1**2 + (n2 - 1) * sd2**2) / (n1 + n2 - 2))
            if (n1 + n2) > 2
            else np.nan
        )
        d = (mean1 - mean2) / sp if sp and sp > 0 else np.nan

    df = pd.DataFrame(
        [
            {
                "n1": n1,
                "mean1": mean1,
                "median1": median1,
                "sd1": sd1,
                "iqr1": iqr1,
                "sem1": sem1,
                "moe1": moe1,
                "n2": n2,
                "mean2": mean2,
                "median2": median2,
                "sd2": sd2,
                "iqr2": iqr2,
                "sem2": sem2,
                "moe2": moe2,
                "U": U,
                "z": z,
                "p": p,
                "cohens_d": d,
            }
        ]
    )
    return df


@app.function
# Function to create initial DataFrame and populate it
def create_stats_df(data_off, data_on, indices):
    stats_df = pd.DataFrame()
    for period, index in indices.items():
        stats_df[f"Mean FR Off {period.capitalize()}"] = pd.Series(
            data_off[index, :].mean(axis=0)
        )
        stats_df[f"Mean FR On {period.capitalize()}"] = pd.Series(
            data_on[index, :].mean(axis=0)
        )
    return stats_df


@app.function
def describe_and_enhance(stats_df):
    described = stats_df.describe()
    confidence_level = 0.95

    for column in stats_df.columns:
        if (
            stats_df[column].dtype.kind in "bifc"
        ):  # Check if the column is numeric
            if (
                len(stats_df[column]) > 1
            ):  # Check if there are at least 2 data points
                try:
                    # Computing median
                    median_val = stats_df[column].median()
                    described.loc["median", column] = median_val

                    # Computing mean and SEM for confidence interval
                    mean_val = stats_df[column].mean()
                    sem_val = stats_df[column].sem()
                    ci_low, ci_high = stats.t.interval(
                        confidence_level,
                        len(stats_df[column]) - 1,
                        loc=mean_val,
                        scale=sem_val,
                    )

                    margin_of_error = (
                        ci_high - mean_val
                    )  # This is the margin of error

                    described.loc["mean", column] = (
                        mean_val  # Just to ensure the mean is explicitly set in the described DataFrame
                    )
                    described.loc["SEM", column] = (
                        sem_val  # Adding the SEM value explicitly
                    )
                    described.loc["CI_low", column] = (
                        ci_low  # Lower bound of the CI
                    )
                    described.loc["CI_high", column] = (
                        ci_high  # Upper bound of the CI
                    )
                    described.loc["MOE", column] = (
                        margin_of_error  # Adding the margin of error
                    )
                    described.loc["IQR", column] = (
                        described.loc["75%", column]
                        - described.loc["25%", column]
                    )  # Interquartile range
                except Exception as e:
                    print(f"Error computing for column {column}: {e}")
            else:
                print(
                    f"Not enough data to compute statistics for column {column}"
                )
        else:
            print(f"Column {column} is not numeric and will be skipped.")

    return described


@app.function
# Statistical testing function
def perform_analysis(df, col1, col2):
    stat, p = stats.wilcoxon(df[col1], df[col2])
    n = (
        df[col1].notna().sum()
    )  # Assuming NaN handling required, adjust as necessary
    z = (stat - n * (n + 1) / 4) / np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    r = abs(z / np.sqrt(n))
    return stat, p, z, r


@app.function
def create_and_describe_df(data, columns):
    df = pd.DataFrame(
        data.T, columns=columns
    )  # Creating DataFrame and transposing data if necessary
    described = df.describe()  # Get basic descriptive statistics

    # Calculate the median and add it to the described DataFrame
    median_values = df.median()
    for column in columns:
        if df[column].dtype.kind in "bifc":  # Check if the column is numeric
            described.loc["median", column] = median_values[column]

            confidence_level = 0.95  # 95% confidence
            if (
                len(df[column].dropna()) > 1
            ):  # Check if there are at least 2 data points after dropping NaN
                mean_val = described.loc["mean", column]
                sem_val = stats.sem(
                    df[column], nan_policy="omit"
                )  # Calculate the SEM, handling NaN values
                ci_low, ci_high = stats.t.interval(
                    confidence_level,
                    len(df[column].dropna()) - 1,
                    loc=mean_val,
                    scale=sem_val,
                )

                margin_of_error = (
                    ci_high - mean_val
                )  # This is the margin of error

                described.loc["SEM", column] = (
                    sem_val  # Adding the SEM value explicitly
                )
                described.loc["CI_low", column] = (
                    ci_low  # Lower bound of the CI
                )
                described.loc["CI_high", column] = (
                    ci_high  # Upper bound of the CI
                )
                described.loc["MOE", column] = (
                    margin_of_error  # Adding the margin of error
                )
                described.loc["IQR", column] = (
                    described.loc["75%", column] - described.loc["25%", column]
                )  # Calculate IQR
            else:
                print(
                    f"Not enough data to compute statistics for column {column}"
                )
        else:
            print(f"Column {column} is not numeric and will be skipped.")

    return df, described


@app.function
# Performing statistical analysis on octaves
def perform_analysis_octaves(off_data, on_data):
    try:
        off = np.asarray(off_data, dtype=float)
        on = np.asarray(on_data, dtype=float)

        stat, p = stats.wilcoxon(
            off, on, nan_policy="omit", zero_method="wilcox", method="auto"
        )

        m = ~np.isnan(off) & ~np.isnan(on)
        d = (on[m] - off[m]).astype(float)
        d = d[d != 0]
        n = d.size
        if n == 0:
            return {
                "Test Statistic": stat,
                "p Value": p,
                "Z Value": np.nan,
                "Effect Size": np.nan,
            }

        absd = np.abs(d)
        ranks = stats.rankdata(absd, method="average")
        Wpos = ranks[d > 0].sum()
        meanW = n * (n + 1) / 4.0
        _, counts = np.unique(absd, return_counts=True)
        tie_term = (
            (counts * (counts + 1) * (2 * counts + 1)).sum()
            if np.any(counts > 1)
            else 0.0
        )
        varW = (n * (n + 1) * (2 * n + 1) - tie_term) / 24.0
        z = (Wpos - meanW) / np.sqrt(varW) if varW > 0 else np.nan
        r = abs(z / np.sqrt(n)) if np.isfinite(z) else np.nan
        return {
            "Test Statistic": stat,
            "p Value": p,
            "Z Value": z,
            "Effect Size": r,
        }
    except ValueError:
        return {
            "Test Statistic": np.nan,
            "p Value": np.nan,
            "Z Value": np.nan,
            "Effect Size": np.nan,
        }


@app.function
def save_dataframes(dfs, base_directory, base_filename):
    for key, df in dfs.items():
        # Construct the full path for the Excel file
        excel_path = os.path.join(
            base_directory, f"{base_filename}_{key}.xlsx"
        )
        # Save DataFrame to Excel
        df.to_excel(excel_path, index=True)

        # Construct the full path for the CSV file
        csv_path = os.path.join(base_directory, f"{base_filename}_{key}.csv")
        # Save DataFrame to CSV
        df.to_csv(csv_path, index=True)


@app.function
def pad_and_create_dataframe(data_lists, columns):
    # Find the maximum length among all lists and pad them
    max_length = max(len(lst) for lst in data_lists)
    padded_data = [
        lst + [np.nan] * (max_length - len(lst)) for lst in data_lists
    ]

    # Create DataFrame
    return pd.DataFrame(
        dict(zip(columns, padded_data)),
        index=[f"Spar{i}" for i in range(1, max_length + 1)],
    )


@app.function
def calculate_statistics(df):
    described = df.describe()

    # Explicitly calculate and add Median if not present
    for column in df.columns:
        if df[column].dtype.kind in "bifc":  # Check if the column is numeric
            median_val = df[column].median()
            described.loc["Median", column] = (
                median_val  # Add Median to the described DataFrame
            )

            # Continue with additional statistics
            n = len(df[column].dropna())  # Number of non-NA values
            if n > 1:
                mean_val = described.loc["mean", column]
                sem_val = df[column].sem()
                ci_low, ci_high = stats.t.interval(
                    0.95, n - 1, loc=mean_val, scale=sem_val
                )
                margin_of_error = ci_high - mean_val

                described.loc["SEM", column] = sem_val
                described.loc["CI_low", column] = ci_low
                described.loc["CI_high", column] = ci_high
                described.loc["MOE", column] = margin_of_error
                described.loc["IQR", column] = (
                    described.loc["75%", column] - described.loc["25%", column]
                )
            else:
                print(
                    f"Not enough data to compute full statistics for column {column}"
                )
        else:
            print(f"Column {column} is not numeric and will be skipped.")

    return described


@app.function
# Fixed stats analysis code
# uses non-zero paired N
# applies tie-corrected variance for N


def perform_statistical_analysis(df, pairs):
    def _wilcoxon_z(off, on):
        # Pairwise NaN drop
        m = ~np.isnan(off) & ~np.isnan(on)
        d = (on[m] - off[m]).astype(float)

        # Drop zero diffs (matches zero_method='wilcox')
        d = d[d != 0]
        n = d.size
        if n == 0:
            return np.nan, 0, np.nan  # z, n, W

        # Rank |d| with average ranks for ties
        absd = np.abs(d)
        ranks = stats.rankdata(absd, method="average")
        W = ranks[d > 0].sum()

        # Mean and tie-corrected variance of W
        meanW = n * (n + 1) / 4.0
        # tie correction on |d|
        _, counts = np.unique(absd, return_counts=True)
        tie_term = (
            (counts * (counts + 1) * (2 * counts + 1)).sum()
            if np.any(counts > 1)
            else 0.0
        )
        varW = (n * (n + 1) * (2 * n + 1) - tie_term) / 24.0
        if varW <= 0:
            return np.nan, n, W

        # Standardized Z (no continuity correction; add -0.5*np.sign(...) if desired)
        z = (W - meanW) / np.sqrt(varW)
        return z, n, W

    results = {}
    for label, (col1, col2) in pairs.items():
        off_data = df[col1].to_numpy(dtype=float)
        on_data = df[col2].to_numpy(dtype=float)
        try:
            # Let SciPy compute W and p (exact/approx as appropriate)
            W, p = stats.wilcoxon(
                off_data,
                on_data,
                nan_policy="omit",
                zero_method="wilcox",
                method="auto",  # exact for small n, normal approx for large n
            )
            z, n_used, _W_check = _wilcoxon_z(off_data, on_data)
            r = (abs(z) / np.sqrt(n_used)) if np.isfinite(z) else np.nan

            results[label] = {
                "Test Statistic": float(W),  # Wilcoxon W
                "p Value": float(p),
                "Z Value": float(z) if np.isfinite(z) else np.nan,
                "Effect Size": float(r) if np.isfinite(r) else np.nan,
            }
        except ValueError:
            results[label] = {
                "Test Statistic": np.nan,
                "p Value": np.nan,
                "Z Value": np.nan,
                "Effect Size": np.nan,
            }
    return pd.DataFrame(results)


@app.cell
def _():
    import marimo as mo
    return


if __name__ == "__main__":
    app.run()
