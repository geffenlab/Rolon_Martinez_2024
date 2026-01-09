import marimo

__generated_with = "0.18.2"
app = marimo.App()


@app.cell
def _():
    # Load the files
    import numpy as np

    # Load the data files
    PV_mTCOff = np.load("PV_mTCOff.npy", allow_pickle=True)
    PV_mTCOn = np.load("PV_mTCOn.npy", allow_pickle=True)
    SST_mTCOff = np.load("SST_mTCOff.npy", allow_pickle=True)
    SST_mTCOn = np.load("SST_mTCOn.npy", allow_pickle=True)

    # Load the uniq frequencies
    uniq_Freq = np.load("uniq_Freq.npy", allow_pickle=True)
    octaves = np.log2(uniq_Freq)

    # Load the facilitation and suppression indices for each
    SST_facilitated_Ind = np.load("SST_facilitated_index.npy", allow_pickle=True)
    SST_suppressed_Ind = np.load("SST_suppressed_index.npy", allow_pickle=True)
    PV_facilitated_Ind = np.load("PV_facilitated_index.npy", allow_pickle=True)
    PV_suppressed_Ind = np.load("PV_suppressed_index.npy", allow_pickle=True)
    return (
        PV_facilitated_Ind,
        PV_mTCOff,
        PV_mTCOn,
        PV_suppressed_Ind,
        SST_facilitated_Ind,
        SST_mTCOff,
        SST_mTCOn,
        SST_suppressed_Ind,
        np,
        octaves,
        uniq_Freq,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Define gaussian and other functions
    """)
    return


@app.cell
def _(np):
    import matplotlib.pyplot as plt
    from sklearn.metrics import r2_score
    from scipy.optimize import curve_fit


    # define gaussian
    def gaussian(x, a, mu, sigma, baseline):
        return a * np.exp(-0.5 * ((x - mu) / sigma) ** 2) + baseline


    def calculate_r2(y_true, y_pred):
        mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
        if np.sum(mask) < 2:
            return np.nan  # not enough points
        return r2_score(y_true[mask], y_pred[mask])
    return curve_fit, gaussian, plt


@app.cell
def _(curve_fit, gaussian, np):
    import pandas as pd


    def fit_gaussians_to_tuning(
        mTC, octaves, *, min_points=3, sigma_init=0.5, maxfev=8000
    ):
        """
        Fits a Gaussian (in octaves) to each column of mTC, trying multiple initializations.
        Allows negative amplitude for inhibited cells.
        Returns: DataFrame and predicted curves.
        """
        n_freqs, n_cells = mTC.shape
        results = []
        yhat_all = np.full_like(mTC, np.nan, dtype=float)
        mu_guess_global = float(np.median(octaves))
        for _cell in range(n_cells):
            y = mTC[:, _cell]
            mask = ~np.isnan(y)
            if np.count_nonzero(mask) < min_points:
                results.append(
                    (_cell, np.nan, np.nan, np.nan, np.nan, np.nan, False)
                )
                continue
            x_fit = octaves[mask]
            y_fit = y[mask]
            try:
                peak_idx = int(np.nanargmax(y))
                baseline_guess = float(np.nanmin(y))
                a_guess = float(y[peak_idx] - baseline_guess)
                init_params = [
                    [a_guess, mu_guess_global, sigma_init, baseline_guess],
                    [a_guess, x_fit[peak_idx], sigma_init, baseline_guess],
                    [a_guess, mu_guess_global, 1.0, baseline_guess],
                    [a_guess, mu_guess_global, 0.25, baseline_guess],
                    [-abs(a_guess), mu_guess_global, sigma_init, baseline_guess],
                ]
                best_fit = None
                best_r2 = -np.inf
                for p0 in init_params:
                    try:
                        popt, _ = curve_fit(
                            gaussian, x_fit, y_fit, p0=p0, maxfev=maxfev
                        )
                        y_pred = gaussian(x_fit, *popt)
                        ss_res = np.sum((y_fit - y_pred) ** 2)
                        ss_tot = np.sum((y_fit - np.mean(y_fit)) ** 2)
                        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
                        if np.isfinite(r2) and r2 > best_r2:
                            best_fit = popt
                            best_r2 = r2
                    except Exception:
                        continue
                if best_fit is not None:
                    a, mu, sigma, baseline = best_fit
                    yhat_all[:, _cell] = gaussian(octaves, *best_fit)
                    results.append((_cell, a, mu, sigma, baseline, best_r2, True))
                else:
                    results.append(
                        (_cell, np.nan, np.nan, np.nan, np.nan, np.nan, False)
                    )
            except Exception as e:
                print(f"Fit failed for cell {_cell}: {e}")
                results.append(
                    (_cell, np.nan, np.nan, np.nan, np.nan, np.nan, False)
                )
        df = pd.DataFrame(
            results,
            columns=["cell", "a", "mu", "sigma", "baseline", "r2", "success"],
        )
        return (df, yhat_all)
    return fit_gaussians_to_tuning, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Compute fits for all the tuning curves (all)
    """)
    return


@app.cell
def _(
    PV_mTCOff,
    PV_mTCOn,
    SST_mTCOff,
    SST_mTCOn,
    fit_gaussians_to_tuning,
    octaves,
):
    # PV Light On
    PV_results_on, PV_yhat_on = fit_gaussians_to_tuning(PV_mTCOn, octaves)

    # PV Light Off
    PV_results_off, PV_yhat_off = fit_gaussians_to_tuning(PV_mTCOff, octaves)

    # SST Light On
    SST_results_on, SST_yhat_on = fit_gaussians_to_tuning(SST_mTCOn, octaves)

    # SST Light Off
    SST_results_off, SST_yhat_off = fit_gaussians_to_tuning(SST_mTCOff, octaves)
    return PV_results_off, PV_results_on, SST_results_off, SST_results_on


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Make PDFs
    """)
    return


@app.cell
def _(
    PV_facilitated_Ind,
    PV_mTCOff,
    PV_mTCOn,
    PV_results_off,
    PV_results_on,
    PV_suppressed_Ind,
    gaussian,
    np,
    plt,
    uniq_Freq,
):
    import matplotlib.ticker as ticker
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib as mpl

    _pdf_filename = "PV_receptivefield_fits_cell55_opt1.pdf"
    _freqs = uniq_Freq
    _off_df = PV_results_off.set_index("cell")
    _on_df = PV_results_on.set_index("cell")
    _mTCOff = PV_mTCOff
    _mTCOn = PV_mTCOn
    # _common_cells = sorted(set(_off_df.index) & set(_on_df.index))
    _common_cells = [55]  # supp example
    # _common_cells = [38] #facil example
    with mpl.rc_context(
        {
            "font.family": "Arial",
            "font.size": 6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    ):
        with PdfPages(_pdf_filename) as _pdf:
            for _cell in _common_cells:
                _row_off = _off_df.loc[_cell]
                _row_on = _on_df.loc[_cell]
                if not _row_off["success"] or not _row_on["success"]:
                    continue
                if np.any(
                    np.isnan(
                        [
                            _row_off["a"],
                            _row_off["mu"],
                            _row_off["sigma"],
                            _row_off["baseline"],
                        ]
                    )
                ):
                    continue
                if np.any(
                    np.isnan(
                        [
                            _row_on["a"],
                            _row_on["mu"],
                            _row_on["sigma"],
                            _row_on["baseline"],
                        ]
                    )
                ):
                    continue
                if _cell in PV_facilitated_Ind:
                    _group_label = "Facilitated"
                elif _cell in PV_suppressed_Ind:
                    _group_label = "Suppressed"
                else:
                    _group_label = "Neither"
                _fit_x_oct = np.linspace(
                    np.log2(_freqs.min()), np.log2(_freqs.max()), 200
                )
                _x_fit_hz = 2.0**_fit_x_oct
                _params_off = (
                    _row_off["a"],
                    _row_off["mu"],
                    _row_off["sigma"],
                    _row_off["baseline"],
                )
                _params_on = (
                    _row_on["a"],
                    _row_on["mu"],
                    _row_on["sigma"],
                    _row_on["baseline"],
                )
                _y_fit_off = gaussian(_fit_x_oct, *_params_off)
                _y_fit_on = gaussian(_fit_x_oct, *_params_on)
                _x_data = _freqs
                _y_data_off = _mTCOff[:, _cell]
                _y_data_on = _mTCOn[:, _cell]
                _fig, _ax = plt.subplots(figsize=(2.2, 2.2))
                _ax.plot(
                    _x_data, _y_data_off, "o", color="grey", label="Light Off Data"
                )
                _ax.plot(
                    _x_fit_hz,
                    _y_fit_off,
                    "-",
                    color="black",
                    label="Light Off Fit",
                )
                _ymin, _ = _ax.get_ylim()
                _ymax = max(_y_fit_off)
                _ax.vlines(
                    [
                        np.power(2.0, _row_off["mu"]),
                        # np.power(2.0, _row_off["mu"] + _row_off["sigma"]),
                        # np.power(2.0, _row_off["mu"] - _row_off["sigma"]),
                    ],
                    _ymin,
                    _ymax,  # [_ymax, _ymax * 0.75, _ymax * 0.75],
                    color="black",
                )
                # """
                _y_sig = _ymax * 0.72
                _ax.hlines(
                    _y_sig,
                    np.power(2.0, _row_off["mu"] - _row_off["sigma"]),
                    np.power(2.0, _row_off["mu"] + _row_off["sigma"]),
                    color="black",
                    linestyles="dotted",
                )
                # """
                _ax.plot(
                    _x_data,
                    _y_data_on,
                    "o",
                    color="#93c5fd",
                    label="Light On Data",
                )
                _ax.plot(
                    _x_fit_hz,
                    _y_fit_on,
                    "-",
                    color="#1d4ed8",
                    label="Light On Fit",
                )
                _ymax = max(_y_fit_on)
                _ax.vlines(
                    [
                        np.power(2.0, _row_on["mu"]),
                        # np.power(2.0, _row_on["mu"] + _row_on["sigma"]),
                        # np.power(2.0, _row_on["mu"] - _row_on["sigma"]),
                    ],
                    _ymin,
                    _ymax,  # [_ymax, _ymax * 0.75, _ymax * 0.75],
                    color="#1d4ed8",
                )
                # """
                _y_sig = _ymax * 0.72
                _ax.hlines(
                    _y_sig,
                    np.power(2.0, _row_on["mu"] - _row_on["sigma"]),
                    np.power(2.0, _row_on["mu"] + _row_on["sigma"]),
                    color="#1d4ed8",
                    linestyles="dotted",
                )
                # """
                _ax.set_xscale("log")
                _ax.set_xlabel("Frequency (Hz, log scale)")
                _ax.set_ylabel("Firing Rate")
                _custom_ticks = [
                    np.min(uniq_Freq),
                    np.median(uniq_Freq),
                    np.max(uniq_Freq),
                ]
                _ax.set_xticks(_custom_ticks)
                _formatter = ticker.ScalarFormatter()
                _formatter.set_scientific(False)
                _formatter.set_useOffset(False)
                _ax.xaxis.set_major_formatter(_formatter)
                _ax.set_title(
                    f"Cell {_cell} ({_group_label})  R² off={_row_off['r2']:.2f}, on={_row_on['r2']:.2f}"
                )
                _ax.legend(frameon=False)
                _ax.spines["top"].set_visible(False)
                _ax.spines["right"].set_visible(False)
                _fig.tight_layout()
                _pdf.savefig(_fig)
                plt.close(_fig)
    print(f"Saved all cell fits to {_pdf_filename}")
    return PdfPages, mpl, ticker


@app.cell
def _(
    PdfPages,
    SST_facilitated_Ind,
    SST_mTCOff,
    SST_mTCOn,
    SST_results_off,
    SST_results_on,
    SST_suppressed_Ind,
    gaussian,
    mpl,
    np,
    plt,
    ticker,
    uniq_Freq,
):
    _pdf_filename = "SST_receptivefield_fits_cell3_opt1.pdf"
    _freqs = uniq_Freq
    _off_df = SST_results_off.set_index("cell")
    _on_df = SST_results_on.set_index("cell")
    _mTCOff = SST_mTCOff
    _mTCOn = SST_mTCOn
    # _common_cells = sorted(set(_off_df.index) & set(_on_df.index))
    _common_cells = [3]
    with mpl.rc_context(
        {
            "font.family": "Arial",
            "font.size": 6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    ):
        with PdfPages(_pdf_filename) as _pdf:
            for _cell in _common_cells:
                _row_off = _off_df.loc[_cell]
                _row_on = _on_df.loc[_cell]
                if not _row_off["success"] or not _row_on["success"]:
                    continue
                if np.any(
                    np.isnan(
                        [
                            _row_off["a"],
                            _row_off["mu"],
                            _row_off["sigma"],
                            _row_off["baseline"],
                        ]
                    )
                ):
                    continue
                if np.any(
                    np.isnan(
                        [
                            _row_on["a"],
                            _row_on["mu"],
                            _row_on["sigma"],
                            _row_on["baseline"],
                        ]
                    )
                ):
                    continue
                if _cell in SST_facilitated_Ind:
                    _group_label = "Facilitated"
                elif _cell in SST_suppressed_Ind:
                    _group_label = "Suppressed"
                else:
                    _group_label = "Neither"
                _fit_x_oct = np.linspace(
                    np.log2(_freqs.min()), np.log2(_freqs.max()), 200
                )
                _x_fit_hz = 2.0**_fit_x_oct
                _params_off = (
                    _row_off["a"],
                    _row_off["mu"],
                    _row_off["sigma"],
                    _row_off["baseline"],
                )
                _params_on = (
                    _row_on["a"],
                    _row_on["mu"],
                    _row_on["sigma"],
                    _row_on["baseline"],
                )
                _y_fit_off = gaussian(_fit_x_oct, *_params_off)
                _y_fit_on = gaussian(_fit_x_oct, *_params_on)
                _x_data = _freqs
                _y_data_off = _mTCOff[:, _cell]
                _y_data_on = _mTCOn[:, _cell]
                _fig, _ax = plt.subplots(figsize=(2.2, 2.2))
                _ax.plot(
                    _x_data, _y_data_off, "o", color="grey", label="Light Off Data"
                )
                _ax.plot(
                    _x_fit_hz,
                    _y_fit_off,
                    "-",
                    color="black",
                    label="Light Off Fit",
                )
                _ymin, _ = _ax.get_ylim()
                _ymax = max(_y_fit_off)
                _ax.vlines(
                    [
                        np.power(2.0, _row_off["mu"]),
                        # np.power(2.0, _row_off["mu"] + _row_off["sigma"]),
                        # np.power(2.0, _row_off["mu"] - _row_off["sigma"]),
                    ],
                    _ymin,
                    _ymax,  # [_ymax, _ymax * 0.75, _ymax * 0.75],
                    color="black",
                )
                # """
                _y_sig = _ymax * 0.72
                _ax.hlines(
                    _y_sig,
                    np.power(2.0, _row_off["mu"] - _row_off["sigma"]),
                    np.power(2.0, _row_off["mu"] + _row_off["sigma"]),
                    color="black",
                    linestyles="dotted",
                )
                # """
                _ax.plot(
                    _x_data,
                    _y_data_on,
                    "o",
                    color="#93c5fd",
                    label="Light On Data",
                )
                _ax.plot(
                    _x_fit_hz,
                    _y_fit_on,
                    "-",
                    color="#1d4ed8",
                    label="Light On Fit",
                )
                _ymax = max(_y_fit_on)
                _ax.vlines(
                    [
                        np.power(2.0, _row_on["mu"]),
                        # np.power(2.0, _row_on["mu"] + _row_on["sigma"]),
                        # np.power(2.0, _row_on["mu"] - _row_on["sigma"]),
                    ],
                    _ymin,
                    _ymax,  # [_ymax, _ymax * 0.75, _ymax * 0.75],
                    color="#1d4ed8",
                )
                # """
                _y_sig = _ymax * 0.72
                _ax.hlines(
                    _y_sig,
                    np.power(2.0, _row_on["mu"] - _row_on["sigma"]),
                    np.power(2.0, _row_on["mu"] + _row_on["sigma"]),
                    color="#1d4ed8",
                    linestyles="dotted",
                )
                # """
                _ax.set_xscale("log")
                _ax.set_xlabel("Frequency (Hz, log scale)")
                _ax.set_ylabel("Firing Rate")
                _custom_ticks = [
                    np.min(uniq_Freq),
                    np.median(uniq_Freq),
                    np.max(uniq_Freq),
                ]
                _ax.set_xticks(_custom_ticks)
                _formatter = ticker.ScalarFormatter()
                _formatter.set_scientific(False)
                _formatter.set_useOffset(False)
                _ax.xaxis.set_major_formatter(_formatter)
                _ax.set_title(
                    f"Cell {_cell} ({_group_label})  R² off={_row_off['r2']:.2f}, on={_row_on['r2']:.2f}"
                )
                _ax.legend(frameon=False)
                _ax.spines["top"].set_visible(False)
                _ax.spines["right"].set_visible(False)
                _fig.tight_layout()
                _pdf.savefig(_fig)
                plt.close(_fig)
    print(f"Saved all cell fits to {_pdf_filename}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Plot histogram of mu/sigmas from fits
    """)
    return


@app.cell
def _(PV_results_off, PV_results_on, SST_results_off, SST_results_on, plt):
    def plot_fit_histograms(results, label, bins_mu=30, bins_sigma=30):
        """
        Plot histograms of mu and sigma from Gaussian fits.

        Parameters
        ----------
        results : pd.DataFrame
            DataFrame with at least 'mu' and 'sigma' columns (mu is in octaves).
        label : str
            Label for the condition (e.g., 'LED Off', 'LED On').
        bins_mu, bins_sigma : int
            Number of bins for histograms.
        """
        mu_hz = results["mu"].dropna()
        sigma = results["sigma"].dropna()
        _fig, axes = plt.subplots(1, 2, figsize=(8, 3))
        axes[0].hist(
            mu_hz, bins=bins_mu, color="gray", edgecolor="black", alpha=0.7
        )
        axes[0].set_xscale("log")
        axes[0].set_xlabel("μ (Hz)")
        axes[0].set_ylabel("Count")
        axes[0].set_title(f"{label}: μ distribution")
        axes[1].hist(
            sigma, bins=bins_sigma, color="gray", edgecolor="black", alpha=0.7
        )
        axes[1].set_xlabel("σ (octaves)")
        axes[1].set_ylabel("Count")
        axes[1].set_title(f"{label}: σ distribution")
        _fig.tight_layout()
        plt.show()


    plot_fit_histograms(PV_results_off, "PV LED Off")
    plot_fit_histograms(PV_results_on, "PV LED On")
    plot_fit_histograms(SST_results_off, "SST LED Off")
    plot_fit_histograms(SST_results_on, "SST LED On")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Plotting mu and sigma changes for suppressed/facilitated cells
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Try out fit params
    """)
    return


@app.cell
def _(np, pd):
    def summarize_good_fits(
        results_off: pd.DataFrame,
        results_on: pd.DataFrame,
        facilitated_idx: list,
        suppressed_idx: list,
        uniq_freq: np.ndarray,
        label: str,
        r2_min: float = 0.6,
        sigma_cap_oct: float = 4.0,
        pad_oct: float = 0.25,
        min_sigma: float = 1e-06,
        only_success: bool = True,
    ):
        """
        Summarize how many cells in each group have good Gaussian fits,
        with filtering consistently in octaves.
        """
        octaves = np.log2(np.asarray(uniq_freq, dtype=float))
        lo_oct = float(octaves.min() - pad_oct)
        hi_oct = float(octaves.max() + pad_oct)

        def count_good(df, idx):
            subset = df[df["cell"].isin(idx)].copy()
            mask = subset["r2"] > r2_min
            if only_success and "success" in subset.columns:
                mask = mask & subset["success"].fillna(False)
            mask = mask & subset["mu"].between(lo_oct, hi_oct)
            mask = mask & (
                (subset["sigma"] > min_sigma) & (subset["sigma"] <= sigma_cap_oct)
            )
            subset = subset[mask]
            return (len(subset), len(df[df["cell"].isin(idx)]))

        print(f"\n=== {label} Summary ===")
        off_good, off_total = count_good(results_off, facilitated_idx)
        on_good, on_total = count_good(results_on, facilitated_idx)
        print(
            f"Facilitated: LED Off {off_good}/{off_total},  LED On {on_good}/{on_total}"
        )  # tested range (Hz -> octaves)
        off_good, off_total = count_good(results_off, suppressed_idx)
        on_good, on_total = count_good(results_on, suppressed_idx)
        print(
            f"Suppressed:  LED Off {off_good}/{off_total},  LED On {on_good}/{on_total}"
        )  # Apply constraints  # Mu in octaves  # Sigma constraints (octaves)  # Facilitated  # Suppressed
    return (summarize_good_fits,)


@app.cell
def _(
    PV_facilitated_Ind,
    PV_results_off,
    PV_results_on,
    PV_suppressed_Ind,
    SST_facilitated_Ind,
    SST_results_off,
    SST_results_on,
    SST_suppressed_Ind,
    summarize_good_fits,
    uniq_Freq,
):
    summarize_good_fits(
        results_off=PV_results_off,
        results_on=PV_results_on,
        facilitated_idx=PV_facilitated_Ind,
        suppressed_idx=PV_suppressed_Ind,
        uniq_freq=uniq_Freq,  # your tested freqs in Hz
        label="PV",
        r2_min=0.6,
        sigma_cap_oct=4.0,
        pad_oct=0.25,
        min_sigma=1e-6,
        only_success=True,
    )

    summarize_good_fits(
        results_off=SST_results_off,
        results_on=SST_results_on,
        facilitated_idx=SST_facilitated_Ind,
        suppressed_idx=SST_suppressed_Ind,
        uniq_freq=uniq_Freq,  # your tested freqs in Hz
        label="SST",
        r2_min=0.6,
        sigma_cap_oct=4.0,
        pad_oct=0.25,
        min_sigma=1e-6,
        only_success=True,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Inspect specific group
    """)
    return


@app.cell
def _(np, pd):
    def inspect_group(
        results_df: pd.DataFrame,
        cell_idx: list,
        uniq_freq,
        label: str,
        r2_min: float = 0.6,
        sigma_cap_oct: float = 4.0,
        pad_oct: float = 0.25,
        min_sigma: float = 1e-06,
        only_success: bool = True,
    ):
        """
        Print detailed info (R², μ, σ) for a group of cells under a single condition.

        Parameters
        ----------
        results_df : pd.DataFrame
            The results table (either LED off or LED on).
            Must have columns ['cell','r2','mu','sigma','success'].
            Note: 'mu' should be in OCTAVES.
        cell_idx : list
            Cell indices (e.g., SST_facilitated_Ind).
        uniq_freq : array
            Tested freqs in Hz.
        label : str
            Label for this group (e.g. "SST facilitated, LED On").
        Other params
            Thresholds for filtering.
        """
        octaves = np.log2(np.asarray(uniq_freq, dtype=float))
        lo_oct = float(octaves.min() - pad_oct)
        hi_oct = float(octaves.max() + pad_oct)
        subset = results_df[results_df["cell"].isin(cell_idx)].copy()
        mask = subset["r2"] > r2_min
        if only_success and "success" in subset.columns:
            mask = mask & subset["success"].fillna(False)
        mask = mask & subset["mu"].between(lo_oct, hi_oct)
        mask = mask & (
            (subset["sigma"] > min_sigma) & (subset["sigma"] <= sigma_cap_oct)
        )
        kept = subset.loc[mask]
        dropped = subset.loc[~mask]
        print(f"\n=== {label} ===")
        print(f"Total cells in group: {len(subset)}")
        print(f"Kept (pass filters): {len(kept)}")
        print(f"Dropped (fail filters): {len(dropped)}")
        if not kept.empty:  # Subset
            print("\n-- Kept cells --")
            print(kept[["cell", "r2", "mu", "sigma"]].to_string(index=False))
        if not dropped.empty:  # Constraints
            print("\n-- Dropped cells --")
            print(dropped[["cell", "r2", "mu", "sigma"]].to_string(index=False))
        return (kept, dropped)
    return (inspect_group,)


@app.cell
def _(PV_facilitated_Ind, PV_results_on, inspect_group, uniq_Freq):
    inspect_group(
        results_df=PV_results_on,
        cell_idx=PV_facilitated_Ind,
        uniq_freq=uniq_Freq,
        label="PV suppressed (LED On)",
    )
    return


@app.cell
def _(PV_facilitated_Ind, PV_results_off, inspect_group, uniq_Freq):
    inspect_group(
        results_df=PV_results_off,
        cell_idx=PV_facilitated_Ind,
        uniq_freq=uniq_Freq,
        label="PV suppressed (LED Off)",
    )
    return


@app.cell
def _(SST_facilitated_Ind, SST_results_on, inspect_group, uniq_Freq):
    inspect_group(
        results_df=SST_results_on,
        cell_idx=SST_facilitated_Ind,
        uniq_freq=uniq_Freq,
        label="SST suppressed (LED On)",
    )
    return


@app.cell
def _(SST_facilitated_Ind, SST_results_off, inspect_group, uniq_Freq):
    # Inspect SST facilitated, LED on
    inspect_group(
        results_df=SST_results_off,
        cell_idx=SST_facilitated_Ind,
        uniq_freq=uniq_Freq,
        label="SST suppressed (LED Off)",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # shared plots
    """)
    return


@app.cell
def _(np, uniq_Freq):
    # Axis stuff
    # middle frequency tested
    print(uniq_Freq)
    print(np.round(np.median(uniq_Freq)))
    return


@app.cell
def _(mpl, np, pd, plt, uniq_Freq):
    from scipy.stats import ttest_rel, wilcoxon
    from matplotlib.ticker import ScalarFormatter


    def build_mu_sigma_subset(
        results_off: pd.DataFrame,
        results_on: pd.DataFrame,
        uniq_freq,
        r2_min: float = 0.6,
        sigma_cap_oct: float = 8.0,
        only_success: bool = True,
        pad_oct: float = 0.25,
        min_sigma: float = 1e-06,
    ) -> pd.DataFrame:
        """
        Return a merged DataFrame with the SAME filtered cells for μ and σ plots.
        Keeps rows where BOTH conditions satisfy:
          - r2 > r2_min
          - (optionally) success == True
          - mu within tested octave range (±pad_oct)
          - 0 < sigma <= sigma_cap_oct  # array of tested freqs in Hz
        """
        df = (
            results_off[["cell", "mu", "sigma", "r2", "success"]]
            .rename(
                columns={
                    "mu": "mu_off",
                    "sigma": "sigma_off",
                    "r2": "r2_off",
                    "success": "success_off",
                }
            )
            .merge(
                results_on[["cell", "mu", "sigma", "r2", "success"]].rename(
                    columns={
                        "mu": "mu_on",
                        "sigma": "sigma_on",
                        "r2": "r2_on",
                        "success": "success_on",
                    }
                ),
                on="cell",
                how="inner",
            )
        )
        octaves = np.log2(np.asarray(uniq_freq, dtype=float))
        lo_oct = float(octaves.min() - pad_oct)
        hi_oct = float(octaves.max() + pad_oct)
        mask = (df["r2_off"] > r2_min) & (df["r2_on"] > r2_min)
        if only_success:
            mask = mask & (
                df["success_off"].fillna(False) & df["success_on"].fillna(False)
            )
        mask = mask & (
            df["mu_off"].between(lo_oct, hi_oct)
            & df["mu_on"].between(lo_oct, hi_oct)
        )
        mask = mask & (
            (df["sigma_off"] > min_sigma) & (df["sigma_off"] <= sigma_cap_oct)
        )
        mask = mask & (
            (df["sigma_on"] > min_sigma) & (df["sigma_on"] <= sigma_cap_oct)
        )
        df_sub = df.loc[mask].copy()
        df_sub["mu_off_Hz"] = np.power(2.0, df_sub["mu_off"])
        df_sub["mu_on_Hz"] = np.power(2.0, df_sub["mu_on"])
        lo_hz, hi_hz = (float(2**lo_oct), float(2**hi_oct))
        df_sub.attrs["limits"] = dict(
            lo_oct=lo_oct, hi_oct=hi_oct, lo_hz=lo_hz, hi_hz=hi_hz
        )
        print(
            f"Subset: kept {len(df_sub)} / {len(df)} cells (R^2>{r2_min}, 0<σ≤{sigma_cap_oct:.2f} oct{(', success only' if only_success else '')})."
        )
        return df_sub


    def plot_mu_from_subset(
        df_sub: pd.DataFrame,
        title: str,
        pdf_path: str | None = None,
        font_family: str = "Arial",
        font_size: int = 10,
    ):
        """Scatter of μ in Hz (log-log) using the shared filtered subset."""
        lims = df_sub.attrs.get("limits", {})
        lo_hz = lims.get(
            "lo_hz", float(df_sub[["mu_off_Hz", "mu_on_Hz"]].min().min())
        )
        hi_hz = lims.get(
            "hi_hz", float(df_sub[["mu_off_Hz", "mu_on_Hz"]].max().max())
        )
        with mpl.rc_context(
            {
                "font.family": font_family,
                "font.size": font_size,
                "pdf.fonttype": 42,
                "ps.fonttype": 42,
            }
        ):
            _fig, _ax = plt.subplots(figsize=(3, 3))
            _ax.scatter(
                df_sub["mu_off_Hz"],
                df_sub["mu_on_Hz"],
                s=40,
                alpha=0.4,
                color="black",
                edgecolors="none",
            )
            _ax.plot([lo_hz, hi_hz], [lo_hz, hi_hz], "--", color="gray", lw=1)
            _ax.set_xlim(lo_hz, hi_hz)
            _ax.set_ylim(lo_hz, hi_hz)
            _ax.set_aspect("equal", adjustable="box")
            _ax.set_xlabel("μ Light Off (Hz)")
            _ax.set_ylabel("μ Light On (Hz)")
            _ax.set_title(title)
            _fig.tight_layout()
            _ax.set_xscale("log")
            _ax.set_yscale("log")
            _ax.tick_params(axis="x", which="minor", bottom=False)
            _ax.tick_params(axis="y", which="minor", left=False)
            _custom_ticks = [
                np.min(uniq_Freq),
                np.median(uniq_Freq),
                np.max(uniq_Freq),
            ]
            _ax.set_xticks(_custom_ticks)
            _ax.set_yticks(_custom_ticks)
            _formatter = ScalarFormatter()
            _formatter.set_scientific(False)
            _formatter.set_useOffset(False)
            _ax.xaxis.set_major_formatter(_formatter)
            _ax.yaxis.set_major_formatter(_formatter)
            if pdf_path:
                _fig.savefig(pdf_path, format="pdf")
            plt.show()
            paired = df_sub[["mu_off_Hz", "mu_on_Hz"]].dropna()
            x = paired["mu_off_Hz"]
            y = paired["mu_on_Hz"]
            t_stat, p_val = ttest_rel(x, y)
            print(f"Paired t-test: t = {t_stat:.3f}, p = {p_val:.3e}")
            if p_val < 0.05:
                print("*")


    def plot_sigma_from_subset(
        df_sub: pd.DataFrame,
        title: str,
        pdf_path: str | None = None,
        font_family: str = "Arial",
        font_size: int = 10,
        as_fwhm: bool = False,
    ):
        """Scatter of σ (octaves) or FWHM (octaves) using the same filtered subset."""
        with mpl.rc_context(
            {
                "font.family": font_family,
                "font.size": font_size,
                "pdf.fonttype": 42,
                "ps.fonttype": 42,
            }
        ):
            x = df_sub["sigma_off"].to_numpy(copy=True)
            y = df_sub["sigma_on"].to_numpy(copy=True)
            label = "σ (octaves)"
            if as_fwhm:
                factor = 2.354820045
                x = x * factor
                y = y * factor
                label = "FWHM (octaves)"
            lims = df_sub.attrs.get("limits", {})
            max_val = float(max(x.max(), y.max()))
            lim0 = 0.0
            lim1 = 2.75
            _fig, _ax = plt.subplots(figsize=(3, 3))
            _ax.scatter(x, y, s=40, alpha=0.4, color="black", edgecolors="none")
            _ax.plot([lim0, lim1], [lim0, lim1], "--", color="gray", lw=1)
            _ax.set_xlim(lim0, lim1)
            _ax.set_aspect("equal", adjustable="box")
            _ax.set_xlabel("σ Light Off (octaves)")
            _ax.set_ylabel("σ Light On (octaves)")
            _ax.set_title(title)
            _fig.tight_layout()
            if pdf_path:
                _fig.savefig(pdf_path, format="pdf")
            plt.show()
            t_stat, p_val = ttest_rel(x, y)
            print(f"Paired t-test: t = {t_stat:.3f}, p = {p_val:.3e}")
            if p_val < 0.05:
                print("*")
    return (
        ScalarFormatter,
        build_mu_sigma_subset,
        plot_mu_from_subset,
        plot_sigma_from_subset,
        ttest_rel,
        wilcoxon,
    )


@app.cell
def _(ScalarFormatter, mpl, np, pd, plt, uniq_Freq):
    import seaborn as sns


    def plot_paired_bar_swarm(
        df_sub: pd.DataFrame,
        off_col: str,
        on_col: str,
        off_color: str,
        on_color: str,
        y_label: str,
        title: str,
        log_y: bool = False,
        pdf_path: str | None = None,
        font_family: str = "Arial",
        font_size: int = 10,
    ):
        """
        Bar (mean ± SE) with beeswarm overlay for paired OFF vs ON data.
        Expects df_sub to contain 'cell', plus the two value columns.
        """
        with mpl.rc_context(
            {
                "font.family": font_family,
                "font.size": font_size,
                "pdf.fonttype": 42,
                "ps.fonttype": 42,
            }
        ):
            long = df_sub[["cell", off_col, on_col]].melt(
                id_vars="cell", var_name="condition", value_name="value"
            )
            long["condition"] = long["condition"].map(
                {off_col: "Off", on_col: "On"}
            )
            _fig, _ax = plt.subplots(figsize=(1.5, 3))
            sns.boxplot(
                data=long,
                x="condition",
                y="value",
                palette={"Off": off_color, "On": on_color},
                fliersize=0,
                linewidth=1,
                ax=_ax,
            )
            sns.stripplot(
                data=long,
                x="condition",
                y="value",
                color="black",
                alpha=0.5,
                size=5,
                jitter=True,
                dodge=True,
                edgecolor="none",
                ax=_ax,
            )
            if log_y:
                _ax.set_yscale("log")
                lims = df_sub.attrs.get("limits", {})
                lo_hz, hi_hz = (lims.get("lo_hz", None), lims.get("hi_hz", None))
                if lo_hz and hi_hz:
                    _ax.set_ylim(lo_hz, hi_hz)
                    _ax.set_yscale("log")
                    _ax.tick_params(axis="y", which="minor", left=False)
                    _custom_ticks = [
                        np.min(uniq_Freq),
                        np.median(uniq_Freq),
                        np.max(uniq_Freq),
                    ]
                    _ax.set_yticks(_custom_ticks)
                    _formatter = ScalarFormatter()
                    _formatter.set_scientific(False)
                    _formatter.set_useOffset(False)
                    _ax.yaxis.set_major_formatter(_formatter)
            _ax.set_xlabel("")
            _ax.set_ylabel(y_label)
            _ax.set_title(title)
            _fig.tight_layout()
            if pdf_path:
                _fig.savefig(pdf_path, format="pdf")
            plt.show()
    return (plot_paired_bar_swarm,)


@app.cell
def _(
    PV_facilitated_Ind,
    PV_results_off,
    PV_results_on,
    PV_suppressed_Ind,
    SST_facilitated_Ind,
    SST_results_off,
    SST_results_on,
    SST_suppressed_Ind,
    build_mu_sigma_subset,
    plot_mu_from_subset,
    plot_paired_bar_swarm,
    plot_sigma_from_subset,
    uniq_Freq,
):
    # PV facilitated group
    pv_fac_off = PV_results_off[PV_results_off["cell"].isin(PV_facilitated_Ind)]
    pv_fac_on = PV_results_on[PV_results_on["cell"].isin(PV_facilitated_Ind)]

    pv_fac_subset = build_mu_sigma_subset(
        pv_fac_off,
        pv_fac_on,
        uniq_Freq,
        r2_min=0.6,
        sigma_cap_oct=4,
        only_success=True,
        pad_oct=0.25,
    )

    plot_mu_from_subset(
        pv_fac_subset, title="PV Facilitated — μ (Hz)", pdf_path="PV_fac_mu.pdf"
    )
    plot_sigma_from_subset(
        pv_fac_subset,
        title="PV Facilitated — σ (octaves)",
        pdf_path="PV_fac_sigma.pdf",
    )

    plot_paired_bar_swarm(
        pv_fac_subset,
        off_col="mu_off_Hz",
        on_col="mu_on_Hz",
        off_color="#002c94",
        on_color="CornflowerBlue",
        y_label="μ (Hz)",
        title="PV Facilitated mu",
        log_y=True,
        pdf_path="PV_Facilitated_mu_bar_swarm.pdf",
    )

    plot_paired_bar_swarm(
        pv_fac_subset,
        off_col="sigma_off",
        on_col="sigma_on",
        off_color="#002c94",
        on_color="CornflowerBlue",
        y_label="σ (octaves)",
        title="PV Facilitated sigma",
        log_y=False,
        pdf_path="PV_Facilitated_sigma_bar_swarm.pdf",
    )

    # PV suppressed group
    pv_sup_off = PV_results_off[PV_results_off["cell"].isin(PV_suppressed_Ind)]
    pv_sup_on = PV_results_on[PV_results_on["cell"].isin(PV_suppressed_Ind)]
    pv_sup_subset = build_mu_sigma_subset(
        pv_sup_off,
        pv_sup_on,
        uniq_Freq,
        r2_min=0.6,
        sigma_cap_oct=4,
        only_success=True,
        pad_oct=0.25,
    )
    plot_mu_from_subset(
        pv_sup_subset, title="PV Suppressed — μ (Hz)", pdf_path="PV_sup_mu.pdf"
    )
    plot_sigma_from_subset(
        pv_sup_subset,
        title="PV Suppressed — σ (octaves)",
        pdf_path="PV_sup_sigma.pdf",
    )

    plot_paired_bar_swarm(
        pv_sup_subset,
        off_col="mu_off_Hz",
        on_col="mu_on_Hz",
        off_color="#A94C54",
        on_color="CornflowerBlue",
        y_label="μ (Hz)",
        title="PV Suppressed mu",
        log_y=True,
        pdf_path="PV_Suppressed_mu_bar_swarm.pdf",
    )

    plot_paired_bar_swarm(
        pv_sup_subset,
        off_col="sigma_off",
        on_col="sigma_on",
        off_color="#A94C54",
        on_color="CornflowerBlue",
        y_label="σ (octaves)",
        title="PV Suppressed sigma",
        log_y=False,
        pdf_path="PV_Suppressed_sigma_bar_swarm.pdf",
    )

    # SST facilitated group
    sst_fac_off = SST_results_off[
        SST_results_off["cell"].isin(SST_facilitated_Ind)
    ]
    sst_fac_on = SST_results_on[SST_results_on["cell"].isin(SST_facilitated_Ind)]
    sst_fac_subset = build_mu_sigma_subset(
        sst_fac_off,
        sst_fac_on,
        uniq_Freq,
        r2_min=0.6,
        sigma_cap_oct=4,
        only_success=True,
        pad_oct=0.25,
    )
    plot_mu_from_subset(
        sst_fac_subset, title="SST Facilitated — μ (Hz)", pdf_path="SST_fac_mu.pdf"
    )
    """
    plot_sigma_from_subset(
        sst_fac_subset,
        title="SST Facilitated — σ (octaves)",
        pdf_path="SST_fac_sigma.pdf",
    )
    """

    plot_paired_bar_swarm(
        sst_fac_subset,
        off_col="mu_off_Hz",
        on_col="mu_on_Hz",
        off_color="#002c94",
        on_color="CornflowerBlue",
        y_label="μ (Hz)",
        title="SST Facilitated mu",
        log_y=True,
        pdf_path="SST_Facilitated_mu_bar_swarm.pdf",
    )

    plot_paired_bar_swarm(
        sst_fac_subset,
        off_col="sigma_off",
        on_col="sigma_on",
        off_color="#002c94",
        on_color="CornflowerBlue",
        y_label="σ (octaves)",
        title="SST Facilitated sigma",
        log_y=False,
        pdf_path="SST_Facilitated_sigma_bar_swarm.pdf",
    )

    # SST suppressed group
    sst_sup_off = SST_results_off[SST_results_off["cell"].isin(SST_suppressed_Ind)]
    sst_sup_on = SST_results_on[SST_results_on["cell"].isin(SST_suppressed_Ind)]
    sst_sup_subset = build_mu_sigma_subset(
        sst_sup_off,
        sst_sup_on,
        uniq_Freq,
        r2_min=0.6,
        sigma_cap_oct=4,
        only_success=True,
        pad_oct=0.25,
    )
    plot_mu_from_subset(
        sst_sup_subset, title="SST Suppressed — μ (Hz)", pdf_path="SST_sup_mu.pdf"
    )
    plot_sigma_from_subset(
        sst_sup_subset,
        title="SST Suppressed — σ (octaves)",
        pdf_path="SST_sup_sigma.pdf",
    )

    plot_paired_bar_swarm(
        sst_sup_subset,
        off_col="mu_off_Hz",
        on_col="mu_on_Hz",
        off_color="#A94C54",
        on_color="CornflowerBlue",
        y_label="μ (Hz)",
        title="SST Suppressed mu",
        log_y=True,
        pdf_path="SST_Suppressed_mu_bar_swarm.pdf",
    )

    plot_paired_bar_swarm(
        sst_sup_subset,
        off_col="sigma_off",
        on_col="sigma_on",
        off_color="#A94C54",
        on_color="CornflowerBlue",
        y_label="σ (octaves)",
        title="SST Suppressed sigma",
        log_y=False,
        pdf_path="SST_Suppressed_sigma_bar_swarm.pdf",
    )
    return pv_fac_subset, pv_sup_subset, sst_fac_subset, sst_sup_subset


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Summarize the fits
    """)
    return


@app.cell
def _(
    PV_facilitated_Ind,
    PV_results_off,
    PV_results_on,
    PV_suppressed_Ind,
    SST_facilitated_Ind,
    SST_results_off,
    SST_results_on,
    SST_suppressed_Ind,
    np,
):
    def summarize_good_fits_1(
        results_off, results_on, facilitated_idx, suppressed_idx, label
    ):
        """
        Summarize how many cells in a condition had good Gaussian fits (R² > 0.6).

        Parameters
        ----------
        results_off : pd.DataFrame
            Results dataframe for LED off.
        results_on : pd.DataFrame
            Results dataframe for LED on.
        facilitated_idx : list
            Cell indices for facilitated group.
        suppressed_idx : list
            Cell indices for suppressed group.
        label : str
            Label for this population (e.g. "PV", "SST")
        """

        def count_good(df, idx):
            subset = df[df["cell"].isin(idx)]
            n_total = len(subset)
            n_good = np.sum(subset["r2"] > 0.6)
            return (n_good, n_total)

        print(f"\n=== {label} Summary ===")
        off_good, off_total = count_good(results_off, facilitated_idx)
        on_good, on_total = count_good(results_on, facilitated_idx)
        print(
            f"Facilitated:  LED Off {off_good}/{off_total},  LED On {on_good}/{on_total}"
        )  # Facilitated
        off_good, off_total = count_good(results_off, suppressed_idx)
        on_good, on_total = count_good(results_on, suppressed_idx)
        print(
            f"Suppressed:  LED Off {off_good}/{off_total},  LED On {on_good}/{on_total}"
        )


    summarize_good_fits_1(
        PV_results_off,
        PV_results_on,
        PV_facilitated_Ind,
        PV_suppressed_Ind,
        label="PV",
    )
    # Example usage:
    summarize_good_fits_1(
        SST_results_off,
        SST_results_on,
        SST_facilitated_Ind,
        SST_suppressed_Ind,
        label="SST",
    )  # Suppressed
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Compare statistical tests
    """)
    return


@app.cell
def _(np, pd, ttest_rel, wilcoxon):
    def _prep_pairs(df, off_col, on_col):
        """Return paired arrays x (off), y (on) with NaNs/Infs removed and aligned."""
        pairs = df[[off_col, on_col]].replace([np.inf, -np.inf], np.nan).dropna()
        x = pairs[off_col].to_numpy()
        y = pairs[on_col].to_numpy()
        return (x, y)


    def _paired_effect_sizes(x, y):
        """Cohen's dz for paired data and summary diffs."""
        d = y - x
        n = d.size
        mean_diff = float(np.mean(d)) if n else np.nan
        median_diff = float(np.median(d)) if n else np.nan
        sd_diff = float(np.std(d, ddof=1)) if n > 1 else np.nan
        dz = (
            mean_diff / sd_diff
            if sd_diff and np.isfinite(sd_diff) and (sd_diff > 0)
            else np.nan
        )
        return (n, mean_diff, median_diff, dz)


    def paired_tests_table(
        groups: dict, off_col: str, on_col: str
    ) -> pd.DataFrame:
        """
        groups: dict like {"PV suppressed": df_sub_sup, "PV facilitated": df_sub_fac, ...}
                Each df must contain off_col & on_col (e.g., 'mu_off_Hz','mu_on_Hz').
        Returns a DataFrame with t-test and Wilcoxon results per group.
        """
        rows = []
        for name, df in groups.items():
            x, y = _prep_pairs(df, off_col, on_col)
            n, mean_diff, median_diff, dz = _paired_effect_sizes(x, y)
            t_stat = t_p = w_stat = w_p = np.nan
            if n >= 2:
                t_stat, t_p = ttest_rel(x, y, nan_policy="omit")
                try:
                    w_stat, w_p = wilcoxon(
                        x,
                        y,
                        zero_method="wilcox",
                        alternative="two-sided",
                        method="auto",
                    )
                except ValueError:  # default placeholders
                    w_stat, w_p = (np.nan, np.nan)
            rows.append(
                {
                    "group": name,
                    "n_pairs": n,
                    "mean_diff": mean_diff,
                    "median_diff": median_diff,
                    "cohens_dz": dz,
                    "t_stat": t_stat,
                    "t_p": t_p,
                    "W_stat": w_stat,
                    "W_p": w_p,
                }
            )
        table = pd.DataFrame(rows)
        table = table[
            [
                "group",
                "n_pairs",
                "mean_diff",
                "median_diff",
                "cohens_dz",
                "t_stat",
                "t_p",
                "W_stat",
                "W_p",
            ]
        ]  # Paired t-test
        return table  # Wilcoxon signed-rank (paired, nonparametric)  # Handle the case where all diffs are zero (scipy raises ValueError)  # zero_method='wilcox' ignores zero-diffs; two-sided by default  # on - off  # Optional: order columns nicely
    return (paired_tests_table,)


@app.cell
def _(
    paired_tests_table,
    pv_fac_subset,
    pv_sup_subset,
    sst_fac_subset,
    sst_sup_subset,
):
    groups_mu = {
        "PV suppressed": pv_sup_subset,  # has columns 'mu_off_Hz','mu_on_Hz'
        "PV facilitated": pv_fac_subset,
        "SST suppressed": sst_sup_subset,
        "SST facilitated": sst_fac_subset,
    }

    mu_table = paired_tests_table(
        groups_mu, off_col="mu_off_Hz", on_col="mu_on_Hz"
    )
    print(mu_table)

    # For sigma (octaves):
    groups_sigma = {
        "PV suppressed": pv_sup_subset,
        "PV facilitated": pv_fac_subset,
        "SST suppressed": sst_sup_subset,
        "SST facilitated": sst_fac_subset,
    }
    sigma_table = paired_tests_table(
        groups_sigma, off_col="sigma_off", on_col="sigma_on"
    )
    print(sigma_table)

    # Save to Excel
    mu_table.to_excel("mu_comparison_table.xlsx", index=False)
    sigma_table.to_excel("sigma_comparison_table.xlsx", index=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Plot FRFs for each group
    """)
    return


@app.cell
def _(mpl, np, plt):
    def plot_centered_rf_by_mu(
        mTCOff,
        mTCOn,
        uniq_Freq,
        df_subset,
        title="",
        pdf_path=None,
        align_to="off",
        off_color="black",
        on_color="CornflowerBlue",
    ):
        """
        Center each cell's tuning curve on its fitted mu (from df_subset),
        then plot mean OFF vs ON with SEM shading.

        df_subset must have columns: 'cell', 'mu_off_Hz', 'mu_on_Hz'
        mTCOff / mTCOn shape: (n_freq, n_cells) or (n_cells, n_freq)
        uniq_Freq: array of tested freqs in Hz (log-spaced preferred)
        align_to: 'off' (use mu_off_Hz) or 'on' (use mu_on_Hz)
        """
        if mTCOff.shape[0] != len(uniq_Freq):
            mTCOff = mTCOff.T
        if mTCOn.shape[0] != len(uniq_Freq):
            mTCOn = mTCOn.T
        cells = df_subset["cell"].to_numpy(dtype=int)
        mu_hz = (
            df_subset["mu_off_Hz"] if align_to == "off" else df_subset["mu_on_Hz"]
        ).to_numpy(dtype=float)
        logF = np.log2(np.asarray(uniq_Freq, dtype=float))
        step = np.median(np.diff(logF))
        nF = len(logF)
        half = (nF - 1) / 2.0
        x_oct = (np.arange(nF) - half) * step
        off_centered = []
        on_centered = []
        for c, mu in zip(cells, mu_hz):
            x_cell = logF - np.log2(mu)
            y_off = mTCOff[:, c].astype(float)
            y_on = mTCOn[:, c].astype(float)
            y_off_c = np.interp(x_oct, x_cell, y_off, left=np.nan, right=np.nan)
            y_on_c = np.interp(x_oct, x_cell, y_on, left=np.nan, right=np.nan)
            off_centered.append(y_off_c)
            on_centered.append(y_on_c)
        off_centered = np.vstack(off_centered)
        on_centered = np.vstack(on_centered)

        def nansem(a, axis=0):
            n = np.sum(~np.isnan(a), axis=axis)
            return np.nanstd(a, axis=axis, ddof=1) / np.sqrt(np.maximum(n, 1))

        def norm_percell(FR_arr):
            out = np.array([])
            for cell_FRs in FR_arr.T:
                maxFR = np.nanmax(cell_FRs)
                FR = cell_FRs / maxFR
                out = np.vstack([out, FR]) if out.size else FR

            return out.T

        mean_off = np.nanmean(off_centered, axis=0)
        sem_off = nansem(off_centered, axis=0)
        mean_on = np.nanmean(on_centered, axis=0)
        sem_on = nansem(on_centered, axis=0)

        # Normalized
        off_centered_norm = norm_percell(off_centered)
        on_centered_norm = norm_percell(on_centered)

        mean_off_norm = np.nanmean(off_centered_norm, axis=0)
        sem_off_norm = nansem(off_centered_norm, axis=0)
        mean_on_norm = np.nanmean(on_centered_norm, axis=0)
        sem_on_norm = nansem(on_centered_norm, axis=0)

        # shift to baseline
        """
        mean_off_norm = mean_off_norm - mean_off_norm.min()
        sem_off_norm = sem_off_norm - mean_off_norm.min()
        mean_on_norm = mean_on_norm - mean_on_norm.min()
        sem_on_norm = sem_on_norm - mean_on_norm.min()
        """

        with mpl.rc_context(
            {
                "font.family": "Arial",
                "font.size": 12,
                "pdf.fonttype": 42,
                "ps.fonttype": 42,
            }
        ):
            _fig, _ax = plt.subplots(figsize=(2.2, 2.2))
            _ax.plot(x_oct, mean_off, label="Off", color=off_color, lw=0.5)
            _ax.fill_between(
                x_oct,
                mean_off - sem_off,
                mean_off + sem_off,
                alpha=0.25,
                color=off_color,
            )
            _ax.plot(x_oct, mean_on, label="On", color=on_color, lw=0.5)
            _ax.fill_between(
                x_oct,
                mean_on - sem_on,
                mean_on + sem_on,
                alpha=0.25,
                color=on_color,
            )
            _ax.axvline(0, lw=1, ls="--", color="k", alpha=0.5)
            _ax.set_xlabel("Octaves from μ")
            # _ax.set_xticks([])
            _ax.set_ylabel("Firing rate (Hz)")
            # _ax.set_yticks([])
            _ax.set_title(title)
            _ax.legend(frameon=False)
            # _ax.legend().set_visible(False)
            _fig.tight_layout()
            if pdf_path:
                _fig.savefig(pdf_path, format="pdf")
            plt.show()
    return (plot_centered_rf_by_mu,)


@app.cell
def _(
    PV_mTCOff,
    PV_mTCOn,
    SST_mTCOff,
    SST_mTCOn,
    plot_centered_rf_by_mu,
    pv_fac_subset,
    pv_sup_subset,
    sst_sup_subset,
    uniq_Freq,
):
    plot_centered_rf_by_mu(
        PV_mTCOff,
        PV_mTCOn,
        uniq_Freq,
        pv_fac_subset,
        title="PV Facilitated",
        pdf_path="PV_fac_centeredRF_byMu.pdf",
        align_to="off",  # or "on" to center by mu_on_Hz
    )

    plot_centered_rf_by_mu(
        PV_mTCOff,
        PV_mTCOn,
        uniq_Freq,
        pv_sup_subset,
        title="PV Suppressed",
        pdf_path="PV_supp_centeredRF_byMu.pdf",
        align_to="off",
    )
    """
    plot_centered_rf_by_mu(
        SST_mTCOff,
        SST_mTCOn,
        uniq_Freq,
        sst_fac_subset,
        title="SST Facilitated",
        pdf_path="SST_fac_centeredRF_byMu.pdf",
        align_to="off",
    )
    """

    plot_centered_rf_by_mu(
        SST_mTCOff,
        SST_mTCOn,
        uniq_Freq,
        sst_sup_subset,
        title="SST Suppressed",
        pdf_path="SST_supp_centeredRF_byMu.pdf",
        align_to="off",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Generate output for the stats table
    """)
    return


@app.cell
def _(np, pd, ttest_rel):
    from scipy.stats import t, iqr


    def _sem(sd: float, n: int) -> float:
        return sd / np.sqrt(n) if n > 0 else np.nan


    # ---------- helpers ----------


    def _moe(sem: float, n: int, alpha: float) -> float:
        if n <= 1 or np.isnan(sem):
            return np.nan
        tcrit = t.ppf(
            1 - alpha / 2, df=n - 1
        )  # 95% MoE by default (uses t critical with df=n-1)
        return tcrit * sem


    def _cohen_dz(diff: np.ndarray) -> float:
        n = diff.size
        if n <= 1:
            return np.nan
        sd = diff.std(
            ddof=1
        )  # dz = mean(diff) / sd(diff), equivalent to t / sqrt(n)
        return np.mean(diff) / sd if sd > 0 else np.nan


    def _hedges_gz_from_dz(dz: float, n: int) -> float:
        if n <= 2 or np.isnan(dz):
            return np.nan
        J = 1 - 3 / (4 * n - 5)
        return dz * J


    # small-sample correction for dz
    def summarize_paired(
        x: np.ndarray,
        y: np.ndarray,
        label: str,
        alpha: float = 0.05,
        decimals: int = 3,
    ) -> pd.DataFrame:
        """
        Summarize Off vs On with paired t-test.
        Returns a tidy 1-row DataFrame with condition summaries + test results.
        """
        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask].astype(float)
        y = y[mask].astype(float)
        n = x.size
        dfree = max(n - 1, 0)
        off_mean = float(np.mean(x)) if n else np.nan
        on_mean = (
            float(np.mean(y)) if n else np.nan
        )  # keep only finite paired observations
        off_median = float(np.median(x)) if n else np.nan
        on_median = float(np.median(y)) if n else np.nan
        off_sd = float(np.std(x, ddof=1)) if n > 1 else np.nan
        on_sd = float(np.std(y, ddof=1)) if n > 1 else np.nan
        off_iqr = float(iqr(x, rng=(25, 75))) if n else np.nan
        on_iqr = float(iqr(y, rng=(25, 75))) if n else np.nan
        off_sem = _sem(off_sd, n)  # per-condition descriptives
        on_sem = _sem(on_sd, n)
        off_moe = _moe(off_sem, n, alpha)
        on_moe = _moe(on_sem, n, alpha)
        diff = y - x
        mean_diff = float(np.mean(diff)) if n else np.nan
        median_diff = float(np.median(diff)) if n else np.nan
        sd_diff = float(np.std(diff, ddof=1)) if n > 1 else np.nan
        sem_diff = _sem(sd_diff, n)
        moe_diff = _moe(sem_diff, n, alpha)
        ci_lo_diff = mean_diff - moe_diff if np.isfinite(moe_diff) else np.nan
        ci_hi_diff = mean_diff + moe_diff if np.isfinite(moe_diff) else np.nan
        if n > 1:
            t_stat, p_val = ttest_rel(y, x, nan_policy="omit")
        else:  # paired differences (On - Off)
            t_stat, p_val = (np.nan, np.nan)
        dz = _cohen_dz(diff) if n > 1 else np.nan
        gz = _hedges_gz_from_dz(dz, n) if n > 1 else np.nan
        out = pd.DataFrame(
            [
                {
                    "measure": label,
                    "N_paired": n,
                    "off_mean": off_mean,
                    "off_median": off_median,
                    "off_sd": off_sd,
                    "off_iqr": off_iqr,
                    "off_sem": off_sem,
                    "off_moe_95": off_moe,
                    "on_mean": on_mean,
                    "on_median": on_median,
                    "on_sd": on_sd,
                    "on_iqr": on_iqr,
                    "on_sem": on_sem,
                    "on_moe_95": on_moe,
                    "mean_diff": mean_diff,
                    "median_diff": median_diff,
                    "sd_diff": sd_diff,
                    "sem_diff": sem_diff,
                    "moe_diff_95": moe_diff,
                    "ci95_diff_lo": ci_lo_diff,
                    "ci95_diff_hi": ci_hi_diff,
                    "t_stat": t_stat,
                    "df": dfree,
                    "p_value": p_val,
                    "cohen_dz": dz,
                    "hedges_gz": gz,
                }
            ]
        )
        num_cols = out.select_dtypes(include=[float, int]).columns
        out[num_cols] = out[num_cols].apply(lambda s: s.round(decimals))
        return out


    def stats_table_from_subset(
        df_sub: pd.DataFrame,
        value: str = "mu",
        as_fwhm: bool = False,
        alpha: float = 0.05,
        decimals: int = 3,
    ) -> pd.DataFrame:
        """# paired t-test
        Build a 1-row stats table for the chosen metric using your filtered df_sub.
        value="mu" uses columns 'mu_off_Hz' and 'mu_on_Hz'
        value="sigma" uses 'sigma_off' and 'sigma_on' (or FWHM if as_fwhm=True).
        """
        if value == "mu":
            x = df_sub["mu_off_Hz"].to_numpy()  # effect sizes (paired)
            y = df_sub["mu_on_Hz"].to_numpy()
            label = "mu (Hz)"
        elif value == "sigma":
            x = df_sub["sigma_off"].to_numpy()
            y = df_sub["sigma_on"].to_numpy()
            if as_fwhm:
                factor = 2.354820045  # Off
                x = x * factor
                y = y * factor
                label = "FWHM (oct)"  # On
            else:
                label = "sigma (oct)"
        else:  # Paired differences (On - Off)
            raise ValueError("value must be 'mu' or 'sigma'.")
        return summarize_paired(x, y, label=label, alpha=alpha, decimals=decimals)


    def stats_table_both_mu_and_sigma(
        df_sub: pd.DataFrame,
        include_fwhm: bool = False,
        alpha: float = 0.05,
        decimals: int = 3,
    ) -> pd.DataFrame:  # Test
        """Convenience wrapper: returns a multi-row table for mu and sigma (±FWHM)."""
        tables = [
            stats_table_from_subset(
                df_sub, value="mu", alpha=alpha, decimals=decimals
            ),
            stats_table_from_subset(
                df_sub,
                value="sigma",
                as_fwhm=False,
                alpha=alpha,
                decimals=decimals,
            ),
        ]  # Effect sizes
        if include_fwhm:
            tables.append(
                stats_table_from_subset(
                    df_sub,
                    value="sigma",
                    as_fwhm=True,
                    alpha=alpha,
                    decimals=decimals,
                )
            )
        # ---------- main entry points ----------
        return pd.concat(
            tables, ignore_index=True
        )  # optional rounding for manuscript readability  # "mu" or "sigma"  # only used when value == "sigma"  # FWHM = 2.354820045 * sigma
    return (stats_table_both_mu_and_sigma,)


@app.cell
def _(
    pd,
    pv_fac_subset,
    pv_sup_subset,
    sst_fac_subset,
    sst_sup_subset,
    stats_table_both_mu_and_sigma,
):
    all_groups_stats = pd.concat(
        [
            stats_table_both_mu_and_sigma(pv_fac_subset, include_fwhm=True).assign(
                group="PV Facilitated"
            ),
            stats_table_both_mu_and_sigma(pv_sup_subset, include_fwhm=True).assign(
                group="PV Suppressed"
            ),
            stats_table_both_mu_and_sigma(
                sst_fac_subset, include_fwhm=True
            ).assign(group="SST Facilitated"),
            stats_table_both_mu_and_sigma(
                sst_sup_subset, include_fwhm=True
            ).assign(group="SST Suppressed"),
        ],
        ignore_index=True,
    )

    print(all_groups_stats)

    # Save to CSV for manuscript
    all_groups_stats.to_csv("all_stats_table.csv", index=False)
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
