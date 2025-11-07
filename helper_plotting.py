import marimo

__generated_with = "0.18.0"
app = marimo.App(width="medium")

with app.setup:
    import numpy as np
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    import cmasher as cmr
    import pandas as pd
    import seaborn as sns
    from scipy.stats import ttest_rel
    from scipy.stats import t as student_t


@app.function
# FIXUP: wrapped these param declarations around a func to easily reuse in the figure modules.
def set_mpl_rcparams(black_color):
    # Global Plotting Parameters
    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.sans-serif"] = "Arial"
    mpl.rcParams["font.size"] = 10
    mpl.rcParams.update({"mathtext.default": "regular"})
    # mpl.rcParams['figure.figsize']: (0.67,0.5)
    mpl.rcParams["figure.titlesize"] = 10
    mpl.rcParams["figure.autolayout"]: True
    mpl.rcParams["figure.constrained_layout.use"] = True

    mpl.rcParams["savefig.dpi"] = 600
    mpl.rcParams["savefig.format"] = "pdf"
    mpl.rcParams["savefig.bbox"] = "tight"
    mpl.rcParams["savefig.pad_inches"] = 0
    mpl.rcParams["savefig.transparent"] = True

    mpl.rcParams["axes.linewidth"] = 0.5
    mpl.rcParams["axes.labelsize"] = 10
    mpl.rcParams["axes.labelpad"] = 2
    mpl.rcParams["xaxis.labellocation"] = "center"
    mpl.rcParams["yaxis.labellocation"] = "center"
    mpl.rcParams["axes.titlesize"] = 10
    mpl.rcParams["axes.titlepad"] = 4

    mpl.rcParams["lines.linewidth"] = 0.5

    mpl.rcParams["legend.loc"] = "upper left"
    mpl.rcParams["legend.frameon"] = False
    mpl.rcParams["legend.fontsize"] = 8

    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    mpl.rcParams["svg.fonttype"] = "none"

    mpl.rcParams["xtick.labelsize"] = 8
    mpl.rcParams["ytick.labelsize"] = 8

    mpl.rcParams["xtick.major.size"] = 2
    mpl.rcParams["xtick.major.width"] = 0.5
    mpl.rcParams["ytick.major.size"] = 2
    mpl.rcParams["ytick.major.width"] = 0.5
    mpl.rcParams["xtick.major.pad"] = 1
    mpl.rcParams["ytick.major.pad"] = 1

    mpl.rcParams["xtick.minor.size"] = 1
    mpl.rcParams["xtick.minor.width"] = 0.5
    mpl.rcParams["ytick.minor.size"] = 1
    mpl.rcParams["ytick.minor.width"] = 0.5
    mpl.rcParams["xtick.minor.pad"] = 1
    mpl.rcParams["ytick.minor.pad"] = 1

    mpl.rcParams["lines.color"] = black_color
    mpl.rcParams["patch.edgecolor"] = black_color
    mpl.rcParams["patch.force_edgecolor"] = False
    mpl.rcParams["boxplot.flierprops.color"] = black_color
    mpl.rcParams["boxplot.flierprops.markeredgecolor"] = black_color
    mpl.rcParams["boxplot.boxprops.color"] = black_color
    mpl.rcParams["boxplot.whiskerprops.color"] = black_color
    mpl.rcParams["boxplot.capprops.color"] = black_color
    mpl.rcParams["boxplot.capprops.linewidth"] = 0.5
    mpl.rcParams["text.color"] = black_color
    mpl.rcParams["axes.edgecolor"] = black_color
    mpl.rcParams["axes.labelcolor"] = black_color
    mpl.rcParams["xtick.color"] = black_color
    mpl.rcParams["ytick.color"] = black_color


@app.function
def plot_cells_psth(
    full_or_mean_psth_off,
    full_or_mean_psth_on,
    time_points,
    start_of_laser,
    start_of_stimulus,
    duration_of_stimulus,
    normalized_psth_off=None,
    normalized_psth_on=None,
    laser_only_psth=None,
    laser_only_duration=None,
    show_plots=True,
):
    if full_or_mean_psth_off.ndim == 3:
        mean_off_responses = full_or_mean_psth_off.mean(axis=0)
        mean_on_responses = full_or_mean_psth_on.mean(axis=0)
    else:
        mean_off_responses = full_or_mean_psth_off
        mean_on_responses = full_or_mean_psth_on

    if laser_only_psth is not None and laser_only_psth.ndim == 3:
        mean_laser_only_responses = laser_only_psth.mean(axis=0)
    elif laser_only_psth is not None:
        mean_laser_only_responses = laser_only_psth

    cell_count = mean_off_responses.shape[1]
    include_normalized_plots = (
        normalized_psth_off is not None and normalized_psth_on is not None
    )

    if show_plots:
        plt.rcParams.update({"figure.max_open_warning": 0})

        for cell_idx in range(cell_count):
            subplot_count = 1 + (
                1
                if include_normalized_plots or laser_only_psth is not None
                else 0
            )
            fig, ax = plt.subplots(
                1, subplot_count, figsize=(2.5 * subplot_count, 2)
            )
            if subplot_count == 1:
                ax = np.array([ax])

            ax[0].plot(time_points, mean_off_responses[:, cell_idx], "k")
            ax[0].plot(time_points, mean_on_responses[:, cell_idx])
            ax[0].axvline(x=start_of_laser, ls="--", c="c")
            ax[0].axvline(x=start_of_stimulus, ls="--", c="k")
            ax[0].axvline(x=duration_of_stimulus, ls="--", c="k")
            ax[0].set(
                xlabel="Time (s)",
                ylabel="FR (Hz)",
                title=f"PSTH for Cell ID = {cell_idx}",
            )
            ax[0].legend(["Laser Off", "Laser On"])

            next_ax_idx = 1

            if include_normalized_plots:
                ax[next_ax_idx].plot(
                    time_points, normalized_psth_off[:, cell_idx], "k"
                )
                ax[next_ax_idx].plot(
                    time_points, normalized_psth_on[:, cell_idx]
                )
                ax[next_ax_idx].axvline(x=start_of_laser, ls="--", c="c")
                ax[next_ax_idx].axvline(x=start_of_stimulus, ls="--", c="k")
                ax[next_ax_idx].axvline(x=duration_of_stimulus, ls="--", c="k")
                ax[next_ax_idx].set(
                    xlabel="Time (s)",
                    ylabel="Normalized FR",
                    title=f"Normalized PSTH for Cell ID = {cell_idx}",
                )
                ax[next_ax_idx].legend(["Laser Off", "Laser On"])
                next_ax_idx += 1

            if laser_only_psth is not None:
                ax[next_ax_idx].plot(
                    time_points, mean_laser_only_responses[:, cell_idx], "k"
                )
                ax[next_ax_idx].axvline(x=start_of_laser, ls="--", c="k")
                ax[next_ax_idx].axvline(x=laser_only_duration, ls="--", c="k")
                ax[next_ax_idx].set(
                    xlabel="Time (s)",
                    ylabel="FR (Hz)",
                    title=f"Laser Only PSTH for Cell ID = {cell_idx}",
                )
                ax[next_ax_idx].legend(["Laser Only"])

            fig.align_labels()

    # Regardless of plotting, always return the mean off and on responses.
    return mean_off_responses, mean_on_responses


@app.function
def plot_data_TC(freqs, mTCcond1, mTCcond2, ax=None, title=""):
    # Check if the ax parameter is None. If so, create a new figure and axes.
    if ax is None:
        fig, ax = plt.subplots()
        fig.set_size_inches(2, 1.5)
        created_fig = True
    else:
        created_fig = (
            False  # To know whether to show the plot and return a figure
        )

    ax.plot(
        freqs, mTCcond1.mean(axis=1), color="black"
    )  # Assuming _new_black was a color
    ax.plot(freqs, mTCcond2.mean(axis=1), color="CornflowerBlue")
    ax.set_ylabel("FR (Hz)")
    ax.set_xscale("log")
    ax.set_xlabel("Frequency (Hz)")
    if title:
        ax.set_title(title)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.set_box_aspect(1)

    # Show the plot if this function created the figure; otherwise, it's the caller's responsibility.
    if created_fig:
        plt.show()
        return fig
    else:
        return ax


@app.function
def plot_individual_mean_tuning_curves(mTCOff_V, mTCOn_V, freqs):
    """
    Plots mean tuning curves for given firing rates and frequencies.

    Parameters:
    - mTCOff: 2D array of mean tuning curves with laser off (conditions x cells).
    - mTCOn: 2D array of mean tuning curves with laser on (conditions x cells).
    - freqs: 1D array of frequencies.
    """
    for cell in range(mTCOff_V.shape[1]):
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(freqs, mTCOff_V[:, cell], "k")
        ax.plot(freqs, mTCOn_V[:, cell], "CornflowerBlue")
        ax.set_ylabel("FR (Hz)")
        ax.set_xscale("log")
        ax.set_title("tc= %i" % cell)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

    plt.show()


@app.function
# BUGFIX: suspected typo ln 18, 'octaves' -> 'octaves_V'
def plot_centered_individual_tuning_curves(
    cFR_V, cFR_On_V, octaves_V, cell_range=None
):
    """
    Plots tuning curves for given firing rates and octaves.

    Parameters:
    - cFR: 2D array of firing rates with laser off (conditions x cells).
    - cFR_On: 2D array of firing rates with laser on (conditions x cells).
    - octaves: 1D array of octaves.
    - cell_range: tuple of (start, end) to specify range of cells to plot. If None, all cells are plotted.
    """
    if cell_range is None:
        cell_range = (0, cFR_V.shape[1])

    for cell in range(*cell_range):
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.errorbar(
            octaves_V,
            cFR_V[:, cell],
            ecolor="k",
            linestyle="-",
            color="k",
            markerfacecolor="k",
            marker="o",
            markersize=8,
            capsize=5,
        )

        ax.errorbar(
            octaves_V,
            cFR_On_V[:, cell],
            ecolor="CornflowerBlue",
            linestyle="-",
            color="CornflowerBlue",
            markerfacecolor="CornflowerBlue",
            marker="o",
            markersize=8,
            capsize=5,
        )

        ax.set_xlim(
            np.round(octaves_V[0] - 0.025, 3),
            np.round(octaves_V[-1] + 0.025, 3),
        )
        ax.set_xticks(np.round(octaves_V, 2))
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%0.2g"))
        ax.set_title("Norm PSTH for Cell ID = %i" % cell)
        ax.set_ylabel("FR (Hz)", labelpad=10)
        ax.set_xlabel("Octaves from Best Frequency", labelpad=10)
        ax.legend(["Laser Off", "Laser On"], bbox_to_anchor=(1, 1))
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        fig.tight_layout()

    plt.show()


@app.function
def plot_raster_responses(
    spike_sort_indices, rasters, freq, title_prefix="Cell"
):
    """
    Plots raster plots for responses to different frequencies.

    Parameters:
    - spike_sort_indices: List of arrays, each containing spike sorting indices for responses.
    - rasters: List of arrays, each containing raster data for responses.
    - freq: Array of unique frequencies.
    - title_prefix: String to prefix the title with, followed by the cell ID.
    """
    cmap = cmr.get_sub_cmap("cmr.tropical", 0.1, 1, N=len(freq))

    for l, spike_indices in enumerate(spike_sort_indices):
        cInd = np.ceil(spike_indices / 20) * 20
        midpoint = np.max(cInd / 2)
        sortind_c_off = spike_indices <= midpoint
        sortind_c_on = spike_indices > midpoint

        if max(spike_indices[sortind_c_off], default=-1) >= len(
            rasters[l]
        ) or max(spike_indices[sortind_c_on], default=-1) >= len(rasters[l]):
            print(f"Index out of bounds for cell {l}. Skipping.")
            continue

        rastoff = np.asarray(rasters[l])[sortind_c_off]
        raston = np.asarray(rasters[l])[sortind_c_on]

        fig, ax = plt.subplots(figsize=(4, 3), constrained_layout=True)
        cax = ax.scatter(
            rastoff,
            sortind_c_off,
            s=0.5,
            c=sortind_c_off,
            cmap=cmap,
            marker="|",
            linewidth=0.5,
        )
        cax2 = ax.scatter(
            raston,
            sortind_c_on,
            s=0.5,
            c=sortind_c_on,
            cmap=cmap,
            marker="|",
            linewidth=0.5,
        )

        cbar = fig.colorbar(cax)
        cbar.ax.yaxis.set_major_locator(
            ticker.LinearLocator(numticks=len(freq))
        )
        cbar.ax.set_yticklabels(
            np.around(freq / 1000, decimals=1), fontsize=10
        )
        cbar.set_label("Frequency (kHz)", fontsize="small")

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Trials")
        ax.set_title(f"{title_prefix} for Cell ID = {l}")
        ax.tick_params(axis="both")
        ax.locator_params(nbins=5, axis="y")
        ax.locator_params(nbins=8, axis="x")
        ax.set_ylim(top=max(spike_indices) + 1)
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%0.2g"))

        plt.show()


@app.function
def plot_window_means_strip(
    time,
    toneOn_diff_bc,
    laser_cells_bc,
    window_s=(0.0, 0.025),
    off_color="seagreen",
    on_color="CornflowerBlue",
    font_family="Arial",
    font_size=8,
    title="0-25 ms means (paired)",
    pdf_path=None,
):
    # 0-25 ms window (time in seconds)
    lo, hi = window_s
    win = (time >= lo) & (time < hi)

    # Per-cell means
    on_vals = toneOn_diff_bc[win, :].mean(
        axis=0
    )  # "On": ToneOn Δ (Laser−NoLaser)
    off_vals = laser_cells_bc[win, :].mean(axis=0)  # "Off": ToneOff (Laser)

    # Paired t-test
    t_stat, p_val = ttest_rel(on_vals, off_vals)

    # Tidy for plotting
    order = ["Off", "On"]
    df = pd.DataFrame(
        {
            "value": np.r_[off_vals, on_vals],
            "condition": [order[0]] * off_vals.size
            + [order[1]] * on_vals.size,
        }
    )

    with mpl.rc_context(
        {
            "font.family": font_family,
            "font.size": font_size,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    ):
        fig, ax = plt.subplots(figsize=(1.8, 2))

        # Box plot (median, IQR, whiskers = 1.5×IQR); no fliers so points underneath are clean
        sns.boxplot(
            data=df,
            x="condition",
            y="value",
            order=order,
            palette={"Off": off_color, "On": on_color},
            width=0.6,
            linewidth=1,
            fliersize=0,
            whis=1.5,
            ax=ax,
            zorder=1,
        )

        # Jittered points (black) over the boxes
        """
        sns.stripplot(
            data=df,
            x="condition",
            y="value",
            order=order,
            color="black",
            alpha=0.5,
            size=2,
            jitter=0.15,
            dodge=False,
            edgecolor="none",
            ax=ax,
            zorder=3,
        )
        """

        ax.set_ylim(top=100)

        ax.set_xticks([0, 1], order)
        ax.set_xlabel("")
        ax.set_ylabel("Firing rate (sp/s)\n(mean 0-25 ms)")
        ax.set_title(f"{title}\npaired t = {t_stat:.2f}, p = {p_val:.3g}")
        fig.tight_layout()
        if pdf_path:
            fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
        plt.show()

    return t_stat, p_val, on_vals, off_vals


@app.function
def mirror_y_ticks_to_x(ax):
    ax.figure.canvas.draw()
    ax.set_xticks(ax.get_yticks())
    try:
        ax.set_xticks(ax.get_yticks(minor=True), minor=True)
    except TypeError:
        pass
    ax.xaxis.set_major_formatter(ax.yaxis.get_major_formatter())


@app.function
def plot_window_means_scatter(
    time,
    toneOn_diff_bc,
    laser_cells_bc,
    window_s=(0.0, 0.025),
    off_color="seagreen",
    on_color="CornflowerBlue",
    font_family="Arial",
    font_size=8,
    title="0-25 ms means (paired)",
    pdf_path=None,
    point_color="black",
    point_size=5,
):
    # 0-25 ms window (time in seconds)
    lo, hi = window_s
    win = (time >= lo) & (time < hi)

    # Per-cell means
    on_vals = toneOn_diff_bc[win, :].mean(axis=0)  # Tone On Δ (Laser−NoLaser)
    off_vals = laser_cells_bc[win, :].mean(axis=0)  # Tone Off (Laser)

    # Paired t-test
    t_stat, p_val = ttest_rel(on_vals, off_vals)

    conf = 0.95
    tcrit = student_t.ppf(0.5 + conf / 2, df=on_vals.size - 1)

    def summarize(arr):
        n = arr.size
        mean = arr.mean()
        median = np.median(arr)
        sd = arr.std(ddof=1)
        sem = sd / np.sqrt(n)
        iqr = np.percentile(arr, 75) - np.percentile(arr, 25)
        moe = tcrit * sem
        return n, mean, median, sd, iqr, sem, moe

    # Per-condition summaries
    n_off, m_off, med_off, sd_off, iqr_off, sem_off, moe_off = summarize(
        off_vals
    )
    n_on, m_on, med_on, sd_on, iqr_on, sem_on, moe_on = summarize(on_vals)

    # Paired-difference (On − Off)
    diff = on_vals - off_vals
    n_diff = diff.size
    mean_diff = diff.mean()
    sd_diff = diff.std(ddof=1)
    sem_diff = sd_diff / np.sqrt(n_diff)
    moe_diff = tcrit * sem_diff
    dz = mean_diff / sd_diff if sd_diff > 0 else np.nan  # Cohen's dz (paired)

    # Print nicely
    print("\n Summary Stats (0-25 ms) ")
    print(
        f"Off  : N={n_off}  mean={m_off:.3f}  median={med_off:.3f}  SD={sd_off:.3f}  "
        f"IQR={iqr_off:.3f}  SEM={sem_off:.3f}  MoE95={moe_off:.3f}"
    )
    print(
        f"On   : N={n_on}   mean={m_on:.3f}   median={med_on:.3f}   SD={sd_on:.3f}   "
        f"IQR={iqr_on:.3f}   SEM={sem_on:.3f}   MoE95={moe_on:.3f}"
    )
    print(
        f"Paired diff (On-Off): mean={mean_diff:.3f}  SD={sd_diff:.3f}  "
        f"SEM={sem_diff:.3f}  MoE95={moe_diff:.3f}  t={t_stat:.3f}  p={p_val:.3g}  dz={dz:.3f}\n"
    )

    with mpl.rc_context(
        {
            "font.family": font_family,
            "font.size": font_size,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    ):
        fig, ax = plt.subplots(figsize=(2.2, 2.2))

        # Scatter: x = Tone On, y = Tone Off (matches axis labels below)
        ax.scatter(
            off_vals,
            on_vals,
            s=point_size,
            c=point_color,
            alpha=0.6,
            edgecolors="none",
        )

        # Identity line y = x
        all_vals = np.r_[on_vals, off_vals]
        vmin, vmax = np.min(all_vals), np.max(all_vals)
        pad = 0.05 * (vmax - vmin if vmax > vmin else 1.0)
        lo_lim, hi_lim = vmin - pad, vmax + pad
        ax.plot(
            [lo_lim, hi_lim],
            [lo_lim, hi_lim],
            ls="--",
            lw=1,
            color="k",
            alpha=0.6,
            zorder=0,
        )

        # Make axes share identical limits and aspect so y=x is at 45°
        if hi_lim > 100:
            hi_lim = 100

        ax.set_xlim(lo_lim, hi_lim)
        ax.set_ylim(lo_lim, hi_lim)
        ax.set_aspect("equal", adjustable="box")

        # Set ticks
        # mirror_y_ticks_to_x(ax)

        # Make plot square
        # ax.set_box_aspect(1)
        # ax.set_aspect('equal', adjustable='datalim')

        ax.set_xlabel("Firing Rate Δ Tone Off (Hz)")
        ax.set_ylabel("Firing Rate Δ Tone On (Hz)")

        fig.tight_layout()
        if pdf_path:
            fig.savefig(pdf_path, format="pdf")
        plt.show()

    return t_stat, p_val, on_vals, off_vals


@app.cell
def _():
    import marimo as mo
    return


if __name__ == "__main__":
    app.run()
