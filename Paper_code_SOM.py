import marimo

__generated_with = "0.18.2"
app = marimo.App(width="medium", auto_download=["ipynb"])


@app.cell
def _():
    import os
    from datetime import datetime
    import numpy as np
    import pandas as pd
    import scipy.stats as stats
    import statsmodels.formula.api as smf

    # Local module imports
    from helper_analysis import (
        load_experiment_data,
        summarize_data,
        load_probe_data,
        map_cluster_depths_to_real_dimensions,
        normalize_psth_data,
        normalize_to_laserOff_TC,
        calculate_mTC,
        calculate_psthTC,
        fit_and_predict,
        calculate_sparseness,
        get_BF_and_uBF,
    )

    from helper_plotting import plot_cells_psth
    return (
        calculate_mTC,
        calculate_psthTC,
        calculate_sparseness,
        datetime,
        fit_and_predict,
        get_BF_and_uBF,
        load_experiment_data,
        load_probe_data,
        map_cluster_depths_to_real_dimensions,
        normalize_psth_data,
        normalize_to_laserOff_TC,
        np,
        os,
        pd,
        plot_cells_psth,
        smf,
        stats,
        summarize_data,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Loading Data
    """)
    return


@app.cell
def _(load_experiment_data, load_probe_data, os, summarize_data):
    # FEAT: This will allow loading regardless of user file structure. works as-is if the data directory is in the cwd, you can also specify a remote location with the start='/path/' kewyord argument.
    dataLoc = os.path.relpath("Data_RolonMartinez2024")

    excel_path = os.path.join(dataLoc, "probe_depth_per_recording.xlsx")
    stim_loc = os.path.join(dataLoc, "Stimuli")

    # Base directory for stimulus files
    main_stim_file = "TuningCurve_50ms_Laser_50ms_Frequencies3_80Hz_02232022_stimInfo"  # Main stimulus file
    laser_stim_file = (
        "LaserStim_Optotag_50ms_400ISI_5Reps_stimInfo"  # Main laser stimulus file
    )

    # Uncomment the line corresponding to the dataset you are working with.

    date = "04192024"
    date_Laser = "04192024"

    # date = '05082024'  # folder for PV Controls
    # date_Laser = '05082024'  # folder for SOM Controls

    # Define the cell type and viral vector used in the experiment.

    # cellType = "PV"
    cellType = "SOM"

    Virus = "stGtACR1"
    # Virus = 'Controls'

    # Use absolute paths for data loading to avoid changing directories within the script.

    # Construct paths for saving or accessing processed spike data related to this analysis.
    data_paper = "Data_Paper"

    # Construct data folder paths
    dataFolder = f"{cellType}_Recordings/{Virus}/{data_paper}/PSTH/PSTH_{main_stim_file.replace('.mat', '')}_{date}"
    dataFolder_Laser = f"{cellType}_Recordings/{Virus}/{data_paper}/PSTH/PSTH_{laser_stim_file.replace('.mat', '')}_{date_Laser}"

    data_path = os.path.join(dataLoc, dataFolder)
    data_path_laser = os.path.join(dataLoc, dataFolder_Laser)

    # Define repetition counts for the stimulus presented.
    nreps = 2  # Number of repetitions for the tuning curve stimulus.
    nreps_laser = 10  # Number of repetitions for the laser-only stimulus.

    laserStart = 0
    tStart = 0

    # Use the load_experiment_data function to load and process experimental data:
    # This will output the PSTH, cluster depth, raster data, trials, spike sorting indices, and laser-specific PSTH.

    (
        allPSTH,
        allPSTH_S,
        Clust_Depth,
        Rasters,
        Trials,
        SpikeSortInd,
        allPSTH_Laser,
        allPSTH_Laser_S,
        trialOrder,
        ITI,
        ITI_laser,
        laserDur,
        laserOnlyDur,
        tDur,
        binSize,
        edges,
        time,
        session_guide,
        session_guide_laser,
    ) = load_experiment_data(
        data_path,
        main_stim_file,
        data_path_laser,
        laser_stim_file,
        stim_loc,
        nreps,
        nreps_laser,
    )

    # Summarize_data function to get number of cells (FullPST (trials, time, neurons)):
    summarize_data(
        Rasters,
        SpikeSortInd,
        Clust_Depth,
        allPSTH,
        allPSTH_S,
        allPSTH_Laser,
        allPSTH_Laser_S,
    )

    # comment out for control data
    probe_data = load_probe_data(excel_path, cellType, Virus)
    return (
        Clust_Depth,
        Rasters,
        SpikeSortInd,
        Trials,
        Virus,
        allPSTH,
        allPSTH_Laser_S,
        allPSTH_S,
        cellType,
        dataLoc,
        data_paper,
        edges,
        laserDur,
        laserOnlyDur,
        laserStart,
        main_stim_file,
        probe_data,
        tDur,
        tStart,
        time,
        trialOrder,
    )


@app.cell
def _(Virus, cellType, dataLoc, data_paper, datetime, main_stim_file, os):
    # FIXUP: joined and simplifed creating output paths
    # Create figure and stat output directories:
    def _():
        date = datetime.now().strftime("%d%m%Y")
        date_time = datetime.now().strftime("%d%m%Y%I%M%p")

        base_path = os.path.join(
            dataLoc, f"{cellType}_Recordings/{Virus}/{data_paper}"
        )

        fig_folder = "Figures_{}".format(cellType)
        fig_folder_name = "Figures_AllCells_{}".format(date)
        fig_folder_name_detailed = "Figures_AllCells_{0}_{1}".format(
            date_time, main_stim_file
        )

        fig_dir = os.path.join(base_path, fig_folder, fig_folder_name)
        fig_dir_time = os.path.join(
            base_path, fig_folder, fig_folder_name, fig_folder_name_detailed
        )

        stats_folder = "Statistics_{}".format(cellType)
        stats_folder_name = "Statistics_AllCells_{}".format(date)
        stats_folder_name_detailed = "Statistics_AllCells_{0}_{1}".format(
            date_time, main_stim_file
        )

        stats_dir = os.path.join(base_path, stats_folder, stats_folder_name)
        stats_dir_time = os.path.join(
            base_path, stats_folder, stats_folder_name, stats_folder_name_detailed
        )

        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)
            os.makedirs(fig_dir_time)
        else:
            os.makedirs(fig_dir_time)

        if not os.path.exists(stats_dir):
            os.makedirs(stats_dir)
            os.makedirs(stats_dir_time)
        else:
            os.makedirs(stats_dir_time)

        return fig_dir, fig_dir_time, stats_dir, stats_dir_time


    fig_dir, fig_dir_time, stats_dir, stats_dir_time = _()

    print("Figure folders for this run created at:", fig_dir_time)
    print("Stats folders for this run created at:", stats_dir_time)
    return fig_dir_time, stats_dir_time


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Start PSTH analysis
    """)
    return


@app.cell
def _(np, tDur, tStart, time, trialOrder):
    # Find unique frequencies and the stimulus index
    uniq_Freq, stim_Ind = np.unique(trialOrder[:, 0], return_inverse=True)
    laserCond, laser_Ind = np.unique(trialOrder[:, 1], return_inverse=True)

    # Directly obtain laserOn and laserOff trial indices based on laser_Ind and the trialOrder array.
    laserOn = np.flatnonzero(trialOrder[:, 1] == 1)
    laserOff = np.flatnonzero(trialOrder[:, 1] == 0)

    # Define the time indeces:
    spontInd = np.where((time <= tStart))[0].squeeze()
    early_evoked_Ind = np.where((time >= tStart) & (time <= 0.025))[0].squeeze()
    late_evoked_Ind = np.where((time > 0.025) & (time <= tDur))[0].squeeze()
    offsetInd = np.where((time > tDur) & (time <= 0.100))[0].squeeze()
    late_offsetInd = np.where((time > 0.100) & (time <= time[-1]))[0].squeeze()
    # early_evoked_Ind = np.where((time >= tStart) & (time < 0.050))[0]
    return (
        early_evoked_Ind,
        laserOff,
        laserOn,
        late_evoked_Ind,
        offsetInd,
        spontInd,
        stim_Ind,
        uniq_Freq,
    )


@app.cell
def _(Clust_Depth, map_cluster_depths_to_real_dimensions, np, probe_data):
    # Constants for the probe
    tip_to_last_channel = 0.075  # 75µm in mm
    channel_span = 0.775  # 775µm in mm

    # Convert depth from micrometers to millimeters and adjust
    probe_data["Total_Probe_End"] = probe_data["Depth"] / 1000
    probe_data["Probe_End"] = (probe_data["Depth"] / 1000) - tip_to_last_channel
    probe_data["Probe_Start"] = probe_data["Total_Probe_End"] - (
        tip_to_last_channel + channel_span
    )

    real_cluster_depths = map_cluster_depths_to_real_dimensions(
        Clust_Depth, probe_data
    )
    real_cluster_depths = np.array(real_cluster_depths)
    return (real_cluster_depths,)


@app.cell
def _(
    Rasters,
    SpikeSortInd,
    Trials,
    allPSTH,
    allPSTH_Laser_S,
    allPSTH_S,
    np,
    real_cluster_depths,
):
    # Sort psth per depth of the probe:
    # Assuming real_cluster_depths is a list of tuples (depth, session_id)
    depths_only = [
        _depth if _depth is not None else float("inf")
        for _depth, _session in real_cluster_depths
    ]
    sorted_indices = np.argsort(depths_only)

    # Example arrays allPSTH, allPSTH_S, etc. need to be defined appropriately
    sortedFullPSTH_unsmoothed = allPSTH[:, :, sorted_indices]
    sortedFullPSTH = allPSTH_S[:, :, sorted_indices]
    sortedFullPSTH_Laser = allPSTH_Laser_S[:, :, sorted_indices]

    depth = [
        real_cluster_depths[_idx] for _idx in sorted_indices
    ]  # This retains the tuple structure
    Rasters_sort = Rasters[
        sorted_indices
    ]  # Assuming Rasters has a compatible shape
    Trials_sort = Trials[sorted_indices]
    SpikeSortInd_sort = SpikeSortInd[sorted_indices]
    return (
        Rasters_sort,
        SpikeSortInd_sort,
        Trials_sort,
        depth,
        sortedFullPSTH,
        sortedFullPSTH_Laser,
    )


@app.cell
def _(
    laserOff,
    laserOn,
    normalize_psth_data,
    np,
    sortedFullPSTH,
    sortedFullPSTH_Laser,
):
    # Normalize the data using the normalize_psth_data function
    norm_PSTH_OFF, norm_PSTH_On = normalize_psth_data(
        sortedFullPSTH, laserOff, laserOn, axis=0
    )

    # Normalize data for laser-only condition
    norm_PSTH_LaserOnly = np.empty_like(sortedFullPSTH_Laser.mean(axis=0))

    # Loop through each cell to normalize the data individually
    for _cell in range(sortedFullPSTH_Laser.shape[2]):
        # Calculate the mean for the current cell across all trials
        mean_cell = sortedFullPSTH_Laser[:, :, _cell].mean(axis=0)

        # Compute the minimum and maximum values for normalization
        min_val, max_val = mean_cell.min(), mean_cell.max()

        # Normalize the mean PSTH data for the current cell
        # Handle division by zero if max_val equals min_val
        norm_PSTH_LaserOnly[:, _cell] = (
            (mean_cell - min_val) / (max_val - min_val)
            if max_val != min_val
            else 0
        )
    return norm_PSTH_LaserOnly, norm_PSTH_OFF, norm_PSTH_On


@app.cell
def _(
    laserOff,
    laserOn,
    laserStart,
    norm_PSTH_OFF,
    norm_PSTH_On,
    plot_cells_psth,
    sortedFullPSTH,
    tDur,
    tStart,
    time,
):
    # get the mean PSTH for both LaserOn and LaserOff conditions for each cell. Optional: plot individual PSTH's by changing show_plots = TRUE.
    meanLaserOff, meanLaserOn = plot_cells_psth(
        sortedFullPSTH[laserOff, :, :],
        sortedFullPSTH[laserOn, :, :],
        time,
        laserStart,
        tStart,
        tDur,
        norm_PSTH_OFF,
        norm_PSTH_On,
        show_plots=False,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Audrey edits
    """)
    return


@app.cell
def _(
    Rasters_sort,
    SpikeSortInd_sort,
    Trials_sort,
    depth,
    early_evoked_Ind,
    laserOff,
    laserOn,
    norm_PSTH_LaserOnly,
    norm_PSTH_OFF,
    norm_PSTH_On,
    np,
    pd,
    smf,
    sortedFullPSTH,
    sortedFullPSTH_Laser,
    spontInd,
):
    # Initialize lists
    peakVal = []
    laser_p_evokedEarly = []
    significant = []

    # Define non-significant value
    non_significant_value = np.nan

    n_trials, _, n_neurons = sortedFullPSTH.shape
    laser_vec = np.zeros(n_trials, dtype=int)
    laser_vec[np.asarray(laserOn, dtype=int)] = 1  # 0=off, 1=on

    # create spont ind that matches early evoked ind length
    spontInd_new = spontInd[0 : early_evoked_Ind.shape[0]]

    for _i in range(n_neurons):
        fr_sp = sortedFullPSTH[:, spontInd_new, _i].mean(axis=1)
        fr_ev = sortedFullPSTH[:, early_evoked_Ind, _i].mean(axis=1)

        fr = np.concatenate([fr_sp, fr_ev])
        sound = np.concatenate([np.zeros(n_trials, int), np.ones(n_trials, int)])
        laser2 = np.concatenate([laser_vec, laser_vec])

        df = pd.DataFrame({"fr": fr, "sound": sound, "laser": laser2})

        # Robust SEs to cover variance differences
        model = smf.ols("fr ~ sound * laser", data=df).fit(cov_type="HC3")

        p_sound = model.pvalues.get("sound", np.nan)
        p_interaction = model.pvalues.get("sound:laser", np.nan)
        coef_inter = model.params.get("sound:laser", np.nan)  # sign for classify
        p_laser_base = model.pvalues.get("laser", np.nan)

        # set alpha to compare to
        alpha = 0.05

        # re-create Solymar's variables
        is_sound = np.isfinite(p_sound) and (p_sound < alpha)
        peakVal.append(1 if is_sound else 0)

        # same as laser p early evoked
        LpEE = (
            float(p_interaction)
            if np.isfinite(p_interaction)
            else non_significant_value
        )
        laser_p_evokedEarly.append(LpEE)

        # Significant only if sound-responsive and interaction significant
        is_sig = bool(is_sound and np.isfinite(LpEE) and (LpEE < alpha))
        significant.append(1 if is_sig else 0)

    # Set the specific index for peakVal=1 and significant=0
    toneIndex = np.squeeze(
        np.where((np.asarray(peakVal) == 1) & (np.asarray(significant) == 0))
    )

    # Set the significance index
    sigtoneindex = np.squeeze(
        np.where((np.asarray(peakVal) == 1) & (np.asarray(significant) == 1))
    )

    print(f"Number of non-significant cells with peaks: {len(toneIndex)}")

    # Select signiticant units to *tone* only:
    toneOnlyRespPSTH = sortedFullPSTH[:, :, toneIndex]
    toneOnlyRespPSTH_Laser = sortedFullPSTH[:, :, toneIndex]

    meanLaserOff_toneonly_masked = toneOnlyRespPSTH[laserOff, :, :].mean(axis=0)
    meanLaserOn_toneonly_masked = toneOnlyRespPSTH[laserOn, :, :].mean(axis=0)

    norm_PSTH_OFF_toneonly_masked = norm_PSTH_OFF[:, toneIndex]
    norm_PSTH_On_toneonly_masked = norm_PSTH_On[:, toneIndex]

    PSTH_Laser_norm_toneonly_masked = norm_PSTH_LaserOnly[:, toneIndex]

    Clust_Depth_toneRespOnly = np.array([depth[_i] for _i in toneIndex])
    Rasters_toneRespOnly = Rasters_sort[toneIndex]
    Trials_toneRespOnly = Trials_sort[toneIndex]
    SpikeSortI_toneRespOnly = SpikeSortInd_sort[toneIndex]

    # Select signiticant units to *tone and laser*:
    toneRespPSTH = sortedFullPSTH[:, :, sigtoneindex]
    toneRespPSTH_Laser = sortedFullPSTH_Laser[:, :, sigtoneindex]

    meanLaserOff_masked = toneRespPSTH[laserOff, :, :].mean(axis=0)
    meanLaserOn_masked = toneRespPSTH[laserOn, :, :].mean(axis=0)

    norm_PSTH_OFF_masked = norm_PSTH_OFF[:, sigtoneindex]
    norm_PSTH_On_masked = norm_PSTH_On[:, sigtoneindex]

    PSTH_Laser_norm_masked = norm_PSTH_LaserOnly[:, sigtoneindex]

    Clust_Depth_toneResp = np.array([depth[_i] for _i in sigtoneindex])
    Rasters_toneResp = Rasters_sort[sigtoneindex]
    Trials_toneResp = Trials_sort[sigtoneindex]
    SpikeSortI_toneResp = SpikeSortInd_sort[sigtoneindex]

    # Print shapes to verify
    print("Shapes after filtering for tone & laser responsive units:")
    print("Tone responsive PSTH shape:", np.shape(toneRespPSTH))
    print("Tone responsive PSTH Laser shape:", np.shape(toneRespPSTH_Laser))
    print("Cluster Depth tone-responsive shape:", Clust_Depth_toneResp.shape)
    print("Rasters tone-responsive shape:", Rasters_toneResp.shape)
    print("Trials tone-responsive shape:", Trials_toneResp.shape)
    print("Spike Sort Index tone-responsive shape:", SpikeSortI_toneResp.shape)
    return (
        Clust_Depth_toneResp,
        Clust_Depth_toneRespOnly,
        PSTH_Laser_norm_masked,
        Rasters_toneResp,
        SpikeSortI_toneResp,
        Trials_toneResp,
        meanLaserOff_masked,
        meanLaserOff_toneonly_masked,
        meanLaserOn_masked,
        meanLaserOn_toneonly_masked,
        norm_PSTH_OFF_masked,
        norm_PSTH_OFF_toneonly_masked,
        norm_PSTH_On_masked,
        norm_PSTH_On_toneonly_masked,
        toneIndex,
        toneOnlyRespPSTH,
        toneOnlyRespPSTH_Laser,
        toneRespPSTH,
        toneRespPSTH_Laser,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Continue Solymar code
    """)
    return


@app.cell
def _(
    early_evoked_Ind,
    late_evoked_Ind,
    norm_PSTH_OFF_masked,
    norm_PSTH_On_masked,
    offsetInd,
    spontInd,
    toneRespPSTH,
):
    # Select the appropriate traces for spontaneous activity, evoked activity & offset
    spontTrace = toneRespPSTH[:, spontInd, :]
    evokedTrace = toneRespPSTH[:, early_evoked_Ind, :]
    late_evokedTrace = toneRespPSTH[:, late_evoked_Ind, :]
    offsetTrace = toneRespPSTH[:, offsetInd, :]

    spontTrace_normOff = norm_PSTH_OFF_masked[spontInd, :]
    evokedTrace_norm_normOff = norm_PSTH_OFF_masked[early_evoked_Ind, :]
    late_evokedTrace_norm_normOff = norm_PSTH_OFF_masked[late_evoked_Ind, :]
    offsetTrace_norm_normOff = norm_PSTH_OFF_masked[offsetInd, :]

    spontTrace_normOn = norm_PSTH_On_masked[spontInd, :]
    evokedTrace_norm_normOn = norm_PSTH_On_masked[early_evoked_Ind, :]
    late_evokedTrace_norm_normOn = norm_PSTH_On_masked[late_evoked_Ind, :]
    offsetTrace_norm_normOn = norm_PSTH_On_masked[offsetInd, :]
    return evokedTrace, late_evokedTrace


@app.cell
def _(
    early_evoked_Ind,
    meanLaserOff_masked,
    meanLaserOff_toneonly_masked,
    meanLaserOn_masked,
    meanLaserOn_toneonly_masked,
    norm_PSTH_OFF_masked,
    norm_PSTH_OFF_toneonly_masked,
    norm_PSTH_On_masked,
    norm_PSTH_On_toneonly_masked,
    np,
    toneOnlyRespPSTH,
    toneRespPSTH,
):
    # Initialize lists to store the results of the difference in mean firing rate laser on - laser off:
    diff_resp_EarlyOn_toneOnly = []
    diff_resp_norm_EarlyOn_toneOnly = []

    for cell in range(len(toneOnlyRespPSTH[0][0])):
        # Calculate the difference between LaserOn and LaserOff
        # Audrey edited these to have the full length of the tone
        diff = meanLaserOn_toneonly_masked[early_evoked_Ind, cell].mean(
            axis=0
        ) - meanLaserOff_toneonly_masked[early_evoked_Ind, cell].mean(axis=0)
        diff_resp_EarlyOn_toneOnly.append(diff)

        # Calculate the normalized difference
        diff_norm = norm_PSTH_On_toneonly_masked[early_evoked_Ind, cell].mean(
            axis=0
        ) - norm_PSTH_OFF_toneonly_masked[early_evoked_Ind, cell].mean(axis=0)
        diff_resp_norm_EarlyOn_toneOnly.append(diff_norm)

    # Convert list to NumPy array for easier processing
    diff_resp_EarlyOn_array_toneOnly = np.array(diff_resp_EarlyOn_toneOnly)
    diff_resp_norm_EarlyOn_array_toneOnly = np.array(diff_resp_norm_EarlyOn_toneOnly)

    diff_resp_EarlyOn = []
    diff_resp_norm_EarlyOn = []

    for cell in range(len(toneRespPSTH[0][0])):
        # Calculate the difference between LaserOn and LaserOff
        # Audrey edited these to have the full length of the tone
        diff = meanLaserOn_masked[early_evoked_Ind, cell].mean(
            axis=0
        ) - meanLaserOff_masked[early_evoked_Ind, cell].mean(axis=0)
        diff_resp_EarlyOn.append(diff)

        # Calculate the normalized difference
        diff_norm = norm_PSTH_On_masked[early_evoked_Ind, cell].mean(
            axis=0
        ) - norm_PSTH_OFF_masked[early_evoked_Ind, cell].mean(axis=0)
        diff_resp_norm_EarlyOn.append(diff_norm)

    # Convert list to NumPy array for easier processing
    diff_resp_EarlyOn_array = np.array(diff_resp_EarlyOn)
    diff_resp_norm_EarlyOn_array = np.array(diff_resp_norm_EarlyOn)

    # Establish facilitation or suppression indices:
    facilitated_Ind = np.where(diff_resp_EarlyOn_array > 0)[0]
    suppressed_Ind = np.where(diff_resp_EarlyOn_array < 0)[0]

    # Compute differences in firing rated durng laserOn - laserOff for sorted facilitated and suppressed data
    diff_Sorted_facil = diff_resp_EarlyOn_array[facilitated_Ind]

    diff_Sorted_supp = diff_resp_EarlyOn_array[suppressed_Ind]
    return (
        diff_Sorted_facil,
        diff_Sorted_supp,
        diff_resp_EarlyOn_array,
        diff_resp_EarlyOn_array_toneOnly,
        diff_resp_norm_EarlyOn_array,
        diff_resp_norm_EarlyOn_array_toneOnly,
        facilitated_Ind,
        suppressed_Ind,
    )


@app.cell
def _(
    Clust_Depth_toneResp,
    Rasters_toneResp,
    SpikeSortI_toneResp,
    Trials_toneResp,
    facilitated_Ind,
    meanLaserOff_masked,
    meanLaserOn_masked,
    norm_PSTH_OFF_masked,
    norm_PSTH_On_masked,
    suppressed_Ind,
    toneRespPSTH_Laser,
):
    # Index firing rates for facilitated and suppressed cells:
    facilitated_cells_Off = meanLaserOff_masked[:, facilitated_Ind]
    facilitated_cells_On = meanLaserOn_masked[:, facilitated_Ind]

    suppressed_cells_Off = meanLaserOff_masked[:, suppressed_Ind]
    suppressed_cells_On = meanLaserOn_masked[:, suppressed_Ind]

    facilitated_cells_Off_norm = norm_PSTH_OFF_masked[:, facilitated_Ind]
    facilitated_cells_On_norm = norm_PSTH_On_masked[:, facilitated_Ind]

    suppressed_cells_Off_norm = norm_PSTH_OFF_masked[:, suppressed_Ind]
    suppressed_cells_On_norm = norm_PSTH_On_masked[:, suppressed_Ind]

    mean_responses_facil_laser_only = toneRespPSTH_Laser[
        :, :, facilitated_Ind
    ].mean(axis=0)  # Mean across neurons
    mean_responses_supp_laser_only = toneRespPSTH_Laser[:, :, suppressed_Ind].mean(
        axis=0
    )  # Mean across neurons

    # Index rasters, trials, and sorted indeces for facilitated and suppressed cells:
    Facil_Rasters_toneResp = Rasters_toneResp[facilitated_Ind]
    Facil_Trials_toneResp = Trials_toneResp[facilitated_Ind]
    Facil_SpikeSortI_toneResp = SpikeSortI_toneResp[facilitated_Ind]

    Supp_Rasters_toneResp = Rasters_toneResp[suppressed_Ind]
    Supp_Trials_toneResp = Trials_toneResp[suppressed_Ind]
    Supp_SpikeSortI_toneResp = SpikeSortI_toneResp[suppressed_Ind]

    # Convert the depth component of each tuple in facilitated_depths to float
    facilitated_depths = Clust_Depth_toneResp[facilitated_Ind]
    facilitated_depths = [
        (float(depth), session) if isinstance(depth, str) else (depth, session)
        for depth, session in facilitated_depths
    ]

    # Convert the depth component of each tuple in suppressed_depths to float
    suppressed_depths = Clust_Depth_toneResp[suppressed_Ind]
    suppressed_depths = [
        (float(depth), session) if isinstance(depth, str) else (depth, session)
        for depth, session in suppressed_depths
    ]
    return (
        Facil_Rasters_toneResp,
        Facil_SpikeSortI_toneResp,
        Supp_Rasters_toneResp,
        Supp_SpikeSortI_toneResp,
        facilitated_cells_Off,
        facilitated_cells_Off_norm,
        facilitated_cells_On,
        facilitated_cells_On_norm,
        facilitated_depths,
        suppressed_cells_Off,
        suppressed_cells_Off_norm,
        suppressed_cells_On,
        suppressed_cells_On_norm,
        suppressed_depths,
    )


@app.cell
def _(
    Clust_Depth_toneResp,
    Clust_Depth_toneRespOnly,
    diff_Sorted_facil,
    diff_Sorted_supp,
    facilitated_depths,
    np,
    suppressed_depths,
):
    depths_numeric = np.array([float(x[0]) for x in Clust_Depth_toneResp])
    depths_numeric_toneOnly = np.array([float(x[0]) for x in Clust_Depth_toneRespOnly])
    depths_numeric_facil = np.array([float(x[0]) for x in facilitated_depths])
    depths_numeric_supp = np.array([float(x[0]) for x in suppressed_depths])

    # Bin the data per depth of the probe
    bin_size = 0.100  # Bin size in µm
    bins_depth = np.arange(
        np.min(depths_numeric), np.max(depths_numeric) + bin_size, bin_size
    )
    bin_centers = (
        bins_depth[:-1] + bins_depth[1:]
    ) / 2  # Calculate bin centers for plotting


    # Digitize your depth data for each category
    bin_indices_facil = np.digitize(depths_numeric_facil, bins_depth)
    bin_indices_supp = np.digitize(depths_numeric_supp, bins_depth)

    bin_indices_facil = bin_indices_facil.astype(int)
    bin_indices_supp = bin_indices_supp.astype(int)

    # Convert to array for compatibility
    diff_Sorted_facil_norm = np.asarray(diff_Sorted_facil)
    diff_Sorted_supp_norm = np.asarray(diff_Sorted_supp)

    # Adjusted to handle empty slices
    facil_agg = np.array(
        [
            np.mean(diff_Sorted_facil_norm[bin_indices_facil == i])
            if np.any(bin_indices_facil == i)
            else np.nan
            for i in range(1, len(bins_depth))
        ]
    )
    supp_agg = np.array(
        [
            np.mean(diff_Sorted_supp_norm[bin_indices_supp == i])
            if np.any(bin_indices_supp == i)
            else np.nan
            for i in range(1, len(bins_depth))
        ]
    )

    # For sum, you might want to return 0 instead of np.nan when the slice is empty
    facil_sum = np.array(
        [
            np.sum(diff_Sorted_facil_norm[bin_indices_facil == i])
            if np.any(bin_indices_facil == i)
            else 0
            for i in range(1, len(bins_depth))
        ]
    )
    supp_sum = np.array(
        [
            np.sum(diff_Sorted_supp_norm[bin_indices_supp == i])
            if np.any(bin_indices_supp == i)
            else 0
            for i in range(1, len(bins_depth))
        ]
    )

    # Count occurrences for each bin for facilitated and suppressed conditions
    facil_counts = np.bincount(
        bin_indices_facil - 1, minlength=len(bins_depth) - 1
    )
    supp_counts = np.bincount(bin_indices_supp - 1, minlength=len(bins_depth) - 1)

    # Combine all depths to define the range of bins
    depth_all = np.concatenate([facilitated_depths, suppressed_depths])
    return (
        bin_centers,
        bin_indices_facil,
        bin_indices_supp,
        bin_size,
        depths_numeric,
        depths_numeric_facil,
        depths_numeric_supp,
        depths_numeric_toneOnly,
        diff_Sorted_facil_norm,
        diff_Sorted_supp_norm,
        facil_agg,
        facil_counts,
        facil_sum,
        supp_agg,
        supp_counts,
        supp_sum,
    )


@app.cell
def _(
    calculate_mTC,
    calculate_psthTC,
    evokedTrace,
    fit_and_predict,
    laserOff,
    laserOn,
    late_evokedTrace,
    normalize_to_laserOff_TC,
    stim_Ind,
    toneRespPSTH,
    uniq_Freq,
):
    tonesOn = uniq_Freq[stim_Ind[laserOn]]
    tonesOff = uniq_Freq[stim_Ind[laserOff]]

    evOff = evokedTrace[laserOff, :, :].mean(axis=1)
    evOn = evokedTrace[laserOn, :, :].mean(axis=1)

    norm_evOff, norm_evOn = normalize_to_laserOff_TC(evOff, evOn)

    mTCOff = calculate_mTC(evOff, tonesOff, uniq_Freq)
    mTCOn = calculate_mTC(evOn, tonesOn, uniq_Freq)

    mTCOff_norm = calculate_mTC(norm_evOff, tonesOff, uniq_Freq)
    mTCOn_norm = calculate_mTC(norm_evOn, tonesOn, uniq_Freq)

    slopes, intercepts, y_pred = fit_and_predict(mTCOff_norm, mTCOn_norm)

    # Example usage for the initial evokedTrace
    Late_evOff = late_evokedTrace[laserOff, :, :].mean(axis=1)
    Late_evOn = late_evokedTrace[laserOn, :, :].mean(axis=1)

    Late_norm_evOff, Late_norm_evOn = normalize_to_laserOff_TC(evOff, evOn)

    Late_mTCOff = calculate_mTC(Late_evOff, tonesOff, uniq_Freq)
    Late_mTCOn = calculate_mTC(Late_evOn, tonesOn, uniq_Freq)

    Late_mTCOff_norm = calculate_mTC(Late_norm_evOff, tonesOff, uniq_Freq)
    Late_mTCOn_norm = calculate_mTC(Late_norm_evOn, tonesOn, uniq_Freq)

    Late_slopes, Late_intercepts, Late_y_pred = fit_and_predict(
        Late_mTCOff_norm, Late_mTCOn_norm
    )

    # Calculate psthTC for both laser off and on conditions
    psthTC, psthTC_On = calculate_psthTC(
        toneRespPSTH, tonesOff, tonesOn, uniq_Freq, laserOff, laserOn
    )
    return (
        Late_evOff,
        Late_evOn,
        Late_mTCOff,
        Late_mTCOff_norm,
        Late_mTCOn,
        Late_mTCOn_norm,
        Late_y_pred,
        evOff,
        evOn,
        mTCOff,
        mTCOff_norm,
        mTCOn,
        mTCOn_norm,
        psthTC,
        psthTC_On,
        y_pred,
    )


@app.cell
def _(
    cellType,
    facilitated_Ind,
    fig_dir_time,
    mTCOff,
    mTCOn,
    np,
    os,
    suppressed_Ind,
    uniq_Freq,
):
    # FIXUP: moved these to save in the fig directories instead of being saved to the base directory. This was done for several figures later in the file as well.
    # Save the variables
    if cellType == "PV":
        np.save(os.path.join(fig_dir_time, "PV_mTCOff.npy"), mTCOff)
        np.save(os.path.join(fig_dir_time, "PV_mTCOn.npy"), mTCOn)
        np.save(os.path.join(fig_dir_time, "uniq_Freq.npy"), uniq_Freq)
        np.save(
            os.path.join(fig_dir_time, "PV_facilitated_index.npy"),
            facilitated_Ind,
        )
        np.save(
            os.path.join(fig_dir_time, "PV_suppressed_index.npy"),
            suppressed_Ind,
        )
    else:
        np.save(os.path.join(fig_dir_time, "SST_mTCOff.npy"), mTCOff)
        np.save(os.path.join(fig_dir_time, "SST_mTCOn.npy"), mTCOn)
        np.save(os.path.join(fig_dir_time, "uniq_Freq.npy"), uniq_Freq)
        np.save(
            os.path.join(fig_dir_time, "SST_facilitated_index.npy"),
            facilitated_Ind,
        )
        np.save(
            os.path.join(fig_dir_time, "SST_suppressed_index.npy"),
            suppressed_Ind,
        )
    return


@app.cell
def _(
    Late_evOff,
    Late_evOn,
    Late_mTCOff,
    Late_mTCOn,
    evOff,
    evOn,
    facilitated_Ind,
    mTCOff,
    mTCOn,
    suppressed_Ind,
):
    evOff_facil = evOff[:, facilitated_Ind]
    evOn_facil = evOn[:, facilitated_Ind]

    Late_evOff_facil = Late_evOff[:, facilitated_Ind]
    Late_evOn_facil = Late_evOn[:, facilitated_Ind]

    evOff_supp = evOff[:, suppressed_Ind]
    evOn_supp = evOn[:, suppressed_Ind]

    Late_evOff_supp = Late_evOff[:, suppressed_Ind]
    Late_evOn_supp = Late_evOn[:, suppressed_Ind]

    mTCOff_facil = mTCOff[:, facilitated_Ind]
    mTCOn_facil = mTCOn[:, facilitated_Ind]

    mTCOff_supp = mTCOff[:, suppressed_Ind]
    mTCOn_supp = mTCOn[:, suppressed_Ind]

    Late_mTCOff_facil = Late_mTCOff[:, facilitated_Ind]
    Late_mTCOn_facil = Late_mTCOn[:, facilitated_Ind]

    Late_mTCOff_supp = Late_mTCOff[:, suppressed_Ind]
    Late_mTCOn_supp = Late_mTCOn[:, suppressed_Ind]
    return (
        Late_mTCOff_facil,
        Late_mTCOff_supp,
        Late_mTCOn_facil,
        Late_mTCOn_supp,
        mTCOff_facil,
        mTCOff_supp,
        mTCOn_facil,
        mTCOn_supp,
    )


@app.cell
def _(
    Late_mTCOff,
    Late_mTCOff_facil,
    Late_mTCOff_supp,
    Late_mTCOn,
    Late_mTCOn_facil,
    Late_mTCOn_supp,
    calculate_sparseness,
    mTCOff,
    mTCOff_facil,
    mTCOff_supp,
    mTCOn,
    mTCOn_facil,
    mTCOn_supp,
    np,
):
    # Calculate Sparseness

    # Calculate for all cells
    (
        SparsenessOff_all,
        SparsenessOn_all,
        Late_SparsenessOff_all,
        Late_SparsenessOn_all,
    ) = calculate_sparseness(mTCOff, mTCOn, Late_mTCOff, Late_mTCOn)

    # Calculate for facilitated cells, ensure to replace mTCOff_facil/mTCOn_facil with your actual variables for facilitated cells
    (
        SparsenessOff_facilitated,
        SparsenessOn_facilitated,
        Late_SparsenessOff_facilitated,
        Late_SparsenessOn_facilitated,
    ) = calculate_sparseness(
        mTCOff_facil, mTCOn_facil, Late_mTCOff_facil, Late_mTCOn_facil
    )

    # Calculate for suppressed cells, ensure to replace mTCOff_supp/mTCOn_supp with your actual variables for suppressed cells
    (
        SparsenessOff_suppressed,
        SparsenessOn_suppressed,
        Late_SparsenessOff_suppressed,
        Late_SparsenessOn_suppressed,
    ) = calculate_sparseness(
        mTCOff_supp, mTCOn_supp, Late_mTCOff_supp, Late_mTCOn_supp
    )

    # Example of filtering NaNs for one of the results if needed
    late_Spars_on_supp = np.asarray(Late_SparsenessOn_suppressed)
    Late_SparsenessOn_suppressed_filt = late_Spars_on_supp[
        ~np.isnan(late_Spars_on_supp)
    ]
    return (
        Late_SparsenessOff_all,
        Late_SparsenessOff_facilitated,
        Late_SparsenessOff_suppressed,
        Late_SparsenessOn_all,
        Late_SparsenessOn_facilitated,
        Late_SparsenessOn_suppressed,
        SparsenessOff_all,
        SparsenessOff_facilitated,
        SparsenessOff_suppressed,
        SparsenessOn_all,
        SparsenessOn_facilitated,
        SparsenessOn_suppressed,
    )


@app.cell
def _(
    Late_mTCOff_norm,
    Late_mTCOn_norm,
    Late_y_pred,
    facilitated_Ind,
    mTCOff_norm,
    mTCOn_norm,
    suppressed_Ind,
    y_pred,
):
    mTCOff_facil_norm = mTCOff_norm[:, facilitated_Ind]
    mTCOn_facil_norm = mTCOn_norm[:, facilitated_Ind]

    mTCOff_supp_norm = mTCOff_norm[:, suppressed_Ind]
    mTCOn_supp_norm = mTCOn_norm[:, suppressed_Ind]

    Late_mTCOff_facil_norm = Late_mTCOff_norm[:, facilitated_Ind]
    Late_mTCOn_facil_norm = Late_mTCOn_norm[:, facilitated_Ind]

    Late_mTCOff_supp_norm = Late_mTCOff_norm[:, suppressed_Ind]
    Late_mTCOn_supp_norm = Late_mTCOn_norm[:, suppressed_Ind]

    y_pred_facil = y_pred[:, facilitated_Ind]
    y_pred_supp = y_pred[:, suppressed_Ind]

    Late_y_pred_facil = Late_y_pred[:, facilitated_Ind]
    Late_y_pred_supp = Late_y_pred[:, suppressed_Ind]
    return


@app.cell
def _(
    Late_mTCOff,
    Late_mTCOff_facil,
    Late_mTCOff_supp,
    Late_mTCOn,
    Late_mTCOn_facil,
    Late_mTCOn_supp,
    get_BF_and_uBF,
    mTCOff,
    mTCOff_facil,
    mTCOff_supp,
    mTCOn,
    mTCOn_facil,
    mTCOn_supp,
    np,
    uniq_Freq,
):
    # Get best frequency and the unpreferred frequency

    sideFreq = 4
    octaves = np.log2(uniq_Freq[4:13]) - np.log2(uniq_Freq[8])

    # For all cells dataset
    BF, FR_BF, cFR, cFR_On, uBF, FR_uBF, cFR_uBF, cFR_On_uBF = get_BF_and_uBF(
        mTCOff, mTCOn, uniq_Freq, sideFreq
    )

    # For the facilitated datasets
    (
        BF_facil,
        FR_BF_facil,
        cFR_facil,
        cFR_On_facil,
        uBF_facil,
        FR_uBF_facil,
        cFR_uBF_facil,
        cFR_On_uBF_facil,
    ) = get_BF_and_uBF(mTCOff_facil, mTCOn_facil, uniq_Freq, sideFreq)

    # For the suppressed datasets
    (
        BF_supp,
        FR_BF_supp,
        cFR_supp,
        cFR_On_supp,
        uBF_supp,
        FR_uBF_supp,
        cFR_uBF_supp,
        cFR_On_uBF_supp,
    ) = get_BF_and_uBF(mTCOff_supp, mTCOn_supp, uniq_Freq, sideFreq)

    # For all cells datasets using "Late" time point
    (
        Late_BF,
        Late_FR_BF,
        Late_cFR,
        Late_cFR_On,
        Late_uBF,
        Late_FR_uBF,
        Late_cFR_uBF,
        Late_cFR_On_uBF,
    ) = get_BF_and_uBF(Late_mTCOff, Late_mTCOn, uniq_Freq, sideFreq)

    # For the facilitated datasets using "Late" time point
    (
        Late_BF_facil,
        Late_FR_BF_facil,
        Late_cFR_facil,
        Late_cFR_On_facil,
        Late_uBF_facil,
        Late_FR_uBF_facil,
        Late_cFR_uBF_facil,
        Late_cFR_On_uBF_facil,
    ) = get_BF_and_uBF(Late_mTCOff_facil, Late_mTCOn_facil, uniq_Freq, sideFreq)

    # For the suppressed datasets using "Late" time point
    (
        Late_BF_supp,
        Late_FR_BF_supp,
        Late_cFR_supp,
        Late_cFR_On_supp,
        Late_uBF_supp,
        Late_FR_uBF_supp,
        Late_cFR_uBF_supp,
        Late_cFR_On_uBF_supp,
    ) = get_BF_and_uBF(Late_mTCOff_supp, Late_mTCOn_supp, uniq_Freq, sideFreq)
    return (
        BF,
        Late_cFR,
        Late_cFR_On,
        Late_cFR_On_facil,
        Late_cFR_On_supp,
        Late_cFR_facil,
        Late_cFR_supp,
        cFR,
        cFR_On,
        cFR_On_facil,
        cFR_On_supp,
        cFR_facil,
        cFR_supp,
        octaves,
    )


@app.cell
def _(
    Late_cFR_On_facil,
    Late_cFR_On_supp,
    Late_cFR_facil,
    Late_cFR_supp,
    cFR_On_facil,
    cFR_On_supp,
    cFR_facil,
    cFR_supp,
    facilitated_Ind,
    facilitated_cells_Off,
    facilitated_cells_Off_norm,
    facilitated_cells_On,
    facilitated_cells_On_norm,
    meanLaserOff_masked,
    meanLaserOff_toneonly_masked,
    meanLaserOn_masked,
    meanLaserOn_toneonly_masked,
    norm_PSTH_OFF_masked,
    norm_PSTH_OFF_toneonly_masked,
    norm_PSTH_On_masked,
    norm_PSTH_On_toneonly_masked,
    suppressed_Ind,
    suppressed_cells_Off,
    suppressed_cells_Off_norm,
    suppressed_cells_On,
    suppressed_cells_On_norm,
    toneOnlyRespPSTH_Laser,
    toneRespPSTH_Laser,
):
    from helper_statistics import calculate_sem

    # Calculate SEM for non-normalized tone-only significantdata
    errorOff_toneOnly = calculate_sem(meanLaserOff_toneonly_masked)
    errorOn_toneOnly = calculate_sem(meanLaserOn_toneonly_masked)

    # Calculate SEM for normalized tone-only significant data
    errorOff_n_toneOnly = calculate_sem(norm_PSTH_OFF_toneonly_masked)
    errorOn_n_toneOnly = calculate_sem(norm_PSTH_On_toneonly_masked)

    errorLaserOnly_toneOnly = calculate_sem(toneOnlyRespPSTH_Laser.mean(axis=0))

    # Calculate SEM for non-normalized data
    errorOff = calculate_sem(meanLaserOff_masked)
    errorOn = calculate_sem(meanLaserOn_masked)

    # Calculate SEM for normalized data
    errorOff_n = calculate_sem(norm_PSTH_OFF_masked)
    errorOn_n = calculate_sem(norm_PSTH_On_masked)

    errorLaserOnly = calculate_sem(toneRespPSTH_Laser.mean(axis=0))

    # Calculate SEM for both groups for raw firing rates & normalized firing rates
    errorlasfacil = calculate_sem(
        toneRespPSTH_Laser.mean(axis=0)[:, facilitated_Ind]
    )
    errorlassupp = calculate_sem(
        toneRespPSTH_Laser.mean(axis=0)[:, suppressed_Ind]
    )

    errorOff_facil = calculate_sem(facilitated_cells_Off)
    errorOn_facil = calculate_sem(facilitated_cells_On)

    errorOff_supp = calculate_sem(suppressed_cells_Off)
    errorOn_supp = calculate_sem(suppressed_cells_On)

    errorOff_facil_norm = calculate_sem(facilitated_cells_Off_norm)
    errorOn_facil_norm = calculate_sem(facilitated_cells_On_norm)

    errorOff_supp_norm = calculate_sem(suppressed_cells_Off_norm)
    errorOn_supp_norm = calculate_sem(suppressed_cells_On_norm)

    # Calculate errors using the function
    errorOff_tc_facil = calculate_sem(cFR_facil)
    errorOn_tc_facil = calculate_sem(cFR_On_facil)

    errorOff_tc_supp = calculate_sem(cFR_supp)
    errorOn_tc_supp = calculate_sem(cFR_On_supp)

    Late_errorOff_tc_facil = calculate_sem(Late_cFR_facil)
    Late_errorOn_tc_facil = calculate_sem(Late_cFR_On_facil)

    Late_errorOff_tc_supp = calculate_sem(Late_cFR_supp)
    Late_errorOn_tc_supp = calculate_sem(Late_cFR_On_supp)
    return (
        Late_errorOff_tc_facil,
        Late_errorOff_tc_supp,
        Late_errorOn_tc_facil,
        Late_errorOn_tc_supp,
        calculate_sem,
        errorLaserOnly,
        errorOff,
        errorOff_facil,
        errorOff_n,
        errorOff_supp,
        errorOff_tc_facil,
        errorOff_tc_supp,
        errorOn,
        errorOn_facil,
        errorOn_n,
        errorOn_supp,
        errorOn_tc_facil,
        errorOn_tc_supp,
        errorlasfacil,
        errorlassupp,
    )


@app.cell
def _(
    early_evoked_Ind,
    facilitated_Ind,
    meanLaserOff_masked,
    meanLaserOn_masked,
    np,
    spontInd,
    suppressed_Ind,
    toneRespPSTH,
):
    # find peaks ?
    from scipy import signal

    pOn = []
    pOnInd = []
    pOff = []
    for _cell in range(len(toneRespPSTH[0][0])):
        #     fig, ax = plt.subplots()
        pOn_t, _ = signal.find_peaks(
            meanLaserOn_masked[early_evoked_Ind, _cell],
            height=np.mean(meanLaserOn_masked[spontInd, _cell])
            + np.std(meanLaserOn_masked[spontInd, _cell]) * 3,
        )
        pOn.append(pOn_t)
        if len(pOn_t) == 0:
            pOn_tI = 0
        else:
            pOn_tI = 1
        pOnInd.append(pOn_tI)
        pOff_t, _ = signal.find_peaks(
            meanLaserOff_masked[early_evoked_Ind, _cell],
            height=np.mean(meanLaserOff_masked[spontInd, _cell])
            + np.std(meanLaserOff_masked[spontInd, _cell]) * 3,
        )
        pOff.append(pOff_t)
    peakInd = np.squeeze(np.where(np.asarray(pOnInd) == 1))
    facil_peakInd, _, _ = np.intersect1d(
        facilitated_Ind, peakInd, return_indices=True
    )
    supp_peakInd, _, _ = np.intersect1d(
        suppressed_Ind, peakInd, return_indices=True
    )
    peakCellstuseOn = meanLaserOn_masked[:, peakInd]
    peakCellstuseOff = meanLaserOff_masked[:, peakInd]
    peakCellstuseOn_facil = meanLaserOn_masked[:, facil_peakInd]
    peakCellstuseOff_facil = meanLaserOff_masked[:, facil_peakInd]
    peakCellstuseOn_supp = meanLaserOn_masked[:, supp_peakInd]
    peakCellstuseOff_supp = meanLaserOff_masked[:, supp_peakInd]
    return (
        peakCellstuseOff,
        peakCellstuseOff_facil,
        peakCellstuseOff_supp,
        peakCellstuseOn,
        peakCellstuseOn_facil,
        peakCellstuseOn_supp,
    )


@app.cell
def _(
    Clust_Depth_toneResp,
    diff_Sorted_facil,
    diff_Sorted_supp,
    diff_resp_EarlyOn_array,
    facilitated_depths,
    fig_dir_time,
    os,
    pd,
    suppressed_depths,
):
    # Convert to DataFrames
    df_diff_Sorted_facil = pd.DataFrame(
        diff_Sorted_facil, columns=["Diff Sorted Facil"]
    )
    df_diff_Sorted_supp = pd.DataFrame(
        diff_Sorted_supp, columns=["Diff Sorted Supp"]
    )
    df_facilitated_depths = pd.DataFrame(
        facilitated_depths,
        columns=["Facilitated Depths Column1", "Facilitated Depths Column2"],
    )
    df_suppressed_depths = pd.DataFrame(
        suppressed_depths,
        columns=["Suppressed Depths Column1", "Suppressed Depths Column2"],
    )
    df_diff_resp_EarlyOn_array = pd.DataFrame(
        diff_resp_EarlyOn_array, columns=["Diff Resp Early On"]
    )
    df_Clust_Depth_toneResp = pd.DataFrame(
        Clust_Depth_toneResp,
        columns=[
            "Clust Depth ToneResp Column1",
            "Clust Depth ToneResp Column2",
        ],
    )

    # Export to CSV
    # FIXUP: redundant exports
    """
    df_diff_Sorted_facil.to_csv('diff_Sorted_facil.csv', index=True)
    df_diff_Sorted_supp.to_csv('diff_Sorted_supp.csv', index=False)
    df_facilitated_depths.to_csv('facilitated_depths.csv', index=False)
    df_suppressed_depths.to_csv('suppressed_depths.csv', index=False)
    df_diff_resp_EarlyOn_array.to_csv('diff_resp_EarlyOn_array.csv', index=False)
    df_Clust_Depth_toneResp.to_csv('Clust_Depth_toneResp.csv', index=False)
    """

    # Save CSV files in the specified directory
    csv_files = {
        "diff_Sorted_facil.csv": df_diff_Sorted_facil,
        "diff_Sorted_supp.csv": df_diff_Sorted_supp,
        "facilitated_depths.csv": df_facilitated_depths,
        "suppressed_depths.csv": df_suppressed_depths,
        "diff_resp_EarlyOn_array.csv": df_diff_resp_EarlyOn_array,
        "Clust_Depth_toneResp.csv": df_Clust_Depth_toneResp,
    }

    for filename, dataframe in csv_files.items():
        full_path = os.path.join(fig_dir_time, filename)  # Create full file path
        dataframe.to_csv(full_path, index=False)  # Save to CSV without the index

    print("CSV files have been saved in:", fig_dir_time)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Figures
    """)
    return


@app.cell
def _():
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    from matplotlib.ticker import (
        AutoMinorLocator,
        LinearLocator,
        FormatStrFormatter,
    )
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    import cmasher as cmr
    import seaborn as sns

    from helper_analysis import find_time_to_peak

    from helper_plotting import (
        set_mpl_rcparams,
        plot_window_means_scatter,
        plot_window_means_strip,
        plot_data_TC,
    )
    return (
        AutoMinorLocator,
        Line2D,
        cmr,
        find_time_to_peak,
        mpl,
        plot_data_TC,
        plot_window_means_scatter,
        plot_window_means_strip,
        plt,
        set_mpl_rcparams,
        sns,
        ticker,
    )


@app.cell
def _(set_mpl_rcparams):
    # FIXUP: removed leading underscore so 'new_black' persists across cells
    new_black = "#373737"
    set_mpl_rcparams(new_black)
    return (new_black,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Tone Controls Analysis
    """)
    return


@app.cell
def _(
    cellType,
    facilitated_Ind,
    facilitated_cells_Off,
    facilitated_cells_On,
    fig_dir_time,
    mpl,
    np,
    os,
    plot_window_means_scatter,
    plot_window_means_strip,
    plt,
    time,
    toneRespPSTH_Laser,
):
    ## comparison of tone on laser on vs tone on laser off, to tone off laser on (e.g. what is the effect of laser on tone vs no tone response?)
    ## for facilitated cells
    def _():
        toneOn_diff = (
            facilitated_cells_On - facilitated_cells_Off
        )  # Tone On (Laser On − Off)

        # Collapse laser across trials to get mean PSTH (time × cells), then subset facilitated
        laser_fac_cells = toneRespPSTH_Laser.mean(axis=0)[
            :, facilitated_Ind
        ]  # shape (time, n_facilitated)

        # verify shapes
        print(toneOn_diff.shape)
        print(laser_fac_cells.shape)

        fig, ax = plt.subplots(figsize=(3, 2))

        # figure out baseline bins (first 50 ms)
        n_base = 25  # 25 bins

        with mpl.rc_context(
            {
                "font.family": "Arial",
                "font.size": 10,
                "pdf.fonttype": 42,
                "ps.fonttype": 42,
            }
        ):
            # ToneOn_diff (already a diff, but still baseline-correct for consistency)
            mean_on = toneOn_diff.mean(axis=1)
            sem_on = toneOn_diff.std(axis=1, ddof=1) / np.sqrt(
                toneOn_diff.shape[1]
            )
            ax.plot(
                time,
                mean_on,
                color="CornflowerBlue",
                label="ToneOn (Laser−NoLaser)",
            )
            ax.fill_between(
                time,
                mean_on - sem_on,
                mean_on + sem_on,
                color="CornflowerBlue",
                alpha=0.3,
            )

            # Laser_fac_cells (baseline corrected by first 50 ms)
            laser_fac_cells_bc = laser_fac_cells - laser_fac_cells[
                :n_base, :
            ].mean(axis=0, keepdims=True)
            mean_laser = laser_fac_cells_bc.mean(axis=1)
            sem_laser = laser_fac_cells_bc.std(axis=1, ddof=1) / np.sqrt(
                laser_fac_cells_bc.shape[1]
            )
            ax.plot(time, mean_laser, color="seagreen", label="ToneOff (Laser)")
            ax.fill_between(
                time,
                mean_laser - sem_laser,
                mean_laser + sem_laser,
                color="seagreen",
                alpha=0.3,
            )

            # Formatting
            ax.axvline(0, color="k", linestyle="--")  # stimulus onset
            ax.set_xlabel("Time (ms)")
            ax.set_ylabel("Firing rate (sp/s)")
            ax.legend()

            # Save the plot
            if cellType == "PV":
                title_str = "PV_facilitated_laser_tone_comparison"
            else:
                title_str = "SST_facilitated_laser_tone_comparison"

            plt.savefig(
                os.path.join(fig_dir_time, title_str) + ".pdf", bbox_inches="tight"
            )

            plt.show()

        # Add Strip Plot
        if cellType == "PV":
            pdfPath = "PV_facilitated_stripplot.pdf"
        else:
            pdfPath = "SST_facilitated_stripplot.pdf"
        plot_window_means_strip(
            time,
            toneOn_diff,
            laser_fac_cells_bc,
            pdf_path=os.path.join(fig_dir_time, pdfPath),
        )

        # Add Scatter Plot
        if cellType == "PV":
            pdfPath = "PV_facilitated_scatterplot.pdf"
        else:
            pdfPath = "SST_facilitated_scatterplot.pdf"
        plot_window_means_scatter(
            time,
            toneOn_diff,
            laser_fac_cells_bc,
            pdf_path=os.path.join(fig_dir_time, pdfPath),
        )


    _()
    return


@app.cell
def _(
    cellType,
    fig_dir_time,
    mpl,
    np,
    os,
    plot_window_means_scatter,
    plot_window_means_strip,
    plt,
    suppressed_Ind,
    suppressed_cells_Off,
    suppressed_cells_On,
    time,
    toneRespPSTH_Laser,
):
    ## comparison of tone on laser on vs tone on laser off, to tone off laser on (e.g. what is the effect of laser on tone vs no tone response?)
    ## for suppressed cells


    def _():
        toneOn_diff = (
            suppressed_cells_On - suppressed_cells_Off
        )  # Tone On (Laser On − Off)

        # Collapse laser across trials to get mean PSTH (time × cells), then subset facilitated
        laser_fac_cells = toneRespPSTH_Laser.mean(axis=0)[
            :, suppressed_Ind
        ]  # shape (time, n_facilitated)

        # verify shapes
        print(toneOn_diff.shape)
        print(laser_fac_cells.shape)

        with mpl.rc_context(
            {
                "font.family": "Arial",
                "font.size": 10,
                "pdf.fonttype": 42,
                "ps.fonttype": 42,
            }
        ):
            fig, ax = plt.subplots(figsize=(3, 2))

            # figure out baseline bins (first 50 ms)
            n_base = 25

            # ToneOn_diff (already a diff, but still baseline-correct for consistency)
            # toneOn_diff_bc = toneOn_diff - toneOn_diff[:n_base,:].mean(axis=0, keepdims=True)
            mean_on = toneOn_diff.mean(axis=1)
            sem_on = toneOn_diff.std(axis=1, ddof=1) / np.sqrt(
                toneOn_diff.shape[1]
            )
            ax.plot(
                time,
                mean_on,
                color="CornflowerBlue",
                label="ToneOn (Laser-NoLaser)",
            )
            ax.fill_between(
                time,
                mean_on - sem_on,
                mean_on + sem_on,
                color="CornflowerBlue",
                alpha=0.3,
            )

            # Laser_fac_cells (baseline corrected by first 50 ms)
            laser_fac_cells_bc = laser_fac_cells - laser_fac_cells[
                :n_base, :
            ].mean(axis=0, keepdims=True)
            mean_laser = laser_fac_cells_bc.mean(axis=1)
            sem_laser = laser_fac_cells_bc.std(axis=1, ddof=1) / np.sqrt(
                laser_fac_cells_bc.shape[1]
            )
            ax.plot(time, mean_laser, color="seagreen", label="ToneOff (Laser)")
            ax.fill_between(
                time,
                mean_laser - sem_laser,
                mean_laser + sem_laser,
                color="seagreen",
                alpha=0.3,
            )

            # Formatting
            ax.axvline(0, color="k", linestyle="--")  # stimulus onset
            ax.set_xlabel("Time (ms)")
            ax.set_ylabel("Firing rate (sp/s)")
            ax.legend()

            # Save the plot
            if cellType == "PV":
                title_str = "PV_suppressed_laser_tone_comparison"
            else:
                title_str = "SST_suppressed_laser_tone_comparison"

            plt.savefig(
                os.path.join(fig_dir_time, title_str) + ".pdf", bbox_inches="tight"
            )

            plt.show()

        # Also plot means and the strip plot

        if cellType == "PV":
            pdfPath = "PV_suppressed_stripplot.pdf"
        else:
            pdfPath = "SST_suppressed_stripplot.pdf"

        plot_window_means_strip(
            time,
            toneOn_diff,
            laser_fac_cells_bc,
            pdf_path=os.path.join(fig_dir_time, pdfPath),
        )

        # Add Scatter Plot
        if cellType == "PV":
            pdfPath = "PV_suppressed_scatterplot.pdf"
        else:
            pdfPath = "SST_suppressed_scatterplot.pdf"
        plot_window_means_scatter(
            time,
            toneOn_diff,
            laser_fac_cells_bc,
            pdf_path=os.path.join(fig_dir_time, pdfPath),
        )


    _()
    return


@app.cell
def _(
    AutoMinorLocator,
    cellType,
    errorLaserOnly,
    errorOff,
    errorOff_n,
    errorOn,
    errorOn_n,
    fig_dir_time,
    laserOnlyDur,
    laserStart,
    meanLaserOff_masked,
    meanLaserOn_masked,
    new_black,
    norm_PSTH_OFF_masked,
    norm_PSTH_On_masked,
    np,
    os,
    plt,
    tDur,
    tStart,
    ticker,
    time,
    toneRespPSTH_Laser,
):
    def _():
        # Plot average PSTH for all cells.
        fig, ax = plt.subplots(1, 3)
        fig.set_size_inches(6, 3)

        ax[0].plot(time, meanLaserOff_masked.mean(axis=1), c=new_black)
        ax[0].plot(time, meanLaserOn_masked.mean(axis=1), "CornflowerBlue")
        ax[0].axvline(x=tStart, ls="--", c=new_black)
        ax[0].axvline(x=tDur, ls="--", c=new_black)
        ax[0].set_xlabel("Time (s)")
        ax[0].set_ylabel("FR (Hz)")
        ax[0].spines["right"].set_visible(False)
        ax[0].spines["top"].set_visible(False)
        ax[0].fill_between(
            time,
            meanLaserOff_masked.mean(axis=1) - errorOff,
            meanLaserOff_masked.mean(axis=1) + errorOff,
            alpha=0.5,
            facecolor=new_black,
        )
        ax[0].fill_between(
            time,
            meanLaserOn_masked.mean(axis=1) - errorOn,
            meanLaserOn_masked.mean(axis=1) + errorOn,
            alpha=0.5,
            facecolor="CornflowerBlue",
        )

        ax[0].fill_between(
            time,
            toneRespPSTH_Laser.mean(axis=0).mean(axis=1) - errorLaserOnly,
            toneRespPSTH_Laser.mean(axis=0).mean(axis=1) + errorLaserOnly,
            alpha=0.5,
            facecolor="seagreen",
        )

        ax[0].set_yticks(np.ceil(ax[0].get_yticks() / 10) * 10)
        ymin0, ymax0 = ax[0].get_ylim()
        ax[0].set_ylim(0, ymax0)
        ax[0].yaxis.set_major_locator(ticker.MultipleLocator(10))
        ax[0].yaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))
        ax[0].set_xlim(np.min(time), np.max(time))
        ax[0].xaxis.set_major_formatter(ticker.FormatStrFormatter("%0.2g"))
        ax[0].xaxis.set_major_locator(ticker.MultipleLocator(0.05))
        ax[0].xaxis.set_minor_locator(AutoMinorLocator(2))
        ax[0].tick_params(which="major", color=new_black)
        ax[0].tick_params(which="minor", color=new_black)

        ax[1].plot(time, norm_PSTH_OFF_masked.mean(axis=1), c=new_black)
        ax[1].plot(time, norm_PSTH_On_masked.mean(axis=1), "CornflowerBlue")
        ax[1].axvline(x=tStart, ls="--", c=new_black)
        ax[1].axvline(x=tDur, ls="--", c=new_black)
        ax[1].set_xlabel("Time (s)")
        ax[1].set_ylabel("Normalized FR")
        ax[1].spines["right"].set_visible(False)
        ax[1].spines["top"].set_visible(False)

        ax[1].fill_between(
            time,
            norm_PSTH_OFF_masked.mean(axis=1) - errorOff_n,
            norm_PSTH_OFF_masked.mean(axis=1) + errorOff_n,
            alpha=0.5,
            facecolor=new_black,
        )
        ax[1].fill_between(
            time,
            norm_PSTH_On_masked.mean(axis=1) - errorOn_n,
            norm_PSTH_On_masked.mean(axis=1) + errorOn_n,
            alpha=0.5,
            facecolor="CornflowerBlue",
        )
        ax[1].set_yticks(np.round(ax[1].get_yticks(), 1))
        ymin1, ymax1 = ax[1].get_ylim()
        ax[1].set_ylim(0, ymax1)
        ax[1].yaxis.set_major_locator(ticker.MultipleLocator(0.1))
        ax[1].yaxis.set_major_formatter(ticker.FormatStrFormatter("%0.1f"))
        ax[1].set_xlim(np.min(time), np.max(time))
        ax[1].xaxis.set_major_locator(ticker.MultipleLocator(0.05))
        ax[1].xaxis.set_major_formatter(ticker.FormatStrFormatter("%0.2g"))
        ax[1].xaxis.set_minor_locator(AutoMinorLocator(2))
        ax[1].tick_params(which="major", color=new_black)
        ax[1].tick_params(which="minor", color=new_black)

        ax[2].plot(time, toneRespPSTH_Laser.mean(axis=0).mean(axis=1), c=new_black)
        ax[2].axvline(x=laserStart, ls="--", c=new_black)
        ax[2].axvline(x=laserOnlyDur, ls="--", c=new_black)
        ax[2].set_xlabel("Time (s)")
        ax[2].set_ylabel("FR (Hz)")
        ax[2].spines["right"].set_visible(False)
        ax[2].spines["top"].set_visible(False)
        ax[2].set_yticks(np.ceil(ax[2].get_yticks() / 10) * 10)
        ymin2, ymax2 = ax[2].get_ylim()
        ax[2].set_ylim(0, ymax2)
        ax[2].yaxis.set_major_locator(ticker.MultipleLocator(10))
        ax[2].yaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))
        ax[2].set_xlim(np.min(time), np.max(time))
        ax[2].xaxis.set_major_formatter(ticker.FormatStrFormatter("%0.2g"))
        ax[2].xaxis.set_major_locator(ticker.MultipleLocator(0.05))
        ax[2].xaxis.set_minor_locator(AutoMinorLocator(2))
        ax[2].tick_params(which="major", color=new_black)
        ax[2].tick_params(which="minor", color=new_black)

        ax[0].set_box_aspect(0.6)
        ax[1].set_box_aspect(0.6)
        ax[2].set_box_aspect(0.6)

        ax[0].spines["left"].set_bounds((0, ymax0))
        ax[1].spines["left"].set_bounds((0, ymax1))
        ax[2].spines["left"].set_bounds((0, ymax2))

        ax[0].set_title("Mean PSTH")
        ax[1].set_title("Mean Normalized PSTH")
        ax[2].set_title("Mean PSTH Laser Only")

        fig.align_ylabels(ax[:])
        fig.align_xlabels(ax[:])

        plot_name = "Mean_PSTH_AllCells_%s.pdf" % cellType
        fig.savefig(os.path.join(fig_dir_time, plot_name))

        plt.show()


    _()
    return


@app.cell
def _(
    cellType,
    early_evoked_Ind,
    fig_dir_time,
    meanLaserOff_masked,
    meanLaserOn_masked,
    new_black,
    norm_PSTH_OFF_masked,
    norm_PSTH_On_masked,
    os,
    plt,
    ticker,
):
    def _():
        # Plot scatter plots of firing rates during tone presentation in laserOff and laserOn conditions.
        fig, ax = plt.subplots(1, 2)
        fig.set_size_inches(4, 1.5)

        # First subplot
        ax[0].scatter(
            meanLaserOff_masked[early_evoked_Ind, :].mean(axis=0),
            meanLaserOn_masked[early_evoked_Ind, :].mean(axis=0),
            s=0.25,
            c=new_black,
            facecolor=new_black,
        )
        ax[0].plot(
            [0, 1], [0, 1], transform=ax[0].transAxes, color="dimgrey", ls="--"
        )
        ax[0].set_xlabel("FR$_{Off}$")
        ax[0].set_ylabel("FR$_{On}$")

        # Adding a best fit line to the first subplot
        """
        m, b = np.polyfit(
            meanLaserOff_masked[early_evoked_Ind, :].mean(axis=0),
            meanLaserOn_masked[early_evoked_Ind, :].mean(axis=0),
            1,
        )
        ax[0].plot(
            meanLaserOff_masked[early_evoked_Ind, :].mean(axis=0),
            m * meanLaserOff_masked[early_evoked_Ind, :].mean(axis=0) + b,
            "CornflowerBlue",
        )
        """

        # Setting ticks and limits for the first subplot
        ax[0].set_ylim(0, 100)
        ax[0].set_xlim(0, 100)
        ax[0].xaxis.set_major_locator(ticker.MultipleLocator(20))
        ax[0].yaxis.set_major_locator(ticker.MultipleLocator(20))
        ax[0].xaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))
        ax[0].yaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))

        # Second subplot
        ax[1].scatter(
            norm_PSTH_OFF_masked[early_evoked_Ind, :].mean(axis=0),
            norm_PSTH_On_masked[early_evoked_Ind, :].mean(axis=0),
            s=0.25,
            c=new_black,
            facecolor=new_black,
        )
        ax[1].plot(
            [0, 1], [0, 1], transform=ax[1].transAxes, color="dimgrey", ls="--"
        )
        ax[1].set_xlabel("Normalized FR$_{Off}$")
        ax[1].set_ylabel("Normalized FR$_{On}$")

        # Adding a best fit line to the second subplot
        """
        m_norm, b_norm = np.polyfit(
            norm_PSTH_OFF_masked[early_evoked_Ind, :].mean(axis=0),
            norm_PSTH_On_masked[early_evoked_Ind, :].mean(axis=0),
            1,
        )
        ax[1].plot(
            norm_PSTH_OFF_masked[early_evoked_Ind, :].mean(axis=0),
            m_norm * norm_PSTH_OFF_masked[early_evoked_Ind, :].mean(axis=0)
            + b_norm,
            "CornflowerBlue",
        )
        """

        # Setting ticks and limits for the second subplot
        ax[1].set_ylim(0, 1)
        ax[1].set_xlim(0, 1)
        ax[1].xaxis.set_major_locator(ticker.MultipleLocator(0.5))
        ax[1].yaxis.set_major_locator(ticker.MultipleLocator(0.5))
        ax[1].xaxis.set_major_formatter(ticker.FormatStrFormatter("%0.1f"))
        ax[1].yaxis.set_major_formatter(ticker.FormatStrFormatter("%0.1f"))

        # Ensuring equal aspect ratio for both subplots
        ax[0].set_box_aspect(1)
        ax[1].set_box_aspect(1)

        # Aligning labels
        fig.align_ylabels(ax[:])
        fig.align_xlabels(ax[:])

        plt.show()

        plot_name = f"ScatterPlots_Evoked_{cellType}.pdf"
        fig.savefig(
            os.path.join(fig_dir_time, plot_name),
            bbox_inches="tight",
            dpi=600,
            transparent=True,
        )


    _()
    return


@app.cell
def _(
    AutoMinorLocator,
    PSTH_Laser_norm_masked,
    cellType,
    early_evoked_Ind,
    fig_dir_time,
    new_black,
    os,
    plt,
    spontInd,
    ticker,
):
    def _():
        # Plot scatter plots of firing rates during laser presentation and spontaneous activity.
        fig, ax = plt.subplots()
        fig.set_size_inches(2, 1.5)

        # Scatter plot with customized settings
        ax.scatter(
            PSTH_Laser_norm_masked[spontInd, :].mean(axis=0),
            PSTH_Laser_norm_masked[early_evoked_Ind, :].mean(axis=0),
            s=0.25,
            c=new_black,
            facecolor=new_black,
            rasterized=True,
        )

        # Diagonal line indicating y=x for reference
        ax.plot([0, 1], [0, 1], color="dimgrey", ls="--")

        # Set labels
        ax.set_xlabel("Normalized FR$_{Spontaneous}$")
        ax.set_ylabel("Normalized FR$_{On}$")

        # Directly set limits and major tick formatting
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%0.1f"))
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%0.1f"))

        # Minor tick adjustments
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.tick_params(which="major", color=new_black)
        ax.tick_params(which="minor", color=new_black)

        # Ensure square aspect ratio
        ax.set_box_aspect(1)

        # Save the figure
        plot_name = f"ScatterPlot_LaserOnly_{cellType}.pdf"
        fig.savefig(
            os.path.join(fig_dir_time, plot_name),
            bbox_inches="tight",
            dpi=600,
            transparent=True,
        )

        plt.show()


    _()
    return


@app.cell
def _(
    cellType,
    facilitated_Ind,
    fig_dir_time,
    os,
    plt,
    suppressed_Ind,
    toneIndex,
):
    def _():
        # Plot pie chart for facilitated and suppressed cells:
        pie_data = [len(facilitated_Ind), len(suppressed_Ind), len(toneIndex)]
        print(pie_data)

        fig, ax = plt.subplots()
        fig.set_size_inches(4, 3)
        labels = "Facilitated", "Suppressed", "Non-Significant"
        colors = ["#002c94", "#85000c", "#787586"]
        patches, texts, pcts = ax.pie(
            pie_data,
            labels=labels,
            wedgeprops={"linewidth": 1.5, "edgecolor": "white"},
            startangle=25,
            colors=colors,
            autopct="%.0f%%",
        )
        plt.setp(pcts, color="white")

        plot_name = "PieChart_%s.pdf" % cellType

        fig.savefig(
            os.path.join(fig_dir_time, "PieChart.pdf"),
            bbox_inches="tight",
            dpi=600,
            transparent=True,
        )

        plt.show()


    _()
    return


@app.cell
def _(
    AutoMinorLocator,
    cellType,
    errorOff_facil,
    errorOff_supp,
    errorOn_facil,
    errorOn_supp,
    errorlasfacil,
    errorlassupp,
    facilitated_Ind,
    facilitated_cells_Off,
    facilitated_cells_On,
    fig_dir_time,
    laserOnlyDur,
    laserStart,
    new_black,
    np,
    os,
    plt,
    suppressed_Ind,
    suppressed_cells_Off,
    suppressed_cells_On,
    tDur,
    tStart,
    ticker,
    time,
    toneRespPSTH_Laser,
):
    def _():
        # Plot mean PSTH for the separated groups during tone-presentation & laser only:

        fig = plt.figure()
        gs = fig.add_gridspec(2, 2)
        fig.set_size_inches(4, 3)

        ax0 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1], sharey=ax0, sharex=ax0)

        ax1 = fig.add_subplot(gs[1, 0])
        ax3 = fig.add_subplot(gs[1, 1], sharey=ax1, sharex=ax1)

        ax0.plot(time, facilitated_cells_Off.mean(axis=1), c=new_black)
        ax0.plot(time, facilitated_cells_On.mean(axis=1), "CornflowerBlue")
        ax0.plot(
            time,
            toneRespPSTH_Laser.mean(axis=0)[:, facilitated_Ind].mean(axis=1),
            c="seagreen",
        )
        ax0.axvline(x=tStart, ls="--", c=new_black)
        ax0.axvline(x=tDur, ls="--", c=new_black)
        ax0.legend(["Laser Off", "Laser On", "Laser Only"])
        ax0.set_title("Facilitated Neurons")
        ax0.set_xlabel("Time (s)")
        ax0.set_ylabel("FR (Hz)")
        ax0.spines["right"].set_visible(False)
        ax0.spines["top"].set_visible(False)
        ax0.fill_between(
            time,
            facilitated_cells_Off.mean(axis=1) - errorOff_facil,
            facilitated_cells_Off.mean(axis=1) + errorOff_facil,
            alpha=0.5,
            facecolor=new_black,
        )
        ax0.fill_between(
            time,
            facilitated_cells_On.mean(axis=1) - errorOn_facil,
            facilitated_cells_On.mean(axis=1) + errorOn_facil,
            alpha=0.5,
            facecolor="CornflowerBlue",
        )
        ax0.fill_between(
            time,
            toneRespPSTH_Laser.mean(axis=0)[:, facilitated_Ind].mean(axis=1)
            - errorlasfacil,
            toneRespPSTH_Laser.mean(axis=0)[:, facilitated_Ind].mean(axis=1)
            + errorlasfacil,
            alpha=0.5,
            facecolor="seagreen",
        )
        ax0.set_yticks(np.floor(ax0.get_yticks() / 20) * 20)
        ymin0, ymax0 = ax0.get_ylim()
        ax0.set_ylim(0, ymax0)
        ax0.yaxis.set_major_locator(ticker.MultipleLocator(20))
        ax0.yaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))
        ax0.set_xlim(np.min(time), np.max(time))
        ax0.xaxis.set_major_locator(ticker.MultipleLocator(0.05))
        ax0.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax0.xaxis.set_major_formatter(ticker.FormatStrFormatter("%0.2g"))
        ax0.tick_params(which="major", color=new_black)
        ax0.tick_params(which="minor", color=new_black)

        ax1.plot(time, suppressed_cells_Off.mean(axis=1), c=new_black)
        ax1.plot(time, suppressed_cells_On.mean(axis=1), "CornflowerBlue")
        ax1.plot(
            time,
            toneRespPSTH_Laser.mean(axis=0)[:, suppressed_Ind].mean(axis=1),
            c="seagreen",
        )
        ax1.axvline(x=tStart, ls="--", c=new_black)
        ax1.axvline(x=tDur, ls="--", c=new_black)
        ax1.legend(["Laser Off", "Laser On"])
        ax1.set_title("Suppressed Neurons")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("FR (Hz)")
        ax1.spines["right"].set_visible(False)
        ax1.spines["top"].set_visible(False)
        ax1.fill_between(
            time,
            suppressed_cells_Off.mean(axis=1) - errorOff_supp,
            suppressed_cells_Off.mean(axis=1) + errorOff_supp,
            alpha=0.5,
            facecolor=new_black,
        )
        ax1.fill_between(
            time,
            suppressed_cells_On.mean(axis=1) - errorOn_supp,
            suppressed_cells_On.mean(axis=1) + errorOn_supp,
            alpha=0.5,
            facecolor="CornflowerBlue",
        )
        ax1.fill_between(
            time,
            toneRespPSTH_Laser.mean(axis=0)[:, suppressed_Ind].mean(axis=1)
            - errorlassupp,
            toneRespPSTH_Laser.mean(axis=0)[:, suppressed_Ind].mean(axis=1)
            + errorlassupp,
            alpha=0.5,
            facecolor="seagreen",
        )
        ax1.set_yticks(np.floor(ax1.get_yticks() / 20) * 20)
        ymin1, ymax1 = ax1.get_ylim()
        ax1.set_ylim(0, ymax1)
        ax1.yaxis.set_major_locator(ticker.MultipleLocator(10))
        ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))
        ax1.set_xlim(np.min(time), np.max(time))
        ax1.xaxis.set_major_formatter(ticker.FormatStrFormatter("%0.2g"))
        ax1.xaxis.set_major_locator(ticker.MultipleLocator(0.05))
        ax1.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax1.tick_params(which="major", color=new_black)
        ax1.tick_params(which="minor", color=new_black)

        ax2.plot(
            time,
            toneRespPSTH_Laser.mean(axis=0)[:, facilitated_Ind].mean(axis=1),
            c=new_black,
        )
        ax2.fill_between(
            time,
            toneRespPSTH_Laser.mean(axis=0)[:, facilitated_Ind].mean(axis=1)
            - errorlasfacil,
            toneRespPSTH_Laser.mean(axis=0)[:, facilitated_Ind].mean(axis=1)
            + errorlasfacil,
            alpha=0.5,
            facecolor=new_black,
        )
        ax2.axvline(x=laserStart, ls="--", c=new_black)
        ax2.axvline(x=laserOnlyDur, ls="--", c=new_black)
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("FR (Hz)")
        ax2.spines["right"].set_visible(False)
        ax2.spines["top"].set_visible(False)
        ax2.set_yticks(np.ceil(ax2.get_yticks() / 20) * 20)
        ymin2, ymax2 = ax2.get_ylim()
        ax2.set_ylim(0, ymax2)
        ax2.yaxis.set_major_locator(ticker.MultipleLocator(20))
        ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))
        ax2.set_xlim(np.min(time), np.max(time))
        ax2.xaxis.set_major_formatter(ticker.FormatStrFormatter("%0.2g"))
        ax2.xaxis.set_major_locator(ticker.MultipleLocator(0.05))
        ax2.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax2.tick_params(which="major", color=new_black)
        ax2.tick_params(which="minor", color=new_black)

        ax3.plot(
            time,
            toneRespPSTH_Laser.mean(axis=0)[:, suppressed_Ind].mean(axis=1),
            c=new_black,
        )
        ax3.fill_between(
            time,
            toneRespPSTH_Laser.mean(axis=0)[:, suppressed_Ind].mean(axis=1)
            - errorlassupp,
            toneRespPSTH_Laser.mean(axis=0)[:, suppressed_Ind].mean(axis=1)
            + errorlassupp,
            alpha=0.5,
            facecolor=new_black,
        )
        ax3.axvline(x=laserStart, ls="--", c=new_black)
        ax3.axvline(x=laserOnlyDur, ls="--", c=new_black)
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("FR (Hz)")
        ax3.spines["right"].set_visible(False)
        ax3.spines["top"].set_visible(False)
        ax3.set_yticks(np.ceil(ax3.get_yticks() / 20) * 20)
        ymin3, ymax3 = ax3.get_ylim()
        ax3.set_ylim(0, ymax3)
        ax3.yaxis.set_major_locator(ticker.MultipleLocator(20))
        ax3.yaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))
        ax3.set_xlim(np.min(time), np.max(time))
        ax3.xaxis.set_major_formatter(ticker.FormatStrFormatter("%0.2g"))
        ax3.xaxis.set_major_locator(ticker.MultipleLocator(0.05))
        ax3.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax3.tick_params(which="major", color=new_black)
        ax3.tick_params(which="minor", color=new_black)

        ax0.set_box_aspect(0.6)
        ax1.set_box_aspect(0.6)
        ax2.set_box_aspect(0.6)
        ax3.set_box_aspect(0.6)

        fig.align_ylabels([ax0, ax1, ax2, ax3])
        fig.align_xlabels([ax0, ax1, ax2, ax3])

        plot_name = "MeanPSTH_Divided_%s.pdf" % cellType
        fig.savefig(os.path.join(fig_dir_time, plot_name))

        plt.show()


    _()
    return


@app.cell
def _(
    cellType,
    diff_resp_norm_EarlyOn_array,
    diff_resp_norm_EarlyOn_array_toneOnly,
    facilitated_Ind,
    fig_dir_time,
    new_black,
    np,
    os,
    plt,
    suppressed_Ind,
    ticker,
):
    def _():
        # Histogram demonstrating the difference in firing rates:

        fig, ax = plt.subplots(figsize=(4, 1.5))

        # Clip values and calculate bins directly within the hist function for brevity
        bins = np.arange(-1.5, 1.5 + 0.071, 0.071)
        # Facilitated hist
        ax.hist(
            np.clip(diff_resp_norm_EarlyOn_array[facilitated_Ind], -1.5, 1.5),
            bins=bins,
            color="#002c94",
            edgecolor="#002c94",
        )
        # Suppressed hist
        ax.hist(
            np.clip(diff_resp_norm_EarlyOn_array[suppressed_Ind], -1.5, 1.5),
            bins=bins,
            color="#85000c",
            edgecolor="#85000c",
        )
        # Non-sig hist
        ax.hist(
            np.clip(diff_resp_norm_EarlyOn_array_toneOnly, -1.5, 5.0),
            bins=bins,
            color="#787586",
            edgecolor="#787586",
            alpha = 0.8,
        )

        # Simplifying axvline and labels in a compact way
        ax.axvline(x=0, color=new_black, linestyle="--")
        ax.set(
            xlabel=r"Mean ${\Delta}$ FR (norm)",
            ylabel="Cell Count",
            axisbelow=True,
        )

        # Minor grid, tick, and aspect adjustments can be made more concise
        # ax.grid(linestyle="--", which="both")
        ax.tick_params(which="major", color=new_black)
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%0.2g"))
        # ax.set_box_aspect(1)

        plot_name = f"Histogram_Diff_{cellType}.pdf"
        fig.savefig(os.path.join(fig_dir_time, plot_name))

        plt.show()


    _()
    return


@app.cell
def _(
    Line2D,
    cellType,
    diff_resp_norm_EarlyOn_array,
    early_evoked_Ind,
    fig_dir_time,
    late_evoked_Ind,
    meanLaserOff_masked,
    meanLaserOn_masked,
    new_black,
    np,
    offsetInd,
    os,
    plt,
    spontInd,
    ticker,
):
    def _():
        # Plot scatter plot with differentiated groups with normalized firing rates:
        colors_ind = [
            "#002c94" if diff > 0 else "#85000c" if diff < 0 else "#787586"
            for diff in diff_resp_norm_EarlyOn_array
        ]

        # Plot scatter plot with differentiated groups for all the time points:
        fig, ax = plt.subplots(2, 2)
        fig.set_size_inches(4, 3)
        ax[0, 0].scatter(
            meanLaserOff_masked[spontInd, :].mean(axis=0),
            meanLaserOn_masked[spontInd, :].mean(axis=0),
            s=0.25,
            color=colors_ind,
        )
        ax[0, 0].plot(
            [0, 1], [0, 1], transform=ax[0, 0].transAxes, color="dimgrey", ls="--"
        )
        ax[0, 0].set_title("Spontaneous Response")
        ax[0, 0].set_xlabel("FR Off (Hz)")
        ax[0, 0].set_ylabel("FR On (Hz)")

        ax[0, 1].scatter(
            meanLaserOff_masked[early_evoked_Ind, :].mean(axis=0),
            meanLaserOn_masked[early_evoked_Ind, :].mean(axis=0),
            s=0.25,
            color=colors_ind,
        )
        ax[0, 1].plot(
            [0, 1], [0, 1], transform=ax[0, 1].transAxes, color="dimgrey", ls="--"
        )
        ax[0, 1].set_title("0-25ms Tone On")
        ax[0, 1].set_xlabel("FR Off (Hz)")
        ax[0, 1].set_ylabel("FR On (Hz)")

        ax[1, 0].scatter(
            meanLaserOff_masked[late_evoked_Ind, :].mean(axis=0),
            meanLaserOn_masked[late_evoked_Ind, :].mean(axis=0),
            s=0.25,
            color=colors_ind,
        )
        ax[1, 0].plot(
            [0, 1], [0, 1], transform=ax[1, 0].transAxes, color="dimgrey", ls="--"
        )
        ax[1, 0].set_title("25-50ms Tone On")
        ax[1, 0].set_xlabel("FR Off (Hz)", loc="center")
        ax[1, 0].set_ylabel("FR On (Hz)", loc="center")

        ax[1, 1].scatter(
            meanLaserOff_masked[offsetInd, :].mean(axis=0),
            meanLaserOn_masked[offsetInd, :].mean(axis=0),
            s=0.25,
            color=colors_ind,
        )
        ax[1, 1].plot(
            [0, 1], [0, 1], transform=ax[1, 1].transAxes, color="dimgrey", ls="--"
        )
        ax[1, 1].set_title("Offset Response")
        ax[1, 1].set_xlabel("FR Off (Hz)")
        ax[1, 1].set_ylabel("FR On (Hz)")

        """
        m_Scat, b_Scat = np.polyfit(
            meanLaserOff_masked[early_evoked_Ind, :].mean(axis=0),
            meanLaserOn_masked[early_evoked_Ind, :].mean(axis=0),
            1,
        )

        ax[0, 1].plot(
            meanLaserOff_masked[early_evoked_Ind, :].mean(axis=0),
            m_Scat * meanLaserOff_masked[early_evoked_Ind, :].mean(axis=0)
            + b_Scat,
            "CornflowerBlue",
        )

        m_Scat1, b_Scat1 = np.polyfit(
            meanLaserOff_masked[late_evoked_Ind, :].mean(axis=0),
            meanLaserOn_masked[late_evoked_Ind, :].mean(axis=0),
            1,
        )

        ax[1, 0].plot(
            meanLaserOff_masked[late_evoked_Ind, :].mean(axis=0),
            m_Scat1 * meanLaserOff_masked[late_evoked_Ind, :].mean(axis=0)
            + b_Scat1,
            "CornflowerBlue",
        )

        m_Scat2, b_Scat2 = np.polyfit(
            meanLaserOff_masked[offsetInd, :].mean(axis=0),
            meanLaserOn_masked[offsetInd, :].mean(axis=0),
            1,
        )

        ax[1, 1].plot(
            meanLaserOff_masked[offsetInd, :].mean(axis=0),
            m_Scat2 * meanLaserOff_masked[offsetInd, :].mean(axis=0) + b_Scat2,
            "CornflowerBlue",
        )

        m_Scat3, b_Scat3 = np.polyfit(
            meanLaserOff_masked[spontInd, :].mean(axis=0),
            meanLaserOn_masked[spontInd, :].mean(axis=0),
            1,
        )

        ax[0, 0].plot(
            meanLaserOff_masked[spontInd, :].mean(axis=0),
            m_Scat3 * meanLaserOff_masked[spontInd, :].mean(axis=0) + b_Scat3,
            "CornflowerBlue",
        )
        """

        ax[0, 0].set_yticks(np.ceil(ax[0, 0].get_yticks() / 10) * 10)
        ymin0, ymax0 = ax[0, 0].get_ylim()
        ax[0, 0].set_ylim(0, 100)
        ax[0, 0].yaxis.set_major_locator(ticker.MultipleLocator(20))
        ax[0, 0].yaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))
        ax[0, 0].set_xticks(np.ceil(ax[0, 0].get_xticks() / 10) * 10)
        xmin0, xmax0 = ax[0, 0].get_xlim()
        ax[0, 0].set_xlim(0, 100)
        ax[0, 0].xaxis.set_major_locator(ticker.MultipleLocator(20))
        ax[0, 0].xaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))
        ax[0, 0].tick_params(which="major", color=new_black)
        ax[0, 0].tick_params(which="minor", color=new_black)

        ax[0, 1].set_yticks(np.ceil(ax[0, 1].get_yticks() / 10) * 10)
        ymin1, ymax1 = ax[0, 1].get_ylim()
        ax[0, 1].set_ylim(0, 120)
        ax[0, 1].yaxis.set_major_locator(ticker.MultipleLocator(20))
        ax[0, 1].yaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))
        ax[0, 1].set_xticks(np.ceil(ax[0, 1].get_xticks() / 10) * 10)
        xmin1, xmax1 = ax[0, 1].get_xlim()
        ax[0, 1].set_xlim(0, 120)
        ax[0, 1].xaxis.set_major_locator(ticker.MultipleLocator(20))
        ax[0, 1].xaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))
        ax[0, 1].tick_params(which="major", color=new_black)
        ax[0, 1].tick_params(which="minor", color=new_black)

        ax[1, 0].set_yticks(np.ceil(ax[1, 0].get_yticks() / 10) * 10)
        ymin2, ymax2 = ax[1, 0].get_ylim()
        ax[1, 0].set_ylim(0, 120)
        ax[1, 0].yaxis.set_major_locator(ticker.MultipleLocator(20))
        ax[1, 0].yaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))
        ax[1, 0].set_xticks(np.ceil(ax[1, 0].get_xticks() / 10) * 10)
        xmin2, xmax2 = ax[1, 0].get_xlim()
        ax[1, 0].set_xlim(0, 120)
        ax[1, 0].xaxis.set_major_locator(ticker.MultipleLocator(20))
        ax[1, 0].xaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))
        ax[1, 0].tick_params(which="major", color=new_black)
        ax[1, 0].tick_params(which="minor", color=new_black)

        ax[1, 1].set_yticks(np.ceil(ax[1, 1].get_yticks() / 10) * 10)
        ymin3, ymax3 = ax[1, 1].get_ylim()
        ax[1, 1].set_ylim(0, 100)
        ax[1, 1].yaxis.set_major_locator(ticker.MultipleLocator(20))
        ax[1, 1].yaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))

        ax[1, 1].set_xticks(np.ceil(ax[1, 1].get_xticks() / 10) * 10)
        xmin3, xmax3 = ax[1, 1].get_xlim()
        ax[1, 1].set_xlim(0, 100)
        ax[1, 1].xaxis.set_major_locator(ticker.MultipleLocator(20))
        ax[1, 1].xaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))

        ax[1, 1].tick_params(which="major", color=new_black)
        ax[1, 1].tick_params(which="minor", color=new_black)
        ax[0, 0].spines["left"].set_bounds((0, 100))
        ax[0, 1].spines["left"].set_bounds((0, 120))
        ax[1, 0].spines["left"].set_bounds((0, 120))
        ax[1, 1].spines["left"].set_bounds((0, 100))

        ax[0, 0].set_box_aspect(1)
        ax[0, 1].set_box_aspect(1)
        ax[1, 0].set_box_aspect(1)
        ax[1, 1].set_box_aspect(1)

        fig.align_ylabels(ax[:])
        fig.align_xlabels(ax[:])

        legend_elements = [
            Line2D(
                [0],
                [0],
                marker=".",
                color="#002c94",
                label="Facilitated Units",
                markerfacecolor="#002c94",
                markersize=5,
                linestyle="None",
            ),
            Line2D(
                [0],
                [0],
                marker=".",
                color="#85000c",
                label="Suppressed Units",
                markerfacecolor="#85000c",
                markersize=5,
                linestyle="None",
            ),
            Line2D(
                [0],
                [0],
                marker=".",
                color="#787586",
                label="No Significant Laser Effect",
                markerfacecolor="#787586",
                markersize=5,
                linestyle="None",
            ),
        ]

        # Adjust the legend placement
        fig.legend(
            handles=legend_elements,
            loc="upper left",
            bbox_to_anchor=(1, 0.5),
            frameon=False,
        )

        plot_name = "ScatterPlots_AllTimes_RawFR%s.pdf" % cellType
        fig.savefig(os.path.join(fig_dir_time, plot_name))

        plt.show()


    _()
    return


@app.cell
def _(
    cellType,
    early_evoked_Ind,
    facilitated_cells_Off,
    facilitated_cells_On,
    fig_dir_time,
    late_evoked_Ind,
    new_black,
    np,
    os,
    plt,
    spontInd,
    suppressed_cells_Off,
    suppressed_cells_On,
    ticker,
):
    def _():
        # Plot bar plots showing Firing rates during laser off and laser on trials during the spontaneous, evoked and late-evoked time:

        fig, axs = plt.subplots(
            1, 3, figsize=(6, 2)
        )  # Adjusting figsize for horizontal layout

        # Common properties and customizations
        boxprops = dict(linewidth=0.2, color=new_black)
        capprops = dict(linestyle="-", linewidth=0.5)
        medianprops = dict(linestyle="-", color=new_black, linewidth=0.25)
        whiskerprops = dict(linestyle="-", color=new_black, linewidth=0.25)
        meanpointprops = dict(
            marker=".",
            markeredgecolor="#f7ab31",
            markerfacecolor="#f7ab31",
            linewidth=0.25,
            markersize=0.25,
        )
        colors = [
            ["#002c94", "CornflowerBlue", "#85000c", "CornflowerBlue"],
            ["#4C6BB4", "#92B4F2", "#A94C54", "#92B4F2"],
            ["#99AAD4", "#B2CAF5", "#C28187", "#B2CAF5"],
        ]
        colors_new = [
            ["#002c94", "CornflowerBlue", "#85000c", "#C28187"],
            ["#002c94", "CornflowerBlue", "#85000c", "#C28187"],
            ["#002c94", "CornflowerBlue", "#85000c", "#C28187"],
        ]
        colors_new2 = [
            ["CornflowerBlue", "#002c94", "#C28187", "#85000c"],
            ["CornflowerBlue", "#002c94", "#C28187", "#85000c"],
            ["CornflowerBlue", "#002c94", "#C28187", "#85000c"],
        ]

        titles = ["Early Evoked", "Late Evoked", "Spontaneous"]

        for i, ax in enumerate(axs):
            # Selecting the correct dataset for each subplot
            if i == 0:
                data_indices = early_evoked_Ind
            elif i == 1:
                data_indices = late_evoked_Ind
            else:  # i == 2
                data_indices = spontInd

            data = [
                facilitated_cells_Off[data_indices, :].mean(axis=0),
                facilitated_cells_On[data_indices, :].mean(axis=0),
                suppressed_cells_Off[data_indices, :].mean(axis=0),
                suppressed_cells_On[data_indices, :].mean(axis=0),
            ]

            # Creating the boxplot
            bplot = ax.boxplot(
                data,
                notch=False,
                showmeans=True,
                patch_artist=True,
                meanprops=meanpointprops,
                meanline=False,
                medianprops=medianprops,
                positions=[0.15, 0.45, 1.05, 1.35],
                widths=0.20,
                boxprops=boxprops,
                capprops=capprops,
                showfliers=False,
                whiskerprops=whiskerprops,
            )

            ax.axvline(x=0.75, color=new_black, linestyle="--")
            for patch, color in zip(bplot["boxes"], colors[i]):
                patch.set_facecolor(color)

            ax.set_yticks(np.floor(ax.get_yticks() / 10) * 10)
            ymin, ymax = ax.get_ylim()
            ax.set_ylim(0, ymax)
            ax.yaxis.set_major_locator(ticker.MultipleLocator(10))
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))
            ax.set_ylabel("FR (Hz)")
            ax.tick_params(which="major", color=new_black)
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.spines["bottom"].set_visible(False)
            ax.spines["left"].set_bounds((0, ymax))
            ax.set_xlim([0, 2.5])
            ax.set_xticklabels(["Off", "On", "Off", "On"])
            ax.set_title(titles[i])

            if (
                i == 0
            ):  # Only add legend to the first plot (or choose according to preference)
                ax.legend(
                    [*bplot["boxes"]],
                    [
                        "Facilitated Off",
                        "Facilitated On",
                        "Suppressed Off",
                        "Suppressed On",
                    ],
                    loc="upper left",
                )

        # Assuming cellType and fig_dir_time are defined
        plot_name = "FR_BPlot_Combined_%s.pdf" % cellType
        fig.savefig(os.path.join(fig_dir_time, plot_name))

        plt.show()


    _()
    return


@app.cell
def _(diff_Sorted_facil_norm, diff_Sorted_supp_norm, plt):
    def _():
        # Plot pie chart with total percentage of cells
        pie_data_sort = [len(diff_Sorted_facil_norm), len(diff_Sorted_supp_norm)]

        fig, ax = plt.subplots()
        fig.set_size_inches(2, 1.5)
        labels = "Facilitated", "Suppressed"
        colors = ["#002c94", "#85000c", "#787586"]
        patches, texts, pcts = ax.pie(
            pie_data_sort,
            labels=labels,
            wedgeprops={"linewidth": 1.5, "edgecolor": "white"},
            startangle=25,
            colors=colors,
            autopct="%.0f%%",
        )
        plt.setp(pcts, color="white")

        plt.show()


    _()
    return


@app.cell
def _(
    Line2D,
    bin_centers,
    bin_indices_facil,
    bin_indices_supp,
    bin_size,
    cellType,
    depths_numeric,
    depths_numeric_facil,
    depths_numeric_supp,
    depths_numeric_toneOnly,
    diff_Sorted_facil,
    diff_Sorted_facil_norm,
    diff_Sorted_supp,
    diff_Sorted_supp_norm,
    diff_resp_EarlyOn_array,
    diff_resp_EarlyOn_array_toneOnly,
    facil_agg,
    facil_counts,
    facil_sum,
    fig_dir_time,
    np,
    os,
    plt,
    supp_agg,
    supp_counts,
    supp_sum,
):
    def _():
        # Create a figure with 2 rows and 3 columns of subplots

        fig, axs = plt.subplots(2, 3, figsize=(17, 11), constrained_layout=False)
        fig.tight_layout(pad=10)

        # Plot 1: scatter plot
        ax = axs[0, 0]
        ax.scatter(diff_resp_EarlyOn_array, depths_numeric, color="#787586", s=10)
        ax.invert_yaxis()
        # ax.set_ylim(-800, 25)
        # ax.set_xlim(-1.05, 1.05)
        ax.axvline(x=0, color="black", linestyle="--")
        ax.set_xlabel(r"$\Delta$ FR (Hz)")
        ax.set_ylabel("Recording Depth (mm)")
        ax.spines["top"].set_visible(True)
        ax.spines["right"].set_visible(True)
        ax.set_title("Scatter plot of difference in FR")

        # Plot 2: Scatter plot with categories
        ax = axs[0, 1]
        ax.scatter(
            diff_resp_EarlyOn_array_toneOnly,
            depths_numeric_toneOnly,
            color="#787586",
            s=10,
            label="Non-Significant",
        )
        ax.scatter(
            diff_Sorted_facil,
            depths_numeric_facil,
            color="#002c94",
            s=10,
            label="Facilitated",
        )
        ax.scatter(
            diff_Sorted_supp,
            depths_numeric_supp,
            color="#85000c",
            s=10,
            label="Suppressed",
        )
        ax.invert_yaxis()
        # ax.set_ylim(-800, 25)
        # ax.set_xlim(-1.05, 1.05)
        ax.axvline(x=0, color="black", linestyle="--")
        ax.set_xlabel(r"$\Delta$ FR (Hz)")
        ax.set_ylabel("Recording Depth (mm)")
        ax.spines["top"].set_visible(True)
        ax.spines["right"].set_visible(True)
        ax.set_title("Categorized scatter plot of difference in FR")

        # Plot 3: Bar plot (aggregated mean)
        ax = axs[0, 2]
        ax.barh(
            bin_centers,
            facil_agg,
            height=bin_size * 0.4,
            label="Facilitated",
            color="#002c94",
        )
        ax.barh(
            bin_centers,
            supp_agg,
            height=bin_size * 0.4,
            label="Suppressed",
            color="#85000c",
        )
        ax.set_xlabel(r"Mean $\Delta$ FR (Hz)")
        ax.set_ylabel("Recording Depth (mm)")
        ax.invert_yaxis()
        ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0)
        ax.spines["top"].set_visible(True)
        ax.spines["right"].set_visible(True)
        ax.set_title("Mean difference in FR per bin")

        # Plot 4: Bar plot (counts)
        ax = axs[1, 0]
        ax.barh(
            bin_centers,
            facil_counts,
            height=bin_size * 0.4,
            label="Facilitated",
            color="#002c94",
        )
        ax.barh(
            bin_centers,
            -supp_counts,
            height=bin_size * 0.4,
            label="Suppressed",
            color="#85000c",
        )
        ax.set_xlabel("Number of Cells")
        ax.set_ylabel("Recording Depth (mm)")
        ax.invert_yaxis()
        ax.spines["top"].set_visible(True)
        ax.spines["right"].set_visible(True)
        ax.set_title("Number of cells per bin")

        # Plot 5: Bar plot (sum)
        ax = axs[1, 1]
        ax.barh(
            bin_centers,
            facil_sum,
            height=bin_size * 0.4,
            label="Facilitated",
            color="#002c94",
        )
        ax.barh(
            bin_centers,
            supp_sum,
            height=bin_size * 0.4,
            label="Suppressed",
            color="#85000c",
        )
        ax.set_xlabel(r"$\sum \text{Mean } \Delta \text{FR (Hz)}$")
        ax.set_ylabel("Recording Depth (mm)")
        ax.invert_yaxis()
        ax.spines["top"].set_visible(True)
        ax.spines["right"].set_visible(True)
        ax.set_title("Sum difference in FR per bin")

        # Plot 6: Scatter plot on top of bars for individual points
        ax = axs[1, 2]
        ax.barh(
            bin_centers,
            facil_sum,
            height=bin_size * 0.4,
            color="#002c94",
            label="Facilitated",
        )
        ax.barh(
            bin_centers,
            supp_sum,
            height=bin_size * 0.4,
            color="#85000c",
            label="Suppressed",
        )

        for i in range(len(bin_centers)):
            # Adjust index access by removing the '-1' since we already start at 0
            facil_points = diff_Sorted_facil_norm[bin_indices_facil == i + 1]
            supp_points = diff_Sorted_supp_norm[bin_indices_supp == i + 1]
            facil_positions = np.full(facil_points.shape, bin_centers[i])
            supp_positions = np.full(supp_points.shape, bin_centers[i])

            # Plot the points; adjust 'alpha' for transparency as needed
            ax.scatter(
                facil_points,
                facil_positions,
                color="white",
                edgecolor="black",
                alpha=0.7,
                label="Facil. Points" if i == 1 else "",
                s=10,
                linewidths=0.5,
            )
            ax.scatter(
                supp_points,
                supp_positions,
                color="black",
                edgecolor="black",
                alpha=0.7,
                label="Supp. Points" if i == 1 else "",
                s=10,
                linewidths=0.5,
            )

        ax.set_ylabel("Recording Depth (mm)")
        ax.invert_yaxis()
        ax.set_xlabel(r"$\sum \text{Mean } \Delta \text{FR (Hz)}$")
        ax.spines["top"].set_visible(True)
        ax.spines["right"].set_visible(True)
        ax.set_title(r"$\sum \text{Mean } \Delta \text{FR (Hz)}$")

        custom_legends = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="white",
                markeredgecolor="black",
                label="Facil. Points",
                markersize=5,
                linewidth=0,
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="black",
                markeredgecolor="black",
                label="Supp. Points",
                markersize=5,
                linewidth=0,
            ),
        ]

        ax.legend(
            handles=custom_legends,
            loc="upper left",
            bbox_to_anchor=(1.05, 1),
            borderaxespad=0,
        )

        # Saving the figure
        plot_name = "Diff_resp_depthofprobe_AllPlots_TEST%s.pdf" % cellType
        fig.savefig(
            os.path.join(fig_dir_time, plot_name),
            bbox_inches="tight",
            dpi=600,
            transparent=True,
        )

        plt.show()


    _()
    return


@app.cell
def _(
    AutoMinorLocator,
    Late_SparsenessOff_facilitated,
    Late_SparsenessOff_suppressed,
    Late_SparsenessOn_facilitated,
    Late_SparsenessOn_suppressed,
    SparsenessOff_facilitated,
    SparsenessOff_suppressed,
    SparsenessOn_facilitated,
    SparsenessOn_suppressed,
    cellType,
    fig_dir_time,
    new_black,
    os,
    plt,
    ticker,
):
    def _():
        fig, ax = plt.subplots(2, 2)
        fig.set_size_inches(4, 3)
        ax[0, 0].scatter(
            SparsenessOff_facilitated,
            SparsenessOn_facilitated,
            s=0.25,
            c=new_black,
        )
        ax[0, 0].plot(
            [0, 1], [0, 1], transform=ax[0, 0].transAxes, color="dimgrey", ls="--"
        )
        # ax.plot([-.35,1.85],[-.35,1.85], transform=ax.transAxes, color='dimgrey', ls = "--")
        ax[0, 0].set_title("Sparseness Facilitated")
        ax[0, 0].set_xlabel("Sparseness FR$_{Off}$")
        ax[0, 0].set_ylabel("Sparseness FR$_{On}$")
        _, ymax = ax[0, 0].get_ylim()
        ax[0, 0].set_ylim([0, 1])
        ax[0, 0].yaxis.set_major_locator(ticker.MultipleLocator(0.5))
        ax[0, 0].yaxis.set_major_formatter(ticker.FormatStrFormatter("%0.1f"))
        _, xmax = ax[0, 0].get_xlim()
        ax[0, 0].set_xlim([0, 1])
        ax[0, 0].xaxis.set_major_locator(ticker.MultipleLocator(0.5))
        ax[0, 0].xaxis.set_major_formatter(ticker.FormatStrFormatter("%0.1f"))
        ax[0, 0].yaxis.set_minor_locator(AutoMinorLocator(2))
        ax[0, 0].xaxis.set_minor_locator(AutoMinorLocator(2))
        ax[0, 0].tick_params(which="major", color=new_black)
        ax[0, 0].tick_params(which="minor", color=new_black)
        ax[0, 0].spines["left"].set_bounds((0, 1))
        ax[0, 0].spines["left"].set_bounds((0, 1))
        ax[0, 0].set_box_aspect(1)

        ax[0, 1].scatter(
            SparsenessOff_suppressed, SparsenessOn_suppressed, s=0.25, c=new_black
        )
        ax[0, 1].plot(
            [0, 1], [0, 1], transform=ax[0, 1].transAxes, color="dimgrey", ls="--"
        )
        # ax.plot([-.35,1.85],[-.35,1.85], transform=ax.transAxes, color='dimgrey', ls = "--")
        ax[0, 1].set_title("Sparseness Suppressed")
        ax[0, 1].set_xlabel("Sparseness FR$_{Off}$")
        ax[0, 1].set_ylabel("Sparseness FR$_{On}$")
        _, ymax = ax[0, 1].get_ylim()
        ax[0, 1].set_ylim([0, 1])
        ax[0, 1].yaxis.set_major_locator(ticker.MultipleLocator(0.5))
        ax[0, 1].yaxis.set_major_formatter(ticker.FormatStrFormatter("%0.1f"))
        _, xmax = ax[0, 1].get_xlim()
        ax[0, 1].set_xlim([0, 1])
        ax[0, 1].xaxis.set_major_locator(ticker.MultipleLocator(0.5))
        ax[0, 1].xaxis.set_major_formatter(ticker.FormatStrFormatter("%0.1f"))
        ax[0, 1].yaxis.set_minor_locator(AutoMinorLocator(2))
        ax[0, 1].xaxis.set_minor_locator(AutoMinorLocator(2))
        ax[0, 1].tick_params(which="major", color=new_black)
        ax[0, 1].tick_params(which="minor", color=new_black)
        ax[0, 1].spines["left"].set_bounds((0, 1))
        ax[0, 1].spines["left"].set_bounds((0, 1))
        ax[0, 1].set_box_aspect(1)

        ax[1, 0].scatter(
            Late_SparsenessOff_facilitated,
            Late_SparsenessOn_facilitated,
            s=0.25,
            c=new_black,
        )
        ax[1, 0].plot(
            [0, 1], [0, 1], transform=ax[1, 0].transAxes, color="dimgrey", ls="--"
        )
        # ax.plot([-.35,1.85],[-.35,1.85], transform=ax.transAxes, color='dimgrey', ls = "--")
        ax[1, 0].set_title("Sparseness Facilitated Late")
        ax[1, 0].set_xlabel("Sparseness FR$_{Off}$")
        ax[1, 0].set_ylabel("Sparseness FR$_{On}$")
        _, ymax = ax[1, 0].get_ylim()
        ax[1, 0].set_ylim([0, 1])
        ax[1, 0].yaxis.set_major_locator(ticker.MultipleLocator(0.5))
        ax[1, 0].yaxis.set_major_formatter(ticker.FormatStrFormatter("%0.1f"))
        _, xmax = ax[1, 0].get_xlim()
        ax[1, 0].set_xlim([0, 1])
        ax[1, 0].xaxis.set_major_locator(ticker.MultipleLocator(0.5))
        ax[1, 0].xaxis.set_major_formatter(ticker.FormatStrFormatter("%0.1f"))
        ax[1, 0].yaxis.set_minor_locator(AutoMinorLocator(2))
        ax[1, 0].xaxis.set_minor_locator(AutoMinorLocator(2))
        ax[1, 0].tick_params(which="major", color=new_black)
        ax[1, 0].tick_params(which="minor", color=new_black)
        ax[1, 0].spines["left"].set_bounds((0, 1))
        ax[1, 0].spines["left"].set_bounds((0, 1))
        ax[1, 0].set_box_aspect(1)

        ax[1, 1].scatter(
            Late_SparsenessOff_suppressed,
            Late_SparsenessOn_suppressed,
            s=0.25,
            c=new_black,
        )
        ax[1, 1].plot(
            [0, 1], [0, 1], transform=ax[1, 1].transAxes, color="dimgrey", ls="--"
        )
        # ax.plot([-.35,1.85],[-.35,1.85], transform=ax.transAxes, color='dimgrey', ls = "--")
        ax[1, 1].set_title("Sparseness Suppressed Late")
        ax[1, 1].set_xlabel("Sparseness FR$_{Off}$")
        ax[1, 1].set_ylabel("Sparseness FR$_{On}$")
        _, ymax = ax[1, 1].get_ylim()
        ax[1, 1].set_ylim([0, 1])
        ax[1, 1].yaxis.set_major_locator(ticker.MultipleLocator(0.5))
        ax[1, 1].yaxis.set_major_formatter(ticker.FormatStrFormatter("%0.1f"))
        _, xmax = ax[1, 1].get_xlim()
        ax[1, 1].set_xlim([0, 1])
        ax[1, 1].xaxis.set_major_locator(ticker.MultipleLocator(0.5))
        ax[1, 1].xaxis.set_major_formatter(ticker.FormatStrFormatter("%0.1f"))
        ax[1, 1].yaxis.set_minor_locator(AutoMinorLocator(2))
        ax[1, 1].xaxis.set_minor_locator(AutoMinorLocator(2))
        ax[1, 1].tick_params(which="major", color=new_black)
        ax[1, 1].tick_params(which="minor", color=new_black)
        ax[1, 1].spines["left"].set_bounds((0, 1))
        ax[1, 1].spines["left"].set_bounds((0, 1))
        ax[1, 1].set_box_aspect(1)

        plot_name = "Sparseness_AllTimePoints_%s.pdf" % cellType
        fig.savefig(os.path.join(fig_dir_time, plot_name))

        plt.show()


    _()
    return


@app.cell
def _(
    Late_mTCOff,
    Late_mTCOff_facil,
    Late_mTCOff_supp,
    Late_mTCOn,
    Late_mTCOn_facil,
    Late_mTCOn_supp,
    cellType,
    fig_dir_time,
    mTCOff,
    mTCOff_facil,
    mTCOff_supp,
    mTCOn,
    mTCOn_facil,
    mTCOn_supp,
    np,
    os,
    plot_data_TC,
    plt,
    uniq_Freq,
):
    def _():
        fig, ax = plt.subplots(2, 3)  # Create a 2x3 grid of subplots
        fig.set_size_inches(12, 4)  # Adjust the overall figure size as needed

        # First row plots
        plot_data_TC(
            uniq_Freq,
            mTCOff_facil,
            mTCOn_facil,
            ax=ax[0, 0],
            title="Early Facilitated Units",
        )
        plot_data_TC(
            uniq_Freq,
            mTCOff_supp,
            mTCOn_supp,
            ax=ax[0, 1],
            title="Early Suppressed Units",
        )
        plot_data_TC(
            uniq_Freq, mTCOff, mTCOn, ax=ax[0, 2], title="Early All Units"
        )

        # Second row plots
        plot_data_TC(
            uniq_Freq,
            Late_mTCOff_facil,
            Late_mTCOn_facil,
            ax=ax[1, 0],
            title="Late Facilitated Units",
        )
        plot_data_TC(
            uniq_Freq,
            Late_mTCOff_supp,
            Late_mTCOn_supp,
            ax=ax[1, 1],
            title="Late Suppressed Units",
        )
        plot_data_TC(
            uniq_Freq, Late_mTCOff, Late_mTCOn, ax=ax[1, 2], title="Late All Units"
        )

        # Assuming the calculation of global_ymin, global_ymax, and yticks as in your previous snippet
        all_ys = np.hstack(
            [a.get_ylim() for row in ax for a in row]
        )  # Flatten the axes array to iterate
        global_ymin, global_ymax = np.min(all_ys), np.max(all_ys)
        tick_spacing = np.ceil((global_ymax - global_ymin) / 5 / 10) * 10
        yticks = np.arange(
            np.floor(global_ymin / 10) * 10,
            np.ceil(global_ymax / 10) * 10 + tick_spacing,
            step=tick_spacing,
        )

        # Apply the calculated y-ticks to each subplot
        for row in ax:
            for a in row:
                a.set_ylim(global_ymin, global_ymax)
                a.set_yticks(yticks)

        fig.align_ylabels(ax[:, 0])  # Align y labels for the first column
        fig.align_xlabels(ax[-1, :])  # Align x labels for the last row

        # Saving the figure
        plot_name = f"Mean_TCs_Divided_{cellType}.pdf"
        fig.savefig(
            os.path.join(fig_dir_time, plot_name),
            bbox_inches="tight",
            dpi=600,
            transparent=True,
        )

        plt.show()


    _()
    return


@app.cell
def _(
    cFR,
    cFR_On,
    cFR_On_facil,
    cFR_On_supp,
    cFR_facil,
    cFR_supp,
    cellType,
    new_black,
    np,
    octaves,
    plt,
    stats,
    ticker,
):
    def _():
        fig, ax = plt.subplots(1, 3)
        fig.set_size_inches(8, 3)

        ax[0].errorbar(
            octaves,
            np.nanmean(cFR_facil, axis=1),
            yerr=stats.sem(cFR_facil, axis=1, nan_policy="omit"),
            ecolor=new_black,
            linestyle="-",
            c=new_black,
            mfc=new_black,
            marker="o",
            markersize=0.25,
            capsize=0.5,
        )

        ax[0].errorbar(
            octaves,
            np.nanmean(cFR_On_facil, axis=1),
            yerr=stats.sem(cFR_On_facil, axis=1, nan_policy="omit"),
            ecolor="CornflowerBlue",
            linestyle="-",
            c="CornflowerBlue",
            mfc="CornflowerBlue",
            marker="o",
            markersize=0.25,
            capsize=0.5,
        )
        ax[0].set_xlim(
            np.round(octaves[0] - 0.025, 3), np.round(octaves[-1] + 0.025, 3)
        )
        ax[0].set_xticks(np.round(octaves, 2))
        ax[0].set_title("Facilitated Neurons")
        ax[0].set_ylabel("FR (Hz)")
        ax[0].set_xlabel("Octaves from Best Frequency")
        ax[0].legend(["Laser Off", "Laser On"])
        ax[0].spines["right"].set_visible(False)
        ax[0].spines["top"].set_visible(False)
        ax[0].set_yticks(np.floor(ax[0].get_yticks() / 10) * 10)
        ymin0, ymax0 = ax[0].get_ylim()
        ax[0].set_ylim(0, ymax0)
        ax[0].yaxis.set_major_locator(ticker.MultipleLocator(20))
        ax[0].yaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))
        ax[0].xaxis.set_major_formatter(ticker.FormatStrFormatter("%0.2g"))

        ax[1].errorbar(
            octaves,
            np.nanmean(cFR_supp, axis=1),
            yerr=stats.sem(cFR_supp, axis=1, nan_policy="omit"),
            ecolor=new_black,
            linestyle="-",
            c=new_black,
            mfc=new_black,
            marker="o",
            markersize=0.25,
            capsize=0.5,
        )

        ax[1].errorbar(
            octaves,
            np.nanmean(cFR_On_supp, axis=1),
            yerr=stats.sem(cFR_On_supp, axis=1, nan_policy="omit"),
            ecolor="CornflowerBlue",
            linestyle="-",
            c="CornflowerBlue",
            mfc="CornflowerBlue",
            marker="o",
            markersize=0.25,
            capsize=0.5,
        )
        ax[1].set_xlim(
            np.round(octaves[0] - 0.025, 3), np.round(octaves[-1] + 0.025, 3)
        )
        ax[1].set_xticks(np.round(octaves, 2))
        ax[1].set_title("Suppressed Neurons")
        ax[1].set_ylabel("FR (Hz)")
        ax[1].set_xlabel("Octaves from Best Frequency")
        ax[1].legend(["Laser Off", "Laser On"])
        ax[1].spines["right"].set_visible(False)
        ax[1].spines["top"].set_visible(False)
        ax[1].set_yticks(np.floor(ax[1].get_yticks() / 10) * 10)
        ymin1, ymax1 = ax[1].get_ylim()
        ax[1].set_ylim(0, ymax1)
        ax[1].yaxis.set_major_locator(ticker.MultipleLocator(20))
        ax[1].yaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))
        ax[1].xaxis.set_major_formatter(ticker.FormatStrFormatter("%0.2g"))

        ax[2].errorbar(
            octaves,
            np.nanmean(cFR, axis=1),
            yerr=stats.sem(cFR, axis=1, nan_policy="omit"),
            ecolor=new_black,
            linestyle="-",
            c=new_black,
            mfc=new_black,
            marker="o",
            markersize=0.25,
            capsize=0.5,
        )

        ax[2].errorbar(
            octaves,
            np.nanmean(cFR_On, axis=1),
            yerr=stats.sem(cFR_On, axis=1, nan_policy="omit"),
            ecolor="CornflowerBlue",
            linestyle="-",
            c="CornflowerBlue",
            mfc="CornflowerBlue",
            marker="o",
            markersize=0.25,
            capsize=0.5,
        )
        ax[2].set_xlim(
            np.round(octaves[0] - 0.025, 3), np.round(octaves[-1] + 0.025, 3)
        )
        ax[2].set_xticks(np.round(octaves, 2))
        ax[2].set_title("All Neurons")
        ax[2].set_ylabel("FR (Hz)")
        ax[2].set_xlabel("Octaves from Best Frequency")
        ax[2].legend(["Laser Off", "Laser On"])
        ax[2].spines["right"].set_visible(False)
        ax[2].spines["top"].set_visible(False)
        ax[2].set_yticks(np.floor(ax[2].get_yticks() / 10) * 10)
        ymin2, ymax2 = ax[2].get_ylim()
        ax[2].set_ylim(0, ymax1)
        ax[2].yaxis.set_major_locator(ticker.MultipleLocator(20))
        ax[2].yaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))
        ax[2].xaxis.set_major_formatter(ticker.FormatStrFormatter("%0.2g"))

        ax[0].tick_params(which="major", color=new_black)
        ax[0].tick_params(which="minor", color=new_black)

        ax[1].tick_params(which="major", color=new_black)
        ax[1].tick_params(which="minor", color=new_black)

        ax[2].tick_params(which="major", color=new_black)
        ax[2].tick_params(which="minor", color=new_black)

        ax[0].set_box_aspect(0.8)
        ax[1].set_box_aspect(0.8)
        ax[2].set_box_aspect(0.8)

        ax[0].spines["left"].set_bounds((0, ymax0))
        ax[1].spines["left"].set_bounds((0, ymax1))
        ax[2].spines["left"].set_bounds((0, ymax2))

        fig.align_ylabels(ax[:])
        fig.align_xlabels(ax[:])

        plot_name = "Centered_TC_divded_%s.pdf" % cellType
        # fig.savefig(os.path.join(fig_dir_time,plot_name))

        plt.show()


    _()
    return


@app.cell
def _(np):
    def norm_percell(FR_arr):
        out = np.array([])
        for cell_FRs in FR_arr.T:
            maxFR = np.nanmax(cell_FRs)
            FR = cell_FRs / maxFR
            out = np.vstack([out, FR]) if out.size else FR

        return out.T
    return (norm_percell,)


@app.cell
def _(
    cFR_On_facil,
    cFR_On_supp,
    cFR_facil,
    cFR_supp,
    calculate_sem,
    cellType,
    errorOff_tc_facil,
    errorOff_tc_supp,
    errorOn_tc_facil,
    errorOn_tc_supp,
    fig_dir_time,
    new_black,
    norm_percell,
    np,
    octaves,
    os,
    plt,
    ticker,
):
    def _():
        fig, ax = plt.subplots(1, 2, figsize=(4, 1.5))

        # Settings for both facilitated and suppressed neurons
        settings = [
            (
                cFR_facil,
                cFR_On_facil,
                errorOff_tc_facil,
                errorOn_tc_facil,
                "Facilitated Neurons",
                ax[0],
            ),
            (
                cFR_supp,
                cFR_On_supp,
                errorOff_tc_supp,
                errorOn_tc_supp,
                "Suppressed Neurons",
                ax[1],
            ),
        ]

        for (
            current_cFR,
            current_cFR_On,
            current_errorOff,
            current_errorOn,
            title,
            axis,
        ) in settings:
            mean_current_cFR = np.nanmean(current_cFR, axis=1)
            mean_current_cFR_On = np.nanmean(current_cFR_On, axis=1)
            # Normalize
            current_cFR_norm = norm_percell(current_cFR)
            current_cFR_On_norm = norm_percell(current_cFR_On)

            mean_current_cFR_norm = np.nanmean(current_cFR_norm, axis=1)
            mean_current_cFR_On_norm = np.nanmean(current_cFR_On_norm, axis=1)

            current_errorOff_norm = calculate_sem(current_cFR_norm)
            current_errorOn_norm = calculate_sem(current_cFR_On_norm)

            # baseline shift
            """
            mean_current_cFR_norm = (
                mean_current_cFR_norm - mean_current_cFR_norm.min()
            )
            mean_current_cFR_On_norm = (
                mean_current_cFR_On_norm - mean_current_cFR_On_norm.min()
            )
            current_errorOff_norm = (
                current_errorOff_norm - mean_current_cFR_norm.min()
            )
            current_errorOn_norm = (
                current_errorOn_norm - mean_current_cFR_On_norm.min()
            )
            """

            axis.plot(
                octaves,
                mean_current_cFR,
                c=new_black,
                linestyle="-",
                linewidth=2,
            )
            axis.fill_between(
                octaves,
                mean_current_cFR - current_errorOff,
                mean_current_cFR + current_errorOff,
                alpha=0.5,
                facecolor=new_black,
            )
            axis.plot(
                octaves,
                mean_current_cFR_On,
                "CornflowerBlue",
                linestyle="-",
                linewidth=2,
            )
            axis.fill_between(
                octaves,
                mean_current_cFR_On - current_errorOn,
                mean_current_cFR_On + current_errorOn,
                alpha=0.5,
                facecolor="CornflowerBlue",
            )
            """
            axis.plot(
                [-1, 1],
                [1, 1],
                c="dimgrey",
                linestyle="--",
                linewidth=1,
            )
            """

            axis.set_xlim(
                np.round(octaves[0] - 0.025, 3), np.round(octaves[-1] + 0.025, 3)
            )
            axis.set_xticks(np.round(octaves, 2))
            axis.set_title(title)
            axis.set_ylabel("FR (norm)")
            axis.set_xlabel("Octaves from Best Frequency")
            axis.legend(["Laser Off", "Laser On"])
            axis.spines["right"].set_visible(False)
            axis.spines["top"].set_visible(False)
            axis.set_yticks(np.floor(axis.get_yticks() / 10) * 10)
            ymin, ymax = axis.get_ylim()
            axis.set_ylim(0, ymax)
            axis.yaxis.set_major_locator(ticker.MultipleLocator(20))
            axis.yaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))
            axis.xaxis.set_major_formatter(ticker.FormatStrFormatter("%0.2g"))
            axis.tick_params(which="major", color=new_black)
            axis.set_box_aspect(0.8)
            axis.spines["left"].set_bounds((0, ymax))
            """
            axis.set_xticks([])
            axis.set_yticks([])
            axis.legend().set_visible(False)
            """
        fig.align_ylabels(ax[:])
        fig.align_xlabels(ax[:])

        plot_name = f"Centered_TC_divided_Patch_{cellType}.pdf"
        fig.savefig(os.path.join(fig_dir_time, plot_name))

        plt.show()


    _()
    return


@app.cell
def _(
    Late_cFR_On_facil,
    Late_cFR_On_supp,
    Late_cFR_facil,
    Late_cFR_supp,
    Late_errorOff_tc_facil,
    Late_errorOff_tc_supp,
    Late_errorOn_tc_facil,
    Late_errorOn_tc_supp,
    calculate_sem,
    cellType,
    fig_dir_time,
    new_black,
    norm_percell,
    np,
    octaves,
    os,
    plt,
    ticker,
):
    def _():
        fig, ax = plt.subplots(1, 2)
        fig.set_size_inches(4, 1.5)

        # Settings for both facilitated and suppressed neurons in the late time period
        late_settings = [
            (
                Late_cFR_facil,
                Late_cFR_On_facil,
                Late_errorOff_tc_facil,
                Late_errorOn_tc_facil,
                "Facilitated NeuronsLate_",
                ax[0],
            ),
            (
                Late_cFR_supp,
                Late_cFR_On_supp,
                Late_errorOff_tc_supp,
                Late_errorOn_tc_supp,
                "Suppressed NeuronsLate_",
                ax[1],
            ),
        ]

        for (
            Late_current_cFR,
            Late_current_cFR_On,
            Late_current_errorOff,
            Late_current_errorOn,
            title,
            axis,
        ) in late_settings:
            Late_mean_current_cFR = np.nanmean(Late_current_cFR, axis=1)
            Late_mean_current_cFR_On = np.nanmean(Late_current_cFR_On, axis=1)
            # Normalize
            Late_current_cFR_norm = norm_percell(Late_current_cFR)
            Late_current_cFR_On_norm = norm_percell(Late_current_cFR_On)

            Late_mean_current_cFR_norm = np.nanmean(Late_current_cFR_norm, axis=1)
            Late_mean_current_cFR_On_norm = np.nanmean(
                Late_current_cFR_On_norm, axis=1
            )

            Late_current_errorOff_norm = calculate_sem(Late_current_cFR_norm)
            Late_current_errorOn_norm = calculate_sem(Late_current_cFR_On_norm)

            # baseline shift
            """
            Late_mean_current_cFR_norm = (
                Late_mean_current_cFR_norm - Late_mean_current_cFR_norm.min()
            )
            Late_mean_current_cFR_On_norm = (
                Late_mean_current_cFR_On_norm - Late_mean_current_cFR_On_norm.min()
            )
            Late_current_errorOff_norm = (
                Late_current_errorOff_norm - Late_mean_current_cFR_norm.min()
            )
            Late_current_errorOn_norm = (
                Late_current_errorOn_norm - Late_mean_current_cFR_On_norm.min()
            )
            """

            axis.plot(
                octaves,
                Late_mean_current_cFR,
                c=new_black,
                linestyle="-",
                linewidth=2,
            )
            axis.fill_between(
                octaves,
                Late_mean_current_cFR - Late_current_errorOff,
                Late_mean_current_cFR + Late_current_errorOff,
                alpha=0.5,
                facecolor=new_black,
            )
            axis.plot(
                octaves,
                Late_mean_current_cFR_On,
                "CornflowerBlue",
                linestyle="-",
                linewidth=2,
            )
            axis.fill_between(
                octaves,
                Late_mean_current_cFR_On - Late_current_errorOn,
                Late_mean_current_cFR_On + Late_current_errorOn,
                alpha=0.5,
                facecolor="CornflowerBlue",
            )
            """
            axis.plot(
                [-1, 1],
                [1, 1],
                c="dimgrey",
                linestyle="--",
                linewidth=1,
            )
            """
            axis.set_xlim(
                np.round(octaves[0] - 0.025, 3), np.round(octaves[-1] + 0.025, 3)
            )
            axis.set_xticks(np.round(octaves, 2))
            axis.set_title(title)
            axis.set_ylabel("FR (norm)")
            axis.set_xlabel("Octaves from Best Frequency")
            axis.legend(["Laser Off", "Laser On"])
            axis.spines["right"].set_visible(False)
            axis.spines["top"].set_visible(False)
            axis.set_yticks(np.floor(axis.get_yticks() / 10) * 10)
            ymin, ymax = axis.get_ylim()
            axis.set_ylim(0, ymax)
            axis.yaxis.set_major_locator(ticker.MultipleLocator(20))
            axis.yaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))
            axis.xaxis.set_major_formatter(ticker.FormatStrFormatter("%0.2g"))
            axis.tick_params(which="major", color=new_black)
            axis.set_box_aspect(0.8)
            axis.spines["left"].set_bounds((0, ymax))
            """
            axis.set_xticks([])
            axis.set_yticks([])
            axis.legend().set_visible(False)
            """
        fig.align_ylabels(ax[:])
        fig.align_xlabels(ax[:])

        plot_name = f"Centered_TC_divided_Patch_LateTimePeriod{cellType}.pdf"
        fig.savefig(os.path.join(fig_dir_time, plot_name))

        plt.show()


    _()
    return


@app.cell
def _(BF, cellType, depths_numeric, fig_dir_time, new_black, os, plt):
    def _():
        from scipy.stats import linregress

        fig, ax = plt.subplots()
        fig.set_size_inches(4, 3)
        ax.scatter(BF, depths_numeric, c=new_black, s=0.75)

        res = linregress(BF, depths_numeric)
        print(f"slope: {res.slope:.6f}")
        print(f"intercept: {res.intercept:.6f}")
        print(f"R-value: {res.rvalue:.6f}")
        print(f"R-squared: {res.rvalue**2:.6f}")
        print(f"p-value: {res.pvalue:.6f}")

        ax.plot(
            [min(BF), max(BF)],
            res.intercept + [res.slope * min(BF), res.slope * max(BF)],
            "red",
        )

        ax.set_box_aspect(1)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Probe Depth (µm)")
        ax.invert_yaxis()
        ax.set_xscale("log")
        # ax.tick_params(which="major", color=new_black)
        # ax.tick_params(which="minor", color=new_black)
        ax.set_xticks([10000, 100000])
        ax.get_xaxis().get_major_formatter().labelOnlyBase = False
        plot_name = "BF_ProbeDepth_%s.pdf" % cellType
        fig.savefig(os.path.join(fig_dir_time, plot_name))

        # fig.savefig(os.path.join(fig_dir_time,"BF_ProbeDepth.pdf"), bbox_inches='tight', dpi=600, transparent = True)

        plt.show()


    _()
    return


@app.cell
def _(
    Late_SparsenessOff_facilitated,
    Late_SparsenessOff_suppressed,
    Late_SparsenessOn_facilitated,
    Late_SparsenessOn_suppressed,
    SparsenessOff_all,
    SparsenessOff_facilitated,
    SparsenessOff_suppressed,
    SparsenessOn_all,
    SparsenessOn_facilitated,
    SparsenessOn_suppressed,
    cellType,
    fig_dir_time,
    new_black,
    os,
    pd,
    plt,
    sns,
):
    def _():
        data_frames = {
            "All": [SparsenessOff_all, SparsenessOn_all],
            "Facilitated_Early": [
                SparsenessOff_facilitated,
                SparsenessOn_facilitated,
            ],
            "Suppressed_Early": [
                SparsenessOff_suppressed,
                SparsenessOn_suppressed,
            ],
            "Facilitated_Late": [
                Late_SparsenessOff_facilitated,
                Late_SparsenessOn_facilitated,
            ],
            "Suppressed_Late": [
                Late_SparsenessOff_suppressed,
                Late_SparsenessOn_suppressed,
            ],
        }

        # Create a combined DataFrame
        df_test = pd.concat(
            [
                pd.DataFrame(
                    {
                        "Sparseness": df,
                        "Condition": "Off" if i % 2 == 0 else "On",
                        "Type": key,
                    }
                )
                for key, dfs in data_frames.items()
                for i, df in enumerate(dfs)
            ],
            ignore_index=True,
        )

        # Define custom parameters to remove top and right spines
        custom_params = {"axes.spines.right": False, "axes.spines.top": False}
        sns.set_theme(style="ticks", rc=custom_params)

        # Define the colors for boxplots
        colors_facilitated = [
            "#002c94",
            "CornflowerBlue",
        ]  # Off, On for facilitated
        colors_suppressed = [
            "#85000c",
            "CornflowerBlue",
        ]  # Off, On for suppressed

        colors_facilitated_new = [
            "#002c94",
            "CornflowerBlue",
        ]  # Off, On for facilitated
        colors_suppressed_new = [
            "#85000c",
            "#C28187",
        ]  # Off, On for suppressed

        colors_facilitated_new2 = [
            "CornflowerBlue",
            "#002c94",
        ]  # Off, On for facilitated
        colors_suppressed_new2 = [
            "#C28187",
            "#85000c",
        ]  # Off, On for suppressed

        # Define the categories and their corresponding color maps for boxplots
        categories = [
            "Facilitated_Early",
            "Facilitated_Late",
            "Suppressed_Early",
            "Suppressed_Late",
        ]
        color_maps = {
            "Facilitated_Early": colors_facilitated,
            "Facilitated_Late": colors_facilitated,
            "Suppressed_Early": colors_suppressed,
            "Suppressed_Late": colors_suppressed,
        }

        # Prepare the plot
        fig, axes = plt.subplots(
            nrows=2, ncols=2, figsize=(5, 10), sharey=True
        )  # Adjusted figsize
        axes = axes.flatten()

        # Loop through each category to plot
        for idx, category in enumerate(categories):
            ax = axes[idx]
            subset = df_test[df_test["Type"] == category]

            # Boxplot
            sns.boxplot(
                data=subset,
                x="Type",
                y="Sparseness",
                hue="Condition",
                palette=color_maps[category],
                ax=ax,
                zorder=1,
            )

            # Stripplot with 'dimgrey' color using palette
            sns.stripplot(
                data=subset,
                x="Type",
                y="Sparseness",
                hue="Condition",
                dodge=True,
                palette=[new_black, new_black],
                ax=ax,
                alpha=0.60,
                jitter=True,
                edgecolor="gray",
                linewidth=0.5,
                zorder=2,
            )

            # Customize legend
            handles, labels = ax.get_legend_handles_labels()
            if idx == 0:
                ax.legend(handles[:2], ["Off", "On"], title="Condition")
            else:
                ax.legend([], [], frameon=False)

            # Set title
            ax.set_title(category)

        # Adjust labels and layout
        plt.tight_layout(pad=3.0)  # Adjusted padding for layout

        plot_name = f"Sparseness_divided_{cellType}.pdf"
        fig.savefig(os.path.join(fig_dir_time, plot_name))

        plt.show()


    _()
    return


@app.cell
def _(
    early_evoked_Ind,
    edges,
    find_time_to_peak,
    peakCellstuseOff,
    peakCellstuseOff_facil,
    peakCellstuseOff_supp,
    peakCellstuseOn,
    peakCellstuseOn_facil,
    peakCellstuseOn_supp,
    spontInd,
):
    # Use the function for each condition
    t2pOff, t2pOn = find_time_to_peak(
        peakCellstuseOff, peakCellstuseOn, edges, early_evoked_Ind, spontInd
    )
    t2pOff_facil, t2pOn_facil = find_time_to_peak(
        peakCellstuseOff_facil,
        peakCellstuseOn_facil,
        edges,
        early_evoked_Ind,
        spontInd,
    )
    t2pOff_supp, t2pOn_supp = find_time_to_peak(
        peakCellstuseOff_supp,
        peakCellstuseOn_supp,
        edges,
        early_evoked_Ind,
        spontInd,
    )
    return t2pOff_facil, t2pOff_supp, t2pOn_facil, t2pOn_supp


@app.cell
def _(
    cellType,
    fig_dir_time,
    new_black,
    os,
    pd,
    plt,
    sns,
    t2pOff_facil,
    t2pOff_supp,
    t2pOn_facil,
    t2pOn_supp,
):
    def _():
        # Creating separate dataframes and adding a new 'Condition' column
        df_facil_off = pd.DataFrame(
            {
                "Time": [time * 1000 for time in t2pOff_facil],
                "Condition": "Facilitated Off",
            }
        )
        df_facil_on = pd.DataFrame(
            {
                "Time": [time * 1000 for time in t2pOn_facil],
                "Condition": "Facilitated On",
            }
        )
        df_supp_off = pd.DataFrame(
            {
                "Time": [time * 1000 for time in t2pOff_supp],
                "Condition": "Suppressed Off",
            }
        )
        df_supp_on = pd.DataFrame(
            {
                "Time": [time * 1000 for time in t2pOn_supp],
                "Condition": "Suppressed On",
            }
        )

        # Concatenating dataframes by group
        df_facilitated = pd.concat([df_facil_off, df_facil_on], ignore_index=True)
        df_suppressed = pd.concat([df_supp_off, df_supp_on], ignore_index=True)

        # Define the colors for boxplots
        colors_facilitated = [
            "#002c94",
            "CornflowerBlue",
        ]  # Off, On for facilitated
        colors_suppressed = [
            "#85000c",
            "CornflowerBlue",
        ]  # Off, On for suppressed

        colors_facilitated_new = [
            "#002c94",
            "CornflowerBlue",
        ]  # Off, On for facilitated
        colors_suppressed_new = [
            "#85000c",
            "#C28187",
        ]  # Off, On for suppressed

        colors_facilitated_new2 = [
            "CornflowerBlue",
            "#002c94",
        ]  # Off, On for facilitated
        colors_suppressed_new2 = [
            "#C28187",
            "#85000c",
        ]  # Off, On for suppressed

        # Plotting for Facilitated Conditions
        fig_facil, ax_facil = plt.subplots(figsize=(2.5, 5))
        sns.boxplot(
            x="Condition",
            y="Time",
            data=df_facilitated,
            palette=colors_facilitated,
            medianprops={"color": new_black, "linewidth": 2},
            width=0.5,
        )
        sns.stripplot(
            x="Condition",
            y="Time",
            data=df_facilitated,
            color=new_black,
            size=4,
            jitter=True,
            alpha=0.8,
        )
        ax_facil.set_title("Time to Peak for Facilitated Conditions")
        ax_facil.set_ylabel("Time to Peak (ms)")
        ax_facil.set_xlabel("Condition")

        # Save the Facilitated plot
        plot_name_facil = f"Time2Peak_Facilitated_{cellType}.pdf"
        fig_facil.savefig(os.path.join(fig_dir_time, plot_name_facil))

        # Plotting for Suppressed Conditions
        fig_supp, ax_supp = plt.subplots(figsize=(2.5, 5))
        sns.boxplot(
            x="Condition",
            y="Time",
            data=df_suppressed,
            palette=colors_suppressed,
            medianprops={"color": new_black, "linewidth": 2},
            width=0.5,
        )
        sns.stripplot(
            x="Condition",
            y="Time",
            data=df_suppressed,
            color=new_black,
            size=4,
            jitter=True,
            alpha=0.8,
        )
        ax_supp.set_title("Time to Peak for Suppressed Conditions")
        ax_supp.set_ylabel("Time to Peak (ms)")
        ax_supp.set_xlabel("Condition")

        # Save the Suppressed plot
        plot_name_supp = f"Time2Peak_Suppressed_{cellType}.pdf"
        fig_supp.savefig(os.path.join(fig_dir_time, plot_name_supp))

        plt.show()


    _()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Example Cells
    """)
    return


@app.cell
def _(facilitated_Ind, psthTC, psthTC_On, suppressed_Ind):
    psthTC_facil = psthTC[:, :, facilitated_Ind]
    psthTC_On_facil = psthTC_On[:, :, facilitated_Ind]

    psthTC_suppressed = psthTC[:, :, suppressed_Ind]
    psthTC_On_suppressed = psthTC_On[:, :, suppressed_Ind]
    return (
        psthTC_On_facil,
        psthTC_On_suppressed,
        psthTC_facil,
        psthTC_suppressed,
    )


@app.cell
def _(
    AutoMinorLocator,
    Facil_Rasters_toneResp,
    Facil_SpikeSortI_toneResp,
    Late_mTCOff_facil,
    Late_mTCOn_facil,
    cellType,
    cmr,
    facilitated_cells_Off,
    facilitated_cells_On,
    fig_dir_time,
    laserDur,
    mTCOff_facil,
    mTCOn_facil,
    new_black,
    np,
    os,
    plt,
    psthTC_On_facil,
    psthTC_facil,
    tDur,
    tStart,
    ticker,
    time,
    trialOrder,
    uniq_Freq,
):
    def _():
        # PLOT EXAMPLE CELLS FACILITATED
        # cellID = 69 #PV
        cellID = 1  # SOM

        fig = plt.figure()
        gs = fig.add_gridspec(2, 4)
        fig.set_size_inches(6, 3)

        ax = fig.add_subplot(gs[0, 0])
        ax.plot(uniq_Freq, mTCOff_facil[:, cellID], c=new_black)
        ax.plot(uniq_Freq, mTCOn_facil[:, cellID], "CornflowerBlue")
        ax.set_xscale("log")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        # ax.set_box_aspect(1)
        # ax[1,0].xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
        ax.set_title("Facil Frequency Response Cell = %i" % cellID)
        ax.set_ylabel("FR (Hz)")
        ax.set_xlabel("Frequency (Hz)")
        ax.legend(["Laser Off", "Laser On"])
        ymin0, ymax0 = ax.get_ylim()
        ax.set_ylim([0, ymax0])
        ax.yaxis.set_major_locator(ticker.MultipleLocator(20))
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))

        ax.tick_params(which="major", color=new_black)
        ax.tick_params(which="minor", color=new_black)
        ax.set_box_aspect(1)

        ax1 = fig.add_subplot(gs[0, 1], sharey=ax, sharex=ax)
        plt.setp(ax1.get_yticklabels(), visible=False)
        ax1.plot(uniq_Freq, Late_mTCOff_facil[:, cellID], c=new_black)
        ax1.plot(uniq_Freq, Late_mTCOn_facil[:, cellID], "CornflowerBlue")
        ax1.set_xscale("log")
        ax1.spines["right"].set_visible(False)
        ax1.spines["top"].set_visible(False)
        # ax.set_box_aspect(1)
        # ax[1,0].xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
        ax1.set_title("Facil Frequency Response Cell = %i" % cellID)
        # ax1.set_ylabel('FR (Hz)')
        ax1.set_xlabel("Frequency (Hz)")
        ax1.legend(["Laser Off", "Laser On"])
        ymin1, ymax1 = ax1.get_ylim()
        ax1.set_ylim([0, ymax1])
        ax1.yaxis.set_major_locator(ticker.MultipleLocator(20))
        ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))

        ax1.tick_params(which="major", color=new_black)
        ax1.tick_params(which="minor", color=new_black)
        ax1.set_box_aspect(1)

        ax2 = fig.add_subplot(gs[1, :-2])
        ax2.plot(time, facilitated_cells_Off[:, cellID], c=new_black)
        ax2.plot(time, facilitated_cells_On[:, cellID], c="CornflowerBlue")
        ax2.axvline(x=tStart, ls="--", c=new_black)
        ax2.axvline(x=tDur, ls="--", c=new_black)
        ax2.legend(["Laser Off", "Laser On"])
        ax2.set(xlabel="Time (s)")
        ax2.set(ylabel="FR (Hz)")
        ax2.set_title("PSTH for Cell ID = %i" % cellID)
        ax2.spines["right"].set_visible(False)
        ax2.spines["top"].set_visible(False)
        ax2.set_yticks(np.ceil(ax2.get_yticks() / 10) * 10)
        ymin2, ymax2 = ax2.get_ylim()
        ax2.set_ylim(0, ymax2)
        ax2.yaxis.set_major_locator(ticker.MultipleLocator(20))
        ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))

        ax2.set_xlim(np.min(time), np.max(time))
        ax2.xaxis.set_major_formatter(ticker.FormatStrFormatter("%0.2g"))
        ax2.xaxis.set_major_locator(ticker.MultipleLocator(0.05))
        ax2.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax2.tick_params(which="major", color=new_black)
        ax2.tick_params(which="minor", color=new_black)
        # ax2.set_box_aspect(.6)

        ax3 = fig.add_subplot(gs[0:, 2])
        for uniq in range(len(uniq_Freq)):
            ax3.plot(time, psthTC_facil[uniq, :, cellID] + 250 * uniq, c=new_black)
            ax3.plot(
                time,
                psthTC_On_facil[uniq, :, cellID] + 250 * uniq,
                "CornflowerBlue",
            )
            ax3.set_box_aspect(2)

        ax3.axvline(x=tStart, ls="--", c=new_black)
        ax3.axvline(x=tDur, ls="--", c=new_black)
        ymin3, ymax3 = ax3.get_ylim()
        ax3.set_ylim(0, ymax3)
        ax3.yaxis.set_major_locator(ticker.LinearLocator(numticks=20))
        ax3.set_yticklabels(
            ticker.FormatStrFormatter("%d").format_ticks(np.ceil(uniq_Freq / 1000))
        )
        ax3.set_xlim(np.min(time), np.max(time))
        ax3.xaxis.set_major_formatter(ticker.FormatStrFormatter("%0.2g"))
        ax3.xaxis.set_major_locator(ticker.MultipleLocator(0.05))
        ax3.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax3.tick_params(which="major", color=new_black)
        ax3.tick_params(which="minor", color=new_black)
        ax3.set_xlabel("Time(s)")
        ax3.set_ylabel("Frequency (kHz)")

        ax4 = fig.add_subplot(gs[0:, 3])

        cmap = cmr.get_sub_cmap("cmr.tropical", 0.1, 1, N=len(uniq_Freq))
        c_IND = np.ceil(Facil_SpikeSortI_toneResp[cellID] / 20) * 20
        spikesortind_c_off = np.where(
            Facil_SpikeSortI_toneResp[cellID] <= np.max(c_IND / 2)
        )
        spikesortind_c_on = np.where(
            Facil_SpikeSortI_toneResp[cellID] > np.max(c_IND / 2)
        )

        spikesortid_off = Facil_SpikeSortI_toneResp[cellID][spikesortind_c_off]
        spikesortid_on = Facil_SpikeSortI_toneResp[cellID][spikesortind_c_on]

        rastoff = np.asarray(Facil_Rasters_toneResp[cellID])[spikesortind_c_off]
        raston = np.asarray(Facil_Rasters_toneResp[cellID])[spikesortind_c_on]

        pp1 = plt.Rectangle(
            (tStart, len(trialOrder) / 2), laserDur, len(trialOrder) / 2, alpha=0.1
        )
        pp2 = plt.Rectangle(
            (tStart, tStart),
            tDur,
            len(trialOrder) / 2,
            facecolor=new_black,
            alpha=0.1,
        )
        ax4.add_patch(pp1)
        ax4.add_patch(pp2)
        #     ax[0].plot(Rasters,spikeSortI,'ko',markersize=2.3)
        cax = ax4.scatter(
            rastoff,
            spikesortid_off,
            0.25,
            c=spikesortid_off,
            cmap=cmap,
            marker="|",
            linewidth=0.5,
        )

        cax2 = ax4.scatter(
            raston,
            spikesortid_on,
            0.25,
            c=spikesortid_on,
            cmap=cmap,
            marker="|",
            linewidth=0.5,
        )

        cbar = fig.colorbar(cax, shrink=0.4)
        #     cbar.ax.ticker.LinearLocator(numticks=None, presets=None)
        cbar.ax.yaxis.set_major_locator(ticker.LinearLocator(19))
        #     cbar.ax.locator_params(axis = 'y', nbins=20)
        # cbar.set_ticks(uniq_Freq/1000)
        cbar.ax.set_yticklabels(
            ticker.FormatStrFormatter("%d").format_ticks(
                np.ceil(uniq_Freq / 1000)
            ),
            fontsize=4,
        )
        cbar.set_label("Frequency(kHz)")
        #     ax[0].scatter(Rasters,spikeSortI,2,'k','o')
        ax4.set_xlabel("Time(s)")
        ax4.set_ylabel("Trials")
        ax4.set_title("Raster Plot for Cell ID = %i" % cellID)
        ax4.tick_params(axis="x")
        ax4.tick_params(axis="y")
        ax4.locator_params(axis="y", nbins=5)
        ax4.locator_params(axis="x", nbins=8)
        ax4.set_ylim([0, len(trialOrder)])
        ax4.yaxis.set_major_locator(ticker.LinearLocator(3))
        ax4.yaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))
        ax4.set_xlim(np.min(time), np.max(time))
        ax4.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax4.xaxis.set_major_formatter(ticker.FormatStrFormatter("%0.2g"))
        ax4.tick_params(which="major", color=new_black)
        ax4.tick_params(which="minor", color=new_black)

        # fig.align_labels(cbar.ax)
        ax4.set_box_aspect(2)

        # ax[0].spines.bottom.set_bounds(x.min(), x.max())
        ax.spines["left"].set_bounds((0, ymax0))
        ax1.spines["left"].set_bounds((0, ymax1))
        ax2.spines["left"].set_bounds((0, ymax2))

        # fig.align_ylabels(ax[:])
        # fig.align_xlabels(ax[:])

        plot_name = "Example_Cell_Facil%s.pdf" % cellType
        fig.savefig(os.path.join(fig_dir_time, plot_name))

        plt.show()


    _()
    return


@app.cell
def _(
    AutoMinorLocator,
    Late_mTCOff_supp,
    Late_mTCOn_supp,
    Supp_Rasters_toneResp,
    Supp_SpikeSortI_toneResp,
    cellType,
    cmr,
    fig_dir_time,
    laserDur,
    mTCOff_supp,
    mTCOn_supp,
    new_black,
    np,
    os,
    plt,
    psthTC_On_suppressed,
    psthTC_suppressed,
    suppressed_cells_Off,
    suppressed_cells_On,
    tDur,
    tStart,
    ticker,
    time,
    trialOrder,
    uniq_Freq,
):
    def _():
        # PLOT EXAMPLE CELLS SUPPRESSED
        # cellID_supp = 53 #PV
        cellID_supp = 1  # som

        fig = plt.figure()
        gs = fig.add_gridspec(2, 4)
        fig.set_size_inches(6, 3)

        ax = fig.add_subplot(gs[0, 0])
        ax.plot(uniq_Freq, mTCOff_supp[:, cellID_supp], c=new_black)
        ax.plot(uniq_Freq, mTCOn_supp[:, cellID_supp], "CornflowerBlue")
        ax.set_xscale("log")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        # ax.set_box_aspect(1)
        # ax[1,0].xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
        ax.set_title("Facil Frequency Response Cell = %i" % cellID_supp)
        ax.set_ylabel("FR (Hz)")
        ax.set_xlabel("Frequency (Hz)")
        ax.legend(["Laser Off", "Laser On"])
        ymin0, ymax0 = ax.get_ylim()
        ax.set_ylim([0, ymax0])
        ax.yaxis.set_major_locator(ticker.MultipleLocator(20))
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))

        ax.tick_params(which="major", color=new_black)
        ax.tick_params(which="minor", color=new_black)
        ax.set_box_aspect(1)

        ax1 = fig.add_subplot(gs[0, 1], sharey=ax, sharex=ax)
        plt.setp(ax1.get_yticklabels(), visible=False)
        ax1.plot(uniq_Freq, Late_mTCOff_supp[:, cellID_supp], c=new_black)
        ax1.plot(uniq_Freq, Late_mTCOn_supp[:, cellID_supp], "CornflowerBlue")
        ax1.set_xscale("log")
        ax1.spines["right"].set_visible(False)
        ax1.spines["top"].set_visible(False)
        # ax.set_box_aspect(1)
        # ax[1,0].xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
        ax1.set_title("Facil Frequency Response Cell = %i" % cellID_supp)
        # ax1.set_ylabel('FR (Hz)')
        ax1.set_xlabel("Frequency (Hz)")
        ax1.legend(["Laser Off", "Laser On"])
        ymin1, ymax1 = ax1.get_ylim()
        ax1.set_ylim([0, ymax1])
        ax1.yaxis.set_major_locator(ticker.MultipleLocator(20))
        ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))

        ax1.tick_params(which="major", color=new_black)
        ax1.tick_params(which="minor", color=new_black)
        ax1.set_box_aspect(1)

        ax2 = fig.add_subplot(gs[1, :-2])
        ax2.plot(time, suppressed_cells_Off[:, cellID_supp], c=new_black)
        ax2.plot(time, suppressed_cells_On[:, cellID_supp], c="CornflowerBlue")
        ax2.axvline(x=tStart, ls="--", c=new_black)
        ax2.axvline(x=tDur, ls="--", c=new_black)
        ax2.legend(["Laser Off", "Laser On"])
        ax2.set(xlabel="Time (s)")
        ax2.set(ylabel="FR (Hz)")
        ax2.set_title("PSTH for Cell ID = %i" % cellID_supp)
        ax2.spines["right"].set_visible(False)
        ax2.spines["top"].set_visible(False)
        ax2.set_yticks(np.ceil(ax2.get_yticks() / 10) * 10)
        ymin2, ymax2 = ax2.get_ylim()
        ax2.set_ylim(0, ymax2)
        ax2.yaxis.set_major_locator(ticker.MultipleLocator(20))
        ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))

        ax2.set_xlim(np.min(time), np.max(time))
        ax2.xaxis.set_major_formatter(ticker.FormatStrFormatter("%0.2g"))
        ax2.xaxis.set_major_locator(ticker.MultipleLocator(0.05))
        ax2.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax2.tick_params(which="major", color=new_black)
        ax2.tick_params(which="minor", color=new_black)
        # ax2.set_box_aspect(.8)

        ax3 = fig.add_subplot(gs[0:, 2])
        for uniq in range(len(uniq_Freq)):
            ax3.plot(
                time,
                psthTC_suppressed[uniq, :, cellID_supp] + 250 * uniq,
                c=new_black,
            )
            ax3.plot(
                time,
                psthTC_On_suppressed[uniq, :, cellID_supp] + 250 * uniq,
                "CornflowerBlue",
            )
            ax3.set_box_aspect(2)

        ax3.axvline(x=tStart, ls="--", c=new_black)
        ax3.axvline(x=tDur, ls="--", c=new_black)
        ymin3, ymax3 = ax3.get_ylim()
        ax3.set_ylim(0, ymax3)
        ax3.yaxis.set_major_locator(ticker.LinearLocator(numticks=20))
        ax3.set_yticklabels(
            ticker.FormatStrFormatter("%d").format_ticks(np.ceil(uniq_Freq / 1000))
        )
        # ax3.set_yticklabels(np.ceil(uniq_Freq/1000));

        ax3.set_xlim(np.min(time), np.max(time))
        ax3.xaxis.set_major_formatter(ticker.FormatStrFormatter("%0.2g"))
        ax3.xaxis.set_major_locator(ticker.MultipleLocator(0.05))
        ax3.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax3.tick_params(which="major", color=new_black)
        ax3.tick_params(which="minor", color=new_black)
        ax3.set_xlabel("Time(s)")
        ax3.set_ylabel("Frequency (kHz)")

        ax4 = fig.add_subplot(gs[0:, 3])

        cmap = cmr.get_sub_cmap("cmr.tropical", 0.1, 1, N=len(uniq_Freq))
        c_IND = np.ceil(Supp_SpikeSortI_toneResp[cellID_supp] / 20) * 20
        spikesortind_c_off = np.where(
            Supp_SpikeSortI_toneResp[cellID_supp] <= np.max(c_IND / 2)
        )
        spikesortind_c_on = np.where(
            Supp_SpikeSortI_toneResp[cellID_supp] > np.max(c_IND / 2)
        )

        spikesortid_off = Supp_SpikeSortI_toneResp[cellID_supp][spikesortind_c_off]
        spikesortid_on = Supp_SpikeSortI_toneResp[cellID_supp][spikesortind_c_on]

        rastoff = np.asarray(Supp_Rasters_toneResp[cellID_supp])[
            spikesortind_c_off
        ]
        raston = np.asarray(Supp_Rasters_toneResp[cellID_supp])[spikesortind_c_on]

        pp1 = plt.Rectangle(
            (tStart, len(trialOrder) / 2), laserDur, len(trialOrder) / 2, alpha=0.1
        )
        pp2 = plt.Rectangle(
            (tStart, tStart),
            tDur,
            len(trialOrder) / 2,
            facecolor=new_black,
            alpha=0.1,
        )
        ax4.add_patch(pp1)
        ax4.add_patch(pp2)
        #     ax[0].plot(Rasters,spikeSortI,'ko',markersize=2.3)
        cax = ax4.scatter(
            rastoff,
            spikesortid_off,
            0.25,
            c=spikesortid_off,
            cmap=cmap,
            marker="|",
            linewidth=0.5,
        )
        cax2 = ax4.scatter(
            raston,
            spikesortid_on,
            0.25,
            c=spikesortid_on,
            cmap=cmap,
            marker="|",
            linewidth=0.5,
        )

        cbar = fig.colorbar(cax, shrink=0.4)
        #     cbar.ax.ticker.LinearLocator(numticks=None, presets=None)
        cbar.ax.yaxis.set_major_locator(ticker.LinearLocator(19))
        #     cbar.ax.locator_params(axis = 'y', nbins=20)
        # cbar.set_ticks(uniq_Freq/1000)
        cbar.ax.set_yticklabels(
            ticker.FormatStrFormatter("%d").format_ticks(
                np.ceil(uniq_Freq / 1000)
            ),
            fontsize=4,
        )
        cbar.set_label("Frequency(kHz)")
        #     ax[0].scatter(Rasters,spikeSortI,2,'k','o')
        ax4.set_xlabel("Time(s)")
        ax4.set_ylabel("Trials")
        ax4.set_title("Raster Plot for Cell ID = %i" % cellID_supp)
        ax4.tick_params(axis="x")
        ax4.tick_params(axis="y")
        ax4.locator_params(axis="y", nbins=5)
        ax4.locator_params(axis="x", nbins=8)
        ax4.set_ylim([0, len(trialOrder)])
        ax4.yaxis.set_major_locator(ticker.LinearLocator(3))
        ax4.yaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))
        ax4.set_xlim(np.min(time), np.max(time))
        ax4.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax4.xaxis.set_major_formatter(ticker.FormatStrFormatter("%0.2g"))
        ax4.tick_params(which="major", color=new_black)
        ax4.tick_params(which="minor", color=new_black)

        # fig.align_labels(cbar.ax)
        ax4.set_box_aspect(2)

        # ax[0].spines.bottom.set_bounds(x.min(), x.max())
        ax.spines["left"].set_bounds((0, ymax0))
        ax1.spines["left"].set_bounds((0, ymax1))
        ax2.spines["left"].set_bounds((0, ymax2))

        # fig.align_ylabels(ax[:])
        # fig.align_xlabels(ax[:])

        plot_name = "Example_Cell_Suppressed%s.pdf" % cellType
        fig.savefig(os.path.join(fig_dir_time, plot_name))

        plt.show()


    _()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Statistics
    """)
    return


@app.cell
def _():
    from helper_statistics import (
        wilcoxon_summary,
        create_stats_df,
        describe_and_enhance,
        perform_analysis,
        perform_analysis_octaves,
        pad_and_create_dataframe,
        create_and_describe_df,
        save_dataframes,
        calculate_statistics,
        perform_statistical_analysis,
    )
    return (
        calculate_statistics,
        create_and_describe_df,
        create_stats_df,
        describe_and_enhance,
        pad_and_create_dataframe,
        perform_analysis,
        perform_analysis_octaves,
        perform_statistical_analysis,
        save_dataframes,
        wilcoxon_summary,
    )


@app.cell
def _(cellType, fig_dir_time, os, toneRespPSTH):
    # Assuming toneRespPSTH.shape returns something like (time, trials, n_cells)
    _, _, n_cells = toneRespPSTH.shape

    # Convert number of cells to string for writing
    n_cells_str = str(n_cells)

    # Prepare file name and path
    cellnum_type = f"N_allCells_{cellType}.txt"
    cellnum_path = os.path.join(fig_dir_time, cellnum_type)

    # Using with statement for better file handling
    with open(cellnum_path, "w") as cellnum_file:
        cellnum_file.write(f"Number of Significant Units: {n_cells_str}")
    return


@app.cell
def _(cellType, depths_numeric_facil, diff_Sorted_facil, wilcoxon_summary):
    def _():
        # FIGURE 2 STATS
        # data is in diff_Sorted_facil and depths_numeric_facil

        bin1 = diff_Sorted_facil[
            (depths_numeric_facil > 2.5) & (depths_numeric_facil < 2.9)
        ]  # first bin
        bin2 = diff_Sorted_facil[
            (depths_numeric_facil > 3.1) & (depths_numeric_facil < 3.5)
        ]  # second bin

        print("facilitated", cellType)
        result = wilcoxon_summary(bin1, bin2)
        print(result.round(3))


    _()
    return


@app.cell
def _(cellType, depths_numeric_supp, diff_Sorted_supp, wilcoxon_summary):
    def _():
        bin1 = diff_Sorted_supp[
            (depths_numeric_supp > 2.5) & (depths_numeric_supp < 2.9)
        ]  # first bin
        bin2 = diff_Sorted_supp[
            (depths_numeric_supp > 3.1) & (depths_numeric_supp < 3.5)
        ]  # second bin

        print("suppressed", cellType)
        result = wilcoxon_summary(bin1, bin2)
        print(result.round(3))


    _()
    return


@app.cell
def _(
    cellType,
    create_stats_df,
    describe_and_enhance,
    early_evoked_Ind,
    facilitated_cells_Off,
    facilitated_cells_On,
    late_evoked_Ind,
    meanLaserOff_masked,
    meanLaserOn_masked,
    os,
    pd,
    perform_analysis,
    spontInd,
    stats_dir_time,
    suppressed_cells_Off,
    suppressed_cells_On,
):
    ##########################################################################
    ############################## Figure 3 & 4 ##############################
    indices = {
        "early": early_evoked_Ind,
        "late": late_evoked_Ind,
        "spont": spontInd,
    }
    data_sources = {
        "All": (meanLaserOff_masked, meanLaserOn_masked),
        "Facilitated": (facilitated_cells_Off, facilitated_cells_On),
        "Suppressed": (suppressed_cells_Off, suppressed_cells_On),
        "Spontaneous": (meanLaserOff_masked, meanLaserOn_masked),
    }
    stats_dfs = {
        key: create_stats_df(off, on, indices)
        for key, (off, on) in data_sources.items()
    }
    # Define indices and data sources once for consistency and reusability
    described_statsDF = {
        key: describe_and_enhance(_df) for key, _df in stats_dfs.items()
    }
    test_statistics = {
        f"{key} {period}": (
            _df,
            f"Mean FR Off {period.capitalize()}",
            f"Mean FR On {period.capitalize()}",
        )
        for key, _df in stats_dfs.items()
        for period in ["early", "late", "spont"]
    }
    results = {
        test_name: perform_analysis(*params)
        for test_name, params in test_statistics.items()
    }
    pVal_FR_DF = pd.DataFrame(
        results, index=["Test Statistic", "p Value", "Z Value", "Effect Size"]
    )
    pVal_FR_DF = pd.DataFrame(results).T
    pVal_FR_DF.columns = [
        "Test Statistic",
        "p Value",
        "Z Value",
        "Effect Size",
    ]
    pVal_FR_DF["p Value"] = pVal_FR_DF["p Value"].map(lambda x: f"{x:.8f}")
    for key, _df in described_statsDF.items():
        described_statsDF_path = os.path.join(
            stats_dir_time, f"StatsDF_FR_{key}_{cellType}.xlsx"
        )
        _df.to_excel(described_statsDF_path, index=True)
    pVal_FR_DF_path = os.path.join(
        stats_dir_time, f"PVal_FR_DF_{cellType}.xlsx"
    )  # Assuming same source for simplicity
    pVal_FR_DF.to_excel(pVal_FR_DF_path, index=True)
    for key, _df in described_statsDF.items():
        described_statsDF_path = os.path.join(
            stats_dir_time, f"StatsDF_FR_{key}_{cellType}.csv"
        )
        # Create all DataFrames
        _df.to_csv(described_statsDF_path, index=True)
    pVal_FR_DF_path = os.path.join(stats_dir_time, f"PVal_FR_DF_{cellType}.csv")
    pVal_FR_DF.to_csv(pVal_FR_DF_path, index=True)
    # Describe and enhance all DataFrames
    # Format the 'p Value' column to eight decimal places
    # described_statsDF is a dictionary of DataFrames
    # pVal_FR_DF as a DataFrame
    print(
        "DataFrames saved to both Excel and CSV files."
    )  # Construct the file path for each DataFrame  # Save each DataFrame to an Excel file  # Construct the file path for each DataFrame  # Save each DataFrame to a CSV file
    return (stats_dfs,)


@app.cell
def _(stats, stats_dfs):
    def _():
        FR1 = stats_dfs["Facilitated"]["Mean FR On Spont"]
        FR2 = stats_dfs["Suppressed"]["Mean FR On Spont"]

        stat, p = stats.mannwhitneyu(FR1, FR2)
        return p


    _()
    return


@app.cell
def _(stats, stats_dfs):
    def _():
        FR1 = stats_dfs["Facilitated"]["Mean FR On Spont"]
        FR2 = stats_dfs["Suppressed"]["Mean FR On Spont"]

        stat, p = stats.ttest_ind(FR1, FR2)
        return p


    _()
    return


@app.cell
def _(stats, stats_dfs):
    def _():
        FR1 = stats_dfs["Facilitated"]["Mean FR On Spont"]
        FR2 = stats_dfs["Facilitated"]["Mean FR Off Spont"]

        stat, p = stats.ttest_ind(FR1, FR2)
        return p


    _()
    return


@app.cell
def _(stats, stats_dfs):
    def _():
        FR1 = stats_dfs["Suppressed"]["Mean FR On Spont"]
        FR2 = stats_dfs["Suppressed"]["Mean FR Off Spont"]

        stat, p = stats.ttest_ind(FR1, FR2)
        return p


    _()
    return


@app.cell
def _(
    Late_cFR,
    Late_cFR_On,
    Late_cFR_On_facil,
    Late_cFR_On_supp,
    Late_cFR_facil,
    Late_cFR_supp,
    cFR,
    cFR_On,
    cFR_On_facil,
    cFR_On_supp,
    cFR_facil,
    cFR_supp,
    cellType,
    create_and_describe_df,
    octaves,
    os,
    pd,
    perform_analysis_octaves,
    save_dataframes,
    stats_dir_time,
):
    ##########################################################################
    ################################## Figure 5 ##############################
    cFR_data_info = [
        ("Early", cFR, cFR_On),
        ("Late", Late_cFR, Late_cFR_On),
        ("facil_Early", cFR_facil, cFR_On_facil),
        ("facil_Late", Late_cFR_facil, Late_cFR_On_facil),
        ("supp_Early", cFR_supp, cFR_On_supp),
        ("supp_Late", Late_cFR_supp, Late_cFR_On_supp),
    ]
    dfs_TC = {}
    TC_described_dfs = {}
    for label, off_data, on_data in cFR_data_info:
        (
            dfs_TC[f"centeredOct_{label}_DF_Off"],
            TC_described_dfs[f"described_centeredOct_{label}_DF_Off"],
        ) = create_and_describe_df(off_data, octaves)
        (
            dfs_TC[f"centeredOct_{label}_DF_On"],
            TC_described_dfs[f"described_centeredOct_{label}_DF_On"],
        ) = create_and_describe_df(on_data, octaves)
    save_dataframes(TC_described_dfs, stats_dir_time, cellType)
    results_TC = {}
    for label, off_data, on_data in cFR_data_info:
        for octave_index in range(off_data.shape[0]):
            result_key = f"{label} Octave {octave_index + 1}"
            # Dictionary to hold DataFrames and their descriptions
            results_TC[result_key] = perform_analysis_octaves(
                off_data[octave_index, :], on_data[octave_index, :]
            )
    results_df_TC = pd.DataFrame(results_TC).T
    results_df_TC["p Value"] = results_df_TC["p Value"].map(lambda x: f"{x:.8f}")
    # Adjust the data preparation loop to use new function
    _excel_path = os.path.join(stats_dir_time, f"ResultsDF_TC_{cellType}.xlsx")
    results_df_TC.to_excel(_excel_path, index=True)
    _csv_path = os.path.join(stats_dir_time, f"ResultsDF_TC_{cellType}.csv")
    # Perform statistical analysis
    # Convert results to a DataFrame for easier viewing and manipulation
    # Save results to files
    results_df_TC.to_csv(_csv_path, float_format="%.8f", index=True)
    return


@app.cell
def _(
    Late_SparsenessOff_all,
    Late_SparsenessOff_facilitated,
    Late_SparsenessOff_suppressed,
    Late_SparsenessOn_all,
    Late_SparsenessOn_facilitated,
    Late_SparsenessOn_suppressed,
    SparsenessOff_all,
    SparsenessOff_facilitated,
    SparsenessOff_suppressed,
    SparsenessOn_all,
    SparsenessOn_facilitated,
    SparsenessOn_suppressed,
    calculate_statistics,
    cellType,
    os,
    pad_and_create_dataframe,
    pd,
    perform_statistical_analysis,
    stats_dir_time,
):
    ##########################################################################
    ############################ Figure 6 ####################################
    columns = [
        "Sparseness Off",
        "Sparseness On",
        "Sparseness Off Late Tone",
        "Sparseness On Late Tone",
        "Sparseness Facilitated Off",
        "Sparseness Facilitated On",
        "Sparseness Facilitated Off Late Tone",
        "Sparseness Facilitated On Late Tone",
        "Sparseness Suppressed Off Tone",
        "Sparseness Suppressed On Tone",
        "Sparseness Suppressed Off Late Tone",
        "Sparseness Suppressed On Late Tone",
    ]
    data_lists = [
        SparsenessOff_all,
        SparsenessOn_all,
        Late_SparsenessOff_all,
        Late_SparsenessOn_all,
        SparsenessOff_facilitated,
        SparsenessOn_facilitated,
        Late_SparsenessOff_facilitated,
        Late_SparsenessOn_facilitated,
        SparsenessOff_suppressed,
        SparsenessOn_suppressed,
        Late_SparsenessOff_suppressed,
        Late_SparsenessOn_suppressed,
    ]
    # Example usage with hypothetical data lists and their corresponding column names
    data_sparseness = pad_and_create_dataframe(data_lists, columns)
    described_data_sparseness = calculate_statistics(data_sparseness)
    pairs = {
        "Early": ("Sparseness Off", "Sparseness On"),
        "Late": ("Sparseness Off Late Tone", "Sparseness On Late Tone"),
        "Facilitated Early": (
            "Sparseness Facilitated Off",
            "Sparseness Facilitated On",
        ),
        "Facilitated Late": (
            "Sparseness Facilitated Off Late Tone",
            "Sparseness Facilitated On Late Tone",
        ),
        "Suppressed Early": (
            "Sparseness Suppressed Off Tone",
            "Sparseness Suppressed On Tone",
        ),
        "Suppressed Late": (
            "Sparseness Suppressed Off Late Tone",
            "Sparseness Suppressed On Late Tone",
        ),
    }
    results_sparseness = perform_statistical_analysis(data_sparseness, pairs)
    results_df_sparseness = pd.DataFrame(results_sparseness).T
    results_df_sparseness["p Value"] = results_df_sparseness["p Value"].map(
        lambda x: f"{x:.8f}"
    )
    _excel_path = os.path.join(
        stats_dir_time, f"ResultsDF_Sparseness_{cellType}.xlsx"
    )
    results_df_sparseness.to_excel(_excel_path, index=True)
    _csv_path = os.path.join(
        stats_dir_time, f"ResultsDF_Sparseness_{cellType}.csv"
    )
    results_df_sparseness.to_csv(_csv_path, float_format="%.8f", index=True)
    data_sparseness.to_csv(
        os.path.join(stats_dir_time, f"data_sparseness_{cellType}.csv"),
        index=True,
    )
    described_data_sparseness.to_csv(
        os.path.join(stats_dir_time, f"described_data_sparseness_{cellType}.csv"),
        index=True,
    )
    results_df_sparseness.to_csv(
        os.path.join(stats_dir_time, f"pVal_sparseness_DF_{cellType}.csv"),
        index=True,
    )
    excel_path_described = os.path.join(
        stats_dir_time, f"DescribedDF_Sparseness_{cellType}.xlsx"
    )
    # Convert results to a DataFrame for easier viewing and manipulation
    # Format p Values for better readability
    # Save results to files
    # Output results to CSV
    described_data_sparseness.to_excel(excel_path_described, index=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Time to peak
    """)
    return


@app.cell
def _(
    calculate_statistics,
    pad_and_create_dataframe,
    perform_statistical_analysis,
    t2pOff_facil,
    t2pOff_supp,
    t2pOn_facil,
    t2pOn_supp,
):
    # Create padded DataFrames
    df_facilitated_stats = pad_and_create_dataframe(
        [t2pOff_facil, t2pOn_facil], ["Facilitated Off", "Facilitated On"]
    )

    df_suppressed_stats = pad_and_create_dataframe(
        [t2pOff_supp, t2pOn_supp], ["Suppressed Off", "Suppressed On"]
    )

    # Convert times from seconds to milliseconds (if needed)
    df_facilitated_stats *= 1000
    df_suppressed_stats *= 1000

    # Calculate descriptive statistics for both DataFrames
    stats_facilitated = calculate_statistics(df_facilitated_stats)
    stats_suppressed = calculate_statistics(df_suppressed_stats)

    # Set pairs for the Wilcoxon test
    pairs_facilitated = {
        "Facilitated Off vs On": ("Facilitated Off", "Facilitated On")
    }
    pairs_suppressed = {
        "Suppressed Off vs On": ("Suppressed Off", "Suppressed On")
    }

    # Perform Wilcoxon tests
    results_facilitated = perform_statistical_analysis(
        df_facilitated_stats, pairs_facilitated
    )
    results_suppressed = perform_statistical_analysis(
        df_suppressed_stats, pairs_suppressed
    )
    return (
        results_facilitated,
        results_suppressed,
        stats_facilitated,
        stats_suppressed,
    )


@app.cell
def _(
    cellType,
    os,
    results_facilitated,
    results_suppressed,
    stats_dir_time,
    stats_facilitated,
    stats_suppressed,
):
    # Define filenames for facilitated results
    facilitated_stats_excel = os.path.join(
        stats_dir_time, f"DescribedDF_T2P_Facilitated_{cellType}.xlsx"
    )
    facilitated_stats_csv = os.path.join(
        stats_dir_time, f"DescribedDF_T2P_Facilitated_{cellType}.csv"
    )

    facilitated_analysis_excel = os.path.join(
        stats_dir_time, f"ResultsDF_T2P_Facilitated_{cellType}.xlsx"
    )
    facilitated_analysis_csv = os.path.join(
        stats_dir_time, f"ResultsDF_T2P_Facilitated_{cellType}.csv"
    )

    # Save facilitated stats to Excel and CSV
    stats_facilitated.to_excel(facilitated_stats_excel, index=True)
    stats_facilitated.to_csv(
        facilitated_stats_csv, float_format="%.8f", index=True
    )

    # Save facilitated analysis to Excel and CSV
    results_facilitated.to_excel(facilitated_analysis_excel, index=True)
    results_facilitated.to_csv(
        facilitated_analysis_csv, float_format="%.8f", index=True
    )

    # Define filenames for suppressed results
    suppressed_stats_excel = os.path.join(
        stats_dir_time, f"DescribedDF_T2P_Suppressed_{cellType}.xlsx"
    )
    suppressed_stats_csv = os.path.join(
        stats_dir_time, f"DescribedDF_T2P_Suppressed_{cellType}.csv"
    )

    suppressed_analysis_excel = os.path.join(
        stats_dir_time, f"ResultsDF_T2P_Suppressed_{cellType}.xlsx"
    )
    suppressed_analysis_csv = os.path.join(
        stats_dir_time, f"ResultsDF_T2P_Suppressed_{cellType}.csv"
    )

    # Save suppressed stats to Excel and CSV
    stats_suppressed.to_excel(suppressed_stats_excel, index=True)
    stats_suppressed.to_csv(suppressed_stats_csv, float_format="%.8f", index=True)

    # Save suppressed analysis to Excel and CSV
    results_suppressed.to_excel(suppressed_analysis_excel, index=True)
    results_suppressed.to_csv(
        suppressed_analysis_csv, float_format="%.8f", index=True
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
