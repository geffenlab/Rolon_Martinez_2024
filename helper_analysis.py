import marimo

__generated_with = "0.18.0"
app = marimo.App(width="full")

with app.setup:
    import os
    from pathlib import Path
    import numpy as np
    import pandas as pd
    from scipy import signal
    from scipy.io import loadmat
    from glob import glob
    from sklearn.linear_model import LinearRegression


@app.function
def load_npy_files(data_loc, subfolder, filename):
    return np.load(os.path.join(data_loc, subfolder, filename))


@app.function
def read_sync_messages(data_loc):
    sync_messages_path = os.path.join(data_loc, "sync_messages.txt")
    try:
        with open(sync_messages_path, "r") as file:
            for last_line in file:
                pass
    except FileNotFoundError:
        print(f"File {sync_messages_path} not found.")
        return None, None

    start_time, sample_rate = map(int, last_line.split("@"))
    sample_rate = sample_rate[:-2]  # Remove Hz from sample rate string
    return start_time, sample_rate


@app.function
def convert_to_seconds(samples, start_time, fs):
    return (samples - start_time) / fs


@app.function
def select_block_events(
    events_timestamps, event_states, block_start, block_end=None
):
    if block_end is None:
        indices = events_timestamps > block_start
    else:
        indices = (events_timestamps > block_start) & (
            events_timestamps < block_end
        )
    return events_timestamps[indices], event_states[indices]


@app.function
def load_cluster_info(data_loc):
    clust_info = pd.read_csv(
        os.path.join(data_loc, "cluster_info.tsv"), delimiter="\t"
    )
    return clust_info[clust_info.group != "noise"]


@app.function
def load_mean_waveforms(data_loc):
    mean_waves = loadmat(os.path.join(data_loc, "mean_waveforms.mat"))
    ycoords = mean_waves["chanMap"]["ycoords"][0][0][0]
    mn = (
        np.squeeze(mean_waves["mn"]) - 1
    )  # Adjusting MATLAB's 1-based indexing
    clust_depth = ycoords[mn]
    max_depth = np.max(mean_waves["chanMap"]["ycoords"][0][0][0])
    clust_depth = np.int32(clust_depth) - np.int32(max_depth)
    return clust_depth


@app.function
def get_events(data_loc):
    msg_text = load_npy_files(data_loc, "messages", "text.npy")
    msg_sample = load_npy_files(data_loc, "messages", "timestamps.npy")
    ev_state = load_npy_files(data_loc, "events", "channel_states.npy")
    ev_sample = load_npy_files(data_loc, "events", "timestamps.npy")
    start_time, fs = read_sync_messages(data_loc)

    if start_time is None or fs is None:
        return None

    ev_ts = convert_to_seconds(ev_sample, start_time, fs)
    msg_ts = convert_to_seconds(msg_sample, start_time, fs)

    print(*msg_text, sep="\n")
    print("Total number of blocks:", len(msg_ts))
    block_start_id = int(
        input(
            "Indicate Block Start here (remember python syntax where 0 = 1): "
        )
    )
    block_start = msg_ts[block_start_id]

    if block_start_id + 1 >= len(msg_ts):
        block_end = None
    else:
        block_end = msg_ts[block_start_id + 1]

    block_ev_ts, block_ev_state = select_block_events(
        ev_ts, ev_state, block_start, block_end
    )

    spikes = load_npy_files(data_loc, "", "spike_times.npy") / fs
    clust = load_npy_files(data_loc, "", "spike_clusters.npy")
    clust_info = load_cluster_info(data_loc)
    clust_depth = load_mean_waveforms(data_loc)

    stimulus_on = block_ev_ts[block_ev_state == 1][
        1:
    ]  # Assuming the first event is not a stimulus onset
    stimulus_off = block_ev_ts[block_ev_state == -1][
        1:
    ]  # Assuming the first event is not a stimulus offset

    return (
        ev_state,
        ev_ts,
        msg_ts,
        block_ev_ts,
        stimulus_on,
        spikes,
        clust,
        clust_depth,
        clust_info,
        stimulus_off,
    )


@app.function
# This function applies a smoothing guassian filter to your data
def SmoothGauss(X, M):
    if X.ndim != 1:
        raise ValueError("X must be a 1D array.")

    sigma = M**0.5
    G = np.arange(-M, M + 1)
    F = np.exp(-(G**2) / ((2 * sigma) ** 2))
    F /= np.sum(F)
    Y = np.convolve(X, F, mode="full")

    start_index = M
    end_index = start_index + len(X)
    Y = Y[start_index:end_index]

    correction_start = np.sum(F) / (np.sum(F[:M]) + np.cumsum(F[M : M * 2]))
    correction_end = np.sum(F) / (
        np.sum(F[:M]) + np.cumsum(F[M : M * 2])[::-1]
    )

    Y[:M] *= correction_start
    Y[-M:] *= correction_end

    return Y


@app.function
# This function extracts spikes and returns your raster, trials and psth variable:
def spike_data(spikes, stimulusOnset, edges, smVar):
    raster = []
    trials = []
    psth = np.empty(([len(stimulusOnset), len(edges) - 1]))
    psth_S = np.empty(([len(stimulusOnset), len(edges) - 1]))

    for s, stim in enumerate(stimulusOnset):
        spks = spikes - stim

        psth[s, :], _ = np.histogram(spks, bins=edges)
        psth[s, :] = psth[s, :] / np.diff(edges).mean()
        psth_S[s, :] = SmoothGauss(psth[s, :], smVar)

        spks = spks[(spks > edges[0]) & (spks < edges[-1])]
        raster.extend(spks)
        trials.extend(np.ones(len(spks)) * (s + 1))

    return raster, trials, psth, psth_S


@app.function
def load_stim_info(stim_loc, file_name, extract_fields=None):
    """
    Load stimulus information from a MATLAB file, handling multi-dimensional arrays appropriately.

    Parameters:
    - stim_loc: The directory where the stimulus file is located.
    - file_name: The name of the MATLAB file containing the stimulus information.
    - extract_fields: Optional list of fields to extract from the stimulus information.
                      If None, all fields are extracted.

    Returns:
    - A dictionary containing the requested stimulus information, or all information if extract_fields is None.
    - The name of the file.
    """
    full_path = os.path.join(stim_loc, file_name)
    try:
        file = loadmat(full_path, appendmat=True)
        stim_inf = file["stimInfo"][0, 0]
        stim_info = {}

        # If no specific fields are requested, extract all.
        if extract_fields is None:
            extract_fields = stim_inf.dtype.names

        for field in extract_fields:
            if field in stim_inf.dtype.names:
                # Handling multi-dimensional arrays without converting to scalar
                value = stim_inf[field]
                if value.size == 1:
                    stim_info[field] = (
                        value.item()
                    )  # For single elements, convert to scalar
                else:
                    stim_info[field] = (
                        value  # Keep as array for multi-dimensional data
                    )
            else:
                print(f"Warning: '{field}' not found in stimulus information.")

        return stim_info, file_name
    except FileNotFoundError:
        print(f"Error: File '{full_path}' not found.")
        return None, None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None


@app.function
def get_stimulus_info(
    stim_loc,
    main_stim_file,
    laser_stim_file,
    nreps,
    nreps_laser,
    extract_fields=None,
):
    main_stim_info, _ = load_stim_info(
        stim_loc, main_stim_file + ".mat", extract_fields
    )
    if main_stim_info is None:
        print(
            f"Failed to load main stimulus information from {main_stim_file}"
        )
        return None

    laser_stim_info, _ = load_stim_info(
        stim_loc, laser_stim_file + ".mat", extract_fields
    )
    if laser_stim_info is None:
        print(
            f"Failed to load laser stimulus information from {laser_stim_file}"
        )
        return None

    # Direct extraction of necessary information without duplication
    return {
        "ITI_laser": laser_stim_info.get("ITI", None),
        "laserOnlyDur": laser_stim_info.get("laserDur", None),
        "ITI": main_stim_info.get("ITI", None),
        "laserDur": main_stim_info.get("laserDur", None),
        "tDur": main_stim_info.get("tDur", None),
        "trialOrder_main": main_stim_info.get("trialOrder", []),
    }


@app.function
def load_probe_data(excel_path, cell_type, virus):
    # Assuming each sheet corresponds to a different cell type or there's a naming pattern
    sheet_name = f"{cell_type}_{virus}_Depths"
    return pd.read_excel(excel_path, sheet_name=sheet_name)


@app.function
def load_and_concatenate_npz(directory, stim):
    files = sorted(glob(os.path.join(directory, f"*{stim}.npz")))
    all_data = [np.load(f, allow_pickle=True) for f in files]

    fullpsth = np.concatenate(
        [data["allPSTH"] for data in all_data if "allPSTH" in data.files],
        axis=2,
    )
    fullpsthS = np.concatenate(
        [data["allPSTH_S"] for data in all_data if "allPSTH_S" in data.files],
        axis=2,
    )

    clustdepth = []
    rasters = []
    trials = []
    spikesortind = []
    session_mapping = {}
    current_index = 0

    for idx, data in enumerate(all_data):
        session_id = os.path.basename(files[idx]).split(" ")[
            2
        ]  # Adjust according to filename format
        if "clustDepth" in data.files and data["clustDepth"].size > 0:
            clust_depth_data = [
                (depth, session_id) for depth in data["clustDepth"].tolist()
            ]
            clustdepth.extend(clust_depth_data)

            session_mapping[session_id] = (
                current_index,
                current_index + len(clust_depth_data),
            )
            current_index += len(clust_depth_data)

        if "rasters" in data.files and data["rasters"].size > 0:
            rasters.extend(data["rasters"].tolist())

        if "trials" in data.files and data["trials"].size > 0:
            trials.extend(data["trials"].tolist())

        if "spikeSortI" in data.files and data["spikeSortI"].size > 0:
            spikesortind.extend(data["spikeSortI"].tolist())

    clustdepth = np.array(clustdepth, dtype=object)
    rasters = np.array(rasters, dtype=object)
    trials = np.array(trials, dtype=object)
    spikesortind = (
        np.array(spikesortind, dtype=object) if spikesortind else None
    )

    return (
        fullpsth,
        fullpsthS,
        clustdepth,
        rasters,
        trials,
        spikesortind,
        session_mapping,
    )


@app.function
def summarize_data(
    rasters,
    spikesortind,
    clustdepth,
    fullpsth,
    fullpsthS,
    fullpsth_L,
    fullpsthS_L,
):
    """
    Summarize the dimensions and shapes of key experimental data arrays.

    Parameters:
    - Rasters: Raster plot data.
    - SpikeSortInd: Spike sorting index.
    - Clust_Depth: Cluster depth information.
    - FullPSTH: Full peri-stimulus time histogram.
    """
    print("Data Summary:")
    print("-------------")
    print(f"Length of Rasters: {len(rasters)}")
    print(f"Length of SpikeSortInd: {len(spikesortind)}")
    print(f"Length of Clust_Depth: {len(clustdepth)}")
    print(f"Shape of FullPSTH: {fullpsth.shape}")
    print(f"Shape of Smoothed FullPSTH: {fullpsthS.shape}")
    print(f"Shape of FullPSTH_Laser: {fullpsth_L.shape}")
    print(f"Shape of Smoothed FullPSTH_Laser: {fullpsthS_L.shape}")


@app.function
def load_experiment_data(
    data_path,
    main_stim_file,
    data_path_laser,
    laser_stim_file,
    stim_loc,
    nreps,
    nreps_laser,
):
    binSize = 0.002
    edges = np.arange(-0.050, 0.20, binSize)
    time = edges[:-1]

    (
        FullPSTH,
        FullPSHT_S,
        Clust_Depth,
        Rasters,
        Trials,
        SpikeSortInd,
        Session_Map,
    ) = load_and_concatenate_npz(data_path, main_stim_file.replace(".mat", ""))

    (
        FullPSTH_Laser,
        FullPSHT_S_Laser,
        Clust_Depth_laser,
        Rasters_laser,
        Trials,
        SpikeSortInd_laser,
        Session_Map_Laser,
    ) = load_and_concatenate_npz(
        data_path_laser, laser_stim_file.replace(".mat", "")
    )

    stim_info = get_stimulus_info(
        stim_loc, main_stim_file, laser_stim_file, nreps, nreps_laser
    )

    # Repetition of trialOrder calculation is now removed and directly fetched from stim_info
    trialOrder = np.matlib.repmat(stim_info["trialOrder_main"], nreps, 1)
    ITI, ITI_laser = stim_info["ITI"], stim_info["ITI_laser"]
    laserDur, laserOnlyDur = stim_info["laserDur"], stim_info["laserOnlyDur"]
    tDur = stim_info["tDur"]

    return (
        FullPSTH,
        FullPSHT_S,
        Clust_Depth,
        Rasters,
        Trials,
        SpikeSortInd,
        FullPSTH_Laser,
        FullPSHT_S_Laser,
        trialOrder,
        ITI,
        ITI_laser,
        laserDur,
        laserOnlyDur,
        tDur,
        binSize,
        edges,
        time,
        Session_Map,
        Session_Map_Laser,
    )


@app.function
def ismember(A, B):
    A = np.asarray(A).astype(int)
    B = np.asarray(B).astype(int)
    res = np.zeros(A.shape)
    for i in np.unique(A):
        res[A == i] = np.argwhere(B == i).squeeze()
    return res


@app.function
def map_cluster_depths_to_real_dimensions(clust_depth, probe_data):
    real_depths = []

    for depth_info in clust_depth:
        depth, session_id = (
            depth_info  # Unpacking depth and session ID from each tuple
        )
        # Get the probe start depth for this session from the probe data
        session_probe_data = probe_data[
            probe_data["Recording Session"] == session_id
        ]
        if not session_probe_data.empty:
            probe_start = session_probe_data["Probe_Start"].iloc[0]

            # Convert the clust_depth value for this session
            # Since clust_depth value is below the start of the probe, convert from µm to mm and adjust from the start
            real_depth = round(probe_start - (depth / 1000), 3)
            real_depths.append((real_depth, session_id))
        else:
            # If no corresponding probe data, use None for depth and keep session ID
            real_depths.append((None, session_id))

    return real_depths


@app.function
def normalize_psth_data(data, laser_off_indices, laser_on_indices, axis=0):
    """
    Calculate the mean for 'laserOn' and 'laserOff' conditions for each cell and normalize the 'laserOn' mean
    to the range derived from the 'laserOff' mean.

    Parameters:
    - data: numpy array, the PSTH data with dimensions that can vary.
    - laser_off_indices: numpy array or list, indices indicating the laser off condition.
    - laser_on_indices: numpy array or list, indices indicating the laser on condition.
    - axis: int, the axis along which the means are calculated and conditions are defined.

    Returns:
    - norm_mean_off: numpy array, normalized mean data for the 'laserOff' condition.
    - norm_mean_on: numpy array, normalized mean data for the 'laserOn' condition to the 'laserOff' range.
    """
    print("size of data:", data.shape)
    print("size of laser off:", laser_off_indices.shape)
    print("size of laser on:", laser_on_indices.shape)

    # Calculate means across the specified axis for both conditions
    mean_off = np.take(data, laser_off_indices, axis=axis).mean(axis=axis)
    mean_on = np.take(data, laser_on_indices, axis=axis).mean(axis=axis)

    # Calculate normalization parameters based on the 'laserOff' mean
    min_off = np.min(mean_off, axis=0, keepdims=True)
    max_off = np.max(mean_off, axis=0, keepdims=True)

    # Normalize both 'laserOff' and 'laserOn' mean data to the 'laserOff' mean range
    norm_mean_off = (mean_off - min_off) / np.where(
        max_off != min_off, max_off - min_off, 1
    )
    norm_mean_on = (mean_on - min_off) / np.where(
        max_off != min_off, max_off - min_off, 1
    )

    return norm_mean_off, norm_mean_on


@app.function
def normalize_to_laserOff_TC(data1, data2):
    norm_data1 = np.empty(data1.shape)
    norm_data2 = np.empty(data2.shape)

    for cell in range(data1.shape[1]):
        min_data1 = np.min(data1[:, cell])
        max_data1 = np.max(data1[:, cell])

        # Normalize data1 to its own range
        norm_data1[:, cell] = (data1[:, cell] - min_data1) / (
            max_data1 - min_data1
        )

        # Normalize data2 to the range of data1
        norm_data2[:, cell] = (data2[:, cell] - min_data1) / (
            max_data1 - min_data1
        )

    return norm_data1, norm_data2


@app.function
def calculate_mTC(data, tones, freqs):
    mTC = np.zeros((len(freqs), data.shape[1]))
    for uniq in range(len(freqs)):
        mTC[uniq, :] = data[tones == freqs[uniq], :].mean(axis=0)
    return mTC


@app.function
def calculate_psthTC(
    PSTH, tonescond1, tonescond2, freqs, lasercond1, lasercond2
):
    num_uniq = len(freqs)
    time_points = PSTH.shape[1]
    num_cells = PSTH.shape[2]
    psthTC1 = np.zeros((num_uniq, time_points, num_cells))
    psthTC2 = np.zeros((num_uniq, time_points, num_cells))

    for uniq in range(num_uniq):
        toneTrial1 = PSTH[lasercond1, :, :][tonescond1 == freqs[uniq]]
        toneTrial_2 = PSTH[lasercond2, :, :][tonescond2 == freqs[uniq]]
        psthTC1[uniq, :, :] = toneTrial1.mean(axis=0)
        psthTC2[uniq, :, :] = toneTrial_2.mean(axis=0)

    return psthTC1, psthTC2


@app.function
def fit_and_predict(TC1, TC2):
    slope = np.zeros(TC1.shape[1])
    intercept = np.zeros(TC1.shape[1])
    y_preds = np.zeros_like(TC1)

    for cell in range(TC1.shape[1]):
        model = LinearRegression(fit_intercept=True)
        modelfit = model.fit(TC1[:, cell].reshape((-1, 1)), TC2[:, cell])
        slope[cell] = modelfit.coef_[0]  # Extract the first element
        intercept[cell] = modelfit.intercept_
        y_preds[:, cell] = modelfit.predict(TC1[:, cell].reshape((-1, 1)))
    return slope, intercept, y_preds


@app.function
def calculate_sparseness(*conditions):
    def calc_sparseness(resp):
        N = len(resp)
        a = ((np.sum(resp) / N) ** 2) / np.sum((resp**2) / N)
        s = (1 - a) / (1 - (1 / N))
        return s

    results = []
    for condition in conditions:
        sparseness_values = [
            calc_sparseness(condition[:, cell])
            for cell in range(condition.shape[1])
        ]
        results.append(sparseness_values)

    return results


@app.function
# FIXUP: this function should take 'edges' and 'early_evoked_Ind' as arguments rather than relying on them being in the global namespace
def find_time_to_peak(
    peak_data_off,
    peak_data_on,
    edges,
    early_evoked_Ind,
    spontInd,
):
    t2pOff, t2pOn = [], []
    for cell in range(len(peak_data_on[0])):
        bOff, _ = signal.find_peaks(
            peak_data_off[early_evoked_Ind, cell],
            height=(
                np.mean(peak_data_off[spontInd, cell])
                + (np.std(peak_data_off[spontInd, cell]) * 3)
            ),
        )
        bOn, _ = signal.find_peaks(
            peak_data_on[early_evoked_Ind, cell],
            height=(
                np.mean(peak_data_on[spontInd, cell])
                + (np.std(peak_data_on[spontInd, cell]) * 3)
            ),
        )

        if bOff.size > 0:
            i_max_peak_Off = bOff[
                np.argmax(peak_data_off[early_evoked_Ind, cell][bOff])
            ]
            x_max_Off = edges[early_evoked_Ind][i_max_peak_Off]
            t2pOff.append(x_max_Off)
        else:
            t2pOff.append(np.nan)  # Handle case where no peak is found

        if bOn.size > 0:
            i_max_peak_On = bOn[
                np.argmax(peak_data_on[early_evoked_Ind, cell][bOn])
            ]
            x_max_On = edges[early_evoked_Ind][i_max_peak_On]
            t2pOn.append(x_max_On)
        else:
            t2pOn.append(np.nan)  # Handle case where no peak is found

    return t2pOff, t2pOn


@app.function
def get_BF_and_uBF(off_responses, on_responses, freqs, side_freq):
    # Nested function to process contextual responses
    def process_contextual_responses(
        i,
        freqs_array,
        freqs,
        side_freq,
        off_responses,
        on_responses,
        off_context_resp,
        on_context_resp,
    ):
        freq_index = np.argwhere(freqs_array[i] == freqs).flatten()[0]
        nearby_indices = np.arange(
            freq_index - side_freq, freq_index + side_freq + 1
        ).astype("float")
        nearby_indices[
            np.logical_or(nearby_indices < 0, nearby_indices >= len(freqs))
        ] = np.nan

        off_context_resp[:, i] = np.nan
        valid = ~np.isnan(nearby_indices)
        valid_indices = nearby_indices[valid].astype(int)
        off_context_resp[valid, i] = off_responses[valid_indices, i]

        on_context_resp[:, i] = np.nan
        on_context_resp[valid, i] = on_responses[valid_indices, i]

    # Initialization for best and un-best frequencies
    best_freqs = np.empty(len(off_responses[0]))
    max_responses = np.array(
        [
            np.max(off_responses[:, cell])
            for cell in range(len(off_responses[0]))
        ]
    ).squeeze()
    off_context_resp_BF = np.empty([2 * side_freq + 1, len(off_responses[0])])
    on_context_resp_BF = np.empty([2 * side_freq + 1, len(off_responses[0])])

    uBF = np.empty(len(off_responses[0]))  # un-best frequencies
    min_responses = np.array(
        [
            np.min(off_responses[:, cell])
            for cell in range(len(off_responses[0]))
        ]
    ).squeeze()
    off_context_resp_uBF = np.empty([2 * side_freq + 1, len(off_responses[0])])
    on_context_resp_uBF = np.empty([2 * side_freq + 1, len(off_responses[0])])

    for cell in range(len(off_responses[0])):
        # Process for best frequencies
        max_resp_freqs = freqs[
            np.where(off_responses[:, cell] == np.max(off_responses[:, cell]))
        ]
        best_freqs[cell] = (
            max_resp_freqs[0] if max_resp_freqs.size else max_resp_freqs
        )

        # Process for un-best frequencies
        min_resp_freqs = freqs[
            np.where(off_responses[:, cell] == np.min(off_responses[:, cell]))
        ]
        uBF[cell] = (
            min_resp_freqs[0] if min_resp_freqs.size else min_resp_freqs
        )

    for i in range(len(off_responses[0])):
        # Handling best frequencies
        process_contextual_responses(
            i,
            best_freqs,
            freqs,
            side_freq,
            off_responses,
            on_responses,
            off_context_resp_BF,
            on_context_resp_BF,
        )

        # Handling un-best frequencies
        process_contextual_responses(
            i,
            uBF,
            freqs,
            side_freq,
            off_responses,
            on_responses,
            off_context_resp_uBF,
            on_context_resp_uBF,
        )

    return (
        best_freqs,
        max_responses,
        off_context_resp_BF,
        on_context_resp_BF,
        uBF,
        min_responses,
        off_context_resp_uBF,
        on_context_resp_uBF,
    )


@app.cell
def _():
    import marimo as mo
    return


if __name__ == "__main__":
    app.run()
