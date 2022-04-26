"""Utils to work with Ephys.

Some general utils to work with paton lab ephys data with some help from
spikeInterface and friends.

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import spikeinterface.toolkit as st
import spikeinterface as si

from os import path
from scipy.interpolate import interp1d
from probeinterface import get_probe
from spikeinterface.widgets import plot_timeseries


# General argparse utils
def str2bool(v):
    """ A naive approach to boolean parsing.

    https://stackoverflow.com/questions/\
    15008758/parsing-boolean-values-with-argparse"""
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


# Utilities in general for nested dicts
# Based on stackoverflow.com
# questions/23981553/get-all-values-from-nested-dictionaries-in-python
# Use yield to make iterators to get relevant info from dictionaries
# Get the result form the iterators into a list by cheating

def _nested_dict_values(d):
    for v in d.values():
        if isinstance(v, dict):
            yield from _nested_dict_values(v)
        else:
            yield v

def _nested_dict_keys(d):
    for k, v in d.items():
        if isinstance(v, dict):
            yield from _nested_dict_keys(v)
        else:
            yield k


def _nested_dict_values_from_key(d, key):
    for k, v in d.items():
        if isinstance(v, dict):
            yield from _nested_dict_values_from_key(v, key)
        else:
            if k == key:
                yield v

def _yield_2_list(args, constructor):
    return list(constructor(*args))


def get_dict_values(d):
    return _yield_2_list(d, _nested_dict_values)

def get_nested_dict_keys(d):
    return _yield_2_list(d, _nested_dict_keys)

def get_nested_dict_values_from_key(d, key):
    args = (d, key)
    return _yield_2_list(args, _nested_dict_values_from_key)


# EPhys and friends related functions
def get_filepath(recording):
    """Get the filepath from a recording.

    Parameters
    ----------
    recording : SpikeInterface.recording
        A recording from spike interface.

    Returns
    -------
    str
       Location on filesystem of the recording.
    """
    # Get the filepath of the original file:
    ret = get_nested_dict_values_from_key(recording.to_dict(), 'file_paths')[0][0]
    return ret


def load_recording(
    ftoken_bin,
    sampling_rate,
    n_channels,
    dtype,
    gain=0.195,
    offset=-6389.76,
    probe_manufacturer="cambridgeneurotech",
    probe_name="ASSY-156-E-2",
    wiring_device="ASSY-156>RHD2164",
):
    """Simplest possible way of loading a recording from binary.

    Parameters
    ----------
    ftoken_bin : _type_
        _description_
    sampling_rate : _type_
        _description_
    n_channels : _type_
        _description_
    dtype : _type_
        _description_
    probe_manufacturer : str, optional
        _description_, by default 'cambridgeneurotech'
    probe_name : str, optional
        _description_, by default 'ASSY-116-P2'
    wiring_device : str, optional
        _description_, by default 'ASSY-156>RHD2164'

    Returns
    -------
    _type_
        _description_
    """
    # Instantiate a recording object
    rec_args = ftoken_bin, sampling_rate, n_channels, dtype
    kwd_bre = dict(gain_to_uV=gain, offset_to_uV=offset)
    recording = si.BinaryRecordingExtractor(*rec_args, **kwd_bre)
    # Get a probe object
    probe = get_probe(probe_manufacturer, probe_name)
    # add the wiring, if known make it
    if isinstance(wiring_device, str):
        probe.wiring_to_device(wiring_device)
    # if give, also use it
    elif isinstance(wiring_device, (list, np.ndarray, tuple)):
        probe.set_device_channel_indices(wiring_device)
    # put the chanel map into the recording
    recording = recording.set_probe(probe)
    # update the channel properties by the probe shanks
    recording = recording.set_probegroup(probe, group_mode="by_shank")
    return recording


def get_wiring(wiring_device):
    # To keep the possible wirings from ephys in one place
    if wiring_device == "ASSY-116>RHD2132":
        ret = [
            16,
            18,
            20,
            22,
            24,
            26,
            28,
            30,
            31,
            29,
            27,
            25,
            23,
            21,
            19,
            17,
            15,
            13,
            11,
            9,
            7,
            5,
            3,
            1,
            0,
            2,
            4,
            6,
            8,
            10,
            12,
            14,
        ]
    elif wiring_device == "ASSY-156>RHD2164":
        ret = wiring_device

    else:
        raise NotImplementedError(
            "Only wiring devices: {`ASSY-116>RHD2132`, `ASSY-156>RHD2164`} are implemented"
        )
    return ret


def lazy_preproc_recording(
    recording,
    pipeline=dict(
        bandpass_filter=dict(freq_min=200, freq_max=9000),
        common_reference=dict(reference="global", operator="median"),
    ),
):
    """Preprocessing pipeline  with spikeinterface.

    Each element in the pipeline is a preprocessor, they are applied sequentialy
    in order. Each key in the dict should point to a preprocessor, and values
    are the keyword-arguments.

    Parameters
    ----------
    recording : _type_
        _description_
    pipeline : _type_, optional
        _description_, by default dict( bandpass_filter=dict(freq_min=200, freq_max=6000), common_reference=dict(reference='global', operator='median'), )

    Returns
    -------
    _type_
        _description_
    """

    # Assert that the pipeline is possible
    preprocessers = st.preprocesser_dict.keys()
    assert all(
        [p in preprocessers for p in pipeline.keys()]
    ), f"Pipeline values outside of {set(preprocessers)}."
    recording_p = recording
    for preproc, kwds_pp in pipeline.items():
        recording_p = getattr(st, preproc)(recording_p, **kwds_pp)
    return recording_p


# Relevant for the data processes
def get_structured_random_traces(recording, num_chunks=[15], chunk_sizes=[60]):
    """Get a structured amount of random data from a recording.

    Get chunk_sizes sized num_chunks amounts of continuous pieces of data in
    the recording. This allows to get random samples that are representative of
    the time structure of the data.

    To get representative samples, will cut the recording size in num_chunk//2
    and get two samples per recordig split. Such that it gets samples all over
    the session.

    The defaul parameters get 10 minutes of samples, splited in:
        - 24 samples of 10 secs, 4 minutes.
        - 4 samples of 30 secs, 2 minutes.
        - 4 samples of 60 secs, 4 minutes.

    Parameters
    ----------
    rec_num_samples : int
        Total number of samples in the recording object.
    rec_samp_freq : int
        Sampling frequency of the recording object.
    num_chunks : list, optional
        How many chunks per chunk size should be extracted, by default [18, 4, 4]
    chunk_sizes : list, optional
        Size of the chunks to be extracted in seconds, by default [10, 30, 60]
    """
    rec_num_samples = recording.get_num_samples()
    rec_samp_freq = recording.get_sampling_frequency()
    kwds = dict(num_chunks=num_chunks, chunk_sizes=chunk_sizes)
    start_end_edges = get_structured_random_traces_edges(
        rec_num_samples, rec_samp_freq, **kwds
    )
    ret = [
        recording.get_traces(start_frame=st, end_frame=ef) for st, ef in start_end_edges
    ]
    return np.vstack(ret)


def get_structured_random_traces_edges(
    rec_num_samples,
    rec_samp_freq,
    num_chunks=[24, 4, 4],
    chunk_sizes=[10, 30, 60],
):
    """Get a structured amount of random data from a recording.

    Get chunk_sizes sized num_chunks amounts of continuous pieces of data in
    the recording. This allows to get random samples that are representative of
    the time structure of the data.

    To get representative samples, will cut the recording size in num_chunk//2
    and get two samples per recordig split. Such that it gets samples all over
    the session.

    The defaul parameters get 10 minutes of samples, splited in:
        - 24 samples of 10 secs, 4 minutes.
        - 4 samples of 30 secs, 2 minutes.
        - 4 samples of 60 secs, 4 minutes.

    Parameters
    ----------
    rec_num_samples : int
        Total number of samples in the recording object.
    rec_samp_freq : int
        Sampling frequency of the recording object.
    num_chunks : list, optional
        How many chunks per chunk size should be extracted, by default [18, 4, 4]
    chunk_sizes : list, optional
        Size of the chunks to be extracted in seconds, by default [10, 30, 60]
    """
    # Ensure that num_chunks and chunk sizes are lists:
    num_chunks, chunk_sizes = [np.asarray(arg) for arg in [num_chunks, chunk_sizes]]
    # For sanity check that both have the same lenght
    assert np.allclose(
        *[len(arg) for arg in [num_chunks, chunk_sizes]]
    ), "Lists num_chunks and chunk_sizes must have the same lenght."
    chunk_sizes_samples = chunk_sizes * rec_samp_freq
    # The total size of the chunk to get, the sample rate by the size of the chunks
    tot_chunk_size = sum(np.product((num_chunks, chunk_sizes), axis=0) * rec_samp_freq)
    assert (
        tot_chunk_size * 5 < rec_num_samples
    ), "The recording is too small for the expected result, use the full recording."
    # To go from larger to smaller chunk sizes:
    idx = np.argsort(-chunk_sizes)
    slices = [
        _get_non_overlaping_samples(rec_num_samples, nc, cs)
        for nc, cs, in zip(num_chunks, chunk_sizes_samples)
    ]
    return [x for xs in slices for x in xs]


def _get_non_overlaping_samples(rec_num_samples, number_chunks, chunk_size):
    """Return non-overlping samples of some size for a sequence of rec_num_samples.

    The issue has been solved by this brute force method:
        https://www.geeksforgeeks.org/python-non-overlapping-random-ranges/

    Here we only add the fact that we split the rec_num_samples in num_chunk
    to split the recording in many chunks and random samples all over it.

    Parameters
    ----------
    rec_num_samples : int
        total size of the time series.
    number_chunks : int
        Number of chunks to return.
    chunk_size : int
        Size of the chunks.
    """
    res = set()
    # how many splits of the data to make
    n_batches = number_chunks
    # Size in samples of each batch
    rec_batches_size = rec_num_samples // n_batches
    max_size = int(rec_batches_size - chunk_size)
    # offsets
    offsets = [max_size * n for n in range(n_batches)]
    for _ in range(number_chunks):
        temp = np.random.randint(0, max_size)
        while any(temp >= idx and temp <= idx + chunk_size for idx in res):
            temp = np.random.randint(0, max_size)
        res.add(temp)
    res = [
        (int(idx + offset), int(idx + chunk_size + offset))
        for idx, offset in zip(res, offsets)
    ]
    return res


# Plotting with spikeinterface assistance
def plot_random_samples(recording, seed=42, samp_dur=10, num_samps=6, pipeline=None):
    # get random pieces around the session and plot the results as channels
    np.random.seed(seed)
    frame_ranges = [
        (np.add(np.random.randint(recording.get_total_duration()-samp_dur), [0, 30])
         * recording.get_sampling_frequency())
        for x in range(num_samps)
    ]
    samp_rec = si.concatenate_recordings([recording.frame_slice(*tr) for tr in frame_ranges])
    ret = plot_recording(samp_rec, seed=seed, full_rec=True, pipeline=pipeline)
    ret[0].suptitle('Random ephys samples sorted by depth')
    return ret


def plot_recording(recording, seed=42, full_rec=False, pipeline=None):
    # get a random minute and plot it
    # seed the PRNG
    np.random.seed(seed)
    # get one minute in the recording
    time_range = np.add(np.random.randint(recording.get_total_duration()-60), [0, 60]) * recording.get_sampling_frequency()
    # to plot the whole recording
    if full_rec:
        time_range = [0, recording.get_total_duration() * recording.get_sampling_frequency()]
    recording = recording.frame_slice(*time_range)

    if pipeline:
        recordings = [recording, lazy_preproc_recording(recording, pipeline)]
        fig, axs = plt.subplots(ncols=2, nrows=1, figsize=[15, 15], dpi=120)
    else:
        recordings = [recording]
        fig, axs = plt.subplots(ncols=1, nrows=1, figsize=[15, 15], dpi=120)
        axs = [axs]
    titles = ['raw', 'preprocessed']
    _ = fig.suptitle('Ephys traces sorted by depth')
    for ax, tbp,title in zip(axs, recordings, titles):
        w = plot_timeseries(
            tbp,
            show_channel_ids=True,
            order_channel_by_depth=True,
            ax=ax,
        )
        ax.set(title=title)
    _ = fig.tight_layout(rect=[0,0,1,.98])
    return fig, axs


def plot_artifacts(recording, thresh=100, samp_artifacts=50, tbta=[-1, 1],):
    # plot a sample of the artifacts detected
    ftoken = get_filepath(recording)
    # Here's the peak detection, if we already did, load peaks and skip
    # unless we enforce new detection!
    ftoken_peaks = path.splitext(ftoken)[0] + "_artifacts_peaks.csv"
    assert path.isfile(ftoken_peaks), "no peak artifacts file"
    uniq_peaks = np.loadtxt(ftoken_peaks).astype(int)
    uniq_peaks = uniq_peaks if uniq_peaks.shape[0] < samp_artifacts else uniq_peaks[np.random.randint(uniq_peaks.shape[0], size=samp_artifacts)]
    tbta = np.multiply(tbta, recording.get_sampling_frequency())
    rec_art = si.concatenate_recordings([recording.frame_slice(*tbta+peak) for peak in uniq_peaks])
    # Check for the threshold
    noise_levels = si.toolkit.get_noise_levels(recording, return_scaled=False)
    abs_threholds = noise_levels * thresh
    fig, ax = plt.subplots(1, figsize=[10, 10], dpi=100)
    traces = rec_art.get_traces()
    _ = ax.plot(traces)
    _ = [ax.axhline(thr, ls='--', c='k') for thr in np.multiply(abs_threholds.mean(), [-1, 1])]
    _ = fig.suptitle('Sample of artifacts and threshold')
    return fig, ax

# Interpolate waveforms

def quantile_interpolate_waves(waves):
    lq_wfrm, avg_wfrm, hq_wfrm = [waves.quantile(q) for q in [0.25, 0.5, 0.75]]
    x_ori = np.linspace(0, 1, avg_wfrm.shape[0])
    x_new = np.linspace(0, 1, 100)
    f_lq, f_av, f_hq = [
        interp1d(x_ori, wfrm, kind='cubic')
        for wfrm in [lq_wfrm, avg_wfrm, hq_wfrm]
    ]
    nlq_wfrm, navg_wfrm, nhq_wfrm = [f(x_new) for f in [f_lq, f_av, f_hq]]
    ret = pd.DataFrame(data=[nlq_wfrm, navg_wfrm, nhq_wfrm], index=[0.25, 0.5, 0.75]).rename_axis(index='percentile')
    return ret
