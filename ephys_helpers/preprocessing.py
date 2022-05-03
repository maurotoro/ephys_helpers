"""Pipeline for doing preprocessing on ephys data.

To run you need to have installed, at least:
- Python 3.8.10
- Python libraries:
    - spikeinterface
- For sorting:
    - kilosort 2.5
    - phy

Need to have a ephys binary file on the computer, or with acces to it from the computer.

Here I'm assuming that the file is recorded from a ASSY-156 probe connected to a RHD2164 headstage

Other probes and headstages are available on request, some are already available.

On a terminal, go to the folder where the file is and take a look at the data:

    python ephys_preproc ephys.file -bvi


TODO:
    - Move all visualization t oephyviewer to have a better look!
"""

import os.path as path
import yaml
import argparse
import inspect
import numpy as np
import spikeinterface as si
import matplotlib.pyplot as plt
import spikeinterface.toolkit as st

from shutil import rmtree

from utils import (get_wiring, load_recording, lazy_preproc_recording, get_structured_random_traces, get_filepath)
from sklearn.metrics import pairwise_distances
from spikeinterface.sortingcomponents.peak_detection import detect_peaks
from spikeinterface.widgets import plot_timeseries, plot_probe_map
from probeinterface import get_probe

plt.ion()



# turn into program
def parse_args():
    # Basic general documentation for the script
    description = """Preprocess ephys data.

    Load a binary datafile and quickly preprocess relying mostly on spikeinterface.

    It will resort the channels into a more meaningful ordering, and then depending
    on your parameters it could:
     - remove bad channels
     - bandpass
     - detect extreme amplitde artifacts
     - remove artifacts (*)
     - common reference

    (*: This step requires a file with the artifact indices in the data)

    A better artifact removal would be to replace the noise with values from a
    distribution similar to surrounding, but still not there... Add it as a PR?

    """
    parser = argparse.ArgumentParser(description=description)

    # File of ephys recording
    parser.add_argument("ftoken_bin", type=str, help="Raw ephys file.")
    ## sampling_rate
    parser.add_argument(
        "-sr",
        "--sampling_rate",
        type=float,
        default=30000,
        help="Sampling rate of the raw signal (samp/s).",
    )
    ## n_channels
    parser.add_argument(
        "-nc",
        "--n_channels",
        type=int,
        default=64,
        help="Number of channels in the recordings.",
    )
    ## dtype
    parser.add_argument(
        "--dtype",
        type=str,
        default="i2",
        help="Data type of the recording as a string. int16=i2, uint16=u2, etc.",
    )
    ## Gain and offsets
    parser.add_argument(
        "--offset",
        type=float,
        default=-32768,
        help="Offline for the digitazion.",
    )
    parser.add_argument(
        "--gain",
        type=float,
        default=0.195,
        help="Gain needed to rescale the recording.",
    )
    # The probe properties
    ## manufacturer
    parser.add_argument(
        "-pm",
        "--probe_manufacturer",
        type=str,
        default="cambridgeneurotech",
        help="Name for the probe manufacturer.",
    )
    ## probe name/model
    parser.add_argument(
        "-pn",
        "--probe_name",
        type=str,
        default="ASSY-156-E-2",
        help="Probe model."
    )
    ## probe wiring
    parser.add_argument(
        "-wd",
        "--wiring_device",
        type=str,
        default="ASSY-156>RHD2164",
        help="Wiring of the probe into the headstage.",
    )
    # BANDPASS FILTERING
    # MIN
    parser.add_argument(
        "-fmin",
        "--freq_min",
        type=int,
        default=200,
        help="Lower bound of the bandpass filter to apply to the signal. By default 200hz.",
    )
    # MAX
    parser.add_argument(
        "-fmax",
        "--freq_max",
        type=int,
        default=9000,
        help="Higher bound of the bandpass filter to apply to the signal. By defaul 9khz.",
    )
    # PIPELINE FOR FINAL PREPROCESSING
    parser.add_argument(
        "-car",
        "--car",
        type=str,
        choices={"global", "local" "group"},
        default="global",
        help="Wheter to use local or global CAR.",
    )
    # GENERALEST VISUALIZATION
    parser.add_argument(
        "-bvi",
        "--basic_visual_inspect",
        action="store_true",
        help="Whether to do a visual inspection of the signal to asses quality.",
    )
    # Modifiable for the bad channel dedetction
    parser.add_argument(
        "-rbc",
        "--remove_bad_channels",
        action="store_true",
        help="Whether to remove known bad channels from the recording.",
    )
    ## Add one horrible channel for each channel to be removed
    parser.add_argument(
        "-rbci",
        "--remove_bad_channels_id",
        type=int,
        default=None,
        action="append",
        help="Channel to be removed from the recording",
    )
    parser.add_argument(
        "-rbct",
        "--remove_bad_channels_threshold",
        type=float,
        default=5,
        help="Threshold for considering a channel bad.",
    )
    parser.add_argument(
        "-rbcv",
        "--remove_bad_channels_visualization",
        action="store_true",
        help="Whether to show a visualization of the remove bad channels pipeline.",
    )
    # Modifiable for the artifact removal
    parser.add_argument(
        "-ar",
        "--artifact_removal",
        action="store_true",
        help="Whether to detect and remove artifacts from recording.",
    )
    parser.add_argument(
        "-arf",
        "--artifact_removal_force",
        action="store_true",
        help="Wheter to force the artifact removal.",
    )
    parser.add_argument(
        "-art",
        "--artifact_removal_threshold",
        type=int,
        default=50,
        help="Threshold for considering an event an artifact.",
    )
    parser.add_argument(
        "-armb",
        "--artifact_removal_ms_before",
        type=int,
        default=50,
        help="Remove this ms before detected artifacts",
    )
    parser.add_argument(
        "-arma",
        "--artifact_removal_ms_after",
        type=int,
        default=50,
        help="Remove this ms after detected artifacts",
    )
    parser.add_argument(
        "-arv",
        "--artifact_removal_visualization",
        action="store_true",
        help="Wheter to show some artifacts and the current threshold.",
    )
    # Save the resulting recording
    parser.add_argument(
        "--save",
        action="store_true",
        help="Whether tosave the whole preprocessing, the arguments and result.",
    )

    return parser.parse_args()


def keep_track_of_args(frame):
    # save a YAML file with all the elements in kwargs into fname with a header.
    # based on the examples here:
    # https://github.com/fabianlee/blogcode/blob/master/python/inspect_func_test.py
    args, _, _, values = inspect.getargvalues(frame)
    # Drop the recording because we already will have that on the main frame
    # Also drop the keep track, we know you are.
    ret = {arg: values[arg] for arg in args if arg not in ["recording", "keep_track"]}
    return ret


def save_args(dargs, fname, header):
    # Dump the args dictionary into a YAML
    with open(fname, "wt") as F:
        header = "# Arguments used" if header is None else header
        print(header, file=F)
        print(yaml.dump(dargs, sort_keys=False), file=F)
    pass


def get_car_pipeline(car, groups):
    if car == 'local':
        ret = dict(
                reference="local",
                operator="median",
                local_radius=(150, 300),
            )
    elif car == "global":
        ret = dict(
                reference="global",
                operator="median",
                local_radius=(150, 300),
            )
    elif car == "group":
        ret = dict(
                reference="global",
                groups=groups,
                operator="median",
                local_radius=(150, 300),
            )
    return ret

# Preprocessing pipes


def remove_bad_channels(
    recording,
    thresh=4,
    force_remove=None,
    time_range=[60 * 10, 60 * 11],
    rand_samp_kwds=dict(num_chunks=[10], chunk_sizes=[30]),
    visualization=False,
    keep_track=False,
):
    """A way of detecting bad channels and other that rhyme with it.

    Parameters
    ----------
    recording : _type_
        _description_
    thresh : int, optional
        _description_, by default 4
    force_remove : list or None
        If a list is given, also remove these channels from the recording.
    time_range : list, optional
        _description_, by default [60*10, 60*11]
    rand_samp_kwds : _type_, optional
        _description_, by default dict(num_chunks=[10], chunk_sizes=[30])
    visualization : bool, optional
        _description_, by default False

    Returns
    -------
    _type_
        _description_
    """
    # to detect bad channels first bandpass and re-reference
    pipeline = dict(
        bandpass_filter=dict(
            freq_min=200,
            freq_max=6000,
        ),
        common_reference=dict(reference="global", operator="median"),
    )
    recording_p = lazy_preproc_recording(recording, pipeline=pipeline)

    # Get the noise levels of the recording
    kwds_gnl = dict(return_scaled=True, chunk_size=30000, num_chunks_per_segment=30)
    noise_levs = st.utils.get_noise_levels(recording_p, **kwds_gnl)
    # Given the threshold, mark all bad channels
    nsn_thresh = np.median(noise_levs) + noise_levs.std() * thresh
    bad_ixs = np.nonzero(noise_levs > nsn_thresh)[0]

    # If there are channels to be removed by choice, add them to bad_ixs
    if all([force_remove, isinstance(force_remove, (list, tuple, np.ndarray))]):
        bad_ixs = np.hstack((bad_ixs, force_remove))

    # Sample data, compare channels and for any too similar,
    data = get_structured_random_traces(recording_p, **rand_samp_kwds)
    d = 1 - pairwise_distances(data.T, metric="cosine")
    # If there's a bad channel, check if there are other similarly bad ones
    if len(bad_ixs) > 0:
        # pairwaise distances of the bad channels
        bad_n_friends = d[bad_ixs]
        # index withouth the diagonals
        ixs = bad_n_friends != 1
        threshold = np.median(bad_n_friends[ixs]) + bad_n_friends[ixs].std() * thresh
        bad_ixs = np.nonzero(bad_n_friends > threshold)[1]
        val_channs = list(set(recording_p.channel_ids).difference(set(bad_ixs)))
    else:
        threshold = 0
        val_channs = recording_p.channel_ids
    ret = recording.channel_slice(val_channs)

    # if plotting wanted
    if visualization:
        fig = plt.figure(figsize=[13, 13])
        gs = plt.GridSpec(nrows=2, ncols=2, width_ratios=[0.5, 1])
        [ax_noise, ax_d, ax_trace,] = [
            fig.add_subplot(gs[ixs])
            for ixs in [slice(0, 1), slice(1, 2), slice(2, None)]
        ]

        _ = ax_noise.plot(noise_levs, "ok")
        _ = ax_noise.axhline(nsn_thresh, c="k", ls="--")

        acr = ax_d.imshow(d, aspect="auto", vmin=0.1, vmax=0.9, cmap="PuBu_r")
        _ = fig.colorbar(acr, ax=ax_d)
        # Plot the new raw traces
        _ = plot_timeseries(
            ret,
            time_range=time_range,
            ax=ax_trace,
            order_channel_by_depth=True,
            show_channel_ids=True,
        )

        # add the sum of distances for all noise_channels and the threshold used
        ax_noise_c = ax_noise.twinx()
        _ = ax_noise_c.plot(d[bad_ixs].mean(axis=0), "+r", label="noise similarity")
        _ = ax_noise_c.axhline(threshold, c="r", ls="--")
        # now some labels
        _ = [
            ax.set(
                xlabel="channels",
            )
            for ax in [ax_noise, ax_d]
        ]
        _ = [
            ax.set(
                ylabel=tit,
            )
            for ax, tit in zip(
                [ax_noise, ax_noise_c], ["chan noise levels", "distance to noisy chans"]
            )
        ]
        _ = [
            ax.set(
                title=tit,
            )
            for ax, tit in zip(
                [ax_noise, ax_d], ["noise values", "pairwise distances by chan"]
            )
        ]
        fig.tight_layout()
        _ = plot_recording(ret, pipeline=pipeline)

    # for tracking purposes
    if keep_track:
        frame = inspect.currentframe()
        dargs = keep_track_of_args(frame)
        ret = [ret, dargs]

    return ret


def remove_artifacts(
    recording,
    pipeline_detect=dict(
        bandpass_filter=dict(freq_min=200, freq_max=6000),
        common_reference=dict(reference="global", operator="median"),
    ),
    threshold=100,
    ms_before=50,
    ms_after=50,
    pipeline_return=dict(
        bandpass_filter=dict(freq_min=200, freq_max=6000),
        common_reference=dict(
            reference="local",
            operator="median",
            local_radius=(50, 100),
        ),
    ),
    mode="zeros",
    force=False,
    keep_track=False,
):
    """Remove artifacts in a recording.

    First preprocess, find a good threshold, save a tmp file,
    load and detect on that. After found peaks, re-process, and add a
    remove artifacts.
    """
    # ensure that the file is unfiltered, or we could mess the process
    assert not recording.is_filtered(), "The recording needs to ne unfiltered."

    # Bandpass and CAR
    recording_p = lazy_preproc_recording(recording, pipeline=pipeline_detect)
    ftoken = get_filepath(recording_p)

    # Here's the peak detection, if we already did, load peaks and skip
    # unless we enforce new detection!
    ftoken_peaks = path.splitext(ftoken)[0] + "_artifacts_peaks.csv"
    if all([path.isfile(ftoken_peaks), not force]):
        uniq_peaks = np.loadtxt(ftoken_peaks).astype(int).tolist()
    else:
        # Check for tmp folder to dump the signal needed for peak detection
        dtoken_temp = path.join(path.split(ftoken)[0], "temp")
        # If there's a dtoken remove it
        if path.isdir(dtoken_temp):
            rmtree(dtoken_temp)
        kwds_save = dict(
            name=None,
            folder=dtoken_temp,
            joblib_backend="loki",
            chunk_size=recording.get_sampling_frequency() * 60 ^ 3,
            progress_bar=True,
            n_jobs=5,
        )
        # Dump the recording for sanity
        print(f"\nSaving a temporary file as {dtoken_temp}.")
        recording_tmp = recording_p.save_to_folder(**kwds_save)
        print("\tDone with the tmp preprocessing and dump.\n")
        # Detect on the temp
        peaks = detect_peaks(
            recording_tmp,
            peak_sign="both",
            detect_threshold=threshold,
            total_memory="3G",
            n_jobs=3,
        )
        uniq_peaks = np.unique(peaks["sample_ind"]).astype(int).tolist()
        # save the peaks
        np.savetxt(ftoken_peaks, uniq_peaks, fmt="%d")
        # Now delete the temp files and the rest!
        rmtree(dtoken_temp)
        del recording_tmp, peaks
        print("All peaks found")

    kwds_ra = dict(ms_before=ms_before, ms_after=ms_after, mode=mode)
    # Again the preprocessing
    recording_p = lazy_preproc_recording(recording, pipeline=pipeline_return)
    # But now remove the artifacts
    ret = st.remove_artifacts(recording_p, uniq_peaks, **kwds_ra)
    if keep_track:
        frame = inspect.currentframe()
        dargs = keep_track_of_args(frame)
        ret = [ret, dargs]
    return ret


def save_prepoc(recording, kwds_save):
    ftoken = get_filepath(recording)
    dtoken_bin = path.splitext(ftoken)[0] + "_preproc"
    print(f"\nSaving the preprocessed file as {dtoken_bin}.")
    kwds_save.update(dict(name=None, folder=dtoken_bin, progress_bar=True))
    recording.save_to_folder(**kwds_save)
    print("\tDone saving the preprocessed file.\n")
    pass


# Prehistoric visualizations
def plot_random_samples(recording, seed=42, samp_dur=10, num_samps=6, pipeline=None):
    # get random pieces around the session and plot the results as channels
    np.random.seed(seed)
    frame_ranges = [
        (np.add(np.random.randint(0, recording.get_total_duration() - (samp_dur)), [0, samp_dur])
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
    # to plot the whole recording
    if full_rec:
        frame_range = [0, recording.get_total_duration() * recording.get_sampling_frequency()]
        time_range = np.divide(frame_range, recording._sampling_frequency)
    else:
        frame_range = np.add(np.random.randint(recording.get_total_duration()-60), [0, 60]) * recording.get_sampling_frequency()
        time_range=[0, 60]
    recording = recording.frame_slice(*frame_range)
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
            time_range=time_range,
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


if __name__ == "__main__":
    # ephys_preproc.py /personal/Documents/datasets/ephys/Vileplume/SNr/Vileplume_EphysRaw_SNr_2021-07-11T11_18_35/Vileplume_EphysRaw_SNr_2021-07-11T11_18_35.bin^Cnc 64 -rbc true -rbcv f  -rbct 2.5 -rbci 17 -rbci 47 -ar t --save false
    args = parse_args()
    args_rec = args.ftoken_bin, args.sampling_rate, args.n_channels, args.dtype
    kwds_rec = dict(
        offset=args.offset,
        gain=args.gain,
        probe_manufacturer=args.probe_manufacturer,
        probe_name=args.probe_name,
        wiring_device=args.wiring_device,
    )
    # to keep track of the pipeline save relevant stuff in a dictionary
    dargs = dict()
    dargs["load_file"] = dict(ftoken_bin=args.ftoken_bin, **kwds_rec)
    kwds_rec["wiring_device"] = get_wiring(args.wiring_device)
    recording = load_recording(*args_rec, **kwds_rec)

    # General values for later
    groups = [(np.nonzero(recording.get_channel_groups() == group)[0]).tolist()  for group in np.unique(recording.get_channel_groups())]
    pipeline_detect = dict(
        bandpass_filter=dict(freq_min=args.freq_min, freq_max=args.freq_max),
        common_reference=dict(reference="global", operator="median"),
    )
    pipeline_return = dict(
        bandpass_filter=dict(freq_min=args.freq_min, freq_max=args.freq_max),
        common_reference=get_car_pipeline(args.car, groups)
    )

    if args.basic_visual_inspect:
        time_range = [60 * 24, 60 * 25]
        w = plot_recording(recording, pipeline=pipeline_return)
        w = plot_random_samples(recording, pipeline=pipeline_return)

    # Find noisy channels
    if args.remove_bad_channels:
        kwds_rbd = dict(
            thresh=args.remove_bad_channels_threshold,
            force_remove=args.remove_bad_channels_id,
            time_range=[60 * 10, 60 * 11],
            rand_samp_kwds=dict(num_chunks=[10], chunk_sizes=[30]),
            visualization=args.remove_bad_channels_visualization,
            keep_track=True,
        )
        recording, dargs["remove_bad_channels"] = remove_bad_channels(
            recording, **kwds_rbd
        )
        if args.remove_bad_channels_visualization:
            w = plot_random_samples(recording, pipeline=pipeline_return)
            p = plot_probe_map(recording, with_channel_ids=True)


    # Find and remove the artifacts, save them JIC
    # The common pipeline to be used all over the places
    if args.artifact_removal:
        kwds_ra = dict(
            pipeline_detect=pipeline_detect,
            threshold=args.artifact_removal_threshold,
            ms_before=args.artifact_removal_ms_before,
            ms_after=args.artifact_removal_ms_after,
            pipeline_return=pipeline_return,
            mode="zeros",
            force=args.artifact_removal_force,
            keep_track=True,
        )
        recording_p, dargs["remove_artifacts"] = remove_artifacts(recording, **kwds_ra)
        if args.artifact_removal_visualization:
            _ = plot_artifacts(recording, thresh=args.artifact_removal_threshold)

    # Save the new processed file
    if args.save:
        kwds_save = dict(
            joblib_backend='loki',
            chunk_size=recording_p.get_sampling_frequency()*60^3,
            total_memory="4G",
            n_jobs=5
        )
        save_prepoc(recording_p, kwds_save)
        ftoken_dargs = path.splitext(args.ftoken_bin)[0] + "_pipeline.yaml"
        save_args(dargs, ftoken_dargs, '# Relevant params for the preprocessing')
