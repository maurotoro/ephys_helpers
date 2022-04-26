"""General tests for ephysviewer based plots

Ephyviewer it;s a bit tricky, this only works as standalone, and not with
matplotlib on the side, so beware. Mybe if setting mpl to be QT based?...
anyway, this is WPI and maybe to be left to rot and fosilize.

"""
import argparse
import numpy as np
import spikeinterface as si
import spikeinterface.toolkit as st


from time import sleep
from ephyviewer import mkQApp, MainViewer, TraceViewer, SpikeInterfaceRecordingSource

from utils import load_recording, get_wiring, lazy_preproc_recording

def tests():
    recording = si.load_extractor("Vileplume_EphysRaw_SNr_2021-07-07T10_06_52_preproc")


    app = mkQApp()
    win = MainViewer(debug=True, show_auto_scale=True)

    sig_source = SpikeInterfaceRecordingSource(recording)

    view = TraceViewer(source=sig_source, name='raw signal')
    view.auto_scale()
    view.params["scale_mode"] = 'same_for_all'
    view.params["display_labels"] = True

    for c in range(sig_source.nb_channel):
        print(f'ch{c}')
        view.by_channel_params[f'ch{c}'] =dict(visible= True)

    win.add_view(view)

    win.show()
    app.exec()


def set_sigs_visible(view, win, bad_channels=[]):
    view.auto_scale()
    view.params["scale_mode"] = 'same_for_all'
    view.params["display_labels"] = True

    for c in range(view.source.nb_channel):
        if c not in bad_channels:
            view.by_channel_params[f'ch{c}'] =dict(visible= True)
    win.add_view(view)
    sleep(.5)


def get_ephy_viewer(recording, bad_channels=[], pipeline=dict(
         
    ),):
    # Make an app, and collect the views
    app = mkQApp()

    # first make a base recording viewer
    sig_raw = SpikeInterfaceRecordingSource(recording)
    view_raw = TraceViewer(source=sig_raw, name='raw signal')
    win = MainViewer(debug=True, show_auto_scale=True)
    set_sigs_visible(view_raw, win, bad_channels=bad_channels)


    # if there are some preprocesses, do them
    if pipeline:
        sig_preproc = SpikeInterfaceRecordingSource(lazy_preproc_recording(recording, pipeline=pipeline))
        view_preproc = TraceViewer(source=sig_preproc, name='preproc signal')
        set_sigs_visible(view_preproc, win, bad_channels=bad_channels)
    sleep(.5)
    return win, app


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
        "-pn", "--probe_name", type=str, default="ASSY-156-E-2", help="Probe model."
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
    parser.add_argument(
        "-fmin",
        "--freq_min",
        type=int,
        default=200,
        help="Lower bound of the bandpass filter to apply to the signal. By default 200hz.",
    )
    parser.add_argument(
        "-fmax",
        "--freq_max",
        type=int,
        default=9000,
        help="Higher bound of the bandpass filter to apply to the signal. By defaul 9khz.",
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
        default=2,
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


if __name__ == "__main__":
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