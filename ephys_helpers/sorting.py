""" Run an automatic spike sorting algorithm on some preprocessed data.

"""

import yaml
import argparse
from os import path
from pathlib import Path
from datetime import datetime
import spikeinterface as si
import spikeinterface.sorters as ss
from spikeinterface.exporters.report import export_report


# for interactive ploting of whatever
from spikeinterface.widgets import plot_probe_map
from utils import plot_random_samples, plot_recording, check_for_file
import matplotlib.pyplot as plt
plt.ion()

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


def parse_args():
    description = """Quick-Auto-Sort ephys data.

    Load a binary datafile and quickly preprocess relying mostly on spikeinterface.

    After preprocessing, makes a kilo friendly pipeline.

    """
    parser = argparse.ArgumentParser(description=description)

    # File of ephys recording
    parser.add_argument("ftoken_si", type=str, help="SpikeInterface ephys data folder.")

    # for sorting with kilo
    parser.add_argument(
        "-ss",
        "--spike_sorter",
        type=str2bool,
        default=True,
        nargs='?',
        const=True,
        help="Whether to do a visual inspection of the signal to asses quality.",
    )
    parser.add_argument(
        "-ssa",
        "--spike_sorter_algorithm",
        type=str,
        default="ks2",
        choices={'ks2', 'ks25'},
        help="What spike sorting algorithm to use. For now only Kilosort 2, 2.5.",
    )
    parser.add_argument(
        "-msr",
        "--make_sorting_report",
        action="store_true",
        default=False,
        help="Whether to save a report to asses sorting quality.",
    )
    parser.add_argument(
        "-msrp",
        "--make_sorting_report_path",
        type=str,
        default=None,
        help="If there's more than one sorting result, use this cue to pick the correct.",
    )
    return parser.parse_args()


def nested_dict_values(d):
    # Taken from stackoverflow.com
    # questions/23981553/get-all-values-from-nested-dictionaries-in-python
    for v in d.values():
        if isinstance(v, dict):
            yield from nested_dict_values(v)
        else:
            yield v


def get_filepath(recording):
    """Get the filepath in a recording.

    Parameters
    ----------
    recording : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    # Get the filepath of the original file:
    d_vals = list(nested_dict_values(recording.to_dict()))
    # Only one filepath on the whole list
    ret = [
        # unravel the values, but keep only the lists
        f
        for ff in d_vals
        if isinstance(ff, list)
        # Keep only things that are not integers
        for f in ff
        if not isinstance(f, (int, float, list, tuple, dict))
    ][0]
    return ret



def set_kilosrt_2_5_params(args):
    params = ss.Kilosort2_5Sorter.default_params()
    params.update(dict(
        car=False, minFR=0.01, minfr_goodchannels=0.1, preclust_threshold=7,
        ntbuff=16, detect_threshold=4, freq_min=300, nfilt_factor=8)
    )
    # Check for params made for this file in particular:
    ftoken_params = check_for_file(args.dtoken_base, 'kilosort2_5')
    if ftoken_params:
        params.update(yaml.safe_load(open(ftoken_params, "r")))
    return params


def set_kilosrt_2_params(args):
    # get spikeInterface default params
    params = ss.Kilosort2Sorter.default_params()
    # Set our defaults
    params.update(dict(
        car=False, minFR=0.01, minfr_goodchannels=0.1, preclust_threshold=7,
        ntbuff=16, detect_threshold=3, freq_min=300, nfilt_factor=8)
    )
    # if there's a file with params, update everything to those.
    ftoken_params = check_for_file(args.dtoken_base, 'kilosort2')
    if ftoken_params:
        params.update(yaml.safe_load(open(ftoken_params, "r")))
    return params


def get_sorter_and_params(args):
    today = args.today.strftime("%Y-%m-%dT%H%M")
    args.dtoken_output = path.join(args.dtoken_base, f'{args.spike_sorter_algorithm}_{today}')
    args.dtoken_wfs = path.join(args.dtoken_base, 'waves')
    args.dtoken_rep = path.join(path.split(args.dtoken_output)[0], "report")
    if args.spike_sorter_algorithm == "ks2":
        args.params = set_kilosrt_2_5_params(args)
        args.sorter = ss.run_kilosort2_5
    if args.spike_sorter_algorithm == "ks2":
        args.params = set_kilosrt_2_params(args)
        args.sorter = ss.run_kilosort2
    return args


def make_sorting_report(recording, args):
    # check that there's already a sorted result,
    base_dir = Path(args.dtoken_base)
    sorting_results = [dtoken for dtoken in base_dir.iterdir() if args.spike_sorter_algorithm in str(dtoken)]
    # if more than one sorting result
    if len(sorting_results) > 1:
        # first check if there's a phy folder, if only one has it, that's the one
        sorting_results = [dtoken for dtoken in sorting_results for dtoken_phy in dtoken.iterdir() if '.phy' in str(dtoken_phy)]
        if len(sorting_results) == 1:
            sorting_results = sorting_results[0]
        # Check if there was a cued one
        elif args.make_sorting_report_path:
            cue = args.make_sorting_report_path
            sorting_results = [dtoken for dtoken in sorting_results if cue in str(dtoken)][0]
        # if these two didn't work, raise error
        if isinstance(sorting_results, (list, tuple)):
            msg = 'there are still too many sorting results, remove one or add a cue.'
            raise(ValueError(msg))
    else:
        sorting_results = sorting_results[0]    
    sorting = ss.read_sorter_folder(sorting_results)
    kwds_jobs = dict(n_jobs=3, total_memory="500M", progress_bar=True)
    waves = si.extract_waveforms(
        recording, sorting, folder=args.dtoken_wfs, overwrite=True,
        **kwds_jobs
         )
    report = export_report(waves, args.dtoken_rep, **kwds_jobs)
    return waves, report


if __name__ == "__main__":
    args = parse_args()
    ftoken_si = args.ftoken_si
    args.dtoken_base = path.split(ftoken_si)[0]
    recording = si.load_extractor(ftoken_si)
    args.today = datetime.today()
    args = get_sorter_and_params(args)

    # Now un the spike sorter
    if args.spike_sorter:
        print("\nStart spike sorting\n")
        sorting = args.sorter(recording, output_folder=args.dtoken_output, **args.params)
        print("\nDone Sorting\n")
    # # Run kilosort, possible ones
    # sorting_KS25 = ss.run_kilosort2_5(recording, output_folder=args.dtoken_output, **args.ks25_params)
    # sorting_KS2 = ss.run_kilosort2(recording, output_folder=args.dtoken_output, **args.ks2_params)

    # # Check sorting quality
    if args.make_sorting_report:
        waves, report = make_sorting_report(recording, args)