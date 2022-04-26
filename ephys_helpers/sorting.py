import yaml
import argparse
from utils import get_filepath
from os import path
from datetime import datetime
# import matplotlib.pyplot as plt

import spikeinterface as si
import spikeinterface.sorters as ss
import spikeinterface.widgets as sw

# plt.ion()

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
        type=str,
        default="ks2",
        choices={'ks2', 'ks25'},
        help="What spike sorting algorithm to use. For now only Kilosort 2, 2.5.",
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



def set_kilosrt_2_5_params(args):
    ks25_params = ss.Kilosort2_5Sorter.default_params()
    ks25_params.update(dict(
        car=False, minFR=0.01, minfr_goodchannels=0.1, preclust_threshold=7,
        ntbuff=16, detect_threshold=4, freq_min=300, nfilt_factor=8)
    )
    params = ks25_params
    return params


def set_kilosrt_2_params(args):
    ks2_params = ss.Kilosort2Sorter.default_params()
    ks2_params.update(dict(
        car=False, minFR=0.01, minfr_goodchannels=0.1, preclust_threshold=7,
        ntbuff=16, detect_threshold=4, freq_min=300, nfilt_factor=8)
    )
    params = ks2_params
    return params


def get_sorter_and_params(args):
    sorter = args.spike_sorter
    today = args.today.strftime("%Y-%m-%dT%H%M")
    args.dtoken_output = path.join(args.dtoken_base, f'{sorter}_{today}')
    args.dtoken_wfs = path.join(args.dtoken_base, 'waves')
    args.dtoken_rep = path.join(path.split(args.dtoken_output)[0], "report")
    if args.spike_sorter == "ks2":
        args.params = set_kilosrt_2_5_params(args)
        args.sorter = ss.run_kilosort2_5
    if args.spike_sorter == "ks2":
        args.params = set_kilosrt_2_params(args)
        args.sorter = ss.run_kilosort2
    return args


def get_sorting_result(recording, params):
    ftoken_bin = get_filepath(recording)


if __name__ == "__main__":
    args = parse_args()
    ftoken_si = args.ftoken_si
    args.dtoken_base = path.split(ftoken_si)[0]
    recording = si.load_extractor(ftoken_si)
    args.today = datetime.today()
    args = get_sorter_and_params(args)

    # Now un the spike sorter
    print("\nStart spike sorting\n")
    sorting = args.sorter(recording, output_folder=args.dtoken_output, **args.params)
    print("\nDone Sorting\n")
    # # Run kilosort, possible ones
    # sorting_KS25 = ss.run_kilosort2_5(recording, output_folder=args.dtoken_output, **args.ks25_params)
    # sorting_KS2 = ss.run_kilosort2(recording, output_folder=args.dtoken_output, **args.ks2_params)


    # # Check sorting quality
    # waves = si.extract_waveforms(
    #     recording, sorting_KS25, folder=dtoken_wfs, progress_bar=True,
    #     n_jobs=1, total_memory="500M", overwrite=True)

    # # check units
    # for id in range(sorting_KS25.get_num_units()):
    #     w = sw.plot_unit_summary(waves, unit_id=id)
