#!/bin/python3.8
# @Author: mauro
# @Date:   2019-10-31T13:13:28+00:00
# @Last modified by:   Mauricio Toro E. (@Nauta)
# @Last modified time: 2021-10-20 16:20:34

"""Extract waveforms from the spike sorting results

Still hacky solution, but can take a proper form and shape later...

for now it works, but requires human renaming of the resulting pickle file.

"""

from os import path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from spikeinterface.exporters import export_report
import spikeinterface as si


def PHY_extractor(ftoken_bin, dtoken_phy, n_chans=32, dtype=np.int16,
                  wave_len=25):
    """Extract the results from phy curation.

    Load a signal and some phy resulting files,
    Outputs a DataFrame to be pickled.

    Parameters
    ----------
    ftoken_bin : str
        Location a a binary file with ephys.
    dtoken_phy : str
        Location of the phy results folder.
    n_chans : int
        Number of channels in the recording.
    dtype : np.dtype
        Data type of the binary recordings.
    wave_len : int
        How many samples around the event to save.
    Returns
    -------
    DataFrame
        A pandas.DataFrame with columns:
            "id" : int
                ID of the neuron.
            "channel" : int
                Channel where the neuron had the highest peak.
            "label" : str
                Label of the neuron. Either `good` or `mua`.
            "sample" : int
                Sample where the detected spike happened.
            "wave" : array
                Samples around the spike to have a sense of the waveform.

    TODO/IMPROVE:
        Add timestamps to the DataFrame? it's cheaper to make it each time.
        But might be easier this way...

    """
    ftoken_clust_inf = path.join(dtoken_phy, "cluster_info.tsv")
    ftoken_spk_t = path.join(dtoken_phy, "spike_times.npy")
    ftoken_spk_c = path.join(dtoken_phy, "spike_clusters.npy")
    clust_info = pd.read_csv(ftoken_clust_inf, sep="\t")\
        .query("group != 'noise'")
    q = ") | ".join(["(cluster == {}".format(i)
                     for i in clust_info.loc[:, "cluster_id"].values]) + ")"
    spk_times = np.load(ftoken_spk_t).ravel()
    spk_clust = np.load(ftoken_spk_c).ravel()
    spks = pd.DataFrame(data=np.vstack([spk_clust, spk_times]).T,
                        columns=["cluster", "times"])\
        .astype(int).query(q).reset_index(drop=1)
    signals = np.memmap(ftoken_bin, dtype=dtype).reshape(-1, n_chans)
    neurons = pd.DataFrame()
    for x, ci in clust_info.iterrows():
        neuron = pd.DataFrame()
        neuron["sample"] = spks.query("cluster == {}".format(ci.cluster_id)).times.values
        neuron["wave"] = [signals[ts-wave_len:ts+wave_len, ci.ch]
                 for ts in spks.query("cluster == {}".format(ci.cluster_id)).times.values]
        neuron["id"] = ci.cluster_id
        neuron["channel"] = ci.ch
        neuron["label"] = ci.group
        neuron = neuron[["id", "channel", "label", "sample", "wave"]]
        neurons = pd.concat([neurons, neuron], axis=0)
    return neurons


def POLS_extractor(ftoken_pols, sample_rate=30000,
                   frmt=["channel", "id", "timestamps",
                         "pc1", "pc2", "pc3", "wave"]):
    """Extract the results from texts files from plexon offline sorter.

    Load the data with format:
        Channel, Unit, Timestamp, PC 1, PC 2, PC 3, WAVEFORM:
    Outputs a DataFrame to be pickled.

    Parameters
    ----------
    ftoken_pols : str
        Location a plexon offline sorter .txt file.
    n_chans : int
        Number of channels in the recording.
    dtype : np.dtype
        Data type of the binary recordings.
    wave_len : int
        How many samples around the event to save.
    Returns
    -------
    DataFrame
        A pandas.DataFrame with columns:
            "id" : int
                ID of the neuron.
            "channel" : int
                Channel where the neuron had the highest peak.
            "label" : str
                Label of the neuron. Either `good` or `mua`.
            "sample" : int
                Sample where the detected spike happened.
            "wave" : array
                Samples around the spike to have a sense of the waveform.

    """
    pols_d = pd.read_csv(ftoken_pols, header=None, skiprows=1, memory_map=True)
    df = pd.DataFrame()
    df["id"] = pols_d[1]
    df["channel"] = pols_d[0]
    df["label"] = "pols"
    df["sample"] = (pols_d[2] * 30000).astype(int)
    df["wave"] = [x for x  in pols_d.loc[:, 6:].values]
    neurons = df[["id", "channel", "label", "sample", "wave"]]
    return neurons


def make_SI_report(dtoken_si, dtoken_phy):
    import spikeinterface.sorters as ss
    recording = si.load_extractor(dtoken_si)
    sorting = ss.read_sorter_folder(dtoken_phy)
    dtoken_waves = path.join(dtoken_phy, "waves")
    dtoken_report = path.join(dtoken_phy, "report")
    kwargs_job = dict(
        progress_bar=True,
        n_jobs=3,
        total_memory="500M",
    )
    print(f"\nExtracting waveforms into {dtoken_waves}")
    waves = si.extract_waveforms(
        recording,
        sorting,
        folder=dtoken_waves,
        overwrite=False,
        **kwargs_job,
    )
    print("\tDone extracting waveforms.\n")
    print(f"\nGenerating sorting report in {dtoken_report}")
    reports = export_report(waves, dtoken_report, **kwargs_job)
    print("\tDone with the report.")
    pass


def parse_args():
    description = \
    """Get the spikes from a sorting result.

    After running some spike sorting algorithm, this allows to extract the
    individual spikes with their asigned cluster, channel ID and waveform.

    This are saved as a pickle ndarray, where each row is one spike, and 5 columns
    ["id", "channel", "label", "sample", "wave"], the last column has wave_len size.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("ftoken_bin", type=str,
                        help="SpikeInterface recording directory.")
    parser.add_argument("dtoken", type=str,
                        help="Location of the phy results folder.")
    parser.add_argument("--extractor", type=str, default="phy",
                        choices={'phy', 'pols'},
                        help="Data type of the recording as a string.")
    parser.add_argument("--dtype", type=str, default="np.int16",
                        help="Data type of the recording as a string.")
    parser.add_argument("-nc", "--n_channels", type=int, default=32,
                        help="Number of channels in the recordings.")
    parser.add_argument("-wl", "--wave_len", type=int, default=30,
                        help="How many samples around the event to save.")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    assert type(eval(args.dtype)) == type, \
        "The dtype must be a valid and existing dtype."
    assert path.isfile(args.ftoken_bin), \
        "The binary file must exist."
    assert path.isdir(args.dtoken), \
        "The phy results directory must exist."
    print("Loading neurons.")
    if args.extractor == "phy":
        neurons = PHY_extractor(args.ftoken_bin, args.dtoken,
                                n_chans=args.n_channels, dtype=eval(args.dtype),
                                wave_len=args.wave_len)
        # Make the spikeinterface report for the sorting results
        dtoken_si = path.split(args.ftoken_bin)[0]
        dtoken_phy = args.dtoken
        make_SI_report(dtoken_si, dtoken_phy)
    elif args.extractor == "pols":
        neurons = POLS_extractor(args.ftoken_bin, sample_rate=30000)
    new_f = path.splitext(args.ftoken_bin)[0] + ".pkl"
    print("Saving the results.")
    neurons.to_pickle(new_f)
    print("Results saved.")
    exit()