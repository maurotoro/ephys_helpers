# ephys_helpers
Some code for spike sorting with the help of Spikeinterface and magick.


# Documentation

Most files contain a good enough and descriptive help, check those and then ask me.


# Pipeline

This are some rough guides of the process to get this tricks to work

## Preprocessing

The general pipeline for preprocessing goes on these lines:

1. Check how a recording session looks in a general way.
2. Remove by hand any noisy channels, let an algo check for other noisy channels and evaluate performance.
3. Find high amplitude peaks on the recording.
4. Save a new file without the noisy channels, bandpassed and re-referenced to a commom median, where all the artifacts detected are censored.

### Example pipeline

#### Check noisy channels
To look at a sample of a recording session done with a Cambridge neurotech 'ASSY-116-P2'
probe, 64 channels in 4 shanks of 16 channels each, connected to an Intan headstage
'RHD2164', sampled at 30khz, with int16 resolution, with the default gain and offsets
of 0.196 and -32768, to convert the data into SI units.

Run this like of code on a terminal:

```bash
python ephys_helpers/preprocess.py session_recording.bin \
    # sampling rate
    --sampling_rate 30000 \
    # number of channels
    --n_channels 64 \
    # data type
    --dtype i2 \
    # gain steps
    --gain 0.195 \
    # offset
    --offset -32768 \
    # Probe manufacturer, must exist on probeinterface
    # or be added
    --probe_manufacturer cambridgeneurotech \
    # Probe name, must exist on probeinterface
    # or be added
    --probe_name ASSY-116-P2 \
    # How the probe connects to the headstage, must exist on probeinterface
    # or be added
    --wiring_device ASSY-156>RHD2164
    # boolean switch to have a basic visual inspection
    --basic_visual_inspect
```

On that visualization one can get a sensse of which channels to censor from the
preprocessing and later steps. The channels could look like too noisy, too narrow
bandwith, or weird unique peaks. The origin of this issues is highly bizzarre,
some can be corrected during recorgind, by rechecking the grounding on the implant
or setup, but some are related to having electrodes on non-recording friendly places,
move them and it could help.
**In any case, do not assume that the wrong channels on one day will transfer for the next, always check them.**


#### Remove noisy channels and other that rhyme with them
Let's say we found that channels 9, 37, 41 are noisy, next we manually remove them and check how the signal looks afterwards

```bash
python ephys_helpers/preprocess.py session_recording.bin \
    # All first part stays the same
    -sr 30000 -nc 64 -dtype i2 -g 0.195 -o -32768 -pm cambridgeneurotech -pn ASSY-116-P2 -wd ASSY-156>RHD216 \
    # Remove bad channels switch
    --remove_bad_channels
    # Now add each noisy channel
    --remove_bad_channels_id 9
    --remove_bad_channels_id 37
    --remove_bad_channels_id 41
    # Channels that are this different (MAD times) from the rest will be removed
    --remove_bad_channels_threshold 4
    # Visualization of the bad channel switch
    --remove_bad_channels_visualization
```


The visualizations are a random minute in the session and 60 random samples of 10
seconds. They should show a better looking signal, where the LFP should be clearly
noticeable and the badwith of all channels should be similar enough. There's another
plot, showing the similarity between the channels and it's ussed to look for channels
that are too similar to the ones labeled as noisy or channels that are too far from
the rest.

#### Detect artifacts

Now, asumming a freely moving session, there will be movement, chewing or other
associated artifacts, they will induce really **HIGH amplitude** noise on the signal,
this will hurt the spike detection as some 'really goo amazing units' are sparsely
present and whatever, they do hurt a lot. Set to zero everything a bit before and
a bit after the artifacts.

```bash
python ephys_helpers/preprocess.py session_recording.bin \
    # All first part stays the same
    -sr 30000 -nc 64 -dtype i2 -g 0.195 -o -32768 -pm cambridgeneurotech -pn ASSY-116-P2 -wd ASSY-156>RHD216 \
    # Remove bad channels pipeline on short notation, no visualization
    -rbc -rbci 9 -rbci 37 -rbci 41 -rbct 4
    # Artifact detection switch
    --artifact_removal
    # Artifact removal threshold, MAD times
    --artifact_removal_threshold 50
    # Artifact removal windows
    --artifact_removal_ms_before 100
    --artifact_removal_ms_after 100
    # Artifact removal visualization
    --artifact_removal_visualization
```

The visualization shows the artifact thresholds and some 40 samples of the artifacts
detected.

#### Save result

If the resulting signal is satisfactory, save it.

```bash
python ephys_helpers/preprocess.py session_recording.bin \
    # All first part stays the same
    -sr 30000 -nc 64 -dtype i2 -g 0.195 -o -32768 -pm cambridgeneurotech -pn ASSY-116-P2 -wd ASSY-156>RHD216 \
    # Remove bad channels pipeline on short notation, no visualization
    -rbc -rbci 9 -rbci 37 -rbci 41 -rbct 4
    # Remove artifacts pipeline on short notation, no visualization
    -ar -art 50 -armb 100 -arma 100
    # Some relevant values for the preprocessing pipeline
    # Low cut for the bandpass
    --freq_min 200
    # High cut for the bandpass
    --freq_max 9000
    # The type of common average referencing to be used
    # one of {global, local group}, by default is global
    --car global
    # Save file switch
    --save
```

#### Run sorting algorithm

Describe how to run the sorting script.



#### Extract spikes

Describe how to run the spike extraction script.


# Comments

When running kilosort-X there might be issues, the main one found up to now is that they are hiding that one can change the length of the waveforms extracted, and this parameter has been empirically shown to be relevant for our recordings. The current way to have access to it is by changing the `ops.nt0` by hand after the creation of the pipeline by spikeInterface, this is subpar but hasn't been added on their end, I'll make a pull request to add it as it could be useful.
