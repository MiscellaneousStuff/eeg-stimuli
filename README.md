# EEG Stimuli

## About
EEG Speech Stimuli (Listening) Decoding Research.
Uses [Brennan 2019](https://doi.org/10.7302/Z29C6VNH) dataset which covers EEG recordings while listening to the first chapter of Alice in Wonderland.

## Dataset
[Brennan 2019](https://doi.org/10.7302/Z29C6VNH)
33 datasets out of 49 were used in the analysis. 8 out of them were excluded due to low performance on the comprehension quiz. \
8 of them come from participants with high noise.

### Usable Files

Many of the datasets were excluded because the pt's performed badly on comprehension tests or the signal contained too much noise.
This is the list of usable datasets:

['S01', 'S03', 'S04', 'S05', 'S06', 'S08', 'S10', 'S11', 'S12', 'S13', 'S14', 'S15', 'S16', 'S17', 'S18', 'S19', 'S20', 'S21', 'S22', 'S25', 'S26', 'S34', 'S35', 'S36', 'S37', 'S38', 'S39', 'S40', 'S41', 'S42', 'S44', 'S45', 'S48']

### Required Files

- audio.zip
- datasets.mat
- AliceChapterOne-EEG.csv
- S__.zip for usable dataset