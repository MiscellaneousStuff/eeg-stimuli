"""
Test loading of phonemes along with the alignment.
"""

import numpy as np
import matplotlib.pyplot as plt
from lib import *

from matplotlib import cm

def plot_mel_spectrogram(mel_spec, title):
    fig, ax = plt.subplots(1)

    ax.set_title(f"Mel Spectogram \"{title}\"")
    pred = np.swapaxes(mel_spec, 0, 1)
    cax = ax.imshow(pred, interpolation='nearest', cmap=cm.coolwarm, origin='lower')

    return fig

dataset = BrennanDataset(
    root_dir="./dataset/",
    phoneme_dir="./phonemes/",
    idx="S13",
    debug=True)

phone_counts = {}
phoneme_dict = dataset.phoneme_dict
for i in range(len(phoneme_dict)):
    phone = phoneme_dict[i]
    phone_counts[phone] = 0

for i in range(1, 2):
    item = dataset[i]
    print("item:", i)
    for phone_id in item["phonemes"]:
        phone = phoneme_dict[phone_id]
        phone_counts[phone] += 1
    fig = plot_mel_spectrogram(item["audio_feats"], i)
    # plt.plot()
    # plt.show()

print(phone_counts)