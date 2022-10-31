"""
Test loading of phonemes along with the alignment.
"""

import numpy as np
import matplotlib.pyplot as plt
from lib import *

from matplotlib import cm

def plot_mel_spectrogram_comp(mel_spec, mel_spec_b, title):
    fig, ax = plt.subplots(2)

    ax[0].set_title(f"Mel Spectogram Old \"{title}\"")
    pred = np.swapaxes(mel_spec, 0, 1)
    cax = ax[0].imshow(pred, interpolation='nearest', cmap=cm.coolwarm, origin='lower')

    ax[1].set_title(f"Mel Spectogram New \"{title}\"")
    pred = np.swapaxes(mel_spec_b, 0, 1)
    cax = ax[1].imshow(pred, interpolation='nearest', cmap=cm.coolwarm, origin='lower')

    return fig

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
    debug=False)

phone_counts = {}
phoneme_dict = dataset.phoneme_dict
for i in range(len(phoneme_dict)):
    phone = phoneme_dict[i]
    phone_counts[phone] = 0

# for i in range(len(dataset)):
for i in range(10): # len(dataset)): # 300, 500):
    item = dataset[i]
    print("item:", i)
    for phone_id in item["phonemes"]:
        phone = phoneme_dict[phone_id]
        phone_counts[phone] += 1
    title = str(i) + " " + item["label"]
    # fig = plot_mel_spectrogram(item["audio_feats"], title)
    phone_labels = [phoneme_dict[ph_id] for ph_id in item["phonemes"]]
    # print(phone_labels, len(phone_labels), item["audio_feats"].shape)
    # plt.show()

print(phone_counts)

"""
for i in range(0, 10):
    item = dataset[i]
    title = str(i) + " " + item["label"]
    fig = plot_mel_spectrogram_comp(
        item["audio_feats_old"],
        item["audio_feats_new"],
        f"{title} {item['audio_feats_old'].shape, item['audio_feats_new'].shape}")
    print()
    plt.show()
"""