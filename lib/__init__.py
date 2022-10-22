import torch
import scipy.io
import os
import librosa

import pandas as pd
import soundfile as sf
import numpy as np

import matplotlib.pyplot as plt

def double_average(x):
    assert len(x.shape) == 1
    f = np.ones(9)/9.0
    v = np.convolve(x, f, mode='same')
    w = np.convolve(v, f, mode='same')
    return w

def get_semg_feats_orig(eeg_data, hop_length=6, frame_length=16, stft=False, debug=False):
    xs = eeg_data - eeg_data.mean(axis=0, keepdims=True)
    frame_features = []
    for i in range(eeg_data.shape[0]):
        x = xs[i, :]
        w = double_average(x)
        p = x - w
        r = np.abs(p)

        w_h = librosa.util.frame(w, frame_length=frame_length, hop_length=hop_length).mean(axis=0)
        p_w = librosa.feature.rms(w, frame_length=frame_length, hop_length=hop_length, center=False)
        p_w = np.squeeze(p_w, 0)
        p_r = librosa.feature.rms(r, frame_length=frame_length, hop_length=hop_length, center=False)
        p_r = np.squeeze(p_r, 0)
        z_p = librosa.feature.zero_crossing_rate(p, frame_length=frame_length, hop_length=hop_length, center=False)
        z_p = np.squeeze(z_p, 0)
        r_h = librosa.util.frame(r, frame_length=frame_length, hop_length=hop_length).mean(axis=0)

        if stft:
            s = abs(librosa.stft(np.ascontiguousarray(x), n_fft=frame_length, hop_length=hop_length, center=False))

        if debug:
            plt.subplot(7,1,1)
            plt.plot(x)
            plt.subplot(7,1,2)
            plt.plot(w_h)
            plt.subplot(7,1,3)
            plt.plot(p_w)
            plt.subplot(7,1,4)
            plt.plot(p_r)
            plt.subplot(7,1,5)
            plt.plot(z_p)
            plt.subplot(7,1,6)
            plt.plot(r_h)

            plt.subplot(7,1,7)
            plt.imshow(s, origin='lower', aspect='auto', interpolation='nearest')

            plt.show()

        frame_features.append(np.stack([w_h, p_w, p_r, z_p, r_h], axis=1))
        if stft:
            frame_features.append(s.T)

    frame_features = np.concatenate(frame_features, axis=1)
    return frame_features.astype(np.float32)

def remove_drift(signal, fs):
    b, a = scipy.signal.butter(3, 2, 'highpass', fs=fs)
    return scipy.signal.filtfilt(b, a, signal)

def notch(signal, freq, sample_frequency):
    b, a = scipy.signal.iirnotch(freq, 30, sample_frequency)
    return scipy.signal.filtfilt(b, a, signal)

def notch_harmonics(signal, freq, sample_frequency):
    for harmonic in range(1,5): # (1,8)
        signal = notch(signal, freq*harmonic, sample_frequency)
    return signal
    
def get_usable_datasets(fname):
    mat = scipy.io.loadmat(fname)
    usable = [x for x in mat['use'][0]] # file listing
    usable = [x[0] for x in usable] # fnames only
    usable = [x.split(".")[0] for x in usable] # dataset ids only
    return [x for x in usable]

def load_eeg(fname):
    raw_eeg = scipy.io.loadmat(fname)
    raw_eeg = raw_eeg["raw"][0][0][3][0][0]
    return raw_eeg

def get_audio_feats(audio,
                    n_mel_channels=128,
                    filter_length=512,
                    win_length=432,
                    hop_length=160,
                    r=16_000):
    audio_features = librosa.feature.melspectrogram(
        audio,
        sr=r,
        n_mels=n_mel_channels,
        center=False,
        n_fft=filter_length,
        win_length=win_length,
        hop_length=hop_length).T
    audio_features = np.log(audio_features + 1e-5)
    return audio_features.astype(np.float32)

def load_audio(fname):
    audio, r = sf.read(fname)
    if r != 16_000:
        audio = librosa.resample(audio, orig_sr = r, target_sr = 16_000)
        r = 16_000
    assert r == 16_000

    return audio


class BrennanDataset(torch.utils.data.Dataset):
    num_features = 60 * 5
    
    def __init__(self, root_dir, idx, audio_idx=1):
        self.root_dir = root_dir
        self.idx  = idx

        metadata_fi = \
            os.path.join(root_dir, "AliceChapterOne-EEG.csv")
        self.metadata = metadata = pd.read_csv(metadata_fi)
        proc = scipy.io.loadmat(
            os.path.join(root_dir, f"proc/{idx}.mat"))
        self.eeg_segments = eeg_segments = proc["proc"][0][0][4]
        self.order_idx_s = order_idx_s = \
            [seg[-1] for seg in eeg_segments]
        segment_labels = metadata[metadata["Order"].isin(order_idx_s)]
        self.labels = segment_labels["Word"]

        eeg_path = os.path.join(root_dir, f"{idx}.mat")
        eeg_data = load_eeg(eeg_path)
        self.eeg_data = eeg_data

        self.audio_raw = load_audio(
            os.path.join(root_dir, \
                f"audio/DownTheRabbitHoleFinal_SoundFile{audio_idx}.wav"))

    def __getitem__(self, i):
        # Metadata Segment
        order_idx = self.order_idx_s[i] # EEG
        label = self.labels[i] # Text Label
        metadata_entry = self.metadata[self.metadata["Order"] == order_idx]

        # Audio Segment
        audio_onset = metadata_entry.iloc[0]["onset"]
        audio_start = max(audio_onset - 0.3, 0)
        audio_end   = min(audio_onset + 1.0, self.audio_raw.shape[0])
        audio_start_idx = int(audio_start * 16_000)
        audio_end_idx   = int(audio_end * 16_000)
        audio_raw   = self.audio_raw[audio_start_idx:audio_end_idx]
        audio_feats = get_audio_feats(audio_raw)

        # EEG Segment
        powerline_freq  = 60 # Assumption based on US recordings
        cur_eeg_segment = [seg for seg in self.eeg_segments
                           if seg[-1] == order_idx][0]
        eeg_start_idx   = int(cur_eeg_segment[0])
        eeg_end_idx     = int(cur_eeg_segment[1])
        print("eeg idxs:", eeg_start_idx, eeg_end_idx)
        # eeg_raw         = self.eeg_data[:, eeg_start_idx:eeg_end_idx]
        eeg_raw         = self.eeg_data[:, eeg_start_idx:eeg_end_idx]
        eeg_raw         = notch_harmonics(eeg_raw, powerline_freq, 500)
        eeg_raw         = remove_drift(eeg_raw, 500)
        print("eeg_raw.shape:", eeg_raw.shape)
        eeg_feats       = get_semg_feats_orig(eeg_raw)
        print("eeg_feats.shape:", eeg_feats.shape)

        # Dict Segment
        data = {
            "label":       label,
            "audio_feats": audio_feats,
            "audio_raw":   audio_raw,
            "eeg_raw":     eeg_raw,
            "eeg_feats":   eeg_feats
        }

        return data

    def __len__(self):
        return len(self.order_idx_s)