"""
Convert data in audio CSV to textgrid word alignments for
Montreal Forced Aligner (MFA).
"""

import pandas as pd

textgrid_dir = "./dataset/audio_phoneme_test/"
csv_path = "./dataset/AliceChapterOne-EEG.csv"
audio_dir = "./dataset/audio/"
phoneme_dir = "./phonemes/"
textgrid_fname = lambda i: f"DownTheRabbitHoleFinal_SoundFile{i}.TextGrid"
metadata = pd.read_csv(csv_path)
segs = list(set(metadata["Segment"]))

file_entry = lambda xmin, xmax, items: f"""File type = "ooTextFile"
Object class = "TextGrid"

xmin = {xmin:.3f}
xmax = {xmax:.3f}
tiers? <exists>
size=1
item []:
{items}"""

item_entry = lambda xmin, xmax, intervals, intervals_len: f"""\titem [1]:
\t\tclass = "IntervalTier"
\t\tname = "words"
\t\txmin = {xmin:.3f}
\t\txmax = {xmax:.3f}
\t\tintervals: size = {intervals_len}
{intervals}"""

interval_entry = lambda interval, xmin, xmax, text: f"""\t\t\tintervals [{interval}]:
\t\t\t\txmin = {xmin:.3f}
\t\t\t\txmax = {xmax:.3f}
\t\t\t\ttext = "{text}\""""

for seg in segs:
    out_fname = audio_dir + textgrid_fname(seg)
    current_tokens = metadata[metadata["Segment"] == seg]
    texts          = current_tokens["Word"]
    xmins          = current_tokens["onset"]
    xmaxs          = current_tokens["onset"].shift(-1)
    xmaxs.iloc[-1] = current_tokens["offset"].iloc[-1]
    intervals = \
        [interval_entry(i+2, xmin, xmax, text)
         for i, (xmin, xmax, text) in enumerate(zip(xmins, xmaxs, texts))]
    intervals = [interval_entry(1, 0, xmins.iloc[0], "")] + intervals
    intervals_txt = "\n".join(intervals)
    items = item_entry(
        xmin=0,
        xmax=xmaxs.iloc[-1],
        intervals=intervals_txt,
        intervals_len=len(intervals))
    file_content = file_entry(0, xmaxs.iloc[-1], items)
    with open(out_fname, "w") as f:
        f.write(file_content)