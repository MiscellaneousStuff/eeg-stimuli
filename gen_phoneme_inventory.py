import os
from textgrids import TextGrid

flatten = lambda lst: [item for sublist in lst for item in sublist]

textgrid_files = os.listdir("./phonemes/")
phones_s = [TextGrid(os.path.join("./phonemes/", fi))
            for fi in textgrid_files]
phones_s = [phone["phones"] for phone in phones_s]
phones = flatten(phones_s)
phones = [phone.text for phone in phones]
phoneme_dict = set(phones)

with open("phoneme_dict.txt", "w") as f:
    lst = list(phoneme_dict)
    txt = "\n".join([f"{i}: {lst[i]}" for i in range(len(lst))])
    f.write(txt)