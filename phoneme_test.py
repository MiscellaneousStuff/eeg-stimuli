"""
Test loading of phonemes along with the alignment.
"""

from lib import *

dataset = BrennanDataset(
    root_dir="./dataset/",
    phoneme_dir="./phonemes/",
    idx="S13")