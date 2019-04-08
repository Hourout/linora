import numpy as np

__all__ = ['select_best_length']

def select_best_length(sequence, sample_rate=0.8):
    t = sorted(map(lambda x:len(x), sequence))
    return t[int(np.ceil(len(sequence)*sample_rate))]
