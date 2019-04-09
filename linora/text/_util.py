import itertools
import numpy as np

__all__ = ['select_best_length', 'word_to_index', 'word_index_sequence', 'pad_sequences']

def select_best_length(sequence, sample_rate=0.8):
    t = sorted(map(lambda x:len(x), sequence))
    return t[int(np.ceil(len(sequence)*sample_rate))]

def word_to_index(sequence):
    return {v: k + 1 for k, v in enumerate(set(itertools.chain.from_iterable(sequence)))}

def word_index_sequence(sequence, word_index_dict):
    t = []
    for i in sequence:
        index = []
        for j in i:
            try:
                index.append(word_index_dict[j])
            except:
                index.append(0)
        t.extend([index])
    return t

def pad_sequences(sequence, maxlen=None, dtype='int32', padding='pre', truncating='pre', value=0):
    t = []
    if padding=='post' and truncating=='post':
        for i in sequence:
            t.extend([i[:maxlen] if len(i)>maxlen else i+[value]*(maxlen-len(i))])
    elif padding=='post' and truncating=='pre':
        for i in sequence:
            t.extend([i[-maxlen:] if len(i)>maxlen else i+[value]*(maxlen-len(i))])
    elif padding=='pre' and truncating=='post':
        for i in sequence:
            t.extend([i[:maxlen] if len(i)>maxlen else [value]*(maxlen-len(i))+i])
    elif padding=='pre' and truncating=='pre':
        for i in sequence:
            t.extend([i[-maxlen:] if len(i)>maxlen else [value]*(maxlen-len(i))+i])
    else:
        raise ValueError('Padding type "%s" not understood or Truncating type "%s" not understood' % (padding, truncating))
    return t
