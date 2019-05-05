import itertools
import numpy as np

__all__ = ['select_best_length', 'word_to_index', 'word_index_sequence', 'pad_sequences',
           'index_vector_matrix']

def select_best_length(sequence, sample_rate=0.8):
    t = sorted(map(lambda x:len(x), sequence))
    return t[int(np.ceil(len(sequence)*sample_rate))]

def word_to_index(sequence):
    return {v: k + 1 for k, v in enumerate(set(itertools.chain.from_iterable(sequence)))}

def word_index_sequence(sequence, word_index_dict, pad_value=0):
    t = []
    for i in sequence:
        index = []
        for j in i:
            try:
                index.append(word_index_dict[j])
            except:
                index.append(pad_value)
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

def index_vector_matrix(word_index_dict, word_vector_dict, embed_dim=300, initialize='zero', dtype='float32'):
    word_count = len(word_index_dict)+1
    init = {'zero':lambda x,y:np.zeros((x,y), dtype=dtype),
            'one':lambda x,y:np.ones((x,y), dtype=dtype),
            'norm':lambda x,y:np.random.normal(size=(x,y), dtype=dtype)}
    index_embed_matrix = init[initialize](word_count, embed_dim)
    for word, index in word_index_dict.items():
        l = len(word_vector_dict[word])
        if l>embed_dim:
            index_embed_matrix[index, :] = word_vector_dict[word][:embed_dim]
        else:
            index_embed_matrix[index, :l] = word_vector_dict[word]
    return index_embed_matrix
