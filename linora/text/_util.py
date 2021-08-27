from itertools import chain
from collections import Counter

import numpy as np

__all__ = ['select_best_length', 'word_to_index', 'sequence_word_index', 'index_vector_matrix',
           'sequence_index_word']

def select_best_length(sequence, sample_rate=0.8):
    """Select best length for sequence with keep rate.
    
    Args:
        sequence: pd.Series or np.array or List of lists, sample sequence.
        sample_rate: float, keep sample rate.
    Returns:
        int length, sample max keep length.
    """
    t = sorted(map(lambda x:len(x), sequence))
    return t[int(np.ceil(len(sequence)*sample_rate))-1]

def word_to_index(sequence):
    """Sequence word transfer to index.
    
    Args:
        sequence: pd.Series or np.array or List of lists, sample sequence.
    Returns:
        dict, {word: index}, The higher the frequency, the higher the ranking.
    """
    t = Counter(chain.from_iterable(sequence))
    t = sorted(t, key=t.get, reverse=True)
    return {'positive':{v: k + 1 for k, v in enumerate(t)}, 'negative':{k + 1:v for k, v in enumerate(t)}}

def sequence_word_index(sequence, word_index_dict, pad_value=0):
    """Sequence word transfer to sequence index.
    
    Args:
        sequence: pd.Series or np.array or List of lists, sample sequence.
        word_index_dict: dict, {word: index}.
        pad_value: fillna value, if word not in word_index_dict, fillna it.
    Returns:
        List of lists, sequence word transfer to sequence index list.
    """
    if isinstance(sequence[0], str):
        return [word_index_dict.get(j, pad_value) for j in sequence]
    return [[word_index_dict.get(j, pad_value) for j in i] for i in sequence]

def sequence_index_word(sequence, index_word_dict, pad_value=' ', join=False):
    """Sequence index transfer to sequence word.
    
    Args:
        sequence: pd.Series or np.array or List of lists, sample sequence.
        index_word_dict: dict, {index: word}.
        pad_value: fillna value, if word not in index_word_dict, fillna it.
        join: whether to merge the converted words.
    Returns:
        List of lists, sequence index transfer to sequence word list.
    """
    if isinstance(sequence[0], int):
        if join:
            return ''.join([index_word_dict.get(j, pad_value) for j in sequence])
        return [index_word_dict.get(j, pad_value) for j in sequence]
    if join:
        return [''.join([index_word_dict.get(j, pad_value) for j in i]) for i in sequence]
    return [[index_word_dict.get(j, pad_value) for j in i] for i in sequence]

def index_vector_matrix(word_index_dict, word_vector_dict, embed_dim=300, initialize='norm', dtype='float32'):
    """Make index vector matrix with shape `(len(word_index_dict), embed_dim)`.
    
    Args:
        word_index_dict: dict, {word: index}.
        word_vector_dict: dict, {word: vector}.
        embed_dim: embedding dim.
        initialize: initialize method, 'zero' or 'one' or 'norm'.
        dtype: default 'float32', data types.
    Returns:
        a index vector matrix with shape `(len(word_index_dict), embed_dim)`.
    """
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
