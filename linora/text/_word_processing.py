import itertools
import collections

__all__ = ['word_count', 'word_low_freq', 'word_filter']

def word_count(sequence):
    return collections.Counter(itertools.chain.from_iterable(sequence))

def word_low_freq(word_count_dict, threshold=3):
    return [i for i,j in word_count_dict.items() if j<=threshold]
    
def word_filter(sequence, filter_word_list):
    return list(filter(lambda x:x not in filter_word_list, sequence))
