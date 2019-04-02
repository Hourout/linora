import itertools
import collections

__all__ = ['WordCount', 'LowFreqWord', 'WordFilter']

def WordCount(sequence):
    return collections.Counter(itertools.chain.from_iterable(sequence))

def LowFreqWord(word_count_dict, threshold=3):
    return [i for i,j in word_count_dict.items() if j<=threshold]
    
def WordFilter(sequence, filter_word_list):
    t = []
    for i in sequence:
        t.extend([[x for x in i if x not in filter_word_list]])
    return t
