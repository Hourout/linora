import itertools
import collections

__all__ = ['sequence_preprocess', 'word_count', 'word_low_freq', 'word_high_freq', 'filter_word',
           'filter_punctuation']

def sequence_preprocess(sequence):
    t = []
    for s in sequence:
        t.extend([''.join([i for i in s.replace(' ', '') if i>= u'\u4e00' and i<= u'\u9fa5'])])
    return t

def word_count(sequence):
    return collections.Counter(itertools.chain.from_iterable(sequence))

def word_low_freq(word_count_dict, threshold=3):
    return [i for i,j in word_count_dict.items() if j<=threshold]

def word_high_freq(word_count_dict, threshold):
    return [i for i,j in word_count_dict.items() if j>=threshold]

def filter_word(sequence, filter_word_list):
    t = []
    for i in sequence:
        t.extend([[x for x in i if x not in filter_word_list]])
    return t

def filter_punctuation(sequence, punctuation=None):
    punc = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~（），。’‘”“！？》《·、】【；：' if punctuation is None else punctuation
    table = str.maketrans('', '', punc)
    t = []
    for s in sequence:
        t.extend([s.translate(table)])
    return t
