from itertools import chain
from collections import Counter

__all__ = ['sequence_preprocess', 'word_count', 'word_low_freq', 'word_high_freq', 'filter_word',
           'filter_punctuation', 'ngrams_iterator']


def sequence_preprocess(sequence):
    """Sequence preprocess, keep only Chinese.
    
    Args:
        sequence: pd.Series or np.array or list, sample feature value.
    Returns:
        a only Chinese list.
    """
    if isinstance(sequence, str):
        return ''.join([i for i in sequence.replace(' ', '') if i>= u'\u4e00' and i<= u'\u9fa5'])
    return [''.join([i for i in s.replace(' ', '') if i>= u'\u4e00' and i<= u'\u9fa5']) for s in sequence]


def word_count(sequence):
    """Sequence word count.
    
    Args:
        sequence: pd.Series or np.array or list of Lists, sample feature value.
    Returns:
        a dict with word count.
    """
    return Counter(chain.from_iterable(sequence))


def word_low_freq(word_count_dict, threshold=3):
    """Filter low frequency words.
    
    Args:
        word_count_dict: a dict with word count.
        threshold: filter low frequency words threshold.
    Returns:
        a list filter low frequency words.
    """
    return [i for i,j in word_count_dict.items() if j<=threshold]


def word_high_freq(word_count_dict, threshold):
    """Filter high frequency words.
    
    Args:
        word_count_dict: a dict with word count.
        threshold: filter high frequency words threshold.
    Returns:
        a list filter high frequency words.
    """
    return [i for i,j in word_count_dict.items() if j>=threshold]


def filter_word(sequence, filter_word_list):
    """Sequence filter words with a filter word list.
    
    Args:
        sequence: pd.Series or np.array or list of Lists, sample feature value.
        filter_word_list: a filter word list.
    Returns:
        a list of Lists.
    """
    if isinstance(sequence[0], str):
        return [x for x in sequence if x not in filter_word_list]
    return [[x for x in i if x not in filter_word_list] for i in sequence]


def filter_punctuation(sequence, punctuation=None):
    """Sequence preprocess, filter punctuation.
    
    Args:
        sequence: pd.Series or np.array or list, sample feature value.
        punctuation: str, filter punctuation.
    Returns:
        a list of Lists.
    """
    punc = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~（），。’‘”“！？》《·、】【；：' if punctuation is None else punctuation
    table = str.maketrans('', '', punc)
    if isinstance(sequence, str):
        return sequence.translate(table)
    return [s.translate(table) for s in sequence]


def ngrams_iterator(token_list, ngrams, join=' '):
    """Return an iterator that yields the given tokens and their ngrams.

    Args:
        token_list: A list of tokens
        ngrams: the number of ngrams.
    """
    for x in token_list:
        yield x
    for n in range(2, ngrams + 1):
        for x in zip(*[token_list[i:] for i in range(n)]):
            yield join.join(x)