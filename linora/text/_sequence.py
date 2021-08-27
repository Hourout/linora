import random

import numpy as np

__all__ = ['skipgrams', 'sequence_pad', 'make_sampling_table']

def skipgrams(sequence, vocabulary_size,
              window_size=4, negative_samples=1., shuffle=True,
              categorical=False, sampling_table=None, seed=None):
    """Generates skipgram word pairs.
    
    This function transforms a sequence of word indexes (list of integers)
    into tuples of words of the form:
    - (word, word in the same window), with label 1 (positive samples).
    - (word, random word from the vocabulary), with label 0 (negative samples).
    Read more about Skipgram in this gnomic paper by Mikolov et al.:
    [Efficient Estimation of Word Representations in
    Vector Space](http://arxiv.org/pdf/1301.3781v3.pdf)
    # Arguments
        sequence: A word sequence (sentence), encoded as a list
            of word indices (integers). If using a `sampling_table`,
            word indices are expected to match the rank
            of the words in a reference dataset (e.g. 10 would encode
            the 10-th most frequently occurring token).
            Note that index 0 is expected to be a non-word and will be skipped.
        vocabulary_size: Int, maximum possible word index + 1
        window_size: Int, size of sampling windows (technically half-window).
            The window of a word `w_i` will be
            `[i - window_size, i + window_size+1]`.
        negative_samples: Float >= 0. 0 for no negative (i.e. random) samples.
            1 for same number as positive samples.
        shuffle: Whether to shuffle the word couples before returning them.
        categorical: bool. if False, labels will be
            integers (eg. `[0, 1, 1 .. ]`),
            if `True`, labels will be categorical, e.g.
            `[[1,0],[0,1],[0,1] .. ]`.
        sampling_table: 1D array of size `vocabulary_size` where the entry i
            encodes the probability to sample a word of rank i.
        seed: Random seed.
    # Returns
        couples, labels: where `couples` are int pairs and
            `labels` are either 0 or 1.
    # Note
        By convention, index 0 in the vocabulary is
        a non-word and will be skipped.
    """
    if sampling_table is None:
        sampling_table = [1]*vocabulary_size
    if not isinstance(sequence[0], list):
        sequence = [sequence]
    couples = []
    labels = []
    for seq in sequence:
        t = [[i, wi, max(0, i-window_size), min(len(seq), i+window_size+1)] 
                   for i, wi in enumerate(seq) if wi and sampling_table[wi]>random.random()]
        couples += [[wi, seq[j]] for (i, wi, m, n) in t for j in range(m, n) if j!=i and seq[j]]
        labels += ([[0, 1]] if categorical else [1])*len(couples) 
    
    if negative_samples > 0:
        num = int(len(labels) * negative_samples)
        words = [c[0] for c in couples]
        random.shuffle(words)
        couples += [[words[i%len(words)], random.randint(1, vocabulary_size-1)] for i in range(num)]
        labels += ([[1, 0]] if categorical else [0])*num

    if shuffle:
        if seed is None:
            seed = random.randint(0, 10e6)
        random.seed(seed)
        random.shuffle(couples)
        random.seed(seed)
        random.shuffle(labels)
    return couples, labels

def sequence_pad(sequence, maxlen, dtype='int32', padding='pre', truncating='pre', value=0):
    """Pads sequences to the same length.
    
    Args:
        sequences: pd.Series or np.array or List of lists, where each element is a sequence.
        maxlen: Int, maximum length of all sequences.
        dtype: Type of the output sequences.
               To pad sequences with variable length strings, you can use `object`.
        padding: String, 'pre' or 'post':
                 pad either before or after each sequence.
        truncating: String, 'pre' or 'post':
                    remove values from sequences larger than
                    `maxlen`, either at the beginning or at the end of the sequences.
        value: Float or String, padding value.
    Returns:
        List of lists with shape `(len(sequences), maxlen)`
    """
    if not isinstance(sequence[0], list):
        sequence = [sequence]
    if padding=='post' and truncating=='post':
        t = [i[:maxlen] if len(i)>maxlen else i+[value]*(maxlen-len(i)) for i in sequence]
    elif padding=='post' and truncating=='pre':
        t = [i[-maxlen:] if len(i)>maxlen else i+[value]*(maxlen-len(i)) for i in sequence]
    elif padding=='pre' and truncating=='post':
        t = [i[:maxlen] if len(i)>maxlen else [value]*(maxlen-len(i))+i for i in sequence]
    elif padding=='pre' and truncating=='pre':
        t = [i[-maxlen:] if len(i)>maxlen else [value]*(maxlen-len(i))+i for i in sequence]
    else:
        raise ValueError('Padding type "%s" not understood or Truncating type "%s" not understood' % (padding, truncating))
    return t

def make_sampling_table(size, sampling_factor=1e-5):
    """Generates a word rank-based probabilistic sampling table.
    Used for generating the `sampling_table` argument for `skipgrams`.
    `sampling_table[i]` is the probability of sampling
    the word i-th most common word in a dataset
    (more common words should be sampled less frequently, for balance).
    The sampling probabilities are generated according
    to the sampling distribution used in word2vec:
    ```
    p(word) = (min(1, sqrt(word_frequency / sampling_factor) /
        (word_frequency / sampling_factor)))
    ```
    We assume that the word frequencies follow Zipf's law (s=1) to derive
    a numerical approximation of frequency(rank):
    `frequency(rank) ~ 1/(rank * (log(rank) + gamma) + 1/2 - 1/(12*rank))`
    where `gamma` is the Euler-Mascheroni constant.
    # Arguments
        size: Int, number of possible words to sample.
        sampling_factor: The sampling factor in the word2vec formula.
    # Returns
        A 1D Numpy array of length `size` where the ith entry
        is the probability that a word of rank i should be sampled.
    """
    gamma = 0.577
    rank = np.arange(size)
    rank[0] = 1
    inv_fq = rank * (np.log(rank) + gamma) + 0.5 - 1. / (12. * rank)
    f = sampling_factor * inv_fq
    return np.minimum(1., f / np.sqrt(f))
