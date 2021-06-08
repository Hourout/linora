import random

__all__ = ['skipgrams']

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
