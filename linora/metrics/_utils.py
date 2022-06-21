import numpy as np

def _sample_weight(y_true, sample_weight):
    if sample_weight is None:
        sample_weight = np.ones(len(y_true))
    elif isinstance(sample_weight, dict):
        sample_weight = np.array([sample_weight[i] for i in y_true])
    else:
        sample_weight = np.array(sample_weight)
    return sample_weight/sample_weight.sum()*len(sample_weight)