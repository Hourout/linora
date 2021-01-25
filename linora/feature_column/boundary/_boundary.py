__all__ = ['uniform', 'quantile']

def uniform(feature, bins):
    """Equal width bin, take a uniform distribution for the sample value range.
    
    Args:
        feature: pd.Series, model feature values.
        bins: int, split bins of feature.
    Returns:
        the list of split threshold of feature.
    """
    t = (feature.max()-feature.min())/bins
    m = feature.min()
    return [t*i+m for i in range(bins)]+[feature.max()]

def quantile(feature, bins):
    """Equal frequency bin, take a uniform distribution of the sample size.
    
    Args:
        feature: pd.Series, model feature values.
        bins: int, split bins of feature.
    Returns:
        the list of split threshold of feature.
    """
    t = feature.sort_values().values
    w = round(len(t)/bins)
    return [t[w*i] for i in range(bins)]+[feature.max()]
