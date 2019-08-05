__all__ = ['uniform', 'quantile']

def uniform(feature, bins):
    """Equal width bin
    
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
    """Equal frequency bin
    
    Args:
        feature: pd.Series, model feature values.
        bins: int, split bins of feature.
    Returns:
        the list of split threshold of feature.
    """
    t = feature.sort_values().values
    w = round(len(t)/bins)
    return [t[w*i] for i in range(bins)]+[feature.max()]
