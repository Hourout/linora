__all__ = ['uniform', 'quantile']

def uniform(feature, bins):
    t = (feature.max()-feature.min())/bins
    m = feature.min()
    return [t*i+m for i in range(bins)]+[feature.max()]

def quantile(feature, bins):
    t = feature.sort_values().values
    w = round(len(t)/bins)
    return [t[w*i] for i in range(bins)]+[feature.max()]
