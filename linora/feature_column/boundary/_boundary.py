__all__ = ['equal_width', 'equal_frequency']

def equal_width(feature, bins):
    t = (feature.max()-feature.min())/bins
    m = feature.min()
    return [t*i+m for i in range(bins)]+[feature.max()]

def equal_frequency(feature, bins):
    t = feature.sort_values().values
    m = feature.min()
    w = round(len(t)/bins)
    return [t[w*i] for i in range(bins)]+[feature.max()]
