__all__ = ['numeric_binarizer', 'numeric_bucketized']

def numeric_binarizer(feature, feature_scale=None):
    scale = feature_scale if feature_scale is not None else feature.mean()
    t = feature.clip(upper=scale).replace({scale:scale+0.1}).clip(scale).replace({scale:0, scale+0.1:1}).astype('int')
    return t, scale

def numeric_bucketized(feature, boundaries):
    t = feature.copy()
    t[feature<l[0]] = 0
    for r, i in enumerate(boundaries):
        t[feature>=i] = r+1
    return t.astype('int')
