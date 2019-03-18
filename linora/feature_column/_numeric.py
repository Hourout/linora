from scipy.stats import norm


__all__ = ['numeric_binarizer', 'numeric_bucketized', 'numeric_padding', 'numeric_outlier']

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

def numeric_padding(feature, method='mean', feature_scale=None):
    assert method in ['mean', 'median'], "`method` should be one of ['mean', 'median']."
    scale_dict = {'mean':feature.mean(), 'median':feature.median()}
    scale = feature_scale if feature_scale is not None else scale_dict[method]
    t = feature.fillna(scale)
    return t, scale

def numeric_outlier(feature, keep_rate=0.9545, mode='right', feature_scale=None):
    assert mode in ['left', 'right', 'both'], "`mode` should be one of ['left', 'right', 'both']."
    scale = feature_scale if feature_scale is not None else (feature.mean(), feature.std())
    if mode=='both':
        clip_dict = (feature.mean()+norm.ppf((1-keep_rate)/2)*feature.std(), feature.mean()+norm.ppf(keep_rate+(1-keep_rate)/2)*feature.std())
    elif mode=='right':
        clip_dict = (feature.min(), feature.mean()+norm.ppf(keep_rate)*feature.std())
    else:
        clip_dict = (feature.mean()+norm.ppf(1-keep_rate)*feature.std(), feature.max())
    t = feature.clip(clip_dict[0], clip_dict[1])
    return t, scale
