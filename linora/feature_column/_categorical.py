__all__ = ['categorical_binarizer', 'categorical_encoder']

def categorical_binarizer(feature, feature_scale=None):
    scale = feature_scale if feature_scale is not None else feature.mean()
    t = feature.clip(upper=scale).replace({scale:scale+0.1}).clip(scale).replace({scale:0, scale+0.1:1}).astype('int')
    return t, scale

def categorical_encoder(feature, feature_scale=None):
    scale = feature_scale if feature_scale is not None else {j:i for i,j in feature.drop_duplicates().reset_index(drop=True).to_dict().items()}
    t = feature.replace(scale)
    return t, scale
