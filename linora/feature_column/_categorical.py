from functools import reduce

__all__ = ['categorical_encoder', 'categorical_hash', 'categorical_crossed']


def categorical_encoder(feature, feature_scale=None):
    scale = feature_scale if feature_scale is not None else {j:i for i,j in feature.drop_duplicates().reset_index(drop=True).to_dict().items()}
    t = feature.replace(scale)
    return t, scale

def categorical_hash(feature, hash_bucket_size):
    return feature.fillna('').astype(str).map(lambda x:hash(x))%hash_bucket_size

def categorical_crossed(feature_list, hash_bucket_size):
    return reduce(lambda x,y:x+y, [i.fillna('').astype(str) for i in feature_list]).map(lambda x:hash(x))%hash_bucket_size
