# API document 文档

- ## la.chart

| API | description | 描述 |
| --- | --- | --- |
| la.chart.ks_curve | KS curve | KS曲线 |
| la.chart.roc_curve | ROC curve | ROC曲线 |
| la.chart.pr_curve | PR curve | PR曲线 |
| la.chart.gain_curve | Gain curve | Gain曲线 |
| la.chart.gini_curve | Gini curve | Gini曲线 |
| la.chart.confusion_matrix_map | Confusion matrix map | 混淆矩阵图 |

- ## la.feature_column

| API | description | 描述 |
| --- | --- | --- |
| la.feature_column.boundary.uniform |  | 等宽分箱 |
| la.feature_column.boundary.quantile |  | 等频分箱 |
| la.feature_column.categorical_encoder |  | 类别特征编码 |
| la.feature_column.categorical_hash |  | 类别特征哈希 |
| la.feature_column.categorical_crossed |  | 类别特征交叉 |
| la.feature_column.categorical_onehot_binarizer |  | 单类别特征onehot |
| la.feature_column.categorical_onehot_multiple |  | 多类别特征onehot |
| la.feature_column.normalize_meanminmax |  | 平均最小最大归一化 |
| la.feature_column.normalize_minmax |  | 最小最大归一化 |
| la.feature_column.normalize_maxabs |  | 绝对值最大归一化 |
| la.feature_column.normalize_max |  | 最大归一化 |
| la.feature_column.normalize_l1 |  | l1正则归一化 |
| la.feature_column.normalize_l2 |  | l2正则归一化 |
| la.feature_column.normalize_norm |  | 正态归一化 |
| la.feature_column.normalize_robust |  | robust归一化 |
| la.feature_column.numeric_binarizer |  | 数值特征二分类 |
| la.feature_column.numeric_bucketized |  | 数值特征分桶 |
| la.feature_column.numeric_padding |  | 数值特征缺失值填补 |
| la.feature_column.numeric_outlier | feature outlier clip. | 数值特征异常值替换 |

- ## la.feature_selection

| API | description | 描述 |
| --- | --- | --- |
| la.feature_selection.woe |  | 特征woe值 |
| la.feature_selection.iv |  | 特征iv值 |
| la.feature_selection.missing_columns |  | 缺失值选择列 |
| la.feature_selection.single_columns |  | 单一值选择列 |
| la.feature_selection.correlation_columns |  | 相关系数选择列 |
| la.feature_selection.cv_columns |  | 变异系数选择列 |

- ## la.metrics

| API | description | 描述 |
| --- | --- | --- |
| la.metrics.distance.euclidean |  |  |
| la.metrics.distance.manhattan |  |  |
| la.metrics.distance.chebyshev |  |  |
| la.metrics.distance.minkowski |  |  |
| la.metrics.distance.hamming |  |  |
| la.metrics.distance.jaccard |  |  |
| la.metrics.distance.pearson |  |  |
| la.metrics.distance.cosine |  |  |
| la.metrics.distance.levenshtein |  |  |
| la.metrics.distance.kl_divergence |  |  |
| la.metrics.distance.js_divergence |  |  |
| la.metrics.distance.mutual_information_rate |  |  |
| la.metrics.distance.pointwise_mutual_information_rate |  |  |
| la.metrics.binary_accuracy |  |  |
| la.metrics.categorical_accuracy |  |  |
| la.metrics.recall |  |  |
| la.metrics.precision |  |  |
| la.metrics.confusion_matrix |  |  |
| la.metrics.fbeta_score |  |  |
| la.metrics.f1_score |  |  |
| la.metrics.auc_roc |  |  |
| la.metrics.auc_pr |  |  |
| la.metrics.binary_crossentropy |  |  |
| la.metrics.categorical_crossentropy |  |  |
| la.metrics.ks |  |  |
| la.metrics.gini |  |  |
| la.metrics.psi |  |  |
| la.metrics.fmi |  |  |
| la.metrics.binary_report |  |  |
| la.metrics.mapk |  |  |
| la.metrics.hit_ratio |  |  |
| la.metrics.mean_reciprocal_rank |  |  |
| la.metrics.normal_loss |  |  |
| la.metrics.mean_absolute_error |  |  |
| la.metrics.mean_squared_error |  |  |
| la.metrics.mean_absolute_percentage_error |  |  |
| la.metrics.hinge |  |  |
| la.metrics.explained_variance_score |  |  |
| la.metrics.median_absolute_error |  |  |
| la.metrics.r2_score |  |  |

- ## la.param_search

| API | description | 描述 |
| --- | --- | --- |
| la.param_search.XGBRegressor.RandomSearch |  |  |
| la.param_search.XGBRegressor.GridSearch |  |  |
| la.param_search.XGBClassifier.RandomSearch |  |  |
| la.param_search.XGBClassifier.GridSearch |  |  |
| la.param_search.XGBRanker.RandomSearch |  |  |
| la.param_search.XGBRanker.GridSearch |  |  |
| la.param_search.GERegressor.RandomSearch |  |  |
| la.param_search.GEClassifier.RandomSearch |  |  |

- ## la.sample

| API | description | 描述 |
| --- | --- | --- |
| la.sample.ImageDataset |  |  |
| la.sample.ImageClassificationFolderDataset |  |  |

- ## la.sample_splits

| API | description | 描述 |
| --- | --- | --- |
| la.sample_splits.kfold |  |  |
| la.sample_splits.train_test_split |  |  |
| la.sample_splits.timeseries_train_test_split |  |  |
| la.sample_splits.timeseries_walk_forward_fold |  |  |
| la.sample_splits.timeseries_kfold |  |  |

- ## la.image

| API | description | 描述 |
| --- | --- | --- |
| la.image.read_image |  |  |
| la.image.save_image |  |  |
| la.image.ImageAug |  |  |
| la.image.RandomBrightness |  |  |
| la.image.RandomContrast |  |  |
| la.image.RandomHue |  |  |
| la.image.RandomSaturation |  |  |
| la.image.RandomGamma |  |  |
| la.image.RandomNoiseGaussian |  |  |
| la.image.RandomNoisePoisson |  |  |
| la.image.RandomNoiseMask |  |  |
| la.image.RandomNoiseSaltPepper |  |  |
| la.image.RandomNoiseRainbow |  |  |
| la.image.RandomFlipLeftRight |  |  |
| la.image.RandomFlipTopBottom |  |  |
| la.image.RandomTranspose |  |  |
| la.image.RandomRotation |  |  |
| la.image.Normalize |  |  |
| la.image.RandomRescale |  |  |
| la.image.RandomCropCentralResize |  |  |
| la.image.RandomCropPointResize |  |  |

- ## la.text

| API | description | 描述 |
| --- | --- | --- |
| la.text.CountVectorizer |  |  |
| la.text.TfidfVectorizer |  |  |
| la.text.select_best_length |  |  |
| la.text.word_to_index |  |  |
| la.text.word_index_sequence |  |  |
| la.text.pad_sequences |  |  |
| la.text.index_vector_matrix |  |  |
| la.text.sequence_preprocess |  |  |
| la.text.word_count |  |  |
| la.text.word_low_freq |  |  |
| la.text.word_high_freq |  |  |
| la.text.filter_word |  |  |
| la.text.filter_punctuation |  |  |
