# API document 文档

- ## la.chart

| API | description | 描述 |
| --- | --- | --- |
| la.chart.ks_curve | KS curve | KS曲线图 |
| la.chart.roc_curve | ROC curve | ROC曲线图 |
| la.chart.pr_curve | PR curve | PR曲线图 |
| la.chart.gain_curve | Gain curve | Gain曲线图 |
| la.chart.gini_curve | Gini curve | Gini曲线图 |
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
| la.metrics.distance.euclidean |  | 欧氏距离 |
| la.metrics.distance.manhattan |  | 曼哈顿距离 |
| la.metrics.distance.chebyshev |  | 切比雪夫距离 |
| la.metrics.distance.minkowski |  | 闵可夫斯基距离 |
| la.metrics.distance.hamming |  | 汉明距离 |
| la.metrics.distance.jaccard |  | 雅可比距离 |
| la.metrics.distance.pearson |  | 皮尔逊距离 |
| la.metrics.distance.cosine |  | 余玄距离 |
| la.metrics.distance.levenshtein |  | 莱文斯坦距离 |
| la.metrics.distance.kl_divergence |  | kl散度 |
| la.metrics.distance.js_divergence |  | js散度 |
| la.metrics.distance.mutual_information_rate |  |  |
| la.metrics.distance.pointwise_mutual_information_rate |  |  |
| la.metrics.binary_accuracy |  | 二分类精准率 |
| la.metrics.categorical_accuracy |  | 多分类精准率 |
| la.metrics.recall |  | 召回率 |
| la.metrics.precision |  | 准确率 |
| la.metrics.confusion_matrix |  | 混淆矩阵 |
| la.metrics.fbeta_score |  | f分数 |
| la.metrics.f1_score |  | f1分数 |
| la.metrics.auc_roc |  | ROC曲线下面积 |
| la.metrics.auc_pr |  | PR曲线下面积 |
| la.metrics.binary_crossentropy |  | 二分类交叉熵 |
| la.metrics.categorical_crossentropy |  | 多分类交叉熵 |
| la.metrics.ks |  | KS值 |
| la.metrics.gini |  | Gini值 |
| la.metrics.psi |  | Psi值 |
| la.metrics.fmi |  | FMI值 |
| la.metrics.binary_report |  | 二分类报告 |
| la.metrics.mapk |  | topK平均准确率 |
| la.metrics.hit_ratio |  | hit比率 |
| la.metrics.mean_reciprocal_rank |  |  |
| la.metrics.normal_loss |  |  |
| la.metrics.mean_absolute_error |  | 平均绝对误差 |
| la.metrics.mean_squared_error |  | 均方误差 |
| la.metrics.mean_absolute_percentage_error |  | 平均绝对百分比误差 |
| la.metrics.hinge |  | hinge距离 |
| la.metrics.explained_variance_score |  |  |
| la.metrics.median_absolute_error |  | 中位数绝对误差 |
| la.metrics.r2_score |  | R方 |

- ## la.param_search

| API | description | 描述 |
| --- | --- | --- |
| la.param_search.XGBRegressor.RandomSearch |  | XGBRegressor随机超参数搜索 |
| la.param_search.XGBRegressor.GridSearch |  | XGBRegressor网格超参数搜索 |
| la.param_search.XGBClassifier.RandomSearch |  | XGBClassifier随机超参数搜索 |
| la.param_search.XGBClassifier.GridSearch |  | XGBClassifier网格超参数搜索 |
| la.param_search.XGBRanker.RandomSearch |  | XGBRanker随机超参数搜索 |
| la.param_search.XGBRanker.GridSearch |  | XGBRanker网格超参数搜索 |
| la.param_search.GERegressor.RandomSearch |  | 通用算法随机超参数搜索 |
| la.param_search.GEClassifier.RandomSearch |  | 通用算法网格超参数搜索 |

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
