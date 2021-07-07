# API document 文档

- ## la.utils

| API | description | 描述 |
| --- | --- | --- |
| la.utils.Config | config function | 参数配置函数 |
| la.utils.Logger | logging function | 日志函数 |
| la.utils.Progbar | progress bar function | 进度条函数 |
| la.utils.Schedulers | schedulers function | 定时任务函数 |

- ## la.chart

| API | description | 描述 |
| --- | --- | --- |
| la.chart.ks_curve | KS curve | KS曲线图 |
| la.chart.roc_curve | ROC curve | ROC曲线图 |
| la.chart.pr_curve | PR curve | PR曲线图 |
| la.chart.lift_curve | Lift curve | Lift曲线图 |
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
| la.feature_column.numeric_outlier | feature outlier clip | 数值特征异常值替换 |

- ## la.feature_selection

| API | description | 描述 |
| --- | --- | --- |
| la.feature_selection.woe | feature woe value | 特征woe值 |
| la.feature_selection.iv | feature iv value | 特征iv值 |
| la.feature_selection.missing_columns |  | 缺失值选择列 |
| la.feature_selection.single_columns |  | 单一值选择列 |
| la.feature_selection.correlation_columns |  | 相关系数选择列 |
| la.feature_selection.cv_columns |  | 变异系数选择列 |

- ## la.metrics

| API | description | 描述 |
| --- | --- | --- |
| la.metrics.distance.euclidean | euclidean distance | 欧氏距离 |
| la.metrics.distance.manhattan | manhattan distance | 曼哈顿距离 |
| la.metrics.distance.chebyshev | chebyshev distance | 切比雪夫距离 |
| la.metrics.distance.minkowski | minkowski distance | 闵可夫斯基距离 |
| la.metrics.distance.hamming | hamming distance | 汉明距离 |
| la.metrics.distance.jaccard | jaccard distance | 雅可比距离 |
| la.metrics.distance.pearson | pearson distance | 皮尔逊距离 |
| la.metrics.distance.cosine | cosine distance | 余弦距离 |
| la.metrics.distance.levenshtein | levenshtein distance | 莱文斯坦距离 |
| la.metrics.distance.kl_divergence | kl divergence | kl散度 |
| la.metrics.distance.js_divergence | js divergence | js散度 |
| la.metrics.distance.mutual_information_rate | mutual information rate. | 互信息率 |
| la.metrics.distance.pointwise_mutual_information_rate | pointwise mutual information rate | 逐点互信息率 |
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
| la.metrics.mean_reciprocal_rank |  | 平均倒数排序误差 |
| la.metrics.normal_loss |  | 平均标准回归误差 |
| la.metrics.mean_absolute_error |  | 平均绝对误差 |
| la.metrics.mean_squared_error |  | 均方误差 |
| la.metrics.mean_absolute_percentage_error |  | 平均绝对百分比误差 |
| la.metrics.hinge | hinge distance | hinge距离 |
| la.metrics.explained_variance_score |  | 解释方差回归损失 |
| la.metrics.median_absolute_error | median absolute error | 中位数绝对误差 |
| la.metrics.r2_score | R square | R方 |

- ## la.parallel

| API | description | 描述 |
| --- | --- | --- |
| la.parallel.ProcessLoom | executes runners using multi-processing. | 多进程并行 |
| la.parallel.ThreadLoom | executes runners using threading. | 多线程并行 |

- ## la.param_search

| API | description | 描述 |
| --- | --- | --- |
| la.param_search.HyperParametersGrid |  | 网格超参数生成 |
| la.param_search.HyperParametersRandom |  | 随机超参数生成 |
| la.param_search.XGBRegressor.RandomSearch | XGBRegressor random hyperparameter search | XGBRegressor随机超参数搜索 |
| la.param_search.XGBRegressor.GridSearch | XGBRegressor grid hyperparameter search | XGBRegressor网格超参数搜索 |
| la.param_search.XGBClassifier.RandomSearch | XGBClassifier random hyperparameter search | XGBClassifier随机超参数搜索 |
| la.param_search.XGBClassifier.GridSearch | XGBClassifier grid hyperparameter search | XGBClassifier网格超参数搜索 |

- ## la.sample

| API | description | 描述 |
| --- | --- | --- |
| la.sample.ImageDataset | Construct an image dataset label index | 图像样本生成 |
| la.sample.ImageClassificationFolderDataset | Construct an image dataset label index | 图像分类样本生成 |
| la.sample.RandomWalker | RandomWalker sample | 随机游走样本生成 |
| la.sample.Dataset | sample Dataset | 样本生成类 |

- ## la.sample_splits

| API | description | 描述 |
| --- | --- | --- |
| la.sample_splits.kfold | K-Folds cross-validator | K折交叉验证 |
| la.sample_splits.train_test_split | Split DataFrame or matrices into random train and test subsets | 划分训练测试集 |
| la.sample_splits.timeseries_train_test_split | Split DataFrame or matrices into random train and test subsets for timeseries | 划分时间序列训练测试集 |
| la.sample_splits.timeseries_walk_forward_fold | Walk Forward Folds cross-validator for timeseries | 向前走折叠时间序列的交叉验证器 |
| la.sample_splits.timeseries_kfold | K-Folds cross-validator for timeseries | 时间序列K折交叉验证 |

- ## la.image

| API | description | 描述 |
| --- | --- | --- |
| la.image.read_image | Convenience function for read image | 读取图像 |
| la.image.save_image | Writes image to the file at input filename | 保存图像 |
| la.image.ImageAug | image augmentation class | 图像增强类 |
| la.image.enhance_brightness | Adjust the brightness of RGB or Grayscale images | 随机调整图像亮度 |
| la.image.enhance_contrast | Adjust contrast of RGB or grayscale images | 随机调整图像对比度 |
| la.image.enhance_color | Adjust hue of an RGB image | 随机调整图像色调 |
| la.image.enhance_sharpness| Adjust sharpness of an RGB image | 随机调整图像清晰度 |
| la.image.RandomGamma | Performs Gamma Correction on the input image | 图像Gamma校正 |
| la.image.noise_gaussian | Gaussian noise apply to image | 随机增加图像高斯噪声 |
| la.image.noise_poisson | Poisson noise apply to image | 随机增加图像泊松噪声 |
| la.image.noise_mask | Mask noise apply to image | 随机增加图像mask噪声 |
| la.image.noise_saltpepper | Salt-Pepper noise apply to image | 随机增加图像椒盐噪声 |
| la.image.noise_rainbow | Rainbowr noise apply to image | 随机增加图像彩虹噪声 |
| la.image.flip_left_right | Randomly flips an image horizontally | 随机图像左右翻转 |
| la.image.flip_up_down | Randomly flips an image vertically | 随机图像上下翻转 |
| la.image.flip_up_left | Randomly flips an image up to left | 随机图像上左翻转 |
| la.image.flip_up_right | Randomly flips an image up to right | 随机图像上右翻转 |
| la.image.translate | Translate image by swapping the height and width dimension | 图像平移 |
| la.image.rotate | Rotate image counter-clockwise by degrees | 随机图像旋转 |
| la.image.normalize_global | Normalize scales `image` to have mean and variance with global| 图像全局正态标准化 |
| la.image.normalize_channel | Normalize scales `image` to have mean and variance with channel| 图像单通道正态标准化 |
| la.image.rescale | Rescale apply to image | 图像数值等比例缩放 |
| la.image.crop_central | Crop the central region of the image and resize specify shape | 随机中心裁剪并resize |
| la.image.crop_point | Crop the any region of the image and resize specify shape | 随机点裁剪并resize |
| la.image.pencil_sketch | Adjust the pencil sketch of RGB | 随机铅笔画塑形 |
| la.image.list_images | Lists all pictures in a directory, including all subdirectories. |  |
| la.image.ColorMode |  |  |
| la.image.color_convert | Transform image color mode |  |
| la.image.image_to_array | Converts a PIL Image instance to a Numpy array. |  |
| la.image.array_to_image | Converts a 3D Numpy array to a PIL Image instance. |  |

- ## la.text

| API | description | 描述 |
| --- | --- | --- |
| la.text.CountVectorizer | convert a collection of text documents to a matrix of token counts | 文档向量计数 |
| la.text.TfidfVectorizer | transform a count matrix to a normalized tf or tf-idf representation | 文档tfidf值 |
| la.text.select_best_length | select best length for sequence with keep rate | 选择文档最佳长度 |
| la.text.word_to_index | sequence word transfer to index | 词转化索引 |
| la.text.word_index_sequence | sequence word transfer to sequence index | 词序列索引化 |
| la.text.pad_sequences | pads sequences to the same length | 序列截断与填充 |
| la.text.index_vector_matrix | make index vector matrix with shape `(len(word_index_dict), embed_dim)` | 索引向量矩阵 |
| la.text.sequence_preprocess | sequence preprocess, keep only Chinese | 序列预处理（只含中文） |
| la.text.word_count | sequence word count | 词计数字典 |
| la.text.word_low_freq | filter low frequency words | 查找词计数字典低频词 |
| la.text.word_high_freq | filter high frequency words | 查找词计数字典高频词 |
| la.text.filter_word | fequence filter words with a filter word list | 过滤指定词 |
| la.text.filter_punctuation | sequence preprocess, filter punctuation | 过滤标点符号 |
