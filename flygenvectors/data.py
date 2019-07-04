import numpy as np
from sklearn.decomposition import PCA


def zscore(data):
    """zscore over axis 0"""
    std = np.std(data, axis=0)
    mean = np.mean(data, axis=0)
    return (np.copy(data) - mean) / std


def cluster(data):
    """reorder axis 1 of a T x N matrix"""

    from scipy.cluster import hierarchy

    # fit pca to get a feature vector for each neuron
    pca = PCA(n_components=40)
    pca.fit(data)
    features = pca.components_.T

    # cluster feature vectors
    # Z = hierarchy.ward(features)
    Z = hierarchy.linkage(
        features, method='single', metric='cosine', optimal_ordering=True)
    leaves = hierarchy.leaves_list(Z)

    # methods:
    # 'single', 'complete', 'average', 'weighted', 'centroid', 'median', 'ward'
    # metrics:
    # ‘braycurtis’, ‘canberra’, ‘chebyshev’, ‘cityblock’, ‘correlation’,
    # ‘cosine’, ‘dice’, ‘euclidean’, ‘hamming’, ‘jaccard’, ‘jensenshannon’,
    # ‘kulsinski’, ‘mahalanobis’, ‘matching’, ‘minkowski’, ‘rogerstanimoto’,
    # ‘russellrao’, ‘seuclidean’, ‘sokalmichener’, ‘sokalsneath’,
    # ‘sqeuclidean’, ‘yule’

    return data[:, leaves]


def pca_reduce(data, n_components, indxs, trial_len):

    pca = PCA(n_components=n_components)
    data_pca_ = pca.fit_transform(data)

    data_pca = {}
    for dtype in ['train', 'test', 'val']:
        data_segs = []
        for indx in indxs[dtype]:
            data_segs.append(
                data_pca_[(indx * trial_len):(indx * trial_len + trial_len)])
        data_pca[dtype] = data_segs

    data_pca['train_all'] = np.concatenate(data_pca['train'], axis=0)
    data_pca['val_all'] = np.concatenate(data_pca['val'], axis=0)

    return data_pca_, data_pca


def subsample_cells(data, cell_indxs, indxs, trial_len):

    data_sub_ = data[:, cell_indxs]

    data_sub = {}
    for dtype in ['train', 'test', 'val']:
        data_segs = []
        for indx in indxs[dtype]:
            data_segs.append(
                data_sub_[(indx * trial_len):(indx * trial_len + trial_len)])
        data_sub[dtype] = data_segs

    data_sub['train_all'] = np.concatenate(data_sub['train'], axis=0)
    data_sub['val_all'] = np.concatenate(data_sub['val'], axis=0)

    return data_sub_, data_sub
