import os
import numpy as np
from sklearn.decomposition import PCA


def load_timeseries(expt_id, base_data_dir=None):
    """
    Helper function to load neural and behavioral data

    Args:
        expt_id (str)
        base_data_dir (str)

    Returns:
        dict
    """

    import scipy.io as sio
    from flygenvectors.utils import get_dirs

    if base_data_dir is None:
        base_data_dir = get_dirs()['data']

    if expt_id == '190424_f3':
        file_name = 'runAndFeedSample.mat'
        file_path = os.path.join(base_data_dir, file_name)
        data_dict = sio.loadmat(file_path)
        # data_dict contents:
        #   trialFlag: indexes running/feeding/running components of the expt
        #   dOO: ratiometric dF/F for every active cell
        #   A: spatial footprint of these cells
        #   legs, stim, and feed: behavioral data
    elif expt_id == '180824_f3r1':
        file_name = 'runningSample.mat'
        file_path = os.path.join(base_data_dir, file_name)
        data_dict = sio.loadmat(file_path)
        # data_dict contents:
        #   trialFlag: indexes running/feeding/running components of the expt
        #   dOO: ratiometric dF/F for every active cell
        #   A: spatial footprint of these cells
        #   legs, stim, and feed: behavioral data
    else:
        # assumes following filename structure:
        # yyyy_mm_dd_flyi/yyyy_mm_dd_Nsyb_NLS6s_walk_flyi.npz

        strs = expt_id.split('_')
        datestr = str('%s_%s_%s' % (strs[0], strs[1], strs[2]))
        fly = strs[3]
        file_name = str('%s_Nsyb_NLS6s_walk_%s.npz' % (datestr, fly))
        file_path = os.path.join(base_data_dir, expt_id, file_name)
        data = np.load(file_path)
        t = np.max(data['time'].shape)
        data_dict = {}
        for key, val in data.items():
            # put time dim first
            data_dict[key] = val if val.shape[0] == t else val.T
        # data_dict contents:
        #   time: timestamps in seconds
        #   trialFlag: index of which imaging run each timestamp corresponds to
        #       (~10 sec between runs)
        #   dFF: bleach-corrected ratio
        #   ball: movement of ball measured in pixel variance
        #   dlc: dlc labels
        #   dims: height, width, depth of imaged volume

    return data_dict


def load_spatial_footprints(expt_id, base_data_dir=None):
    raise NotImplementedError


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
