import os
import glob
import numpy as np
from sklearn.decomposition import PCA
import pdb


def load_timeseries_simple(expt_id, fly_num, base_data_dir=None):
    """
    Helper function to load neural and behavioral data

    Args:
        expt_id (str)
        base_data_dir (str)

    Returns:
        dict
    """

    import scipy.io as sio
    from scipy import sparse
    from flygenvectors.utils import get_dirs

    exp_folder = expt_id + '_' + fly_num + '/'
    file_name_main = expt_id + '*' + fly_num + '.npz'
    file_name_A = expt_id + '*' + fly_num + '_A.npz'

    file_path_main = glob.glob(base_data_dir+exp_folder+file_name_main)[0]
    file_path_A = glob.glob(base_data_dir+exp_folder+file_name_A)[0]

    data = np.load(file_path_main)
    data_dict = {}
    for key, val in data.items():
        data_dict[key] = val

    data_dict['A'] = sparse.load_npz(file_path_A)

    return data_dict


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
    from scipy import sparse
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
        file_name = str('%s*%s.npz' % (datestr, fly))
        file_path = glob.glob(os.path.join(base_data_dir, expt_id, file_name))[0]
        data = np.load(file_path)
        t = np.max(data['time'].shape)
        data_dict = {}
        for key, val in data.items():
            # put time dim first for items with a temporal dimension
            # pdb.set_trace()
            data_dict[key] = val.T if val.shape[1] == t else val

        # load
        file_name = str('%s_Nsyb_NLS6s_walk_%s_A.npz' % (datestr, fly))
        file_path = os.path.join(base_data_dir, expt_id, file_name)
        data_dict['A'] = sparse.load_npz(file_path)

        # data_dict contents:
        #   time: timestamps in seconds
        #   trialFlag: index of which imaging run each timestamp corresponds to
        #       (~10 sec between runs)
        #   dFF: bleach-corrected ratio
        #   ball: movement of ball measured in pixel variance
        #   dlc: dlc labels
        #   dims: height, width, depth of imaged volume
        #   A: spatial footprint of these cells

    return data_dict


def load_dlc_from_csv(expt_id, base_data_dir=None):
    """
    Helper function to load dlc labels from raw video (uninterpolated)

    Args:
        expt_id (str)
        base_data_dir (str)

    Returns:
        array: x1, y1, like1, x2, y2, like2, ...
    """

    from numpy import genfromtxt
    from flygenvectors.utils import get_dirs

    if base_data_dir is None:
        base_data_dir = get_dirs()['data']

    file_path = glob.glob(
        os.path.join(base_data_dir, expt_id, '*DeepCut*.csv'))[0]
    dlc = genfromtxt(file_path, delimiter=',', dtype=None)
    dlc = dlc[3:, 1:].astype('float')  # get rid of headers, etc.

    return dlc


def load_video(expt_id, base_data_dir=None):
    """
    Helper function to load videos

    Args:
        expt_id (str):
        base_data_dir (str):

    Returns:
        np array (T x y_pix x x_pix)
    """

    import cv2
    from flygenvectors.utils import get_dirs

    if base_data_dir is None:
        base_data_dir = get_dirs()['data']

    file_path = glob.glob(
        os.path.join(base_data_dir, expt_id, '*crop.mp4'))[0]

    # read file
    cap = cv2.VideoCapture(file_path)

    # Check if file opened successfully
    if not cap.isOpened():
        print('Error opening video stream or file')

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    videodata = np.zeros((total_frames, height, width), dtype='uint8')

    # read until video is completed
    fr = 0
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret:
            videodata[fr, :, :] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            fr += 1
        else:
            break

    # When everything done, release the video capture object
    cap.release()

    return videodata


def clean_dlc_labels(labels, thresh=0.8):
    """
    For any dlc label whose likelihood is below `thresh` for a single
    timepoint, linearly interpolate dlc coordinates from surrounding (good)
    coordinates.

    Args:
        labels (dict): keys 'x', 'y', 'l'
        thresh (int):

    Returns:
        dict
    """

    T = labels['x'].shape[0]
    for t in range(1, T - 1):
        is_t = np.where(labels['l'][t, :] < thresh)[0]
        for i in is_t:
            if ((labels['l'][t - 1, i] > thresh) & (
                    labels['l'][t + 1, i] > thresh)):
                for c in ['x', 'y', 'l']:
                    labels[c][t, i] = np.mean(
                        [labels[c][t - 1, i], labels[c][t + 1, i]])
    return labels


def remove_artifact_cells(data, threshold=10, footprints=None):
    mx = np.amax(data, axis=0)
    good_cells = mx < threshold
    data = data[:, good_cells]
    if footprints is not None:
        footprints = footprints[:, good_cells]
        return data, footprints
    else:
        return data


def estimate_noise_variance(y, range_ff=[0.25, 0.5], method='mean'):
    """
    Estimate noise power through the power spectral density over the range of
    large frequencies; adapted from GetSn() function from the OASIS package for
    online deconvolution of calcium imaging data:
    https://github.com/j-friedrich/OASIS/blob/master/oasis/functions.py

    Parameters
    ----------
    y : array, shape (T,)
        One dimensional array containing the fluorescence intensities with
        one entry per time-bin.
    range_ff : (1,2) array, nonnegative, max value <= 0.5
        range of frequency (x Nyquist rate) over which the spectrum is averaged
    method : string, optional, default 'mean'
        method of averaging: Mean, median, exponentiated mean of logvalues

    Returns
    -------
    sn : noise standard deviation
    """
    from scipy.signal import welch

    ff, Pxx = welch(y)
    ind1 = ff > range_ff[0]
    ind2 = ff < range_ff[1]
    ind = np.logical_and(ind1, ind2)
    Pxx_ind = Pxx[ind]
    sn = {
        'mean': lambda Pxx_ind: np.sqrt(np.mean(Pxx_ind / 2)),
        'median': lambda Pxx_ind: np.sqrt(np.median(Pxx_ind / 2)),
        'logmexp': lambda Pxx_ind: np.sqrt(
            np.exp(np.mean(np.log(Pxx_ind / 2))))
    }[method](Pxx_ind)

    return sn


def zscore(data):
    """zscore over axis 0"""
    std = np.std(data, axis=0)
    mean = np.mean(data, axis=0)
    return (np.copy(data) - mean) / std


def cluster(data, returnInd=False, cTh=0.1):
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

    if(returnInd):
        clustInd = hierarchy.fcluster(Z, cTh, 'distance')
        return data[:, leaves], clustInd
    else:
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

def trim_dynamic_range(data,q_min,q_max):
    bmin = np.quantile(data, q_min)
    bmax = np.quantile(data, q_max)
    data = (data-bmin)/(bmax-bmin)
    data[data<0]=0
    data[data>1]=1
    return data

def estimate_neuron_behav_tau(data_dict):
    # find optimal time constant PER NEURON with which to filter ball trace to maximize correlation
    dFF = data_dict['dFF']
    scanRate = data_dict['scanRate']

    tauList = np.logspace(np.log10(scanRate),np.log10(100*scanRate),num=200)
    xlen = round(-tauList[-1]*np.log(0.05)).astype(int)
    x = np.array([i for i in range(xlen)])

    # tauList = np.logspace(0,5,num=300) #range(10,1000,10)
    a = np.zeros((dFF.shape[0],len(tauList)))
    for i in range(len(tauList)):
        # if(not np.mod(i,20)):
        #     print(str(np.round(100*i/len(tauList)))+'%', end =" ")
        cTau = tauList[i]
        eFilt = np.exp(-x/cTau)

        c = np.convolve(eFilt,data_dict['ball'],'valid')#,'same')
        for j in range(dFF.shape[0]):
            a[j,i] = np.corrcoef(dFF[j,len(eFilt)-1:], c)[0,1] 
    return tauList, a
