import os, sys
import glob
import numpy as np
from sklearn.decomposition import PCA
from scipy.optimize import minimize
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
    # from flygenvectors.utils import get_dirs

    exp_folder = expt_id + '_' + fly_num + '/'
    file_name_main = expt_id + '*' + fly_num + '.npz'
    file_name_A = expt_id + '*' + fly_num + '_A.npz'

    file_path_main = glob.glob(base_data_dir+exp_folder+file_name_main)[0]
    file_path_A = glob.glob(base_data_dir+exp_folder+file_name_A)[0]

    data = np.load(file_path_main, allow_pickle=True)
    data_dict = {}
    for key, val in data.items():
        data_dict[key] = val

    data_dict['A'] = sparse.load_npz(file_path_A)
    if 'drink' not in data_dict.keys():
        data_dict['drink'] = np.zeros(data_dict['ball'].shape)
    if 'stim' not in data_dict.keys():
        data_dict['stim'] = np.zeros(data_dict['ball'].shape)

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

    file_path = os.path.join(base_data_dir, 'behavior', 'videos_cropped', '%s.avi' % expt_id)

    # read file
    cap = cv2.VideoCapture(file_path)

    # Check if file opened successfully
    if not cap.isOpened():
        print('Error opening video stream or file from %s' % file_path)

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



def estimate_neuron_behav_tau(data_dict):
    # THIS IS THE LEGACY VERSION OF estimate_neuron_behav_reg_model in regression_model.py
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


def binarize_timeseries(data_in, means_init=[[0.5],[0.8]]):
    # use GMM to turn scalar-valued timeseries into binary
    from sklearn.mixture import GaussianMixture

    gmm = GaussianMixture(n_components=2, means_init=means_init)
    log_beh = np.log(data_in)-np.log(data_in).min()
    log_beh = log_beh/log_beh.max()
    gmm_beh_fit = gmm.fit_predict(log_beh)
    mean_0 = log_beh[gmm_beh_fit==0].mean()
    mean_1 = log_beh[gmm_beh_fit==1].mean()
    if (mean_0<mean_1):
        beh = gmm_beh_fit
    else:
        beh = np.logical_not(gmm_beh_fit)
    return beh
    # data_dict['ball'] = data_dict['ball'].flatten()


def binarize_timeseries_by_outliers(data_in, sig=1):
    # simplistic approach works well for sparse timeseries (better than GMM)
    m = data_in.mean()
    s = data_in.std()
    data_out = data_in>(m+sig*s)
    return data_out



def get_dFF_ica(data_dict):
    # given dual color neural data (dynamic green + static red), uses ICA to remove presumed motion artifacts from green channel
    from sklearn.decomposition import FastICA
    dFF_ica = np.zeros(data_dict['dYY'].shape)
    ica = FastICA(n_components=2, max_iter=5000, fun='cube') #,whiten=False, )

    for i in range(data_dict['dYY'].shape[0]):
        print(i,end=' ')
        data = np.array([ data_dict['dYY'][i,:], data_dict['dRR'][i,:] ]) #[self.YsmoothData[i,:],self.RsmoothData[i,:]]
        m = data.mean(axis=1)
        m = m[:,np.newaxis]
        v = data.var(axis=1)
        s = np.diag(np.reciprocal(np.sqrt(v)))
        X = s@(data - m)
        ic = ica.fit_transform(data.T).T  # Reconstruct signals
        
        mm = ic.mean(axis=1)
        mm = mm[:,np.newaxis]
        ss = ic.std(axis=1)
        ss = np.diag(np.reciprocal(ss))
        Xic = ss@(ic - mm)
        cc = (Xic@X.T)/X.shape[1] #

        sigcom_prop = np.argmax(np.abs(cc[:,0])) # signal component is the one more correlated with green channel (ignoring sign)
        if np.abs(cc[sigcom_prop,0])>np.abs(cc[sigcom_prop,1]):
            sigcom = sigcom_prop # if IC more corr w/ green is more correlated with green than red, keep choice
        else:
            sigcom = int(not sigcom_prop) #if IC more corr w/ green is more correlated with red than green, flip choice
        if (cc[sigcom,0]<0): 
            I = -ic[sigcom,:] #np.matmul(-s_[sigcom,:],1/s[sigcom,sigcom]) + m[sigcom]
            icflipped = 1
        else:
            I = ic[sigcom,:] #np.matmul(s_[sigcom,:],1/s[sigcom,sigcom]) + m[sigcom]

        dFF_ica[i,:] = I*np.sqrt(v[0])/I.std() + m[0]
    return dFF_ica


def get_dlc_motion_energy(data_dict):
    dlc = data_dict['dlc']
    dlc_energy = np.zeros((dlc.shape[0]-1,8))

    for i in range(8):
        xdataChunk = np.diff(dlc[:,(i-1)*2]); 
        ydataChunk = np.diff(dlc[:,1+(i-1)*2]);
        legEnergy = xdataChunk**2 + ydataChunk**2;
        m = np.quantile(legEnergy, 0.01)
        M = np.quantile(legEnergy, 0.99)
        legEnergy[legEnergy<m]=m
        legEnergy[legEnergy>M]=M
        dlc_energy[:,i] = legEnergy
    return dlc_energy


def split_trials(
        n_trials, rng_seed=0, trials_tr=5, trials_val=1, trials_test=1,
        trials_gap=1):
    """
    Split trials into train/val/test blocks.
    The data is split into blocks that have gap trials between tr/val/test:
    train tr | gap tr | val tr | gap tr | test tr | gap tr
    Args:
        n_trials (int): number of trials to use in the split
        rng_seed (int): numpy random seed for reproducibility
        trials_tr (int): train trials per block
        trials_val (int): validation trials per block
        trials_test (int): test trials per block
        trials_gap (int): gap trials between tr/val/test; there will be a total
            of 3 * `trials_gap` gap trials per block
    Returns:
        (dict)
    """

    # same random seed for reproducibility
    np.random.seed(rng_seed)

    tr_per_block = \
        trials_tr + trials_gap + trials_val + trials_gap + trials_test + trials_gap

    n_blocks = int(np.floor(n_trials / tr_per_block))
    leftover_trials = n_trials - tr_per_block * n_blocks
    if leftover_trials > 0:
        offset = np.random.randint(0, high=leftover_trials)
    else:
        offset = 0
    indxs_block = np.random.permutation(n_blocks)

    batch_indxs = {'train': [], 'test': [], 'val': []}
    for block in indxs_block:
        curr_tr = block * tr_per_block + offset
        batch_indxs['train'].append(np.arange(curr_tr, curr_tr + trials_tr))
        curr_tr += (trials_tr + trials_gap)
        batch_indxs['val'].append(np.arange(curr_tr, curr_tr + trials_val))
        curr_tr += (trials_val + trials_gap)
        batch_indxs['test'].append(np.arange(curr_tr, curr_tr + trials_test))

    for dtype in ['train', 'val', 'test']:
        batch_indxs[dtype] = np.concatenate(batch_indxs[dtype], axis=0)

    return batch_indxs


def trim_time(data_dict, before_stim=True):
    """
    Crop time from neural and behav data.
    """
    import copy
    if before_stim:
        buf = 60 #sec
        buf_frames = int(round(data_dict['scanRate']*buf))
        stim_idx = np.flatnonzero(data_dict['stim']==1)[0]
        cut_pt = stim_idx-buf_frames

    data_dict_new = copy.deepcopy(data_dict)
    data_dict_new['dFF'] = data_dict_new['dFF'][:,:cut_pt]
    data_dict_new['dYY'] = data_dict_new['dYY'][:,:cut_pt]
    data_dict_new['dRR'] = data_dict_new['dRR'][:,:cut_pt]
    data_dict_new['ball'] = data_dict_new['ball'][:cut_pt,:]
    data_dict_new['behavior'] = data_dict_new['behavior'][:cut_pt,:]
    data_dict_new['time'] = data_dict_new['time'][:cut_pt,:]
    data_dict_new['trialFlag'] = data_dict_new['trialFlag'][:cut_pt,:]
    data_dict_new['tPl'] = data_dict_new['tPl'][:cut_pt,:]
    data_dict_new['stim'] = data_dict_new['stim'][:cut_pt,:]
    data_dict_new['drink'] = data_dict_new['drink'][:cut_pt,:]
    data_dict_new['dlc'] = data_dict_new['dlc'][:cut_pt,:]
    if len(data_dict_new['beh_labels'])>0:
        data_dict_new['beh_labels'] = data_dict_new['beh_labels'][:cut_pt,:]
    return data_dict_new


def extract_state_runs(states, sort_by='ending', min_length=20):
    """
    Find contiguous chunks of data with the same state

    Args:
        states (list):
        min_length (int):
        sort_by (None, 'ending', 'beginning')
    Returns:
        list
    """

    K = len(np.unique(np.concatenate([np.unique(s) for s in states])))
    if sort_by is None:
        state_snippets = [[] for _ in range(K)]
    else:
        state_snippets = [ [[] for _ in range(K)] for _ in range(K)]
        

    i_beg = 0
    curr_len = 1
    curr_state = states[0]
    old_state = 0

    for i, next_state in enumerate(states, start=1):        
        if next_state != curr_state:
            # record indices if state duration long enough
            if curr_len >= min_length:
                if sort_by == 'ending':
                    state_snippets[curr_state][next_state].append(
                        [j-1 for j in range(i_beg,i)])
                elif sort_by == 'beginning':
                    state_snippets[old_state][curr_state].append(
                        [j-1 for j in range(i_beg,i)])
                else:
                    state_snippets[curr_state].append(
                        [j-1 for j in range(i_beg,i)])
            i_beg = i
            old_state = curr_state
            curr_state = next_state
            curr_len = 1
        else:
            curr_len += 1
        # end of trial cleanup
        # if next_state == curr_state:
        #     # record indices if state duration long enough
        #     if curr_len >= min_length:
        #         state_snippets[curr_state].append(curr_indxs[i_beg:i])
    return state_snippets



class Logger(object):
    # for printing stdout to both screen and logfile
    def __init__(self, fname="logfile.log"):
        self.terminal = sys.stdout
        self.log = open(fname, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        pass    
