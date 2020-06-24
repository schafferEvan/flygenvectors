import os
import glob
import numpy as np
import copy


def split_runs(indxs, dtypes, dtype_lens):
    """

    Args:
        indxs (list):
        dtypes (list of strs):
        dtype_lens (list of ints):

    Returns:
        dict
    """

    # first sort, then split according to ratio
    i_sorted = np.argsort([len(i) for i in indxs])

    indxs_split = {dtype: [] for dtype in dtypes}
    dtype_indx = 0
    dtype_curr = dtypes[dtype_indx]
    counter = 0
    for indx in reversed(i_sorted):
        if counter == dtype_lens[dtype_indx]:
            # move to next dtype
            dtype_indx = (dtype_indx + 1) % len(dtypes)
            while dtype_lens[dtype_indx] == 0:
                dtype_indx = (dtype_indx + 1) % len(dtypes)
            dtype_curr = dtypes[dtype_indx]
            counter = 0
        indxs_split[dtype_curr].append(indxs[indx])
        counter += 1

    return indxs_split


class DLCLabels(object):

    def __init__(self, expt_id, verbose=True, algo='dgp'):
        """
        Args:
            expt_id:
            verbose:
            algo (str): 'dlc' | 'dgp'
        """
        from flygenvectors.utils import get_dirs

        self.expt_id = expt_id
        self.base_data_dir = get_dirs()['data']
        self.algo = algo

        self.labels = {'x': [], 'y': [], 'l': []}
        self.labels_dict = {}

        self.preproc = None
        self.means = {'x': [], 'y': []}
        self.stds = {'x': [], 'y': []}
        self.mins = {'x': [], 'y': []}
        self.maxs = {'x': [], 'y': []}

        self.dtypes = []
        self.dtype_lens = []
        self.skip_idxs = None
        self.idxs_valid = []  # "good" indices
        self.idxs_dict = []  # indices separated by train/test/val

        self.verbose = verbose

    def _get_filename(self):
        return glob.glob(os.path.join(self.base_data_dir, self.expt_id, '*DeepCut*.csv'))[0]

    def load_from_csv(self, filename=None):
        from numpy import genfromtxt
        if filename is None:
            filename = self._get_filename()
        if self.verbose:
            print('loading labels from %s...' % filename, end='')
        dlc = genfromtxt(filename, delimiter=',', dtype=None, encoding=None)
        dlc = dlc[3:, 1:].astype('float')  # get rid of headers, etc.
        self.labels['x'] = dlc[:, 0::3]
        self.labels['y'] = dlc[:, 1::3]
        self.labels['l'] = dlc[:, 2::3]
        if self.verbose:
            print('done')
            print('total time points: %i' % dlc.shape[0])

    def load_from_mat(self):
        """Load from mat file output by DGP."""

        from scipy.io import loadmat

        # define paths
        label_dir = os.path.join(self.base_data_dir, self.expt_id, 'crops_labels')
        if self.algo == 'dlc':
            label_files = glob.glob(os.path.join(label_dir, 'dlc_mu*.mat'))
            key = 'xy'
            prefix = 'dlc_mu'
        elif self.algo == 'dgp':
            label_files = glob.glob(os.path.join(label_dir, 'mu*.mat'))
            key = 'mu_ns'
            prefix = 'mu'
        else:
            raise NotImplementedError

        if len(label_files) == 0:
            raise IOError('no label files found in %s' % label_dir)
        else:
            if self.verbose:
                print('loading labels from %s...' % label_dir, end='')

        segments = len(label_files) - 1  # assume anqi always returns train labels
        labels_x = []
        labels_y = []
        for segment_num in range(segments):
            label_file = os.path.join(label_dir, '%s_%04i_flyball2.mat' % (prefix, segment_num))
            labels_seg = loadmat(label_file)[key]
            labels_x.append(labels_seg[:, :, 0])
            labels_y.append(labels_seg[:, :, 1])
        self.labels['x'] = np.vstack(labels_x)
        self.labels['y'] = np.vstack(labels_y)
        if self.verbose:
            print('done')
            print('total time points: %i' % self.labels['x'].shape[0])

    def load_from_pkl(self):
        """Load from pkl file output by DGP."""
        import pickle

        label_file = os.path.join(
            '/media/mattw/fly/behavior/labels/resnet-50_ws=0.0e+00_wt=0.0e+00/',
            '%s_labeled.pkl' % self.expt_id)

        if self.verbose:
            print('loading labels from %s...' % label_file, end='')

        with open(label_file, 'rb') as f:
            labels = pickle.load(f)

        self.labels['x'] = labels['x']
        self.labels['y'] = labels['y']

        if self.verbose:
            print('done')
            print('total time points: %i' % self.labels['x'].shape[0])

    def preprocess(self, preproc_dict):
        self.preproc = copy.deepcopy(preproc_dict)
        for func_str, kwargs in preproc_dict.items():
            if func_str == 'standardize':
                self.standardize(**kwargs)
            elif func_str == 'unitize':
                self.unitize(**kwargs)
            elif func_str == 'filter':
                self.filter(**kwargs)
            else:
                raise ValueError('"%s" is not a valid preprocessing function' % func_str)

    def standardize(self, by_label=False):
        """subtract off mean and divide by variance across all labels"""
        if self.verbose:
            print('standardizing labels...', end='')

        for c in ['x', 'y']:
            self.means[c] = np.mean(self.labels[c], axis=0)

        if by_label:
            for c in ['x', 'y']:
                self.stds[c] = np.std(self.labels[c], axis=0)
        else:
            self.stds['x'] = self.stds['y'] = np.std(
                np.concatenate([self.labels['x'], self.labels['y']], axis=0))

        for c in ['x', 'y']:
            self.labels[c] = (self.labels[c] - self.means[c]) / self.stds[c]
        if self.verbose:
            print('done')

    def unitize(self, **kwargs):
        """place each label (mostly) in [0, 1]"""
        if self.verbose:
            print('unitizing labels...', end='')
        for c in ['x', 'y']:
            self.mins[c] = np.quantile(self.labels[c], 0.05, axis=0)
            self.maxs[c] = np.quantile(self.labels[c], 0.95, axis=0)
            self.labels[c] = (self.labels[c] - self.mins[c]) / (self.maxs[c] - self.mins[c])
        if self.verbose:
            print('done')

    def filter(self, type='median', **kwargs):
        if self.verbose:
            print('applying %s filter to labels...' % type, end='')
        if type == 'median':
            from scipy.signal import medfilt
            kernel_size = 5 if 'kernel_size' not in kwargs else kwargs['kernel_size']
            for c in ['x', 'y']:
                for i in range(self.labels[c].shape[1]):
                    self.labels[c][:, i] = medfilt(
                        self.labels[c][:, i], kernel_size=kernel_size)
        elif type == 'savgol':
            from scipy.signal import savgol_filter
            window_length = 5 if 'window_size' not in kwargs else kwargs['window_size']
            polyorder = 2 if 'order' not in kwargs else kwargs['order']
            for c in ['x', 'y']:
                for i in range(self.labels[c].shape[1]):
                    self.labels[c][:, i] = savgol_filter(
                        self.labels[c][:, i], window_length=window_length, polyorder=polyorder)
        else:
            raise NotImplementedError
        if self.verbose:
            print('done')

    def extract_runs_by_length(self, max_length, return_vals=False, verbose=None):
        """
        Find contiguous chunks of data

        Args:
            max_length (int): maximum length of high likelihood runs; once a
                run surpasses this threshold a new run is started
            return_vals (bool): return list of indices if `True`, otherwise
                store in object as `indxs_valid`
            verbose (bool or NoneType)

        Returns:
            list
        """
        if verbose is None:
            verbose = self.verbose
        if verbose:
            print('extracting runs of length %i...' % max_length, end='')
        n_t = self.labels['x'].shape[0]
        begs = np.arange(0, n_t, max_length)
        ends = np.concatenate([np.arange(max_length, n_t, max_length), [n_t]])
        idxs = [np.arange(begs[i], ends[i]) for i in range(len(begs))]
        if verbose:
            print('done')
            print('extracted %i runs for a total of %i time points' % (
                len(idxs), np.sum([len(i) for i in idxs])))
        if return_vals:
            return idxs
        else:
            self.idxs_valid = idxs

    def split_labels(self, dtypes, dtype_lens):
        if self.verbose:
            print('splitting labels into {}...'.format(dtypes), end='')
        # get train/text/val indices
        self.idxs_dict = split_runs(self.idxs_valid, dtypes, dtype_lens)
        # split labels into train/test/val using index split above
        dlc_array = self.get_label_array()
        self.labels_dict = {dtype: [] for dtype in self.idxs_dict.keys()}
        for dtype, didxs in self.idxs_dict.items():
            for didx in didxs:
                self.labels_dict[dtype].append(dlc_array[didx, :])
        if self.verbose:
            print('done')
            for dtype in dtypes:
                print('\t%s: %i time points in %i trials' % (
                    dtype, np.sum([len(i) for i in self.labels_dict[dtype]]),
                    len(self.labels_dict[dtype])))

    def get_label_array(self):
        """concatenate x/y labels into a single array"""
        return np.concatenate([self.labels['x'], self.labels['y']], axis=1)
