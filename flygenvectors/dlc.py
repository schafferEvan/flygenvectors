import os
import glob
import numpy as np
from flygenvectors.ssmutils import split_runs


class DLCLabels(object):

    def __init__(self, expt_id, verbose=True):

        from flygenvectors.utils import get_dirs

        self.expt_id = expt_id
        self.base_data_dir = get_dirs()['data']

        self.labels = {'x': [], 'y': [], 'l': []}
        self.labels_dict = {}

        self.preproc = None
        self.means = {'x': [], 'y': []}
        self.stds = {'x': [], 'y': []}
        self.mins = {'x': [], 'y': []}
        self.maxs = {'x': [], 'y': []}

        self.dtypes = []
        self.dtype_lens = []
        self.idxs_valid = []  # "good" indices
        self.idxs_dict = []  # indices separated by train/test/val

        self.verbose = verbose

    def _get_filename(self):
        return glob.glob(os.path.join(
            self.base_data_dir, self.expt_id, '*DeepCut*.csv'))[0]

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

    def preprocess(self, preproc_dict):
        """

        Args:
            preproc_dict (dict):
        """
        # store preprocessing steps
        self.preproc = preproc_dict
        for func_str, kwargs in preproc_dict.items():
            if func_str == 'label_interpolation':
                self.interpolate_labels(**kwargs)
            elif func_str == 'standardize':
                self.standardize(**kwargs)
            elif func_str == ''

    def interpolate_labels(self, thresh=0.8):
        pass

    def interpolate_single_bad_labels(self, thresh=0.8):
        """
        For any label whose likelihood is below `thresh` for a single
        timepoint, linearly interpolate coordinates from surrounding (good)
        coordinates.

        Args:
            thresh (int):
        """
        if self.verbose:
            print('linearly interpolating single bad labels...', end='')
        old_likelihoods = np.copy(self.labels['l'])
        T = self.labels['x'].shape[0]
        for t in range(1, T - 1):
            is_t = np.where(self.labels['l'][t, :] < thresh)[0]
            for i in is_t:
                if ((self.labels['l'][t - 1, i] > thresh) &
                        (self.labels['l'][t + 1, i] > thresh)):
                    for c in ['x', 'y', 'l']:
                        self.labels[c][t, i] = np.mean(
                            [self.labels[c][t - 1, i],
                             self.labels[c][t + 1, i]])
        if self.verbose:
            print('done')
            print('linearly interpolated %i labels' %
                  np.sum(((old_likelihoods - self.labels['l']) != 0) * 1.0))

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

    def unitize(self):
        """place each label (mostly) in [0, 1]"""
        if self.verbose:
            print('unitizing labels...', end='')
        for c in ['x', 'y']:
            self.mins[c] = np.quantile(self.labels[c], 0.05, axis=0)
            self.maxs[c] = np.quantile(self.labels[c], 0.95, axis=0)
            self.labels[c] = (self.labels[c] - self.mins[c]) / \
                             (self.maxs[c] - self.mins[c])
        if self.verbose:
            print('done')

    def filter(self, filter_type):
        if self.verbose:
            print('applying %s filter to labels...' % filter_type, end='')
        if filter_type == 'median':
            from scipy.signal import medfilt
            kernel_size = 5
            for c in ['x', 'y']:
                for i in range(self.labels[c].shape[1]):
                    self.labels[c][:, i] = medfilt(
                        self.labels[c][:, i], kernel_size=kernel_size)
        else:
            raise NotImplementedError
        if self.verbose:
            print('done')

    def print_likelihood_info(self):
        # percentage of time points below a certain threshold per label
        thresh = 0.9
        bad_frac = np.sum((self.labels['l'] < thresh) * 1.0, axis=0) / \
            self.labels['l'].shape[0]
        print('percentage of time points with likelihoods below {}: {}'.format(
              thresh, bad_frac))
        # percentage of time points below a certain threshold for any label
        thresh = 0.5
        bad_frac = np.sum((np.min(self.labels['l'], axis=1) < thresh)*1.0) / \
            self.labels['l'].shape[0]
        print('percentage of time points with likelihoods below {} for any '
              'label: {}'.format(thresh, bad_frac))

    def extract_runs_by_likelihood(
            self, l_thresh, min_length, max_length, comparison='>=',
            dims='all', skip_idxs=None, return_vals=False):
        """
        Find contiguous chunks of data with likelihoods consistent with a
        given condition

        Args:
            l_thresh (float): minimum likelihood threshold
            min_length (int): minimum length of high likelihood runs
            max_length (int): maximum length of high likelihood runs; once a
                run surpasses this threshold a new run is started
            comparison (str): comparison operator to use for data ? l_thresh
                '>' | '>=' | '<' | '<='
            dims (str): define whether any or all dims must meet requirement
                'any' | 'all'
            skip_idxs (np bool array or NoneType, optional): same size as
                `likelihoods`, `True` indices will be counted as a negative
                comparison
            return_vals (bool): return list of indices if `True`, otherwise
                store in object as `indxs_valid`
        """
        import operator

        if self.verbose:
            print('extracting runs of labels %s likelihood=%1.2f...' %
                  (comparison, l_thresh), end='')

        if comparison == '>':
            op = operator.gt
        elif comparison == '>=':
            op = operator.ge
        elif comparison == '<':
            op = operator.lt
        elif comparison == '<=':
            op = operator.le
        else:
            raise ValueError(
                '"%s" is an invalid comparison operator' % comparison)

        if dims == 'any':
            bool_check = np.any
        elif dims == 'all':
            bool_check = np.all
        else:
            raise ValueError('"%s" is an invalid boolean check' % dims)

        T = self.labels['l'].shape[0]
        if skip_idxs is None:
            skip_idxs = np.full(shape=(T,), fill_value=False)

        idxs = []

        run_len = 1
        i_beg = 0
        i_end = 1

        reset_run = False
        save_run = False
        for t in range(1, T):

            if bool_check(op(self.labels['l'][t], l_thresh)) and not \
                    skip_idxs[t]:
                run_len += 1
                i_end = t + 1
            else:
                if run_len >= min_length:
                    save_run = True
                reset_run = True
            if run_len == max_length:
                save_run = True
                reset_run = True

            if save_run:
                idxs.append(np.arange(i_beg, i_end))
                save_run = False
            if reset_run:
                run_len = 0
                i_beg = t + 1
                i_end = t + 1
                reset_run = False

        # final run
        if run_len >= min_length:
            idxs.append(np.arange(i_beg, i_end))

        if self.verbose:
            print('done')
            print('extracted %i runs for a total of %i time points' % (
                len(idxs), np.sum([len(i) for i in idxs])))

        if return_vals:
            return idxs
        else:
            self.idxs_valid = idxs

    def split_labels(self, dtypes, dtype_lens, diff=False):
        if self.verbose:
            print('splitting labels into {}...'.format(dtypes), end='')
        # get train/text/val indices
        self.idxs_dict = split_runs(self.idxs_valid, dtypes, dtype_lens)
        # split labels into train/test/val using index split above
        dlc_array = self.get_label_array()
        if diff:
            dlc_array = np.concatenate([
                np.zeros(shape=(1, dlc_array.shape[1])),
                np.diff(dlc_array, axis=0)])
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
