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

        self.means = {'x': [], 'y': []}
        self.stds = None

        self.dtypes = []
        self.dtype_lens = []
        self.idxs_valid = []  # "good" indices
        self.idxs_dict = []  # indices separated by

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
        dlc = genfromtxt(filename, delimiter=',', dtype=None)
        dlc = dlc[3:, 1:].astype('float')  # get rid of headers, etc.
        self.labels['x'] = dlc[:, 0::3]
        self.labels['y'] = dlc[:, 1::3]
        self.labels['l'] = dlc[:, 2::3]
        if self.verbose:
            print('done')
            print('total time points: %i' % dlc.shape[0])

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

    def standardize(self):
        """subtract off mean and divide by variance across all labels"""
        if self.verbose:
            print('standardizing labels...', end='')
        for c in ['x', 'y']:
            self.means[c] = np.mean(self.labels[c], axis=0)
        self.stds = np.mean(
            np.concatenate([self.labels['x'], self.labels['x']], axis=0))
        for c in ['x', 'y']:
            self.labels[c] = (self.labels[c] - self.means[c]) / self.stds
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
            dims='all', skip_indxs=None):
        """
        Find contiguous chunks of data with likelihoods larger than a given val

        Args:
            l_thresh (float): minimum likelihood threshold
            min_length (int): minimum length of high likelihood runs
            max_length (int): maximum length of high likelihood runs; once a
                run surpasses this threshold a new run is started
            comparison (str): comparison operator to use for data ? l_thresh
                '>' | '>=' | '<' | '<='
            dims (str): define whether any or all dims must meet requirement
                'any' | 'all'
            skip_indxs (np bool array or NoneType, optional): same size as
                `likelihoods`, `True` indices will be counted as a negative
                comparison
        """
        if self.verbose:
            print('extracting runs of labels above likelihood=%1.2f...' %
                  l_thresh, end='')

        import operator
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
        if skip_indxs is None:
            skip_indxs = np.full(shape=(T,), fill_value=False)

        idxs = []

        run_len = 1
        i_beg = 0
        i_end = 1

        reset_run = False
        save_run = False
        for t in range(1, T):

            if bool_check(op(self.labels['l'][t], l_thresh)) and not \
                    skip_indxs[t]:
                run_len += 1
                i_end += 1
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
                run_len = 1
                i_beg = t
                i_end = t + 1
                reset_run = False

        # final run
        if run_len - 1 >= min_length:
            idxs.append(np.arange(i_beg, i_end - 1))

        self.idxs_valid = idxs

        if self.verbose:
            print('done')
            print('extracted %i runs for a total of %i time points' % (
                len(idxs), np.sum([len(i) for i in idxs])))

    def split_labels(self, dtypes, dtype_lens):
        if self.verbose:
            print('splitting labels into {}...'.format(dtypes), end='')
        # get train/text/val indices
        self.idxs_dict = split_runs(self.idxs_valid, dtypes, dtype_lens)
        # split labels into train/test/val using index split above
        dlc_array = self.get_label_array()
        self.labels_dict = {dtype: [] for dtype in self.idxs_dict.keys()}
        for dtype, dindxs in self.idxs_dict.items():
            for dindx in dindxs:
                self.labels_dict[dtype].append(dlc_array[dindx, :])
        if self.verbose:
            print('done')
            for dtype in dtypes:
                print('\t%s: %i time points in %i trials' % (
                    dtype, np.sum([len(i) for i in self.labels_dict[dtype]]),
                    len(self.labels_dict[dtype])))

    def get_label_array(self):
        """concatenate x/y labels into a single array"""
        return np.concatenate([self.labels['x'], self.labels['y']], axis=1)
