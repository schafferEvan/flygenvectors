import numpy as np
import os
import pickle
import subprocess

from flygenvectors.data import load_video as video_loader
from flygenvectors.dlc import preprocess_and_split_data
from flygenvectors.dlc import shuffle_data
import flygenvectors.plotting as plotting
from flygenvectors.segmentation import heuristic_segmentation_v4, heuristic_segmentation_v5


FPS = 70

# NOTES
# - include_neural: decision made by evan
# - include_behavior: subjective decision made by me by examining dlc traces and
#   picking sessions that "lots" of movement
# - max_frames: if None, use all frames from video; else use this number of frames;
#   this is so that we can include sessions with lots of behavior at the beginning,
#   but less so at the end (which we'll just crop off)
# - x_off/y_off: upper left corner of cropping window; currently same for all
# - scale: scalar that multiplies pixel values to adjust histograms (brighten frames);
#   a cropped image was chosen from each behavior session (first frame) and the
#   99.9th percentile was chosen as the new white value (normalized to be in [0, 1])

database = {
    '2019_04_18_fly2_2': {
        'include_behavior': False,
        'include_neural': False,
        'n_frames': np.nan,
        'max_frames': None,
        'x_off': 80,
        'y_off': 50,
        'x_pix': 320,
        'y_pix': 240,
        'scale': 1,
        'notes': 'off ball and flailing'
    },
    '2019_06_26_fly1': {
        'include_behavior': False,
        'include_neural': False,
        'n_frames': 132455,
        'max_frames': None,
        'x_off': 200,
        'y_off': 20,
        'scale': 0.690,
        'notes': 'new addition (2021-04); omitting from further analyses'
    },
    '2019_06_26_fly2_0': {
        'include_behavior': True,
        'include_neural': False,
        'n_frames': 86714,
        'max_frames': None,
        'x_off': 200,
        'y_off': 20,
        'scale': 0.651,
        'notes': ''
    },
    '2019_06_26_fly2': {
        'include_behavior': True,
        'include_neural': False,
        'n_frames': 86714,
        'max_frames': 45000,
        'x_off': 200,
        'y_off': 20,
        'scale': 0.651,
        'notes': ''
    },
    '2019_06_28_fly2': {
        'include_behavior': True,
        'include_neural': True,
        'n_frames': 88212,
        'max_frames': None,
        'x_off': 200,
        'y_off': 20,
        'scale': 0.694,
        'notes': 'right forelimb glued to headplate'
    },
    '2019_06_28_fly3': {
        'include_behavior': True,
        'include_neural': False,
        'n_frames': 133418,
        'max_frames': None,
        'x_off': 200,
        'y_off': 20,
        'scale': 0.643,
        'notes': 'new addition (2021-04); poor video quality'
    },
    '2019_06_30_fly1': {
        'include_behavior': True,
        'include_neural': False,
        'n_frames': 279476,
        'max_frames': 180000,
        'x_off': 200,
        'y_off': 20,
        'scale': 0.736,
        'notes': 'right forelimb glued to itself'
    },
    '2019_07_01_fly2': {
        'include_behavior': True,
        'include_neural': True,
        'n_frames': 192980,
        'max_frames': None,
        'x_off': 200,
        'y_off': 20,
        'scale': 0.661,
        'notes': ''
    },
    '2019_08_06_fly1': {
        'include_behavior': False,
        'include_neural': False,
        'n_frames': 46016,
        'max_frames': None,
        'x_off': 200,
        'y_off': 20,
        'scale': None,
        'notes': ''
    },
    '2019_08_06_fly2': {
        'include_behavior': False,
        'include_neural': False,
        'n_frames': 67630,
        'max_frames': None,
        'x_off': 200,
        'y_off': 60,
        'scale': 0.627,
        'notes': 'new addition (2021-04); head occluded'
    },
    '2019_08_07_fly2': {
        'include_behavior': True,
        'include_neural': False,
        'n_frames': 153314,
        'max_frames': 50000,
        'x_off': 200,
        'y_off': 20,
        'scale': 0.733,
        'notes': ''
    },
    '2019_08_08_fly1': {
        'include_behavior': True,
        'include_neural': False,
        'n_frames': 94960,
        'max_frames': None,
        'x_off': 200,
        'y_off': 20,
        'scale': 0.620,
        'notes': ''
    },
    '2019_08_08_fly1_1': {
        'include_behavior': True,
        'include_neural': False,
        'n_frames': 109669,
        'max_frames': None,
        'x_off': 200,
        'y_off': 20,
        'scale': 0.694,
        'notes': 'lots of struggling'
    },
    '2019_08_13_fly3': {
        'include_behavior': True,
        'include_neural': False,
        'n_frames': 241653,
        'max_frames': None,
        'x_off': 200,
        'y_off': 20,
        'scale': 0.690,
        'notes': 'lots of struggling'
    },
    '2019_08_14_fly1': {
        'include_behavior': True,
        'include_neural': True,
        'n_frames': 124144,
        'max_frames': None,
        'x_off': 200,
        'y_off': 20,
        'scale': 0.973,
        'notes': ''
    },
    '2019_08_14_fly2': {
        'include_behavior': True,
        'include_neural': False,
        'n_frames': 41478,
        'max_frames': None,
        'x_off': 200,
        'y_off': 20,
        'scale': 1,
        'notes': ''
    },
    '2019_08_14_fly3_2': {
        'include_behavior': True,
        'include_neural': False,
        'n_frames': 30911,
        'max_frames': None,
        'x_off': 200,
        'y_off': 20,
        'scale': 1,
        'notes': 'new addition (2021-01)'
    },
    '2019_08_19_fly1_1': {
        'include_behavior': True,
        'include_neural': False,
        'n_frames': 31708,
        'max_frames': None,
        'x_off': 200,
        'y_off': 20,
        'scale': 1,
        'notes': 'right forelimb caught in headplate glue at end'
    },
    '2019_08_20_fly2': {
        'include_behavior': True,
        'include_neural': False,
        'n_frames': 65108,
        'max_frames': None,
        'x_off': 200,
        'y_off': 20,
        'scale': 0.992,
        'notes': ''
    },
    '2019_08_20_fly3': {
        'include_behavior': True,
        'include_neural': False,
        'n_frames': 73055,
        'max_frames': None,
        'x_off': 200,
        'y_off': 20,
        'scale': 1,
        'notes': ''
    },
    '2019_10_02_fly2': {
        'include_behavior': True,
        'include_neural': False,
        'n_frames': 117510,
        'max_frames': None,
        'x_off': 180,
        'y_off': 20,
        'scale': 0.579,
        'notes': 'new addition (2021-04)'
    },
    '2019_10_10_fly3': {
        'include_behavior': True,
        'include_neural': True,
        'n_frames': 141042,
        'max_frames': None,
        'x_off': 200,
        'y_off': 20,
        'scale': 0.549,
        'notes': 'bad video quality'
    },
    '2019_10_14_fly2': {
        'include_behavior': True,
        'include_neural': True,
        'n_frames': 139925,
        'max_frames': None,
        'x_off': 200,
        'y_off': 20,
        'scale': 0.606,
        'notes': 'bad video quality'
    },
    '2019_10_14_fly3': {
        'include_behavior': True,
        'include_neural': True,
        'n_frames': 140929,
        'max_frames': None,
        'x_off': 200,
        'y_off': 20,
        'scale': 0.584,
        'notes': 'bad video quality'
    },
    '2019_10_14_fly4': {
        'include_behavior': True,
        'include_neural': False,
        'n_frames': 136529,
        'max_frames': None,
        'x_off': 200,
        'y_off': 20,
        'scale': 0.631,
        'notes': 'new addition (2021-04)'
    },
    '2019_10_18_fly2': {
        'include_behavior': True,
        'include_neural': False,
        'n_frames': 120239,
        'max_frames': None,
        'x_off': 180,
        'y_off': 20,
        'scale': 0.620,
        'notes': 'new addition (2021-04)'
    },
    '2019_10_18_fly3': {
        'include_behavior': True,
        'include_neural': True,
        'n_frames': 141226,
        'max_frames': None,
        'x_off': 200,
        'y_off': 20,
        'scale': 0.518,
        'notes': ''
    },
    '2019_10_21_fly1': {
        'include_behavior': True,
        'include_neural': True,
        'n_frames': 142554,
        'max_frames': None,
        'x_off': 200,
        'y_off': 20,
        'scale': 0.580,
        'notes': ''
    },
    '2022_01_05_fly1': {
        'include_behavior': True,
        'include_neural': True,
        'n_frames': 51496,
        'max_frames': None,
        'x_off': 200,
        'y_off': 10,
        'scale': 0.602,
        'notes': ''
    },
    '2022_01_05_fly2': {
        'include_behavior': True,
        'include_neural': True,
        'n_frames': 135381,
        'max_frames': None,
        'x_off': 220,
        'y_off': 10,
        'scale': 0.549,
        'notes': ''
    },
    '2022_01_08_fly1': {
        'include_behavior': True,
        'include_neural': True,
        'n_frames': 134545,
        'max_frames': None,
        'x_off': 220,
        'y_off': 10,
        'scale': 0.545,
        'notes': ''
    },
    '2022_01_08_fly2': {
        'include_behavior': True,
        'include_neural': True,
        'n_frames': 90198,
        'max_frames': None,
        'x_off': 210,
        'y_off': 10,
        'scale': 0.576,
        'notes': ''
    },
    '2022_01_14_fly1': {
        'include_behavior': True,
        'include_neural': True,
        'n_frames': 134620,
        'max_frames': None,
        'x_off': 110,
        'y_off': 120,
        'scale': 0.451,
        'notes': ''
    },
    '2022_01_14_fly2': {
        'include_behavior': True,
        'include_neural': True,
        'n_frames': 133330,
        'max_frames': None,
        'x_off': 80,
        'y_off': 130,
        'scale': 0.565,
        'notes': ''
    },
    '2022_01_19_fly1': {
        'include_behavior': True,
        'include_neural': True,
        'n_frames': 134971,
        'max_frames': None,
        'x_off': 80,
        'y_off': 130,
        'scale': 0.486,
        'notes': ''
    },
    '2022_01_21_fly2': {
        'include_behavior': True,
        'include_neural': True,
        'n_frames': 129220,
        'max_frames': None,
        'x_off': 80,
        'y_off': 120,
        'scale': 0.451,
        'notes': ''
    },
    '2022_01_25_fly2': {
        'include_behavior': True,
        'include_neural': True,
        'n_frames': 129861,
        'max_frames': None,
        'x_off': 70,
        'y_off': 120,
        'scale': 0.620,
        'notes': ''
    },
}


def compute_me(file):
    from sklearn.decomposition import PCA
    from skimage.restoration import denoise_tv_chambolle
    video = load_video(file)
    video_diff = np.diff(np.reshape(video, (video.shape[0], -1), order='C'), axis=0)
    me = np.concatenate([[0], np.mean(np.square(video_diff), axis=1)])
    # smooth
    denoised = denoise_tv_chambolle(me, weight=10)
    denoised /= np.percentile(denoised, 99)
    new_file = file[:-4] + '.npy'
    np.save(new_file, denoised)


def load_video(file):
    """Helper function to load videos"""
    import cv2
    # read file
    cap = cv2.VideoCapture(file)
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


def crop_and_equalize(
        in_file, out_file, y_pix, x_pix, y_off, x_off, equalize=True, quality=4, scale=1,
        max_frames=None):

    if not os.path.exists(os.path.dirname(out_file)):
        os.makedirs(os.path.dirname(out_file))

    # spatial crop
    filter_str = str('crop=%i:%i:%i:%i' % (x_pix, y_pix, x_off, y_off))

    # equalization
    if equalize and scale < 1:
        filter_str += str(', curves=all=\'0/0 %f/1\'' % scale)

    # temporal crop
    if max_frames is not None:
        dur_str = '-to %f' % (max_frames / FPS)
    else:
        dur_str = ''

    # put it all together
    call_str = 'ffmpeg -i %s %s -vsync 0 -filter:v "%s" -q:v %i %s' % (
        in_file, dur_str, filter_str, quality, out_file)

    # run process
    print(call_str)
    subprocess.call(['/bin/bash', '-c', call_str])

    return None


def crop_and_compute_ball_me(
        in_file, out_file, y_pix, x_pix, y_off, x_off, quality=4, max_frames=None):

    if not os.path.exists(os.path.dirname(out_file)):
        os.makedirs(os.path.dirname(out_file))

    # spatial crop
    filter_str = str('crop=%i:%i:%i:%i, scale=128:-1' % (x_pix, y_pix, x_off, y_off))

    # temporal crop
    if max_frames is not None:
        dur_str = '-to %f' % (max_frames / FPS)
    else:
        dur_str = ''

    # put it all together
    call_str = 'ffmpeg -i %s %s -vsync 0 -filter:v "%s" -q:v %i %s' % (
        in_file, dur_str, filter_str, quality, out_file)

    # run process
    print(call_str)
    subprocess.call(['/bin/bash', '-c', call_str])

    # compute motion energy on cropped video
    compute_me(out_file)

    return None


def make_sample_videos_for_labeling(in_file, out_file, sample_start=0, sample_len=10, quality=4):

    if not os.path.exists(os.path.dirname(out_file)):
        os.makedirs(os.path.dirname(out_file))

    # temporal crop
    start_str = '-ss %i' % sample_start  # start time in seconds
    dur_str = '-t %i' % sample_len  # duation in seconds

    call_str = 'ffmpeg %s -i %s %s -vsync 0 -q:v %i %s' % (
        start_str, in_file, dur_str, quality, out_file)

    # run process
    print(call_str)
    subprocess.call(['/bin/bash', '-c', call_str])

    return None


def label_videos(
        in_file, out_file, proj_cfg_file, dgp_model_file, dgp_label_script, shuffle, gpu_id=0,
        sample_video=True):

    call_str = \
        str('source /home/mattw/anaconda3/etc/profile.d/conda.sh; ') + \
        str('conda activate dgp; python %s ' % dgp_label_script) + \
        str('--proj_cfg_file %s ' % proj_cfg_file) + \
        str('--dgp_model_file %s ' % dgp_model_file) + \
        str('--video_file %s ' % in_file) + \
        str('--output_dir %s ' % os.path.dirname(out_file)) + \
        str('--shuffle %i ' % shuffle) + \
        str('--gpu_id %i ' % gpu_id)

    if sample_video:
        call_str += '--sample'

    # run process
    print(call_str)
    subprocess.run(['/bin/bash', '-c', call_str], check=True)

    return None


def compute_heuristic_labels(
        expt_id, out_file, ball_me_dir, labels_dir, states_dir, state_ids=[],
        walk_thresh=0.5, still_thresh=0.05, groom_thresh=0.02, ab_thresh=0.9, fidget_thresh=0.5,
        create_syllable_movies=False, create_labeled_movies=False, version=None):

    if os.path.exists(out_file):
        print('%s\n%s already exists; skipping\n' % (expt_id, out_file))
        if create_syllable_movies or create_labeled_movies:
            with open(out_file, 'rb') as f:
                tmp = pickle.load(f)
            states = tmp['states']
            state_mapping = tmp['state_labels']
            n_t = states.shape[0]
            print('loading behavioral video...', end='')
            video = video_loader(expt_id)
            print('done')
    else:

        # load data
        labels_file = os.path.join(labels_dir, '%s_labeled.h5' % expt_id)
        preprocess_list = {
            'filter': {'type': 'savgol', 'window_size': 5, 'order': 2},
            'unitize': {},  # scale labels in [0, 1]
        }
        dlc_obj = preprocess_and_split_data(
            expt_id, preprocess_list, algo='dgp', load_from='h5', filenames=labels_file)
        labels = dlc_obj[0].get_label_array()

        print('loading behavioral video...', end='')
        video = video_loader(expt_id)
        print('done')

        print('loading ball motion energy...', end='')
        ball_me = np.load(os.path.join(ball_me_dir, expt_id + '.npy'))
        print('done')

        n_t = np.min([video.shape[0], labels.shape[0]])  # sometimes index is off by 1
        labels = labels[:n_t]
        ball_me = ball_me[:n_t]

        # perform segmentation
        if version is None or version == 5:
            states, state_mapping = heuristic_segmentation_v5(
                labels, ball_me, walk_thresh=walk_thresh, still_thresh=still_thresh,
                groom_thresh=groom_thresh, ab_thresh=ab_thresh, fidget_thresh=fidget_thresh,
                run_smoother=False)
        elif version == 4:
            states, state_mapping = heuristic_segmentation_v4(
                labels, ball_me, walk_thresh=walk_thresh, still_thresh=still_thresh,
                groom_thresh=groom_thresh, ab_thresh=ab_thresh, run_smoother=False)
        else:
            raise NotImplementedError

        # save data
        data = {'states': states, 'state_labels': state_mapping}
        if not os.path.exists(os.path.dirname(out_file)):
            os.makedirs(os.path.dirname(out_file))
        with open(out_file, 'wb') as f:
            print('saving states to %s' % out_file)
            pickle.dump(data, f)
            print('\n\n')

    # create syllable movie
    if create_syllable_movies:
        if len(state_ids) == 0:
            # put all in one video
            state_ids = [None]
        for state_id in state_ids:
            # split videos by behavior
            if state_id is not None:
                syll_str = state_mapping[state_id]
            else:
                syll_str = 'all'
            save_file = os.path.join(
                states_dir, 'syllable-videos', '%s_%s.mp4' % (expt_id, syll_str))
            if os.path.exists(save_file):
                print('%s\n%s already exists; skipping\n' % (expt_id, save_file))
                continue
            plotting.make_syllable_movie(
                save_file, [states], video, [np.arange(n_t)], single_state=state_id,
                min_threshold=70, n_pre_frames=0, n_buffer=10, plot_n_frames=1000)

    # create labeled movie
    if create_labeled_movies:
        pass
        # l = 1000
        # idxs_chunk = [0, 7, 10]  # , 17, 49, 50, 61, 72, 73]
        # for idx_chunk in idxs_chunk:
        #     save_file = os.path.join(
        #         states_dir, expt_id, str('labeled-video_%03i_wmarkers_v2.mp4' % idx_chunk))
        #     beg = idx_chunk * l
        #     idxs_ = np.arange(beg, beg + l)
        #     plotting.make_labeled_movie_wmarkers(
        #         save_file, states, video, labels, idxs_, state_mapping)
