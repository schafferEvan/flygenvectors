import os


def get_user():
    """Get name of user running this function"""
    import pwd
    return pwd.getpwuid(os.getuid()).pw_name


def get_dirs():
    username = get_user()
    if username == 'evan':
        dirs = {
            'data': '/Users/evan/Dropbox/_AxelLab/__flygenvectors/dataShare/_main/',
            'results': '/Users/evan/Dropbox/_AxelLab/__flygenvectors/figs/'
        }
    elif username == 'mattw':
        dirs = {
            # 'data': '/media/mattw/data/_flygenvectors',  # base data dir
            'data': '/media/mattw/fly/',  # base data dir
            'results': '/media/mattw/fly/behavior/classifiers'  # base results dir
        }
    else:
        raise ValueError(
            'must update flygenvectors.utils.get_dirs() to include user %s' % username)
    return dirs


def get_subdirs(path):
    """Get all first-level subdirectories in a given path (no recursion).

    Parameters
    ----------
    path : :obj:`str`
        absolute path

    Returns
    -------
    :obj:`list`
        first-level subdirectories in :obj:`path`

    """
    if not os.path.exists(path):
        raise ValueError('%s is not a path' % path)
    try:
        s = next(os.walk(path))[1]
    except StopIteration:
        raise StopIteration('%s does not contain any subdirectories' % path)
    if len(s) == 0:
        raise StopIteration('%s does not contain any subdirectories' % path)
    return s


def get_fig_dirs(exp_id, d=None):
    if d is None:
        d = get_dirs()
    pkl_dir = d['data']+exp_id+'/models/'
    fig_folder = d['results']+exp_id+'/'
    clustfig_folder = fig_folder+'clusters/'
    regfig_folder = fig_folder+'regmodel/'
    pcfig_folder = fig_folder+'reg_pcs/'
    if not os.path.exists(fig_folder):
        os.mkdir(fig_folder)
    if not os.path.exists(clustfig_folder):
        os.mkdir(clustfig_folder)
    if not os.path.exists(regfig_folder):
        os.mkdir(regfig_folder)
    if not os.path.exists(pkl_dir):
        os.mkdir(pkl_dir)
    if not os.path.exists(pcfig_folder):
        os.mkdir(pcfig_folder)
    return {'pkl_dir':pkl_dir, 'fig_folder':fig_folder, 'clustfig_folder':clustfig_folder, 
            'regfig_folder':regfig_folder, 'pcfig_folder':pcfig_folder}





