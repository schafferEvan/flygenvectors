import os


def get_user():
    """Get name of user running this function"""
    import pwd
    return pwd.getpwuid(os.getuid()).pw_name


def get_dirs():
    username = get_user()
    if username == 'evan':
        dirs = {
            'data': '',
            'results': ''
        }
    elif username == 'mattw':
        dirs = {
            # 'data': '/media/mattw/data/_flygenvectors',  # base data dir
            'data': '/media/mattw/fly/',  # base data dir
            'results': '/media/mattw/fly/arhmm_results'  # base results dir
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
