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
            'data': '/media/mattw/data/flygenvectors/',  # base data dir
            'results': '/home/mattw/results/fly/'  # base results dir
        }
    else:
        raise ValueError(
            'must update flygenvectors.utils.get_dirs() to include user %s' % username)
    return dirs
