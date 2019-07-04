from distutils.core import setup

setup(
    name='flygenvectors',
    version='0.0.0',
    description='sandbox for analysis of whole-brain imaging in flies',
    author='Evan Schaffer',
    author_email='',
    url='http://www.github.com/schafferEvan/flygenvectors',
    install_requires=[
        'numpy', 'matplotlib', 'sklearn', 'scipy==1.1.0', 'jupyter', 'seaborn'],
    # dependency_links=[
        # 'http://github.com/slinderman/ssm/tarball/master#egg=ssm-0.0.1'],
        # 'git+ssh://git@github.com/slinderman/ssm.git#egg=ssm-master'],
        # 'git+git://github.com/slinderman/ssm@master#egg=ssm'],
    packages=['flygenvectors'],
)
