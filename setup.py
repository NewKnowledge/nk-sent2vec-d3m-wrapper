# from setuptools import setup
from distutils.core import setup

setup(
    name='nk_sent2vec_wrapper',
    version='1.0.0',
    description='wrapper for interacting with a dockerized sent2vec text embedding primitive',
    author='New Knowledge',
    packages=['nk_sent2vec_wrapper'],
    include_package_data=True,
    install_requires=[
        'pandas>=0.22.0',
        'numpy>=1.14.1',
        'nose>=1.3.7',
        'nk_sent2vec>=1.0.0'
    ],
    dependency_links=[
        "git+https://github.com/NewKnowledge/nk-sent2vec#egg=nk_sent2vec-1.0.0"
    ],
    entry_points={
        'd3m.primitives': [
            'distil.nk_sent2vec = nk_sent2vec_wrapper:nk_sent2vec'
        ],
    },
)
