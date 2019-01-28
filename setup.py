from distutils.core import setup

setup(
    name='nk_sent2vec_wrapper',
    version='1.0.0',
    description='wrapper for interacting with a dockerized sent2vec text embedding primitive',
    author='New Knowledge',
    packages=['nk_sent2vec_wrapper'],
    include_package_data=True,
    install_requires=[
        'pandas>=0.22.0, <=0.23.4',
        'numpy>=1.14.1, <=1.15.4',
        'nose>=1.3.7',
        'nk_sent2vec>=1.1.0'
    ],
    dependency_links=[
        "git+https://github.com/NewKnowledge/nk-sent2vec@fc5acdf05bc76f5fd2dd7845f00e21413c18d4e2#egg=nk_sent2vec-1.1.0"
    ], 
    entry_points={
        'd3m.primitives': [
            'feature_extraction.nk_sent2vec.Nk_s2v = nk_sent2vec_wrapper:nk_s2v'
        ],
    },
)
