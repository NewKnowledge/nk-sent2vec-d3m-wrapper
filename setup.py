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
        'nk_sent2vec>=1.1.0'
    ],
    dependency_links=[
        "git+https://github.com/NewKnowledge/nk-sent2vec@78d3e79ab6f59e2fac65581bb560e19fc164d640#egg=nk_sent2vec-1.1.0"
    ], 
    entry_points={
        'd3m.primitives': [
            'distil.nk_sent2vec = nk_sent2vec_wrapper:nk_s2v'
        ],
    },
)
