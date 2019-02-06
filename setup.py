from distutils.core import setup

setup(
    name='S2Vd3mWrapper',
    version='1.1.0',
    description='wrapper for interacting with a dockerized sent2vec text embedding primitive',
    author='New Knowledge',
    packages=['S2Vd3mWrapper'],
    include_package_data=True,
    install_requires=[
        'pandas>=0.22.0, <=0.23.4',
        'numpy>=1.14.1, <=1.15.4',
        'nose>=1.3.7',
        'd3m_sent2vec>=1.2.0'
    ],
    dependency_links=[
        "git+https://github.com/NewKnowledge/d3m_sent2vec@414ed6aeea7c1893c05f32ea00d4d899f9d49a4f#egg=d3m_sent2vec-1.2.0"
    ], 
    entry_points={
        'd3m.primitives': [
            'feature_extraction.nk_sent2vec.D3m_s2v = S2Vd3mWrapper:d3m_s2v'
        ],
    },
)
