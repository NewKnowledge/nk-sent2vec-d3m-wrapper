from distutils.core import setup

setup(
    name="sent2vec",
    version="1.3.0",
    description="wrapper for interacting with a dockerized sent2vec text embedding primitive",
    author="New Knowledge",
    packages=["sent2vec"],
    include_package_data=True,
    install_requires=[
        # "pandas>=0.22.0, <=0.23.4",
        # "numpy>=1.14.1, <=1.15.4",
        # "pytest>=1.3.7",
        "pandas",
        "numpy",
        "pytest",
        "nk_sent2vec @ git+https://github.com/NewKnowledge/nk-sent2vec@8976b22e6c86843626d81a2e0d08cb84d1e041ca#egg=nk_sent2vec",
    ],
    # TODO point to version above
    entry_points={
        "d3m.primitives": [
            "feature_extraction.nk_sent2vec.Sent2Vec = sent2vec:Sent2Vec"
        ]
    },
)
