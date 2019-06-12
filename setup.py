from distutils.core import setup

setup(
    name="sent2vec_wrapper",
    version="1.3.0",
    description="wrapper for interacting with a dockerized sent2vec text embedding primitive",
    author="New Knowledge",
    packages=["sent2vec_wrapper"],
    include_package_data=True,
    install_requires=[
        "pandas",
        "numpy",
        "pytest",
        "nk_sent2vec @ git+https://github.com/NewKnowledge/nk-sent2vec@85cdd7538c41ea8edf49d15ab749d258656eff00#egg=nk_sent2vec",
    ],
    # TODO point to version above
    entry_points={
        "d3m.primitives": [
            "feature_extraction.nk_sent2vec.Sent2Vec = sent2vec_wrapper:Sent2Vec"
        ]
    },
)
