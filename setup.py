from setuptools import setup

setup(
    name='nk_sent2vec_wrapper',
    version='1.0.0',
    description='Embeds text documents using sent2vec',
    author='New Knowledge',
    packages=['nk_sent2vec_wrapper'],
    include_package_data=True,
    install_requires=[
        # 'Cython>=0.28.3',
        # 'numpy>=1.14.1',
        'nose>=1.3.7',
    ],
)
