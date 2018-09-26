import os.path
import numpy as np
import pandas as pd
import pickle
# import requests
# import ast
import typing
from json import JSONDecoder
from typing import List
import sys

from nk_sent2vec import Sent2Vec

from d3m.primitive_interfaces.base import PrimitiveBase, CallResult

from d3m import container, utils
from d3m.metadata import hyperparams, base as metadata_base, params

__author__ = 'Distil'
__version__ = '1.0.0'

Inputs = container.pandas.DataFrame
Outputs = container.pandas.DataFrame


class Params(params.Params):
    pass


class Hyperparams(hyperparams.Hyperparams):
    target_columns = hyperparams.Set(
        elements=hyperparams.Hyperparameter[str](''),
        default=(),
        max_size=sys.maxsize,
        min_size=0,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description='names of columns with strings to embed'
    )


class nk_s2v(PrimitiveBase[Inputs, Outputs, Params, Hyperparams]):
    metadata = metadata_base.PrimitiveMetadata({
        # Simply an UUID generated once and fixed forever. Generated using "uuid.uuid4()".
        'id': "cf450079-9333-4a3f-aed4-b77a4e8c7be7",
        'version': __version__,
        'name': "nk_sent2vec",
        # Keywords do not have a controlled vocabulary. Authors can put here whatever they find suitable.
        'keywords': ['Sent2Vec', 'Embedding', 'NLP', 'Natural Language Processing'],
        'source': {
            'name': __author__,
            'uris': [
                # Unstructured URIs.
                "https://github.com/NewKnowledge/nk-sent2vec-d3m-wrapper",
            ],
        },
        # A list of dependencies in order. These can be Python packages, system packages, or Docker images.
        # Of course Python packages can also have their own dependencies, but sometimes it is necessary to
        # install a Python package first to be even able to run setup.py of another package. Or you have
        # a dependency which is not on PyPi.
        'installation': [
            {
                'type': metadata_base.PrimitiveInstallationType.PIP,  # TODO value for egg= below?
                'package_uri': 'git+https://github.com/NewKnowledge/nk-sent2vec.git@8878a622df9f01c15d3cdf1f9be23e909d9438ac#egg=nk_sent2vec'
            },
            {
                'type': metadata_base.PrimitiveInstallationType.PIP,  # TODO value for egg= below?
                'package_uri': 'git+https://github.com/NewKnowledge/nk-sent2vec-d3m-wrapper.git@{git_commit}#egg=nk_sent2vec_wrapper'.format(
                    git_commit=utils.current_git_commit(os.path.dirname(__file__)),
                ),
            }
        ],
        # The same path the primitive is registered with entry points in setup.py.
        'python_path': 'd3m.primitives.distil.nk_sent2vec',
        # Choose these from a controlled vocabulary in the schema. If anything is missing which would
        # best describe the primitive, make a merge request.
        'algorithm_types': [
            metadata_base.PrimitiveAlgorithmType.CONVOLUTIONAL_NEURAL_NETWORK,  # TODO
        ],
        'primitive_family': metadata_base.PrimitiveFamily.DATA_CLEANING,  # TODO
    })

    def __init__(self, *, hyperparams: Hyperparams, random_seed: int = 0)-> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed)

        self._decoder = JSONDecoder()
        self._params = {}

    def fit(self) -> None:
        pass

    def get_params(self) -> Params:
        return self._params

    def set_params(self, *, params: Params) -> None:
        self.params = params

    def set_training_data(self, *, inputs: Inputs, outputs: Outputs) -> None:
        pass

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        """
        Produce primitive's best guess for the structural type of each input column.

        Parameters
        ----------
        inputs : Input pandas dataframe

        Returns
        -------
        Outputs
            The output is a pandas dataframe
        """
        
        frame = inputs
      
        try:
            vectorizer = Sent2Vec(path='/home/nk-sent2vec-d3m-wrapper/models/torontobooks_unigrams.bin')
            frame = frame.ix[:,0].tolist()
            EmbedSentences = vectorizer.embed_sentences(sentences=frame)
            # print(EmbedSentences)
            index = ['Sentence'+str(i) for i in range(1, len(EmbedSentences)+1)]
            df_output = pd.DataFrame(EmbedSentences, index=index)
            
            # return df_output
        except:
            # Should probably do some more sophisticated error logging here
            return "Failed document embedding"
    
if __name__ == '__main__':
    client = nk_s2v(hyperparams={})
    # make sure to read dataframe as string!
    # frame = pd.read_csv("https://s3.amazonaws.com/d3m-data/merged_o_data/o_4550_merged.csv",dtype='str')
    docs = ['this is a test', 'this is a trap']
    frame = pd.DataFrame(docs, columns=['sentences'])
    result = client.produce(inputs = frame)
    print(result)
