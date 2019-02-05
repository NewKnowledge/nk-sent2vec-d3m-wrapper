import os.path
import numpy as np
import pandas as pd
import pickle
import typing
from json import JSONDecoder
from typing import List
import sys

from d3m_sent2vec import Sent2Vec

from d3m import container, utils
from d3m.primitive_interfaces.transformer import TransformerPrimitiveBase
from d3m.primitive_interfaces.base import CallResult

from d3m.container import DataFrame as d3m_DataFrame
from d3m.metadata import hyperparams, base as metadata_base, params
# from common_primitives import dataset_to_dataframe as DatasetToDataFrame

__author__ = 'Distil'
__version__ = '1.1.0'
__contact__ = 'mailto:nklabs@newknowledge.com'

Inputs = container.pandas.DataFrame
Outputs = container.pandas.DataFrame

class Hyperparams(hyperparams.Hyperparams):
    target_columns = hyperparams.Set(
        elements=hyperparams.Hyperparameter[str](''),
        default=(),
        max_size=sys.maxsize,
        min_size=0,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description='names of columns with strings to embed'
    )


class d3m_s2v(TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
    """
        Produce numerical representations (features) for short texts or sentences.

        Parameters
        ----------
        inputs : Input pandas dataframe

        Returns
        -------
        Outputs
            The output is a pandas dataframe
        """
    metadata = metadata_base.PrimitiveMetadata({
        # Simply an UUID generated once and fixed forever. Generated using "uuid.uuid4()".
        'id': "cf450079-9333-4a3f-aed4-b77a4e8c7be7",
        'version': __version__,
        'name': "nk_sent2vec",
        # Keywords do not have a controlled vocabulary. Authors can put here whatever they find suitable.
        'keywords': ['Sent2Vec', 'Embedding', 'NLP', 'Natural Language Processing'],
        'source': {
            'name': __author__,
            'contact': __contact__,
            'uris': [
                # Unstructured URIs.
                "https://github.com/NewKnowledge/d3m_sent2vec",
            ],
        },
        # A list of dependencies in order. These can be Python packages, system packages, or Docker images.
        # Of course Python packages can also have their own dependencies, but sometimes it is necessary to
        # install a Python package first to be even able to run setup.py of another package. Or you have
        # a dependency which is not on PyPi.
        'installation': [
            {
                'type': metadata_base.PrimitiveInstallationType.PIP,  
                'package_uri': 'git+https://github.com/NewKnowledge/sent2vec-d3m-wrapper.git@{git_commit}#egg=S2Vd3mWrapper'.format(
                    git_commit=utils.current_git_commit(os.path.dirname(__file__)),
                ),
            },
            
            {
                "type": "FILE",
                "key": "d3m_sent2vec_model",
                "file_uri": "http://public.datadrivendiscovery.org/twitter_bigrams.bin",
                "file_digest":"9e8ccfea2aaa4435ca61b05b11b60e1a096648d56fff76df984709339f423dd6"
        },
        ],
        # The same path the primitive is registered with entry points in setup.py.
        'python_path': 'd3m.primitives.feature_extraction.nk_sent2vec.D3m_s2v',
        # Choose these from a controlled vocabulary in the schema. If anything is missing which would
        # best describe the primitive, make a merge request.
        'algorithm_types': [
            metadata_base.PrimitiveAlgorithmType.RECURRENT_NEURAL_NETWORK, 
        ],
        'primitive_family': metadata_base.PrimitiveFamily.FEATURE_EXTRACTION,
    })

    def __init__(self, *, hyperparams: Hyperparams, random_seed: int = 0, volumes: typing.Dict[str,str]=None)-> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed, volumes=volumes)

        self._decoder = JSONDecoder()
        self.volumes = volumes

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        """
        Produce numerical representations (features) for short texts or sentences.

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
            vectorizer = Sent2Vec(path=self.volumes["d3m_sent2vec_model"])
            frame = frame.ix[:,0].tolist()
            EmbedSentences = vectorizer.embed_sentences(sentences=frame)
            # print(EmbedSentences)
            index = ['Sentence'+str(i) for i in range(1, len(EmbedSentences)+1)]
            df_output = pd.DataFrame(EmbedSentences, index=index)
            
            return df_output
        except:
            # Should probably do some more sophisticated error logging here
            return "Failed document embedding"
    
if __name__ == '__main__':
    volumes = {} # d3m large primitive architecture dictionary of large files
    volumes["d3m_sent2vec_model"]='/home/twitter_bigrams.bin'
    docs = ['this is a test', 'this is a trap']
    frame = pd.DataFrame(docs, columns=['sentences'])
    df = d3m_DataFrame(frame)  
    client = d3m_s2v(hyperparams={}, volumes=volumes)
    result = client.produce(inputs = df)
    print(result)
