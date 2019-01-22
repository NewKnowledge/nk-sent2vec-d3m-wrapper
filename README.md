# nk-sent2vec-wrapper

D3M wrapper for nk-sent2vec. The base libary for this wrapper can be found at [nk-sent2vec](https://github.com/NewKnowledge/nk-sent2vec).

```
from nk_sent2vec_wrapper import nk_s2v
import pandas as pd

docs = ['this is a test', 'this is a trap']
frame = pd.DataFrame(docs, columns=['sentences'])

nk_s2v(hyperparams={}).produce(inputs=frame)

```
