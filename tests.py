from .wrapper import nk_sent2vec
import pandas as pd


def test():
    s2v = nk_sent2vec(hyperparams={'target_columns': ['text']})
    # make sure to read dataframe as string!
    # frame = pd.read_csv("https://query.data.world/s/10k6mmjmeeu0xlw5vt6ajry05",dtype='str')
    # input_df = pd.read_csv("https://s3.amazonaws.com/d3m-data/merged_o_data/o_4550_merged.csv", dtype='str')
    # print('input:', input_df)
    # print('input shape:', input_df.shape)
    # print('input cols:', input_df.columns)
    input_df = pd.read_csv("test.csv", header=None, names=['id', 'text'])
    result = s2v.produce(inputs=input_df)
    print('output:', result)
