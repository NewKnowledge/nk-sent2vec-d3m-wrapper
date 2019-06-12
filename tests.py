from sent2vec_wrapper import Sent2Vec
import pandas as pd


def test():
    s2v = Sent2Vec(hyperparams={"target_columns": ["text"]})
    # make sure to read dataframe as string!
    # frame = pd.read_csv("https://query.data.world/s/10k6mmjmeeu0xlw5vt6ajry05",dtype='str')
    # input_df = pd.read_csv("https://s3.amazonaws.com/d3m-data/merged_o_data/o_4550_merged.csv", dtype='str')
    # print('input:', input_df)
    # print('input shape:', input_df.shape)
    # print('input cols:', input_df.columns)
    input_df = pd.read_csv("test.csv", header=None, names=["id", "text"])
    result = s2v.produce(inputs=input_df)
    print("output:", result)

    # # LOAD DATA AND PREPROCESSING
    # input_dataset = container.Dataset.load('file:///home/datasets/seed_datasets_current/185_baseball/185_baseball_dataset/datasetDoc.json')
    # ds2df_client = DatasetToDataFrame.DatasetToDataFramePrimitive(hyperparams={"dataframe_resource":"learningData"})
    # df = d3m_DataFrame(ds2df_client.produce(inputs=input_dataset).value)
    # hyperparams_class = rffeatures.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
    # client = rffeatures(hyperparams=hyperparams_class.defaults())
    # result = client.produce(inputs = df)
    # print(result.value)
