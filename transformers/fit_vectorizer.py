if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test
from sklearn.feature_extraction import DictVectorizer


@transformer
def transform(data, *args, **kwargs):
    data = data['transform_taxi_data']
    print(data)
    categorical = ['PULocationID', 'DOLocationID']
    numerical = ['trip_distance']

    dv = DictVectorizer()

    train_dicts = data[categorical + numerical].to_dict(orient='records')
    X_train = dv.fit_transform(train_dicts)


    return X_train


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
