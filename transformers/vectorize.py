if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test
import mlflow
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error



mlflow.set_tracking_uri("http://mlflow:8012")



@transformer
def transform(data, *args, **kwargs):
    client = mlflow.MlflowClient()
    run = client.create_run(2)
    run = mlflow.start_run(run_id = run.info.run_id)


    categorical = ['PULocationID', 'DOLocationID']
    numerical = ['trip_distance']

    dv = DictVectorizer()

    train_dicts = data[categorical + numerical].to_dict(orient='records')
    X_train = dv.fit_transform(train_dicts)
    print('vectorizer made')

    target = 'duration'
    y_train = data[target].values


    lr = LinearRegression()
    lr.fit(X_train, y_train)
    print('model trained')
    y_train_pred = lr.predict(X_train)
    rmse_train = mean_squared_error(y_train, y_train_pred, squared=False)

    mlflow.log_metric("rmse", rmse_train)
    mlflow.log_metric("intercept", lr.intercept_)
    mlflow.sklearn.log_model(lr, "model")
    mlflow.sklearn.log_model(dv, "vectorizer")
    mlflow.end_run()

    return dv, lr


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
