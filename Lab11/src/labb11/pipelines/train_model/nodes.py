"""
This is a boilerplate pipeline 'train_model'
generated using Kedro 0.18.11
"""

import logging
from datetime import datetime
from typing import Dict

import mlflow
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.svm import SVR
from xgboost import XGBRegressor


def split_data(data: pd.DataFrame, params: Dict):

    shuffled_data = data.sample(frac=1, random_state=params["random_state"])
    rows = shuffled_data.shape[0]

    train_ratio = params["train_ratio"]
    valid_ratio = params["valid_ratio"]

    train_idx = int(rows * train_ratio)
    valid_idx = train_idx + int(rows * valid_ratio)

    assert rows > valid_idx, "test split should not be empty"

    target = params["target"]
    X = shuffled_data.drop(columns=target)
    y = shuffled_data[[target]]

    X_train, y_train = X[:train_idx], y[:train_idx]
    X_valid, y_valid = X[train_idx:valid_idx], y[train_idx:valid_idx]
    X_test, y_test = X[valid_idx:], y[valid_idx:]

    return X_train, X_valid, X_test, y_train, y_valid, y_test


def get_best_model(experiment_id):
    runs = mlflow.search_runs(experiment_id)
    best_model_id = runs.sort_values("metrics.valid_mae")["run_id"].iloc[0]
    best_model = mlflow.sklearn.load_model("runs:/" + best_model_id + "/model")

    return best_model


def train_model(X_train, X_valid, y_train, y_valid):

    #  se definen e inicializan los modelos
    lreg = LinearRegression()
    rfor = RandomForestRegressor()
    svr = SVR()
    xgbr = XGBRegressor()
    lgbmr = LGBMRegressor()

    modelos = [lreg, rfor, svr, xgbr, lgbmr]

    #  nuevo experimento
    experiment_name = "experimento" + str(datetime.now())
    experiment_id = mlflow.create_experiment(experiment_name)

    for modelo in modelos:

        #  iniciamos una run por cada modelo
        with mlflow.start_run(experiment_id=experiment_id, run_name=str(modelo)):

            #  guardamos en mlflow los parametros del modelo
            mlflow.autolog(log_models=True, silent=True)

            #  entrenamos el modelo
            modelo.fit(X_train, y_train)
            y_pred = modelo.predict(X_valid)

            #  calculamos el MAE
            mae = mean_absolute_error(y_valid, y_pred)
            mlflow.log_metric("valid_mae", mae)
            mlflow.sklearn.log_model(modelo, "model")

        #  finalizamos el experimento
        mlflow.end_run()

    #  obtenemos el mejor modelo
    model = get_best_model(mlflow.get_experiment_by_name(experiment_name).experiment_id)

    return model


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series):
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    logger = logging.getLogger(__name__)
    logger.info(f"Model has a Mean Absolute Error of {mae} on test data.")
