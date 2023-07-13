"""
This is a boilerplate pipeline 'train_model'
generated using Kedro 0.18.11
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import evaluate_model, split_data, train_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=split_data,
                inputs=["model_input", "params:split_params"],
                outputs=[
                    "X_train",
                    "X_valid",
                    "X_test",
                    "y_train",
                    "y_valid",
                    "y_test",
                ],
                name="split_data",
            ),
            node(
                func=train_model,
                inputs=["X_train", "X_valid", "y_train", "y_valid"],
                outputs="model",
                name="train_models",
            ),
            node(
                func=evaluate_model,
                inputs=["model", "X_test", "y_test"],
                outputs=None,
                name="Evaluate_model",
            ),
        ]
    )  # type: ignore
