"""
This is a boilerplate pipeline 'data_prep'
generated using Kedro 0.18.11
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    create_model_input_table,
    get_data,
    preprocess_companies,
    preprocess_shuttles,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=get_data,
                inputs=None,
                outputs=["companies", "shuttles", "reviews"],
                name="get_raw_data",
            ),
            node(
                func=preprocess_companies,
                inputs="companies",
                outputs="companies_preprocessed",
                name="preoprocess_companies",
            ),
            node(
                func=preprocess_shuttles,
                inputs="shuttles",
                outputs="shuttles_preprocessed",
                name="preprocess_shuttles",
            ),
            node(
                func=create_model_input_table,
                inputs=["shuttles_preprocessed", "companies_preprocessed", "reviews"],
                outputs="model_input",
                name="Model_input_table",
            ),
        ]
    )  # type: ignore
