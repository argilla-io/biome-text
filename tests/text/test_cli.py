import pytest

from biome.text import Pipeline
from biome.text.cli.serve import _serve


@pytest.mark.skip("Please execute this test manually and check your localhost:9999")
def test_serve():
    """Needs to be automatized this test!"""
    pipeline = Pipeline.from_config(
        {
            "name": "serve_test",
            "head": {"type": "TextClassification", "labels": ["a", "b"]},
        }
    )

    _serve(pipeline)
