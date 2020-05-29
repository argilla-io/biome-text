from biome.text.modules.heads.classification.text_classification import (
    TextClassificationSpec,
)


def test_component_spec_config_with_type():
    head = TextClassificationSpec(
        pooler="boe",
        labels=[
            "toxic",
            "severe_toxic",
            "obscene",
            "threat",
            "insult",
            "identity_hate",
        ],
        multilabel=True,
    )

    assert "type" in head.config
    assert head.config["type"] == "TextClassification"
