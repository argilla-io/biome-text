from typing import Any, List, Optional, Union

from astroid import Instance

from biome.text import Pipeline, PipelineConfiguration
from biome.text.configuration import FeaturesConfiguration
from biome.text.modules.heads import TaskHeadSpec, TextClassification


class MyCustomHead(TextClassification):
    """Just a head renaming the original TextClassification head"""

    def inputs(self) -> Optional[List[str]]:
        return ["text", "second_text"]

    def featurize(
            self,
            text: Any,
            second_text: Optional[Any] = None,
            label: Optional[Union[int, str, List[Union[int, str]]]] = None
    ) -> Optional[Instance]:
        instance = self.backbone.featurizer(
            {"text": text, "text-2": second_text},
            to_field=self.forward_arg_name,
            aggregate=True,
            exclude_record_keys=True,
        )
        return self.add_label(instance, label, to_field=self.label_name)


def test_check_pipeline_inputs_and_output():
    config = PipelineConfiguration(
        "test-pipeline",
        features=FeaturesConfiguration(),
        head=TaskHeadSpec(
            type=MyCustomHead,
            labels=[
                "blue-collar",
                "technician",
                "management",
                "services",
                "retired",
                "admin.",
            ],
        ),
    )

    pipeline = Pipeline.from_config(config)

    assert pipeline.inputs == ["text", "second_text"]
    assert pipeline.output == "label"