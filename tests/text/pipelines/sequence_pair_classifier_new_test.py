import json
import multiprocessing
import os
import tempfile
from time import sleep
from typing import Optional

import pytest
import requests

from biome.text import Pipeline
from biome.text.defs import PipelineDefinition
from biome.text.environment import ES_HOST
from biome.text.pipelines._impl.allennlp.classifier.pipeline import (
    AllenNlpTextClassifierPipeline,
)
from biome.text.pipelines.defs import ExploreConfig, ElasticsearchConfig
from tests import DaskSupportTest
from tests.test_context import TEST_RESOURCES

BASE_CONFIG_PATH = os.path.join(
    TEST_RESOURCES, "resources/models/sequence_pair_classifier"
)


class SequencePairClassifierTest(DaskSupportTest):
    output_dir = tempfile.mkdtemp()
    # output_dir = os.path.join(BASE_CONFIG_PATH, "experiment")
    model_archive = os.path.join(output_dir, "model.tar.gz")

    name = "sequence_pair_classifier"

    model_path = os.path.join(BASE_CONFIG_PATH, "model.new.yml")
    trainer_path = os.path.join(BASE_CONFIG_PATH, "trainer.new.yml")
    training_data = os.path.join(BASE_CONFIG_PATH, "train.data.new.yml")
    validation_data = os.path.join(BASE_CONFIG_PATH, "validation.data.new.yml")

    def test_model_workflow(self):
        self.check_train()
        # Check explore metadata override
        self.check_predictions()
        self.check_explore(
            extra_metadata=dict(model="other-model", project="test-project")
        )
        self.check_serve()

    def check_train(self):
        classifier = AllenNlpTextClassifierPipeline.from_config(
            PipelineDefinition.from_file(self.model_path)
        )

        classifier.learn(
            trainer=self.trainer_path, train=self.training_data, output=self.output_dir
        )
        classifier.learn(
            trainer=self.trainer_path,
            train=self.training_data,
            validation=self.validation_data,
            output=self.output_dir,
        )

        self.assertTrue(classifier._predictor is not None)

        prediction = classifier.predict(
            record1={"a": "mike", "b": "Farrys"}, record2={"a": "mike", "b": "Farrys"}
        )
        self.assertTrue("logits" in prediction, f"Not in {prediction}")

    def check_explore(self, extra_metadata: Optional[dict] = None):
        index = self.name
        es_host = os.getenv(ES_HOST, "http://localhost:9200")
        pipeline = AllenNlpTextClassifierPipeline.load(self.model_archive)
        es_config = ElasticsearchConfig(es_host=es_host, es_index=index)
        pipeline.explore(
            ds_path=self.validation_data,
            config=ExploreConfig(prediction_cache_size=100, **extra_metadata or {}),
            es_config=es_config,
        )

        data = es_config.client.search(index)
        self.assertIn("hits", data, msg=f"Must exist hits in es response {data}")
        self.assertTrue(len(data["hits"]) > 0, "No data indexed")

    def check_serve(self):
        port = 18000
        output_dir = os.path.join(self.output_dir, "predictions")

        pipeline = AllenNlpTextClassifierPipeline.load(self.model_archive)

        process = multiprocessing.Process(
            target=pipeline.serve,
            daemon=True,
            kwargs=dict(port=port, predictions=output_dir),
        )
        process.start()
        sleep(5)

        response = requests.post(
            f"http://localhost:{port}/predict",
            json={
                "record1": "Herbert Brandes-Siller",
                "record2": "Herbert Brandes-Siller",
            },
        )
        self.assertTrue(response.json() is not None)
        assert os.path.isfile(os.path.join(output_dir, Pipeline.PREDICTION_FILE_NAME))

        process.terminate()

    def check_predictions(self):
        pipeline = AllenNlpTextClassifierPipeline.load(self.model_archive)
        inputs = [
            {"record1": "Herbert Brandes-Siller", "record2": "Herbert Brandes-Siller"},
            {
                "record1": ["Herbert", "Brandes-Siller"],
                "record2": "Herbert Brandes-Siller",
            },
        ]

        def test_single_input():
            for inputs_i in inputs:
                result = pipeline.predict(**inputs_i)
                classes = result.get("classes")

                for the_class in ["duplicate", "not_duplicate"]:
                    self.assertIn(the_class, classes)

                self.assertTrue(all(prob > 0 for _, prob in classes.items()))

        def test_cached_predictions():
            pipeline.init_prediction_cache(1)
            with pytest.warns(RuntimeWarning):
                pipeline.init_prediction_cache(1)

            for i, inputs_i in enumerate(inputs):
                pipeline.init_prediction_cache(1)
                pipeline.predict(**inputs_i)
                pipeline.predict(**inputs_i)

                # for every input we get one more hit in the cache_info stats
                assert pipeline._predict_with_cache.cache_info()[0] == i + 1

        def test_prediction_logger():
            output_dir = os.path.join(self.output_dir, "predictions")

            pipeline.init_prediction_logger(output_dir)
            pipeline.predict(**inputs[0])

            output_file = os.path.join(output_dir, Pipeline.PREDICTION_FILE_NAME)
            with open(output_file) as file:
                json_dict = json.loads(file.readlines()[-1])
                assert all([key in json_dict for key in ["inputs", "annotation"]])

        def test_input_that_make_me_cry():
            self.assertRaises(
                Exception,
                pipeline.predict,
                {"label": "duplicate", "record1": "Herbert Brandes-Siller"},
            )

        test_single_input()
        test_input_that_make_me_cry()
        test_cached_predictions()
        test_prediction_logger()
