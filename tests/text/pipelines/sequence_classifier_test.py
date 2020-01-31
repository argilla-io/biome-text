import multiprocessing
import os
import tempfile
from time import sleep
from typing import Optional

import pandas as pd
import pytest
import requests
import yaml
from elasticsearch import Elasticsearch

from biome.text import Pipeline
from biome.text.commands.explore.explore import explore
from biome.text.commands.serve.serve import serve
from biome.text.environment import ES_HOST
from biome.text.pipelines.sequence_classifier import SequenceClassifier
from tests import DaskSupportTest
from tests.test_context import TEST_RESOURCES

BASE_CONFIG_PATH = os.path.join(TEST_RESOURCES, "resources/models/sequence_classifier")


@pytest.fixture
def training_data_yaml(tmpdir):
    data_file = tmpdir.join("sentences.csv")
    df = pd.DataFrame(
        {
            "tokens": ["Two simple sentences. Split by a dot.", "One simple sentence."],
            "label": [1, 0],
        }
    )
    df.to_csv(data_file, index=False)

    yaml_file = tmpdir.join("training.yml")
    yaml_dict = {
        "source": str(data_file),
        "mapping": {"tokens": "tokens", "label": "label"},
    }
    with yaml_file.open("w") as f:
        yaml.safe_dump(yaml_dict, f)
    return str(yaml_file)


@pytest.fixture
def pipeline_yaml(tmpdir):
    yaml_dict = {
        "type": "sequence_classifier",
        "pipeline": {
            "token_indexers": {"tokens": {"type": "single_id"}},
            "segment_sentences": True,
        },
        "architecture": {
            "text_field_embedder": {
                "tokens": {"type": "embedding", "embedding_dim": 2}
            },
            "seq2vec_encoder": {
                "type": "gru",
                "input_size": 2,
                "hidden_size": 2,
                "bidirectional": False,
            },
            "multifield_seq2vec_encoder": {
                "type": "gru",
                "input_size": 2,
                "hidden_size": 2,
                "bidirectional": False,
            },
        },
    }

    yaml_file = tmpdir.join("pipeline.yml")
    with yaml_file.open("w") as f:
        yaml.safe_dump(yaml_dict, f)

    return str(yaml_file)


@pytest.fixture
def trainer_yaml(tmpdir):
    yaml_dict = {
        "iterator": {
            "batch_size": 2,
            "cache_instances": True,
            "max_instances_in_memory": 2,
            "sorting_keys": [["tokens", "num_fields"]],
            "type": "bucket",
        },
        "trainer": {
            "type": "default",
            "cuda_device": -1,
            "num_serialized_models_to_keep": 1,
            "num_epochs": 1,
            "optimizer": {"type": "adam", "amsgrad": True, "lr": 0.01},
        },
    }

    yaml_file = tmpdir.join("trainer.yml")
    with yaml_file.open("w") as f:
        yaml.safe_dump(yaml_dict, f)

    return str(yaml_file)


def test_segment_sentences(
    training_data_yaml, pipeline_yaml, trainer_yaml, tmpdir, tmpdir_factory
):
    pipeline = SequenceClassifier.from_config(pipeline_yaml)

    pipeline.learn(
        trainer=trainer_yaml,
        train=training_data_yaml,
        validation="",
        output=str(tmpdir.join("output")),
    )


class SequenceClassifierTest(DaskSupportTest):
    output_dir = tempfile.mkdtemp()
    model_archive = os.path.join(output_dir, "model.tar.gz")

    name = "sequence_classifier"

    model_path = os.path.join(BASE_CONFIG_PATH, "model.yml")
    trainer_path = os.path.join(BASE_CONFIG_PATH, "trainer.yml")
    training_data = os.path.join(BASE_CONFIG_PATH, "train.data.yml")
    validation_data = os.path.join(BASE_CONFIG_PATH, "validation.data.yml")

    def test_model_workflow(self):
        self.check_train()
        # Check explore metadata override
        self.check_explore(
            extra_metadata=dict(model="other-model", project="test-project")
        )
        self.check_serve()
        self.check_predictor()

    def check_train(self):
        classifier = Pipeline.from_config(self.model_path)
        self.assertIsInstance(classifier, SequenceClassifier)

        # learn without validation
        classifier.learn(
            trainer=self.trainer_path, train=self.training_data, output=self.output_dir
        )

        classifier.learn(
            trainer=self.trainer_path,
            train=self.training_data,
            validation=self.validation_data,
            output=self.output_dir,
        )

        self.assertTrue(classifier.model is not None)

        prediction = classifier.predict("mike Farrys")
        self.assertTrue("logits" in prediction, f"Not in {prediction}")

    def check_explore(self, extra_metadata: Optional[dict] = None):
        index = self.name
        es_host = os.getenv(ES_HOST, "http://localhost:9200")
        explore(
            binary=self.model_archive,
            source_path=self.validation_data,
            es_host=es_host,
            es_index=index,
            interpret=True,  # Enable interpret
            **extra_metadata or {},
        )

        client = Elasticsearch(hosts=es_host, http_compress=True)
        data = client.search(index)
        self.assertIn("hits", data, msg=f"Must exist hits in es response {data}")
        self.assertTrue(len(data["hits"]) > 0, "No data indexed")

    def check_serve(self):
        port = 18000
        process = multiprocessing.Process(
            target=serve, daemon=True, kwargs=dict(binary=self.model_archive, port=port)
        )
        process.start()
        sleep(5)

        response = requests.post(
            f"http://localhost:{port}/predict", json={"tokens": "Mike Farrys"}
        )
        self.assertTrue(response.json() is not None)
        process.terminate()

    def check_predictor(self):
        predictor = SequenceClassifier.load(self.model_archive)

        def test_batch_input():
            inputs = [{"tokens": "Herbert Brandes-Siller", "label": "duplicate"}]

            results = predictor.predict_batch_json(inputs)
            result = results[0]
            classes = result.get("classes")

            for the_class in ["duplicate", "not_duplicate"]:
                self.assertIn(the_class, classes)

            self.assertTrue(all(prob > 0 for _, prob in classes.items()))
            self.assertEqual(1, len(results))

        def test_label_input():
            inputs = {"tokens": "Herbert Brandes-Siller", "label": "duplicate"}

            result = predictor.predict_json(inputs)
            classes = result.get("classes")

            for the_class in ["duplicate", "not_duplicate"]:
                self.assertIn(the_class, classes)

            self.assertTrue(all(prob > 0 for _, prob in classes.items()))

        def test_input_that_make_me_cry():
            self.assertRaises(
                Exception,
                predictor.predict_json,
                {"label": "duplicate", "record1": "Herbert Brandes-Siller"},
            )

        test_batch_input()
        test_input_that_make_me_cry()
        test_label_input()
