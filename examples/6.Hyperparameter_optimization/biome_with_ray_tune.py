import json
import os
import shutil
from tempfile import mkdtemp

from ray import tune
from ray.tune import Trainable
from ray.tune.schedulers import ASHAScheduler
from ray.tune.result import DONE

from biome.text.api_new import Pipeline
from biome.text.api_new import TrainerConfiguration


class BiomeTrainable(Trainable):

    """Simple tune trainable class for biome pipelines hyperparameter optimization"""

    def _setup(self, config):
        env = config["env"]
        metric_name = env["metric_name"]
        self._trainer = TrainerConfiguration(
            validation_metric="+" if env["metric_mode"] == "max" else "-" + metric_name,
            optimizer={"lr": config.get("lr", 0.001), "type": "adam"},
            batch_size=config.get("batch_size", 32),
        )
        self._model_path = env["model_path"]
        self._training = env["training"]
        self._validation = env.get("validation")
        self._test = env.get("test")

        self._output = mkdtemp()
        self._metric = metric_name

        # Maybe here, vocabulary creation

    def _train(self):
        pipeline = Pipeline.from_file(self._model_path)
        pipeline.train(
            output=self._output,
            trainer=self._trainer,
            training=self._training,
            validation=self._validation,
            test=self._test,
        )

        with open(os.path.join(self._output, "metrics.json")) as metrics:
            data_metrics = json.load(metrics)
            return {
                self._metric: data_metrics.get(f"best_validation_{self._metric}"),
                DONE: True,
            }

    def _save(self, tmp_checkpoint_dir):
        checkpoint_path = os.path.join(tmp_checkpoint_dir, "model.tar.gz")
        shutil.copy(os.path.join(self._output, "model.tar.gz"), checkpoint_path)
        return checkpoint_path

    def _restore(self, checkpoint):
        return Pipeline.from_pretrained(checkpoint)


if __name__ == "__main__":

    metric = "loss"
    mode = "min"
    sched = ASHAScheduler(metric=metric, mode=mode)

    analisys = tune.run(
        BiomeTrainable,
        name="test_hpOpts",
        scheduler=sched,
        checkpoint_at_end=True,
        config={
            "env": {
                "model_path": "/Users/frascuchon/recognai/biome/biome-text/examples/1.text_classifier/text_classifier.yaml",
                "training": "/Users/frascuchon/recognai/biome/biome-text/examples/1.text_classifier/train.data.yml",
                "validation": "/Users/frascuchon/recognai/biome/biome-text/examples/1.text_classifier/validation.data.yml",
                "metric_name": metric,
                "metric_mode": mode,
            },
            "lr": 0.02,
            "batch_size": 16,
        },
    )

    print(analisys.get_best_config(metric=metric))
    print(analisys.get_best_trial(metric=metric))
