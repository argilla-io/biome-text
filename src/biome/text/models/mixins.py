from typing import Optional, Dict
import torch
import numpy as np
from allennlp.training.metrics import CategoricalAccuracy, F1Measure


class BiomeClassifierMixin:
    """A mixin class for biome classifiers.

    Inheriting from this class allows you to use Biome's awesome UIs.
    It standardizes the `decode` and `get_metrics` methods.
    Some stuff to be aware of:
    - make sure your forward's output_dict has a "class_probability" key
    - use the `_biome_classifier_metrics` dict in the forward method to record the metrics
    - the forward signature must be compatible with the text_to_instance method of your DataReader
    - the `decode` and `get_metrics` methods override the allennlp.models.model.Model methods

    Parameters
    ----------
    vocab
        Used to initiate the F1 measures for each label. It is also passed on to the model.
    accuracy
        The accuracy you want to use. By default, we choose a categorical top-1 accuracy.
    kwargs
        Passed on to the model class init

    Examples
    --------
    An example of how to implement an AllenNLP model in biome-text to be able to use Biome's UIs:
    >>> from allennlp.models.bert_for_classification import BertForClassification
    >>>
    >>> @Model.register("biome_bert_classifier")
    >>> class BiomeBertClassifier(BiomeClassifierMixin, BertForClassification):
    >>>     def __init__(self, vocab, bert_model, num_labels, index, label_namespace,
    >>>                  trainable, initializer, regularizer, accuracy):
    >>>         super().__init__(accuracy=accuracy, vocab=vocab, bert_model=bert_model, num_labels=num_labels,
    >>>                          index=index, label_namespace=label_namespace, trainable=trainable,
    >>>                          initializer=initializer, regularizer=regularizer)
    >>>
    >>>     @overrides
    >>>     def forward(self, tokens, label = None):
    >>>         output_dict = super().forward(tokens=tokens, label=label)
    >>>         output_dict["class_probabilities"] = output_dict.pop("probs")
    >>>         if label is not None:
    >>>             for metric in self._biome_classifier_metrics.values():
    >>>                 metric(logits, label)
    >>>         return output_dict
    """

    def __init__(self, vocab, accuracy: Optional[CategoricalAccuracy] = None, **kwargs):
        self.vocab = vocab
        super().__init__(vocab=vocab, **kwargs)

        # metrics, some AllenNLP models use the names _accuracy or _metrics, so we have to be more specific.
        self._biome_classifier_metrics = {"accuracy": accuracy or CategoricalAccuracy()}
        self._biome_classifier_metrics.update(
            {
                label: F1Measure(index)
                for index, label in self.vocab.get_index_to_token_vocabulary(
                    "labels"
                ).items()
            }
        )

    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Does a simple position-wise argmax over each token, converts indices to string labels, and
        adds a ``"tags"`` key to the dictionary with the result.
        """
        all_predictions = output_dict["class_probabilities"]
        if not isinstance(all_predictions, np.ndarray):
            all_predictions = all_predictions.data.numpy()

        output_map_probs = []
        max_classes = []
        max_classes_prob = []
        for i, probs in enumerate(all_predictions):
            argmax_i = np.argmax(probs)
            label = self.vocab.get_token_from_index(argmax_i, namespace="labels")
            label_prob = 0.0

            output_map_probs.append({})
            for j, prob in enumerate(probs):
                label_key = self.vocab.get_token_from_index(j, namespace="labels")
                output_map_probs[i][label_key] = prob
                if label_key == label:
                    label_prob = prob

            max_classes.append(label)
            max_classes_prob.append(label_prob)

        return_dict = {
            "logits": output_dict.get("logits"),
            "classes": output_map_probs,
            "max_class": max_classes,
            "max_class_prob": max_classes_prob,
        }
        # having loss == None in dict (when no label is present) fails
        if "loss" in output_dict.keys():
            return_dict["loss"] = output_dict.get("loss")
        return return_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        """Get the metrics of our classifier, see :func:`~allennlp_2.models.Model.get_metrics`.

        Parameters
        ----------
        reset
            Reset the metrics after obtaining them?

        Returns
        -------
        A dictionary with all metric names and values.
        """
        all_metrics = {}

        total_f1 = 0.0
        total_precision = 0.0
        total_recall = 0.0
        for metric_name, metric in self._biome_classifier_metrics.items():
            if metric_name == "accuracy":
                all_metrics["accuracy"] = metric.get_metric(reset)
            else:
                precision, recall, f1 = metric.get_metric(
                    reset
                )  # pylint: disable=invalid-name
                total_f1 += f1
                total_precision += precision
                total_recall += recall
                all_metrics[metric_name + "/f1"] = f1
                all_metrics[metric_name + "/precision"] = precision
                all_metrics[metric_name + "/recall"] = recall

        num_metrics = len(self._biome_classifier_metrics)
        all_metrics["average/f1"] = total_f1 / num_metrics
        all_metrics["average/precision"] = total_precision / num_metrics
        all_metrics["average/recall"] = total_recall / num_metrics

        return all_metrics
