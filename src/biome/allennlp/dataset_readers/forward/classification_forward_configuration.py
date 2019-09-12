from typing import Union, Dict


class ClassificationForwardConfiguration(object):
    """
        This ``ClassificationForwardConfiguration`` represents  forward operations for label
        configuration in classification problems.

        Parameters
        ----------

        label:
            The label configuration from forward definition
        target:
            (deprecated) Just an alias of label
    """

    def __init__(self, label: Union[str, dict] = None, target: dict = None):
        self._label = None
        self._default_label = None
        self._metadata = None

        if target and not label:
            label = target

        if label:
            if isinstance(label, str):
                self._label = label
            else:
                self._label = (
                    label.get("name") or label.get("label") or label.get("gold_label")
                )
                if not self._label:
                    raise RuntimeError("I am missing the label name!")
                self._default_label = label.get(
                    "default", label.get("use_missing_label")
                )
                self._metadata = (
                    self.load_metadata(label.get("metadata_file"))
                    if label.get("metadata_file")
                    else None
                )

    @staticmethod
    def load_metadata(path: str) -> Dict[str, str]:
        with open(path) as metadata_file:
            classes = [line.rstrip("\n").rstrip() for line in metadata_file]

        mapping = {idx + 1: cls for idx, cls in enumerate(classes)}
        # mapping variant with integer numbers
        mapping = {**mapping, **{str(key): value for key, value in mapping.items()}}

        return mapping

    @property
    def label(self) -> str:
        return self._label

    @property
    def default_label(self):
        return self._default_label

    @property
    def metadata(self):
        return self._metadata

    def sanitize_label(self, label: str) -> str:
        label = label.strip()
        if self.default_label:
            label = label if label else self.default_label
        if self.metadata:
            label = self.metadata.get(label, label)

        return label
