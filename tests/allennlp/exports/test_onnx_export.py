import unittest

import torch
from allennlp.models import load_archive
from torch.autograd import Variable


@unittest.skip(reason="is a WIP")
class OnnxExportTest(unittest.TestCase):
    def test_export(self):
        archive = load_archive(
            "/Users/frascuchon/Downloads/tmp/experiment/model.tar.gz"
        )

        input_names = ["tokens"]
        output_names = ["annotation"]
        inputs = dict(bert=Variable(torch.randn(3, 1, 28, 28), requires_grad=True))

        torch.onnx.export(
            archive.model,
            inputs,
            "alexnet.onnx",
            verbose=False,
            input_names=input_names,
            output_names=output_names,
        )
