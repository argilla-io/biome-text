import json
import unittest

import pyhocon


class DatasetReaderTest(unittest.TestCase):

    def test_combine(self):
        config = pyhocon.ConfigFactory.parse_string("""{"dataset_reader": {
            "type": "parallel_dataset_reader",
            "dataset_format": {
                "type": "csv",
                "delimiter": ","
            },
            "transformations": {
                "inputs": [
                    "label"
                ],
                "gold_label": {
                    "field": "category of dataset"
                }
            },
            "token_indexers": {
                "token_characters": {
                    "type": "characters",
                    "character_tokenizer": {
                        "byte_encoding": "utf-8"
                    }
                }
            }
        }}""")

        overrides_hocon = pyhocon.ConfigFactory.parse_string("""
        {
            "dataset_reader": {
                "transformations": {
                    "inputs": ["new-label"],
                    "gold_label": {
                        "field": "new-gold"
                    }
                }
            }
        }
        """)

        combined_hocon = overrides_hocon.with_fallback(config)
        print(json.dumps(combined_hocon))
