from biome.text.dataset_readers.entity_classifier_reader import EntityClassifierReader


def test_text_to_instance():
    reader = EntityClassifierReader(as_text_field=True)

    json_dict = {
        "kind": "textual",
        "context": "this is a test",
        "position": (5, 7),
        "label": "verb"
    }
    instance = reader.text_to_instance(**json_dict)
    # we force as_text_field to False for the EntityClassifier
    assert not reader._as_text_field
    assert instance.fields["tokens"][0][0].text == "textual"
    assert instance.fields["tokens"][1][0].text == "this"
    assert instance.fields["tokens"][2][0].text == "is"
    assert instance.fields["label"] == "verb"

    json_dict = {
        "kind": "tabular",
        "context": [["this", "is", "a", "test"], ["test"]],
        "position": (3, 0),
        "label": "noun"
    }
    instance = reader.text_to_instance(**json_dict)
    assert instance.fields["tokens"][0][0].text == "tabular"
    assert instance.fields["tokens"][1][0].text == "this"
    assert instance.fields["tokens"][2][0].text == "test"
    assert instance.fields["label"] == "noun"

