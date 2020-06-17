from biome.text import Pipeline, TrainerConfiguration, VocabularyConfiguration
from biome.text.data import DataSource

if __name__ == "__main__":
    path = "/Users/dani/recognai/airbus/smart-orders-models/classification/"
    pipeline = Pipeline.from_yaml(
        "/Users/dani/recognai/airbus/smart-orders-models/classification/models/record_classifier.yml"
    )
    train_ds = DataSource.from_yaml(path + "data/train_record.yml")
    pipeline.create_vocabulary(
        VocabularyConfiguration(sources=[train_ds], min_count={"word": 1})
    )
