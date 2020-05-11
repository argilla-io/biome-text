from biome.text import Pipeline
from biome.text.configuration import TrainerConfiguration, PipelineConfiguration, TokenizerConfiguration, \
    FeaturesConfiguration, VocabularyConfiguration
from biome.text.modules.heads.defs import TaskHeadSpec
from biome.text.modules.specs import Seq2SeqEncoderSpec


if __name__ == "__main__":

    tokenizer = TokenizerConfiguration(lang='de')
    
    words = {'embedding_dim': 300}#, 'weights_file': 'https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.es.300.vec.gz'}
    chars = {'embedding_dim': 64, 'encoder': {'type': 'gru', 'hidden_size': 100, 'num_layers': 2}}
    
    features = FeaturesConfiguration(words,chars)
    
    head = TaskHeadSpec(**{'type': 'LanguageModelling'})
    
    config = PipelineConfiguration(
        name='twitter-model',
        features=features,
        head=head,
        tokenizer=tokenizer,
        encoder=Seq2SeqEncoderSpec(**{'type': 'lstm', 'hidden_size': 300})
    )
    
    pipeline = Pipeline.from_config(config)
    
    pipeline.train(
        output='experiment_pretraining',
        trainer=TrainerConfiguration(optimizer='adam'),
        training='configs/train.data.yml',
        validation='configs/val.data.yml',
        extend_vocab=VocabularyConfiguration(sources=['configs/train.data.yml', 'configs/val.data.yml'])
    )