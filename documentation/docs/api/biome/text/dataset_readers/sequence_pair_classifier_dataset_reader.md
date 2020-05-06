# biome.text.dataset_readers.sequence_pair_classifier_dataset_reader <Badge text="Module"/>
<dl>
<h2 id="biome.text.dataset_readers.sequence_pair_classifier_dataset_reader.SequencePairClassifierReader">SequencePairClassifierReader <Badge text="Class"/></h2>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
    <code>
<span class="token keyword">class</span> <span class="ident">SequencePairClassifierReader</span> (</span>
    <span>tokenizer: allennlp.data.tokenizers.tokenizer.Tokenizer = None</span><span>,</span>
    <span>token_indexers: Dict[str, allennlp.data.token_indexers.token_indexer.TokenIndexer] = None</span><span>,</span>
    <span>segment_sentences: Union[bool, allennlp.data.tokenizers.sentence_splitter.SentenceSplitter] = False</span><span>,</span>
    <span>as_text_field: bool = True</span><span>,</span>
    <span>skip_empty_tokens: bool = False</span><span>,</span>
    <span>max_sequence_length: int = None</span><span>,</span>
    <span>max_nr_of_sentences: int = None</span><span>,</span>
    <span>text_transforms: <a title="biome.text.dataset_readers.text_transforms.TextTransforms" href="text_transforms.html#biome.text.dataset_readers.text_transforms.TextTransforms">TextTransforms</a> = None</span><span>,</span>
<span>)</span>
    </code></pre></div>
</dt>
<dd>
<div class="desc"><p>A DatasetReader for the SequencePairClassifier model.</p></div>
<h3>Ancestors</h3>
<ul class="hlist">
<li><a title="biome.text.dataset_readers.datasource_reader.DataSourceReader" href="datasource_reader.html#biome.text.dataset_readers.datasource_reader.DataSourceReader">DataSourceReader</a></li>
<li>allennlp.data.dataset_readers.dataset_reader.DatasetReader</li>
<li>allennlp.common.registrable.Registrable</li>
<li>allennlp.common.from_params.FromParams</li>
<li><a title="biome.text.dataset_readers.mixins.CacheableMixin" href="mixins.html#biome.text.dataset_readers.mixins.CacheableMixin">CacheableMixin</a></li>
</ul>
<h3>Inherited members</h3>
<ul class="hlist">
<li><code><b><a title="biome.text.dataset_readers.datasource_reader.DataSourceReader" href="datasource_reader.html#biome.text.dataset_readers.datasource_reader.DataSourceReader">DataSourceReader</a></b></code>:
<ul class="hlist">
<li><code><a title="biome.text.dataset_readers.datasource_reader.DataSourceReader.build_textfield" href="datasource_reader.html#biome.text.dataset_readers.datasource_reader.DataSourceReader.build_textfield">build_textfield</a></code></li>
<li><code><a title="biome.text.dataset_readers.datasource_reader.DataSourceReader.get" href="mixins.html#biome.text.dataset_readers.mixins.CacheableMixin.get">get</a></code></li>
<li><code><a title="biome.text.dataset_readers.datasource_reader.DataSourceReader.set" href="mixins.html#biome.text.dataset_readers.mixins.CacheableMixin.set">set</a></code></li>
<li><code><a title="biome.text.dataset_readers.datasource_reader.DataSourceReader.signature" href="datasource_reader.html#biome.text.dataset_readers.datasource_reader.DataSourceReader.signature">signature</a></code></li>
<li><code><a title="biome.text.dataset_readers.datasource_reader.DataSourceReader.text_to_instance" href="datasource_reader.html#biome.text.dataset_readers.datasource_reader.DataSourceReader.text_to_instance">text_to_instance</a></code></li>
</ul>
</li>
</ul>
</dd>
</dl>