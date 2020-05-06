# biome.text.dataset_readers.sequence_classifier_dataset_reader <Badge text="Module"/>
<dl>
<h2 id="biome.text.dataset_readers.sequence_classifier_dataset_reader.SequenceClassifierReader">SequenceClassifierReader <Badge text="Class"/></h2>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
    <code>
<span class="token keyword">class</span> <span class="ident">SequenceClassifierReader</span> (</span>
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
<div class="desc"><p>A DatasetReader for the SequenceClassifier model.</p></div>
<h3>Ancestors</h3>
<ul class="hlist">
<li><a title="biome.text.dataset_readers.datasource_reader.DataSourceReader" href="datasource_reader.html#biome.text.dataset_readers.datasource_reader.DataSourceReader">DataSourceReader</a></li>
<li>allennlp.data.dataset_readers.dataset_reader.DatasetReader</li>
<li>allennlp.common.registrable.Registrable</li>
<li>allennlp.common.from_params.FromParams</li>
<li><a title="biome.text.dataset_readers.mixins.CacheableMixin" href="mixins.html#biome.text.dataset_readers.mixins.CacheableMixin">CacheableMixin</a></li>
</ul>
<dl>
<h3 id="biome.text.dataset_readers.sequence_classifier_dataset_reader.SequenceClassifierReader.text_to_instance">text_to_instance <Badge text="Method"/></h3>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">text_to_instance</span> (</span>
   self,
   tokens: Union[str, List[str], dict],
   label: Union[str, NoneType] = None,
)  -> Union[allennlp.data.instance.Instance, NoneType]
</code>
        </pre>
</div>
</dt>
<dd>
<div class="desc"><p>Extracts the forward parameters from the example and transforms them to an <code>Instance</code></p>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>tokens</code></strong></dt>
<dd>The input tokens key,values (or the text string)</dd>
<dt><strong><code>label</code></strong></dt>
<dd>The label value</dd>
</dl>
<h2 id="returns">Returns</h2>
<dl>
<dt><code>instance</code></dt>
<dd>Returns <code>None</code> if cannot generate an new Instance.</dd>
</dl></div>
</dd>
</dl>
<h3>Inherited members</h3>
<ul class="hlist">
<li><code><b><a title="biome.text.dataset_readers.datasource_reader.DataSourceReader" href="datasource_reader.html#biome.text.dataset_readers.datasource_reader.DataSourceReader">DataSourceReader</a></b></code>:
<ul class="hlist">
<li><code><a title="biome.text.dataset_readers.datasource_reader.DataSourceReader.build_textfield" href="datasource_reader.html#biome.text.dataset_readers.datasource_reader.DataSourceReader.build_textfield">build_textfield</a></code></li>
<li><code><a title="biome.text.dataset_readers.datasource_reader.DataSourceReader.get" href="mixins.html#biome.text.dataset_readers.mixins.CacheableMixin.get">get</a></code></li>
<li><code><a title="biome.text.dataset_readers.datasource_reader.DataSourceReader.set" href="mixins.html#biome.text.dataset_readers.mixins.CacheableMixin.set">set</a></code></li>
<li><code><a title="biome.text.dataset_readers.datasource_reader.DataSourceReader.signature" href="datasource_reader.html#biome.text.dataset_readers.datasource_reader.DataSourceReader.signature">signature</a></code></li>
</ul>
</li>
</ul>
</dd>
</dl>