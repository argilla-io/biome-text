# biome.text.dataset_readers.datasource_reader <Badge text="Module"/>
<dl>
<h2 id="biome.text.dataset_readers.datasource_reader.DataSourceReader">DataSourceReader <Badge text="Class"/></h2>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
    <code>
<span class="token keyword">class</span> <span class="ident">DataSourceReader</span> (</span>
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
<div class="desc"><p>A DataSetReader as base for read instances from <code>DataSource</code></p>
<p>The subclasses must implements their own way to transform input data to <code>Instance</code>
in the text_to_field method</p>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>tokenizer</code></strong></dt>
<dd>By default we use a WordTokenizer with the SpacyWordSplitter</dd>
<dt><strong><code>token_indexers</code></strong></dt>
<dd>By default we use the following dict {'tokens': SingleIdTokenIndexer}</dd>
<dt><strong><code>segment_sentences</code></strong></dt>
<dd>If True, we will first segment the text into sentences using SpaCy and then tokenize words.</dd>
<dt><strong><code>as_text_field</code></strong></dt>
<dd>Flag indicating how to generate the <code>TextField</code>. If enabled, the output Field
will be a <code>TextField</code> with text concatenation, else the result field will be
a <code>ListField</code> of <code>TextField</code>s, one per input data value</dd>
<dt><strong><code>skip_empty_tokens</code></strong></dt>
<dd>Should i silently skip empty tokens?</dd>
<dt><strong><code>max_sequence_length</code></strong></dt>
<dd>If you want to truncate the text input to a maximum number of characters</dd>
<dt><strong><code>max_nr_of_sentences</code></strong></dt>
<dd>Use only the first max_nr_of_sentences when segmenting the text into sentences</dd>
<dt><strong><code>text_transforms</code></strong></dt>
<dd>By default we use the as 'rm_spaces' registered class, which just removes useless, leading and trailing spaces
from the text before embedding it in a <code>TextField</code>.</dd>
</dl></div>
<h3>Ancestors</h3>
<ul class="hlist">
<li>allennlp.data.dataset_readers.dataset_reader.DatasetReader</li>
<li>allennlp.common.registrable.Registrable</li>
<li>allennlp.common.from_params.FromParams</li>
<li><a title="biome.text.dataset_readers.mixins.CacheableMixin" href="mixins.html#biome.text.dataset_readers.mixins.CacheableMixin">CacheableMixin</a></li>
</ul>
<h3>Subclasses</h3>
<ul class="hlist">
<li><a title="biome.text.dataset_readers.sequence_classifier_dataset_reader.SequenceClassifierReader" href="sequence_classifier_dataset_reader.html#biome.text.dataset_readers.sequence_classifier_dataset_reader.SequenceClassifierReader">SequenceClassifierReader</a></li>
<li><a title="biome.text.dataset_readers.sequence_pair_classifier_dataset_reader.SequencePairClassifierReader" href="sequence_pair_classifier_dataset_reader.html#biome.text.dataset_readers.sequence_pair_classifier_dataset_reader.SequencePairClassifierReader">SequencePairClassifierReader</a></li>
</ul>
<h3>Instance variables</h3>
<dl>
<dt id="biome.text.dataset_readers.datasource_reader.DataSourceReader.signature"><code class="name">var <span class="ident">signature</span> : dict</code></dt>
<dd>
<div class="desc"><p>Describe de input signature for the pipeline predictions</p>
<h2 id="returns">Returns</h2>
<pre><code>A list of expected inputs with information about if input is optional or nor.

For example, for the signature
&gt;&gt;def text_to_instance(a:str,b:str, c:str=None)

This method will return:
&gt;&gt;{"a":{"optional": False},"b":{"optional": False},"c":{"optional": True}}
</code></pre></div>
</dd>
</dl>
<dl>
<h3 id="biome.text.dataset_readers.datasource_reader.DataSourceReader.build_textfield">build_textfield <Badge text="Method"/></h3>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">build_textfield</span> (</span>
   self,
   data: Iterable,
)  -> Union[allennlp.data.fields.list_field.ListField, allennlp.data.fields.text_field.TextField, NoneType]
</code>
        </pre>
</div>
</dt>
<dd>
<div class="desc"><p>Embeds the record in a TextField or ListField depending on the _as_text_field parameter.</p>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>data</code></strong></dt>
<dd>Record to be embedded.</dd>
</dl>
<h2 id="returns">Returns</h2>
<dl>
<dt><code>field</code></dt>
<dd>Either a TextField or a ListField containing the record.
Returns None if <code>data</code> is not a str or a dict.</dd>
</dl></div>
</dd>
<h3 id="biome.text.dataset_readers.datasource_reader.DataSourceReader.text_to_instance">text_to_instance <Badge text="Method"/></h3>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">text_to_instance</span> (</span>
   self,
   **inputs,
)  -> allennlp.data.instance.Instance
</code>
        </pre>
</div>
</dt>
<dd>
<div class="desc"><p>Convert an input text data into a allennlp Instance</p></div>
</dd>
</dl>
<h3>Inherited members</h3>
<ul class="hlist">
<li><code><b><a title="biome.text.dataset_readers.mixins.CacheableMixin" href="mixins.html#biome.text.dataset_readers.mixins.CacheableMixin">CacheableMixin</a></b></code>:
<ul class="hlist">
<li><code><a title="biome.text.dataset_readers.mixins.CacheableMixin.get" href="mixins.html#biome.text.dataset_readers.mixins.CacheableMixin.get">get</a></code></li>
<li><code><a title="biome.text.dataset_readers.mixins.CacheableMixin.set" href="mixins.html#biome.text.dataset_readers.mixins.CacheableMixin.set">set</a></code></li>
</ul>
</li>
</ul>
</dd>
</dl>