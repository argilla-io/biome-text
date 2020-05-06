# biome.text.api_new.configuration <Badge text="Module"/>
<dl>
<h2 id="biome.text.api_new.configuration.FeaturesConfiguration">FeaturesConfiguration <Badge text="Class"/></h2>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
    <code>
<span class="token keyword">class</span> <span class="ident">FeaturesConfiguration</span> (</span>
    <span>words: Union[Dict[str, Any], NoneType] = None</span><span>,</span>
    <span>chars: Union[Dict[str, Any], NoneType] = None</span><span>,</span>
    <span>**extra_params</span><span>,</span>
<span>)</span>
    </code></pre></div>
</dt>
<dd>
<div class="desc"><p>Features configuration spec</p></div>
<h3>Ancestors</h3>
<ul class="hlist">
<li>allennlp.common.from_params.FromParams</li>
</ul>
<dl>
<h3 id="biome.text.api_new.configuration.FeaturesConfiguration.from_params">from_params <Badge text="Static method"/></h3>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">from_params</span> (</span>
   params: allennlp.common.params.Params,
   **extras,
)  -> <a title="biome.text.api_new.configuration.FeaturesConfiguration" href="#biome.text.api_new.configuration.FeaturesConfiguration">FeaturesConfiguration</a>
</code>
        </pre>
</div>
</dt>
<dd>
<div class="desc"><p>This is the automatic implementation of <code>from_params</code>. Any class that subclasses <code>FromParams</code>
(or <code>Registrable</code>, which itself subclasses <code>FromParams</code>) gets this implementation for free.
If you want your class to be instantiated from params in the "obvious" way &ndash; pop off parameters
and hand them to your constructor with the same names &ndash; this provides that functionality.</p>
<p>If you need more complex logic in your from <code>from_params</code> method, you'll have to implement
your own method that overrides this one.</p></div>
</dd>
</dl>
<dl>
<h3 id="biome.text.api_new.configuration.FeaturesConfiguration.compile">compile <Badge text="Method"/></h3>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">compile</span></span>(<span>self) -> <a title="biome.text.api_new.featurizer.InputFeaturizer" href="featurizer.html#biome.text.api_new.featurizer.InputFeaturizer">InputFeaturizer</a></span>
</code>
        </pre>
</div>
</dt>
<dd>
<div class="desc"><p>Build input featurizer from configuration</p></div>
</dd>
</dl>
</dd>
<h2 id="biome.text.api_new.configuration.TokenizerConfiguration">TokenizerConfiguration <Badge text="Class"/></h2>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
    <code>
<span class="token keyword">class</span> <span class="ident">TokenizerConfiguration</span> (</span>
    <span>lang: str = 'en'</span><span>,</span>
    <span>skip_empty_tokens: bool = False</span><span>,</span>
    <span>max_sequence_length: int = None</span><span>,</span>
    <span>max_nr_of_sentences: int = None</span><span>,</span>
    <span>text_cleaning: Union[Dict[str, Any], NoneType] = None</span><span>,</span>
    <span>segment_sentences: Union[bool, Dict[str, Any]] = False</span><span>,</span>
<span>)</span>
    </code></pre></div>
</dt>
<dd>
<div class="desc"><p>"Tokenization configuration</p></div>
<h3>Ancestors</h3>
<ul class="hlist">
<li>allennlp.common.from_params.FromParams</li>
</ul>
<dl>
<h3 id="biome.text.api_new.configuration.TokenizerConfiguration.compile">compile <Badge text="Method"/></h3>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">compile</span></span>(<span>self) -> <a title="biome.text.api_new.tokenizer.Tokenizer" href="tokenizer.html#biome.text.api_new.tokenizer.Tokenizer">Tokenizer</a></span>
</code>
        </pre>
</div>
</dt>
<dd>
<div class="desc"><p>Build tokenizer object from its configuration</p></div>
</dd>
</dl>
</dd>
<h2 id="biome.text.api_new.configuration.PipelineConfiguration">PipelineConfiguration <Badge text="Class"/></h2>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
    <code>
<span class="token keyword">class</span> <span class="ident">PipelineConfiguration</span> (</span>
    <span>name: str</span><span>,</span>
    <span>features: <a title="biome.text.api_new.configuration.FeaturesConfiguration" href="#biome.text.api_new.configuration.FeaturesConfiguration">FeaturesConfiguration</a></span><span>,</span>
    <span>head: <a title="biome.text.api_new.modules.heads.defs.TaskHeadSpec" href="modules/heads/defs.html#biome.text.api_new.modules.heads.defs.TaskHeadSpec">TaskHeadSpec</a></span><span>,</span>
    <span>tokenizer: Union[biome.text.api_new.configuration.TokenizerConfiguration, NoneType] = None</span><span>,</span>
    <span>encoder: Union[biome.text.api_new.modules.specs.allennlp_specs.Seq2SeqEncoderSpec, NoneType] = None</span><span>,</span>
<span>)</span>
    </code></pre></div>
</dt>
<dd>
<div class="desc"><p>Pipeline configuration attributes</p></div>
<h3>Ancestors</h3>
<ul class="hlist">
<li>allennlp.common.from_params.FromParams</li>
</ul>
<dl>
<h3 id="biome.text.api_new.configuration.PipelineConfiguration.as_dict">as_dict <Badge text="Method"/></h3>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">as_dict</span></span>(<span>self) -> Dict[str, Any]</span>
</code>
        </pre>
</div>
</dt>
<dd>
<div class="desc"></div>
</dd>
</dl>
</dd>
<h2 id="biome.text.api_new.configuration.TrainerConfiguration">TrainerConfiguration <Badge text="Class"/></h2>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
    <code>
<span class="token keyword">class</span> <span class="ident">TrainerConfiguration</span> (</span>
    <span>optimizer: Dict[str, Any]</span><span>,</span>
    <span>validation_metric: str = '-loss'</span><span>,</span>
    <span>patience: Union[int, NoneType] = None</span><span>,</span>
    <span>shuffle: bool = True</span><span>,</span>
    <span>num_epochs: int = 20</span><span>,</span>
    <span>cuda_device: int = -1</span><span>,</span>
    <span>grad_norm: Union[float, NoneType] = None</span><span>,</span>
    <span>grad_clipping: Union[float, NoneType] = None</span><span>,</span>
    <span>learning_rate_scheduler: Union[Dict[str, Any], NoneType] = None</span><span>,</span>
    <span>momentum_scheduler: Union[Dict[str, Any], NoneType] = None</span><span>,</span>
    <span>moving_average: Union[Dict[str, Any], NoneType] = None</span><span>,</span>
    <span>batch_size: Union[int, NoneType] = None</span><span>,</span>
    <span>cache_instances: bool = True</span><span>,</span>
    <span>in_memory_batches: int = 2</span><span>,</span>
    <span>data_bucketing: bool = True</span><span>,</span>
<span>)</span>
    </code></pre></div>
</dt>
<dd>
<div class="desc"><p>Trainer configuration</p></div>
</dd>
<h2 id="biome.text.api_new.configuration.VocabularyConfiguration">VocabularyConfiguration <Badge text="Class"/></h2>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
    <code>
<span class="token keyword">class</span> <span class="ident">VocabularyConfiguration</span> (</span>
    <span>sources: List[str]</span><span>,</span>
    <span>min_count: Dict[str, int] = None</span><span>,</span>
    <span>max_vocab_size: Union[int, Dict[str, int]] = None</span><span>,</span>
    <span>pretrained_files: Union[Dict[str, str], NoneType] = None</span><span>,</span>
    <span>only_include_pretrained_words: bool = False</span><span>,</span>
    <span>tokens_to_add: Dict[str, List[str]] = None</span><span>,</span>
    <span>min_pretrained_embeddings: Dict[str, int] = None</span><span>,</span>
<span>)</span>
    </code></pre></div>
</dt>
<dd>
<div class="desc"><p>Configures a <code>Vocabulary</code> before it gets created from data</p>
<p>Use this to configure a Vocabulary using specific arguments from <code>allennlp.data.Vocabulary</code></p>
<p>See <a href="https://docs.allennlp.org/master/api/data/vocabulary/#vocabulary]">AllenNLP Vocabulary docs</a></p>
<h1 id="parameters">Parameters</h1>
<pre><code>sources: &lt;code&gt;List\[str]&lt;/code&gt;
min_count: &lt;code&gt;Dict\[str, int]&lt;/code&gt;
max_vocab_size: &lt;code&gt;Union\[int, Dict\[str, int]]&lt;/code&gt;
pretrained_files: &lt;code&gt;Optional\[Dict\[str, str]]&lt;/code&gt;
only_include_pretrained_words: &lt;code&gt;bool&lt;/code&gt;
tokens_to_add: &lt;code&gt;Dict\[str, List\[str]]&lt;/code&gt;
min_pretrained_embeddings: &lt;code&gt;Dict\[str, int]&lt;/code&gt;
</code></pre></div>
</dd>
</dl>