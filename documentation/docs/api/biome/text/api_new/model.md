# biome.text.api_new.model <Badge text="Module"/>
<dl>
<h2 id="biome.text.api_new.model.Model">Model <Badge text="Class"/></h2>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
    <code>
<span class="token keyword">class</span> <span class="ident">Model</span> (</span>
    <span>vocab: allennlp.data.vocabulary.Vocabulary</span><span>,</span>
    <span>tokenizer: <a title="biome.text.api_new.tokenizer.Tokenizer" href="tokenizer.html#biome.text.api_new.tokenizer.Tokenizer">Tokenizer</a></span><span>,</span>
    <span>featurizer: <a title="biome.text.api_new.featurizer.InputFeaturizer" href="featurizer.html#biome.text.api_new.featurizer.InputFeaturizer">InputFeaturizer</a></span><span>,</span>
    <span>encoder: Union[biome.text.api_new.modules.specs.allennlp_specs.Seq2SeqEncoderSpec, NoneType] = None</span><span>,</span>
<span>)</span>
    </code></pre></div>
</dt>
<dd>
<div class="desc"><p>Model definition. All models used in pipelines must configure this model class</p>
<p>Initializes internal Module state, shared by both nn.Module and ScriptModule.</p></div>
<h3>Ancestors</h3>
<ul class="hlist">
<li>torch.nn.modules.module.Module</li>
</ul>
<h3>Instance variables</h3>
<dl>
<dt id="biome.text.api_new.model.Model.embedder"><code class="name">var <span class="ident">embedder</span> : allennlp.modules.text_field_embedders.text_field_embedder.TextFieldEmbedder</code></dt>
<dd>
<div class="desc"></div>
</dd>
<dt id="biome.text.api_new.model.Model.features"><code class="name">var <span class="ident">features</span> : Dict[str, allennlp.data.token_indexers.token_indexer.TokenIndexer]</code></dt>
<dd>
<div class="desc"></div>
</dd>
</dl>
<dl>
<h3 id="biome.text.api_new.model.Model.forward">forward <Badge text="Method"/></h3>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">forward</span> (</span>
   self,
   text: Dict[str, torch.Tensor],
   mask: torch.Tensor,
   num_wrapping_dims: int = 0,
)  -> torch.Tensor
</code>
        </pre>
</div>
</dt>
<dd>
<div class="desc"><p>Applies embedding + encoder layers</p></div>
</dd>
<h3 id="biome.text.api_new.model.Model.featurize">featurize <Badge text="Method"/></h3>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">featurize</span> (</span>
   self,
   record: Union[str, List[str], Dict[str, Any]],
   to_field: str = 'record',
   aggregate: bool = False,
)  -> allennlp.data.instance.Instance
</code>
        </pre>
</div>
</dt>
<dd>
<div class="desc"><p>Generate a allennlp Instance from a record input.</p>
<p>If aggregate flag is enabled, the resultant instance will contains a single TextField's
with all record fields; otherwhise, a ListField of TextFields.</p></div>
</dd>
</dl>
</dd>
</dl>