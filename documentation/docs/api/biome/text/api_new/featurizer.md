# biome.text.api_new.featurizer <Badge text="Module"/>
<dl>
<h2 id="biome.text.api_new.featurizer.InputFeaturizer">InputFeaturizer <Badge text="Class"/></h2>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
    <code>
<span class="token keyword">class</span> <span class="ident">InputFeaturizer</span> (</span>
    <span>words: Union[Dict[str, Any], NoneType] = None</span><span>,</span>
    <span>chars: Union[Dict[str, Any], NoneType] = None</span><span>,</span>
    <span>**kwargs: Dict[str, Dict[str, Any]]</span><span>,</span>
<span>)</span>
    </code></pre></div>
</dt>
<dd>
<div class="desc"><p>The input features class. Centralize the token_indexers and embedder configurations, since are very coupled.</p>
<p>This class define two input features: words and chars for embeddings at word and character level. In those cases,
the required configuration is specified in <code>_WordFeaturesSpecs</code> and <code>_CharacterFeaturesSpec</code> respectively</p>
<p>You can provide addittional features by manually specify <code>indexer</code> and <code>embedder</code> configurations.</p></div>
<h3>Class variables</h3>
<dl>
<dt id="biome.text.api_new.featurizer.InputFeaturizer.WORDS"><code class="name">var <span class="ident">WORDS</span></code></dt>
<dd>
<div class="desc"></div>
</dd>
<dt id="biome.text.api_new.featurizer.InputFeaturizer.CHARS"><code class="name">var <span class="ident">CHARS</span></code></dt>
<dd>
<div class="desc"></div>
</dd>
</dl>
<dl>
<h3 id="biome.text.api_new.featurizer.InputFeaturizer.from_params">from_params <Badge text="Static method"/></h3>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">from_params</span></span>(<span>params: allennlp.common.params.Params) -> <a title="biome.text.api_new.featurizer.InputFeaturizer" href="#biome.text.api_new.featurizer.InputFeaturizer">InputFeaturizer</a></span>
</code>
        </pre>
</div>
</dt>
<dd>
<div class="desc"><p>Load a input featurizer from allennlp params</p></div>
</dd>
</dl>
<h3>Instance variables</h3>
<dl>
<dt id="biome.text.api_new.featurizer.InputFeaturizer.config"><code class="name">var <span class="ident">config</span></code></dt>
<dd>
<div class="desc"><p>The data module configuration</p></div>
</dd>
<dt id="biome.text.api_new.featurizer.InputFeaturizer.feature_keys"><code class="name">var <span class="ident">feature_keys</span></code></dt>
<dd>
<div class="desc"><p>The configured feature names ("words", "chars", &hellip;)</p></div>
</dd>
<dt id="biome.text.api_new.featurizer.InputFeaturizer.features"><code class="name">var <span class="ident">features</span> : Dict[str, allennlp.data.token_indexers.token_indexer.TokenIndexer]</code></dt>
<dd>
<div class="desc"><p>Get configured input features in terms of allennlp token indexers</p></div>
</dd>
</dl>
<dl>
<h3 id="biome.text.api_new.featurizer.InputFeaturizer.build_embedder">build_embedder <Badge text="Method"/></h3>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">build_embedder</span> (</span>
   self,
   vocab: allennlp.data.vocabulary.Vocabulary,
)  -> allennlp.modules.text_field_embedders.text_field_embedder.TextFieldEmbedder
</code>
        </pre>
</div>
</dt>
<dd>
<div class="desc"><p>Build the allennlp <code>TextFieldEmbedder</code> from configured embedding features</p></div>
</dd>
</dl>
</dd>
</dl>