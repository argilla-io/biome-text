# biome.text.api_new.modules.specs.defs <Badge text="Module"/>
<dl>
<h2 id="biome.text.api_new.modules.specs.defs.ComponentSpec">ComponentSpec <Badge text="Class"/></h2>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
    <code>
<span class="token keyword">class</span> <span class="ident">ComponentSpec</span> (*args, **kwds)</span>
    </code></pre></div>
</dt>
<dd>
<div class="desc"><p>The layer spec component allows create Pytorch modules lazily,
and instantiate them inside a context (Model or other component) dimension layer chain.</p>
<p>The layer spec wraps a component params and will generate an instance of type T once the input_dim is set.</p></div>
<h3>Ancestors</h3>
<ul class="hlist">
<li>typing.Generic</li>
<li>allennlp.common.from_params.FromParams</li>
</ul>
<h3>Subclasses</h3>
<ul class="hlist">
<li><a title="biome.text.api_new.modules.heads.defs.TaskHeadSpec" href="../heads/defs.html#biome.text.api_new.modules.heads.defs.TaskHeadSpec">TaskHeadSpec</a></li>
<li><a title="biome.text.api_new.modules.heads.doc_classification.DocumentClassificationSpec" href="../heads/doc_classification.html#biome.text.api_new.modules.heads.doc_classification.DocumentClassificationSpec">DocumentClassificationSpec</a></li>
<li><a title="biome.text.api_new.modules.heads.language_modelling.LanguageModellingSpec" href="../heads/language_modelling.html#biome.text.api_new.modules.heads.language_modelling.LanguageModellingSpec">LanguageModellingSpec</a></li>
<li><a title="biome.text.api_new.modules.heads.record_classification.RecordClassificationSpec" href="../heads/record_classification.html#biome.text.api_new.modules.heads.record_classification.RecordClassificationSpec">RecordClassificationSpec</a></li>
<li><a title="biome.text.api_new.modules.heads.text_classification.TextClassificationSpec" href="../heads/text_classification.html#biome.text.api_new.modules.heads.text_classification.TextClassificationSpec">TextClassificationSpec</a></li>
<li><a title="biome.text.api_new.modules.heads.token_classification.TokenClassificationSpec" href="../heads/token_classification.html#biome.text.api_new.modules.heads.token_classification.TokenClassificationSpec">TokenClassificationSpec</a></li>
<li><a title="biome.text.api_new.modules.specs.allennlp_specs.FeedForwardSpec" href="allennlp_specs.html#biome.text.api_new.modules.specs.allennlp_specs.FeedForwardSpec">FeedForwardSpec</a></li>
<li><a title="biome.text.api_new.modules.specs.allennlp_specs.Seq2SeqEncoderSpec" href="allennlp_specs.html#biome.text.api_new.modules.specs.allennlp_specs.Seq2SeqEncoderSpec">Seq2SeqEncoderSpec</a></li>
<li><a title="biome.text.api_new.modules.specs.allennlp_specs.Seq2VecEncoderSpec" href="allennlp_specs.html#biome.text.api_new.modules.specs.allennlp_specs.Seq2VecEncoderSpec">Seq2VecEncoderSpec</a></li>
<li><a title="biome.text.api_new.modules.specs.defs.ComponentSpec" href="#biome.text.api_new.modules.specs.defs.ComponentSpec">ComponentSpec</a></li>
<li><a title="biome.text.api_new.modules.specs.defs.ComponentSpec" href="#biome.text.api_new.modules.specs.defs.ComponentSpec">ComponentSpec</a></li>
<li><a title="biome.text.api_new.modules.specs.defs.ComponentSpec" href="#biome.text.api_new.modules.specs.defs.ComponentSpec">ComponentSpec</a></li>
<li><a title="biome.text.api_new.modules.specs.defs.ComponentSpec" href="#biome.text.api_new.modules.specs.defs.ComponentSpec">ComponentSpec</a></li>
<li><a title="biome.text.api_new.modules.specs.defs.ComponentSpec" href="#biome.text.api_new.modules.specs.defs.ComponentSpec">ComponentSpec</a></li>
<li><a title="biome.text.api_new.modules.specs.defs.ComponentSpec" href="#biome.text.api_new.modules.specs.defs.ComponentSpec">ComponentSpec</a></li>
<li><a title="biome.text.api_new.modules.specs.defs.ComponentSpec" href="#biome.text.api_new.modules.specs.defs.ComponentSpec">ComponentSpec</a></li>
<li><a title="biome.text.api_new.modules.specs.defs.ComponentSpec" href="#biome.text.api_new.modules.specs.defs.ComponentSpec">ComponentSpec</a></li>
<li><a title="biome.text.api_new.modules.specs.defs.ComponentSpec" href="#biome.text.api_new.modules.specs.defs.ComponentSpec">ComponentSpec</a></li>
</ul>
<dl>
<h3 id="biome.text.api_new.modules.specs.defs.ComponentSpec.from_params">from_params <Badge text="Static method"/></h3>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">from_params</span> (</span>
   params: allennlp.common.params.Params,
   **extras,
)  -> ~T
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
<h3>Instance variables</h3>
<dl>
<dt id="biome.text.api_new.modules.specs.defs.ComponentSpec.config"><code class="name">var <span class="ident">config</span> : Dict[str, Any]</code></dt>
<dd>
<div class="desc"><p>Component read-only configuration</p></div>
</dd>
</dl>
<dl>
<h3 id="biome.text.api_new.modules.specs.defs.ComponentSpec.input_dim">input_dim <Badge text="Method"/></h3>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">input_dim</span> (</span>
   self,
   input_dim: int,
)  -> <a title="biome.text.api_new.modules.specs.defs.ComponentSpec" href="#biome.text.api_new.modules.specs.defs.ComponentSpec">ComponentSpec</a>
</code>
        </pre>
</div>
</dt>
<dd>
<div class="desc"><p>Sets the input dimension attribute for this layer configuration</p></div>
</dd>
<h3 id="biome.text.api_new.modules.specs.defs.ComponentSpec.compile">compile <Badge text="Method"/></h3>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">compile</span> (</span>
   self,
   **extras,
)  -> ~T
</code>
        </pre>
</div>
</dt>
<dd>
<div class="desc"><p>Using the wrapped configuration and the input dimension, generates a
instance of type T representing the layer configuration</p></div>
</dd>
</dl>
</dd>
</dl>