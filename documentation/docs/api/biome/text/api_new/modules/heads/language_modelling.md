# biome.text.api_new.modules.heads.language_modelling <Badge text="Module"/>
<dl>
<h2 id="biome.text.api_new.modules.heads.language_modelling.SoftmaxLoss">SoftmaxLoss <Badge text="Class"/></h2>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
    <code>
<span class="token keyword">class</span> <span class="ident">SoftmaxLoss</span> (num_words: int, embedding_dim: int)</span>
    </code></pre></div>
</dt>
<dd>
<div class="desc"><p>Given some embeddings and some targets, applies a linear layer
to create logits over possible words and then returns the
negative log likelihood.
TODO: copied from allennlp master branch, remove when 1.0 is released</p>
<p>Initializes internal Module state, shared by both nn.Module and ScriptModule.</p></div>
<h3>Ancestors</h3>
<ul class="hlist">
<li>torch.nn.modules.module.Module</li>
</ul>
<dl>
<h3 id="biome.text.api_new.modules.heads.language_modelling.SoftmaxLoss.forward">forward <Badge text="Method"/></h3>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">forward</span> (</span>
   self,
   embeddings: torch.Tensor,
   targets: torch.Tensor,
)  -> torch.Tensor
</code>
        </pre>
</div>
</dt>
<dd>
<div class="desc"><p>Defines the computation performed at every call.</p>
<p>Should be overridden by all subclasses.</p>
<div class="admonition note">
<p class="admonition-title">Note</p>
<p>Although the recipe for forward pass needs to be defined within
this function, one should call the :class:<code>Module</code> instance afterwards
instead of this since the former takes care of running the
registered hooks while the latter silently ignores them.</p>
</div></div>
</dd>
</dl>
</dd>
<h2 id="biome.text.api_new.modules.heads.language_modelling.LanguageModelling">LanguageModelling <Badge text="Class"/></h2>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
    <code>
<span class="token keyword">class</span> <span class="ident">LanguageModelling</span> (model: <a title="biome.text.api_new.model.Model" href="../../model.html#biome.text.api_new.model.Model">Model</a>, dropout: float = None)</span>
    </code></pre></div>
</dt>
<dd>
<div class="desc"><p>Task head for next-token language modelling, i.e., a model to predict the next token
in a sequence of tokens.</p>
<p>Initializes internal Module state, shared by both nn.Module and ScriptModule.</p></div>
<h3>Ancestors</h3>
<ul class="hlist">
<li><a title="biome.text.api_new.modules.heads.defs.TaskHead" href="defs.html#biome.text.api_new.modules.heads.defs.TaskHead">TaskHead</a></li>
<li>torch.nn.modules.module.Module</li>
<li>allennlp.common.registrable.Registrable</li>
<li>allennlp.common.from_params.FromParams</li>
</ul>
<h3>Inherited members</h3>
<ul class="hlist">
<li><code><b><a title="biome.text.api_new.modules.heads.defs.TaskHead" href="defs.html#biome.text.api_new.modules.heads.defs.TaskHead">TaskHead</a></b></code>:
<ul class="hlist">
<li><code><a title="biome.text.api_new.modules.heads.defs.TaskHead.extend_labels" href="defs.html#biome.text.api_new.modules.heads.defs.TaskHead.extend_labels">extend_labels</a></code></li>
<li><code><a title="biome.text.api_new.modules.heads.defs.TaskHead.featurize" href="defs.html#biome.text.api_new.modules.heads.defs.TaskHead.featurize">featurize</a></code></li>
<li><code><a title="biome.text.api_new.modules.heads.defs.TaskHead.forward" href="defs.html#biome.text.api_new.modules.heads.defs.TaskHead.forward">forward</a></code></li>
<li><code><a title="biome.text.api_new.modules.heads.defs.TaskHead.get_metrics" href="defs.html#biome.text.api_new.modules.heads.defs.TaskHead.get_metrics">get_metrics</a></code></li>
<li><code><a title="biome.text.api_new.modules.heads.defs.TaskHead.inputs" href="defs.html#biome.text.api_new.modules.heads.defs.TaskHead.inputs">inputs</a></code></li>
<li><code><a title="biome.text.api_new.modules.heads.defs.TaskHead.labels" href="defs.html#biome.text.api_new.modules.heads.defs.TaskHead.labels">labels</a></code></li>
<li><code><a title="biome.text.api_new.modules.heads.defs.TaskHead.num_labels" href="defs.html#biome.text.api_new.modules.heads.defs.TaskHead.num_labels">num_labels</a></code></li>
<li><code><a title="biome.text.api_new.modules.heads.defs.TaskHead.prediction_explain" href="defs.html#biome.text.api_new.modules.heads.defs.TaskHead.prediction_explain">prediction_explain</a></code></li>
<li><code><a title="biome.text.api_new.modules.heads.defs.TaskHead.process_output" href="defs.html#biome.text.api_new.modules.heads.defs.TaskHead.process_output">process_output</a></code></li>
<li><code><a title="biome.text.api_new.modules.heads.defs.TaskHead.register" href="defs.html#biome.text.api_new.modules.heads.defs.TaskHead.register">register</a></code></li>
<li><code><a title="biome.text.api_new.modules.heads.defs.TaskHead.task_name" href="defs.html#biome.text.api_new.modules.heads.defs.TaskHead.task_name">task_name</a></code></li>
</ul>
</li>
</ul>
</dd>
<h2 id="biome.text.api_new.modules.heads.language_modelling.LanguageModellingSpec">LanguageModellingSpec <Badge text="Class"/></h2>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
    <code>
<span class="token keyword">class</span> <span class="ident">LanguageModellingSpec</span> (*args, **kwds)</span>
    </code></pre></div>
</dt>
<dd>
<div class="desc"><p>Spec for language model head components</p></div>
<h3>Ancestors</h3>
<ul class="hlist">
<li><a title="biome.text.api_new.modules.specs.defs.ComponentSpec" href="../specs/defs.html#biome.text.api_new.modules.specs.defs.ComponentSpec">ComponentSpec</a></li>
<li>typing.Generic</li>
<li>allennlp.common.from_params.FromParams</li>
</ul>
<h3>Inherited members</h3>
<ul class="hlist">
<li><code><b><a title="biome.text.api_new.modules.specs.defs.ComponentSpec" href="../specs/defs.html#biome.text.api_new.modules.specs.defs.ComponentSpec">ComponentSpec</a></b></code>:
<ul class="hlist">
<li><code><a title="biome.text.api_new.modules.specs.defs.ComponentSpec.compile" href="../specs/defs.html#biome.text.api_new.modules.specs.defs.ComponentSpec.compile">compile</a></code></li>
<li><code><a title="biome.text.api_new.modules.specs.defs.ComponentSpec.config" href="../specs/defs.html#biome.text.api_new.modules.specs.defs.ComponentSpec.config">config</a></code></li>
<li><code><a title="biome.text.api_new.modules.specs.defs.ComponentSpec.from_params" href="../specs/defs.html#biome.text.api_new.modules.specs.defs.ComponentSpec.from_params">from_params</a></code></li>
<li><code><a title="biome.text.api_new.modules.specs.defs.ComponentSpec.input_dim" href="../specs/defs.html#biome.text.api_new.modules.specs.defs.ComponentSpec.input_dim">input_dim</a></code></li>
</ul>
</li>
</ul>
</dd>
</dl>