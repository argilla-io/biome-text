# biome.text.api_new.modules.heads.defs <Badge text="Module"/>
<dl>
<h2 id="biome.text.api_new.modules.heads.defs.TaskOutput">TaskOutput <Badge text="Class"/></h2>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
    <code>
<span class="token keyword">class</span> <span class="ident">TaskOutput</span> (</span>
    <span>logits: torch.Tensor</span><span>,</span>
    <span>probs: torch.Tensor</span><span>,</span>
    <span>loss: Union[torch.Tensor, NoneType] = None</span><span>,</span>
    <span>**extra_data</span><span>,</span>
<span>)</span>
    </code></pre></div>
</dt>
<dd>
<div class="desc"><p>Task output data class</p>
<p>A task output will contains almost the logits and probs properties</p></div>
<dl>
<h3 id="biome.text.api_new.modules.heads.defs.TaskOutput.as_dict">as_dict <Badge text="Method"/></h3>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">as_dict</span></span>(<span>self) -> Dict[str, torch.Tensor]</span>
</code>
        </pre>
</div>
</dt>
<dd>
<div class="desc"><p>Dict reprentation of task output</p></div>
</dd>
</dl>
</dd>
<h2 id="biome.text.api_new.modules.heads.defs.TaskName">TaskName <Badge text="Class"/></h2>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
    <code>
<span class="token keyword">class</span> <span class="ident">TaskName</span> (</span>
    <span>value</span><span>,</span>
    <span>names=None</span><span>,</span>
    <span>*</span><span>,</span>
    <span>module=None</span><span>,</span>
    <span>qualname=None</span><span>,</span>
    <span>type=None</span><span>,</span>
    <span>start=1</span><span>,</span>
<span>)</span>
    </code></pre></div>
</dt>
<dd>
<div class="desc"><p>The task name enum structure</p></div>
<h3>Ancestors</h3>
<ul class="hlist">
<li>enum.Enum</li>
</ul>
<h3>Class variables</h3>
<dl>
<dt id="biome.text.api_new.modules.heads.defs.TaskName.none"><code class="name">var <span class="ident">none</span></code></dt>
<dd>
<div class="desc"></div>
</dd>
<dt id="biome.text.api_new.modules.heads.defs.TaskName.text_classification"><code class="name">var <span class="ident">text_classification</span></code></dt>
<dd>
<div class="desc"></div>
</dd>
<dt id="biome.text.api_new.modules.heads.defs.TaskName.token_classification"><code class="name">var <span class="ident">token_classification</span></code></dt>
<dd>
<div class="desc"></div>
</dd>
<dt id="biome.text.api_new.modules.heads.defs.TaskName.language_modelling"><code class="name">var <span class="ident">language_modelling</span></code></dt>
<dd>
<div class="desc"></div>
</dd>
</dl>
</dd>
<h2 id="biome.text.api_new.modules.heads.defs.TaskHead">TaskHead <Badge text="Class"/></h2>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
    <code>
<span class="token keyword">class</span> <span class="ident">TaskHead</span> (model: <a title="biome.text.api_new.model.Model" href="../../model.html#biome.text.api_new.model.Model">Model</a>)</span>
    </code></pre></div>
</dt>
<dd>
<div class="desc"><p>Base task head class</p>
<p>Initializes internal Module state, shared by both nn.Module and ScriptModule.</p></div>
<h3>Ancestors</h3>
<ul class="hlist">
<li>torch.nn.modules.module.Module</li>
<li>allennlp.common.registrable.Registrable</li>
<li>allennlp.common.from_params.FromParams</li>
</ul>
<h3>Subclasses</h3>
<ul class="hlist">
<li><a title="biome.text.api_new.modules.heads.classification.defs.ClassificationHead" href="classification/defs.html#biome.text.api_new.modules.heads.classification.defs.ClassificationHead">ClassificationHead</a></li>
<li><a title="biome.text.api_new.modules.heads.language_modelling.LanguageModelling" href="language_modelling.html#biome.text.api_new.modules.heads.language_modelling.LanguageModelling">LanguageModelling</a></li>
<li><a title="biome.text.api_new.modules.heads.token_classification.TokenClassification" href="token_classification.html#biome.text.api_new.modules.heads.token_classification.TokenClassification">TokenClassification</a></li>
</ul>
<dl>
<h3 id="biome.text.api_new.modules.heads.defs.TaskHead.register">register <Badge text="Static method"/></h3>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">register</span> (</span>
   overrides: bool = False,
   **kwargs,
) 
</code>
        </pre>
</div>
</dt>
<dd>
<div class="desc"><p>Enables the task head component for pipeline loading</p></div>
</dd>
</dl>
<h3>Instance variables</h3>
<dl>
<dt id="biome.text.api_new.modules.heads.defs.TaskHead.labels"><code class="name">var <span class="ident">labels</span> : List[str]</code></dt>
<dd>
<div class="desc"><p>The configured vocab labels</p></div>
</dd>
<dt id="biome.text.api_new.modules.heads.defs.TaskHead.num_labels"><code class="name">var <span class="ident">num_labels</span></code></dt>
<dd>
<div class="desc"><p>The number of vocab labels</p></div>
</dd>
</dl>
<dl>
<h3 id="biome.text.api_new.modules.heads.defs.TaskHead.extend_labels">extend_labels <Badge text="Method"/></h3>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">extend_labels</span> (</span>
   self,
   labels: List[str],
) 
</code>
        </pre>
</div>
</dt>
<dd>
<div class="desc"><p>Extends the number of labels</p></div>
</dd>
<h3 id="biome.text.api_new.modules.heads.defs.TaskHead.task_name">task_name <Badge text="Method"/></h3>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">task_name</span></span>(<span>self) -> <a title="biome.text.api_new.modules.heads.defs.TaskName" href="#biome.text.api_new.modules.heads.defs.TaskName">TaskName</a></span>
</code>
        </pre>
</div>
</dt>
<dd>
<div class="desc"><p>The task head name</p></div>
</dd>
<h3 id="biome.text.api_new.modules.heads.defs.TaskHead.inputs">inputs <Badge text="Method"/></h3>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">inputs</span></span>(<span>self) -> Union[List[str], NoneType]</span>
</code>
        </pre>
</div>
</dt>
<dd>
<div class="desc"><p>The expected inputs names for data featuring. If no defined,
will be automatically calculated from featurize signature</p></div>
</dd>
<h3 id="biome.text.api_new.modules.heads.defs.TaskHead.forward">forward <Badge text="Method"/></h3>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">forward</span> (</span>
   self,
   *args: Any,
   **kwargs: Any,
)  -> <a title="biome.text.api_new.modules.heads.defs.TaskOutput" href="#biome.text.api_new.modules.heads.defs.TaskOutput">TaskOutput</a>
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
<h3 id="biome.text.api_new.modules.heads.defs.TaskHead.get_metrics">get_metrics <Badge text="Method"/></h3>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">get_metrics</span> (</span>
   self,
   reset: bool = False,
)  -> Dict[str, float]
</code>
        </pre>
</div>
</dt>
<dd>
<div class="desc"><p>Metrics dictionary for training task</p></div>
</dd>
<h3 id="biome.text.api_new.modules.heads.defs.TaskHead.featurize">featurize <Badge text="Method"/></h3>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">featurize</span> (</span>
   self,
   *args,
   **kwargs,
)  -> Union[allennlp.data.instance.Instance, NoneType]
</code>
        </pre>
</div>
</dt>
<dd>
<div class="desc"><p>Converts incoming data into an allennlp <code>Instance</code>, used for pyTorch tensors generation</p></div>
</dd>
<h3 id="biome.text.api_new.modules.heads.defs.TaskHead.process_output">process_output <Badge text="Method"/></h3>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">process_output</span> (</span>
   self,
   output: <a title="biome.text.api_new.modules.heads.defs.TaskOutput" href="#biome.text.api_new.modules.heads.defs.TaskOutput">TaskOutput</a>,
)  -> <a title="biome.text.api_new.modules.heads.defs.TaskOutput" href="#biome.text.api_new.modules.heads.defs.TaskOutput">TaskOutput</a>
</code>
        </pre>
</div>
</dt>
<dd>
<div class="desc"><p>Build extra parameters over basic task output</p></div>
</dd>
<h3 id="biome.text.api_new.modules.heads.defs.TaskHead.prediction_explain">prediction_explain <Badge text="Method"/></h3>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">prediction_explain</span> (</span>
   self,
   prediction: Dict[str, <built-in function array>],
   instance: allennlp.data.instance.Instance,
)  -> Dict[str, Any]
</code>
        </pre>
</div>
</dt>
<dd>
<div class="desc"><p>Adds embedding explanations information to prediction output</p></div>
</dd>
</dl>
</dd>
<h2 id="biome.text.api_new.modules.heads.defs.TaskHeadSpec">TaskHeadSpec <Badge text="Class"/></h2>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
    <code>
<span class="token keyword">class</span> <span class="ident">TaskHeadSpec</span> (*args, **kwds)</span>
    </code></pre></div>
</dt>
<dd>
<div class="desc"><p>Layer spec for TaskHead components</p></div>
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