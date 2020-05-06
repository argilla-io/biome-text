# biome.text.api_new.modules.heads.classification.defs <Badge text="Module"/>
<dl>
<h2 id="biome.text.api_new.modules.heads.classification.defs.ClassificationHead">ClassificationHead <Badge text="Class"/></h2>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
    <code>
<span class="token keyword">class</span> <span class="ident">ClassificationHead</span> (</span>
    <span>model: <a title="biome.text.api_new.model.Model" href="../../../model.html#biome.text.api_new.model.Model">Model</a></span><span>,</span>
    <span>labels: List[str]</span><span>,</span>
    <span>multilabel: bool = False</span><span>,</span>
<span>)</span>
    </code></pre></div>
</dt>
<dd>
<div class="desc"><p>Base abstract class for classification problems</p>
<p>Initializes internal Module state, shared by both nn.Module and ScriptModule.</p></div>
<h3>Ancestors</h3>
<ul class="hlist">
<li><a title="biome.text.api_new.modules.heads.defs.TaskHead" href="../defs.html#biome.text.api_new.modules.heads.defs.TaskHead">TaskHead</a></li>
<li>torch.nn.modules.module.Module</li>
<li>allennlp.common.registrable.Registrable</li>
<li>allennlp.common.from_params.FromParams</li>
</ul>
<h3>Subclasses</h3>
<ul class="hlist">
<li><a title="biome.text.api_new.modules.heads.doc_classification.DocumentClassification" href="../doc_classification.html#biome.text.api_new.modules.heads.doc_classification.DocumentClassification">DocumentClassification</a></li>
<li><a title="biome.text.api_new.modules.heads.text_classification.TextClassification" href="../text_classification.html#biome.text.api_new.modules.heads.text_classification.TextClassification">TextClassification</a></li>
</ul>
<dl>
<h3 id="biome.text.api_new.modules.heads.classification.defs.ClassificationHead.add_label">add_label <Badge text="Method"/></h3>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">add_label</span> (</span>
   self,
   instance: allennlp.data.instance.Instance,
   label: Union[List[str], List[int], str, int],
   to_field: str = 'label',
)  -> Union[allennlp.data.instance.Instance, NoneType]
</code>
        </pre>
</div>
</dt>
<dd>
<div class="desc"><p>Includes the label field for classification into the instance data</p></div>
</dd>
<h3 id="biome.text.api_new.modules.heads.classification.defs.ClassificationHead.get_metrics">get_metrics <Badge text="Method"/></h3>
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
<div class="desc"><p>Get the metrics of our classifier, see :func:<code>~allennlp_2.models.Model.get_metrics</code>.</p>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>reset</code></strong></dt>
<dd>Reset the metrics after obtaining them?</dd>
</dl>
<h2 id="returns">Returns</h2>
<p>A dictionary with all metric names and values.</p></div>
</dd>
<h3 id="biome.text.api_new.modules.heads.classification.defs.ClassificationHead.single_label_output">single_label_output <Badge text="Method"/></h3>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">single_label_output</span> (</span>
   self,
   logits: torch.Tensor,
   label: Union[torch.IntTensor, NoneType] = None,
)  -> <a title="biome.text.api_new.modules.heads.defs.TaskOutput" href="../defs.html#biome.text.api_new.modules.heads.defs.TaskOutput">TaskOutput</a>
</code>
        </pre>
</div>
</dt>
<dd>
<div class="desc"></div>
</dd>
<h3 id="biome.text.api_new.modules.heads.classification.defs.ClassificationHead.multi_label_output">multi_label_output <Badge text="Method"/></h3>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">multi_label_output</span> (</span>
   self,
   logits: torch.Tensor,
   label: Union[torch.IntTensor, NoneType] = None,
)  -> <a title="biome.text.api_new.modules.heads.defs.TaskOutput" href="../defs.html#biome.text.api_new.modules.heads.defs.TaskOutput">TaskOutput</a>
</code>
        </pre>
</div>
</dt>
<dd>
<div class="desc"></div>
</dd>
</dl>
<h3>Inherited members</h3>
<ul class="hlist">
<li><code><b><a title="biome.text.api_new.modules.heads.defs.TaskHead" href="../defs.html#biome.text.api_new.modules.heads.defs.TaskHead">TaskHead</a></b></code>:
<ul class="hlist">
<li><code><a title="biome.text.api_new.modules.heads.defs.TaskHead.extend_labels" href="../defs.html#biome.text.api_new.modules.heads.defs.TaskHead.extend_labels">extend_labels</a></code></li>
<li><code><a title="biome.text.api_new.modules.heads.defs.TaskHead.featurize" href="../defs.html#biome.text.api_new.modules.heads.defs.TaskHead.featurize">featurize</a></code></li>
<li><code><a title="biome.text.api_new.modules.heads.defs.TaskHead.forward" href="../defs.html#biome.text.api_new.modules.heads.defs.TaskHead.forward">forward</a></code></li>
<li><code><a title="biome.text.api_new.modules.heads.defs.TaskHead.inputs" href="../defs.html#biome.text.api_new.modules.heads.defs.TaskHead.inputs">inputs</a></code></li>
<li><code><a title="biome.text.api_new.modules.heads.defs.TaskHead.labels" href="../defs.html#biome.text.api_new.modules.heads.defs.TaskHead.labels">labels</a></code></li>
<li><code><a title="biome.text.api_new.modules.heads.defs.TaskHead.num_labels" href="../defs.html#biome.text.api_new.modules.heads.defs.TaskHead.num_labels">num_labels</a></code></li>
<li><code><a title="biome.text.api_new.modules.heads.defs.TaskHead.prediction_explain" href="../defs.html#biome.text.api_new.modules.heads.defs.TaskHead.prediction_explain">prediction_explain</a></code></li>
<li><code><a title="biome.text.api_new.modules.heads.defs.TaskHead.process_output" href="../defs.html#biome.text.api_new.modules.heads.defs.TaskHead.process_output">process_output</a></code></li>
<li><code><a title="biome.text.api_new.modules.heads.defs.TaskHead.register" href="../defs.html#biome.text.api_new.modules.heads.defs.TaskHead.register">register</a></code></li>
<li><code><a title="biome.text.api_new.modules.heads.defs.TaskHead.task_name" href="../defs.html#biome.text.api_new.modules.heads.defs.TaskHead.task_name">task_name</a></code></li>
</ul>
</li>
</ul>
</dd>
</dl>