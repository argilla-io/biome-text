# biome.text.api_new.modules.heads.text_classification <Badge text="Module"/>
<dl>
<h2 id="biome.text.api_new.modules.heads.text_classification.TextClassification">TextClassification <Badge text="Class"/></h2>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
    <code>
<span class="token keyword">class</span> <span class="ident">TextClassification</span> (</span>
    <span>model: <a title="biome.text.api_new.model.Model" href="../../model.html#biome.text.api_new.model.Model">Model</a></span><span>,</span>
    <span>pooler: <a title="biome.text.api_new.modules.specs.allennlp_specs.Seq2VecEncoderSpec" href="../specs/allennlp_specs.html#biome.text.api_new.modules.specs.allennlp_specs.Seq2VecEncoderSpec">Seq2VecEncoderSpec</a></span><span>,</span>
    <span>labels: List[str]</span><span>,</span>
    <span>feedforward: Union[biome.text.api_new.modules.specs.allennlp_specs.FeedForwardSpec, NoneType] = None</span><span>,</span>
    <span>multilabel: bool = False</span><span>,</span>
<span>)</span>
    </code></pre></div>
</dt>
<dd>
<div class="desc"><p>Task head for text classification</p>
<p>Initializes internal Module state, shared by both nn.Module and ScriptModule.</p></div>
<h3>Ancestors</h3>
<ul class="hlist">
<li><a title="biome.text.api_new.modules.heads.classification.defs.ClassificationHead" href="classification/defs.html#biome.text.api_new.modules.heads.classification.defs.ClassificationHead">ClassificationHead</a></li>
<li><a title="biome.text.api_new.modules.heads.defs.TaskHead" href="defs.html#biome.text.api_new.modules.heads.defs.TaskHead">TaskHead</a></li>
<li>torch.nn.modules.module.Module</li>
<li>allennlp.common.registrable.Registrable</li>
<li>allennlp.common.from_params.FromParams</li>
</ul>
<h3>Class variables</h3>
<dl>
<dt id="biome.text.api_new.modules.heads.text_classification.TextClassification.label_name"><code class="name">var <span class="ident">label_name</span></code></dt>
<dd>
<div class="desc"></div>
</dd>
<dt id="biome.text.api_new.modules.heads.text_classification.TextClassification.forward_arg_name"><code class="name">var <span class="ident">forward_arg_name</span></code></dt>
<dd>
<div class="desc"></div>
</dd>
</dl>
<h3>Inherited members</h3>
<ul class="hlist">
<li><code><b><a title="biome.text.api_new.modules.heads.classification.defs.ClassificationHead" href="classification/defs.html#biome.text.api_new.modules.heads.classification.defs.ClassificationHead">ClassificationHead</a></b></code>:
<ul class="hlist">
<li><code><a title="biome.text.api_new.modules.heads.classification.defs.ClassificationHead.add_label" href="classification/defs.html#biome.text.api_new.modules.heads.classification.defs.ClassificationHead.add_label">add_label</a></code></li>
<li><code><a title="biome.text.api_new.modules.heads.classification.defs.ClassificationHead.extend_labels" href="defs.html#biome.text.api_new.modules.heads.defs.TaskHead.extend_labels">extend_labels</a></code></li>
<li><code><a title="biome.text.api_new.modules.heads.classification.defs.ClassificationHead.featurize" href="defs.html#biome.text.api_new.modules.heads.defs.TaskHead.featurize">featurize</a></code></li>
<li><code><a title="biome.text.api_new.modules.heads.classification.defs.ClassificationHead.forward" href="defs.html#biome.text.api_new.modules.heads.defs.TaskHead.forward">forward</a></code></li>
<li><code><a title="biome.text.api_new.modules.heads.classification.defs.ClassificationHead.get_metrics" href="classification/defs.html#biome.text.api_new.modules.heads.classification.defs.ClassificationHead.get_metrics">get_metrics</a></code></li>
<li><code><a title="biome.text.api_new.modules.heads.classification.defs.ClassificationHead.inputs" href="defs.html#biome.text.api_new.modules.heads.defs.TaskHead.inputs">inputs</a></code></li>
<li><code><a title="biome.text.api_new.modules.heads.classification.defs.ClassificationHead.labels" href="defs.html#biome.text.api_new.modules.heads.defs.TaskHead.labels">labels</a></code></li>
<li><code><a title="biome.text.api_new.modules.heads.classification.defs.ClassificationHead.num_labels" href="defs.html#biome.text.api_new.modules.heads.defs.TaskHead.num_labels">num_labels</a></code></li>
<li><code><a title="biome.text.api_new.modules.heads.classification.defs.ClassificationHead.prediction_explain" href="defs.html#biome.text.api_new.modules.heads.defs.TaskHead.prediction_explain">prediction_explain</a></code></li>
<li><code><a title="biome.text.api_new.modules.heads.classification.defs.ClassificationHead.process_output" href="defs.html#biome.text.api_new.modules.heads.defs.TaskHead.process_output">process_output</a></code></li>
<li><code><a title="biome.text.api_new.modules.heads.classification.defs.ClassificationHead.register" href="defs.html#biome.text.api_new.modules.heads.defs.TaskHead.register">register</a></code></li>
<li><code><a title="biome.text.api_new.modules.heads.classification.defs.ClassificationHead.task_name" href="defs.html#biome.text.api_new.modules.heads.defs.TaskHead.task_name">task_name</a></code></li>
</ul>
</li>
</ul>
</dd>
<h2 id="biome.text.api_new.modules.heads.text_classification.TextClassificationSpec">TextClassificationSpec <Badge text="Class"/></h2>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
    <code>
<span class="token keyword">class</span> <span class="ident">TextClassificationSpec</span> (*args, **kwds)</span>
    </code></pre></div>
</dt>
<dd>
<div class="desc"><p>Spec for classification head components</p></div>
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