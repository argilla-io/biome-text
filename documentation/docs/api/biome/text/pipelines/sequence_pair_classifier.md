# biome.text.pipelines.sequence_pair_classifier <Badge text="Module"/>
<dl>
<h2 id="biome.text.pipelines.sequence_pair_classifier.SequencePairClassifierPipeline">SequencePairClassifierPipeline <Badge text="Class"/></h2>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
    <code>
<span class="token keyword">class</span> <span class="ident">SequencePairClassifierPipeline</span> (*args, **kwds)</span>
    </code></pre></div>
</dt>
<dd>
<div class="desc"><p>This class combine the different allennlp components that make possible a <code>`Pipeline</code>,
understanding as a model, not only the definition of the neural network architecture,
but also the transformation of the input data to Instances and the evaluation of
predictions on new data</p>
<p>The base idea is that this class contains the model and the dataset reader (as a predictor does),
and allow operations of learning, predict and save</p>
<h2 id="parameters">Parameters</h2>
<p>model`
The class:~allennlp.models.Model architecture</p>
<dl>
<dt><strong><code>reader</code></strong></dt>
<dd>The class:allennlp.data.DatasetReader</dd>
</dl></div>
<h3>Ancestors</h3>
<ul class="hlist">
<li><a title="biome.text.pipelines.pipeline.Pipeline" href="pipeline.html#biome.text.pipelines.pipeline.Pipeline">Pipeline</a></li>
<li>typing.Generic</li>
<li>allennlp.predictors.predictor.Predictor</li>
<li>allennlp.common.registrable.Registrable</li>
<li>allennlp.common.from_params.FromParams</li>
</ul>
<dl>
<h3 id="biome.text.pipelines.sequence_pair_classifier.SequencePairClassifierPipeline.predict">predict <Badge text="Method"/></h3>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">predict</span> (</span>
   self,
   record1: Union[str, List[str], dict],
   record2: Union[str, List[str], dict],
) 
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
<li><code><b><a title="biome.text.pipelines.pipeline.Pipeline" href="pipeline.html#biome.text.pipelines.pipeline.Pipeline">Pipeline</a></b></code>:
<ul class="hlist">
<li><code><a title="biome.text.pipelines.pipeline.Pipeline.config" href="pipeline.html#biome.text.pipelines.pipeline.Pipeline.config">config</a></code></li>
<li><code><a title="biome.text.pipelines.pipeline.Pipeline.empty_pipeline" href="pipeline.html#biome.text.pipelines.pipeline.Pipeline.empty_pipeline">empty_pipeline</a></code></li>
<li><code><a title="biome.text.pipelines.pipeline.Pipeline.extend_labels" href="pipeline.html#biome.text.pipelines.pipeline.Pipeline.extend_labels">extend_labels</a></code></li>
<li><code><a title="biome.text.pipelines.pipeline.Pipeline.from_config" href="pipeline.html#biome.text.pipelines.pipeline.Pipeline.from_config">from_config</a></code></li>
<li><code><a title="biome.text.pipelines.pipeline.Pipeline.get_gradients" href="pipeline.html#biome.text.pipelines.pipeline.Pipeline.get_gradients">get_gradients</a></code></li>
<li><code><a title="biome.text.pipelines.pipeline.Pipeline.get_output_labels" href="pipeline.html#biome.text.pipelines.pipeline.Pipeline.get_output_labels">get_output_labels</a></code></li>
<li><code><a title="biome.text.pipelines.pipeline.Pipeline.init_prediction_cache" href="pipeline.html#biome.text.pipelines.pipeline.Pipeline.init_prediction_cache">init_prediction_cache</a></code></li>
<li><code><a title="biome.text.pipelines.pipeline.Pipeline.init_prediction_logger" href="pipeline.html#biome.text.pipelines.pipeline.Pipeline.init_prediction_logger">init_prediction_logger</a></code></li>
<li><code><a title="biome.text.pipelines.pipeline.Pipeline.json_to_labeled_instances" href="pipeline.html#biome.text.pipelines.pipeline.Pipeline.json_to_labeled_instances">json_to_labeled_instances</a></code></li>
<li><code><a title="biome.text.pipelines.pipeline.Pipeline.learn" href="pipeline.html#biome.text.pipelines.pipeline.Pipeline.learn">learn</a></code></li>
<li><code><a title="biome.text.pipelines.pipeline.Pipeline.load" href="pipeline.html#biome.text.pipelines.pipeline.Pipeline.load">load</a></code></li>
<li><code><a title="biome.text.pipelines.pipeline.Pipeline.model" href="pipeline.html#biome.text.pipelines.pipeline.Pipeline.model">model</a></code></li>
<li><code><a title="biome.text.pipelines.pipeline.Pipeline.model_class" href="pipeline.html#biome.text.pipelines.pipeline.Pipeline.model_class">model_class</a></code></li>
<li><code><a title="biome.text.pipelines.pipeline.Pipeline.name" href="pipeline.html#biome.text.pipelines.pipeline.Pipeline.name">name</a></code></li>
<li><code><a title="biome.text.pipelines.pipeline.Pipeline.predict_json" href="pipeline.html#biome.text.pipelines.pipeline.Pipeline.predict_json">predict_json</a></code></li>
<li><code><a title="biome.text.pipelines.pipeline.Pipeline.predictions_to_labeled_instances" href="pipeline.html#biome.text.pipelines.pipeline.Pipeline.predictions_to_labeled_instances">predictions_to_labeled_instances</a></code></li>
<li><code><a title="biome.text.pipelines.pipeline.Pipeline.reader" href="pipeline.html#biome.text.pipelines.pipeline.Pipeline.reader">reader</a></code></li>
<li><code><a title="biome.text.pipelines.pipeline.Pipeline.reader_class" href="pipeline.html#biome.text.pipelines.pipeline.Pipeline.reader_class">reader_class</a></code></li>
<li><code><a title="biome.text.pipelines.pipeline.Pipeline.signature" href="pipeline.html#biome.text.pipelines.pipeline.Pipeline.signature">signature</a></code></li>
</ul>
</li>
</ul>
</dd>
</dl>