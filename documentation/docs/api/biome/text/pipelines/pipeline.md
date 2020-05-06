# biome.text.pipelines.pipeline <Badge text="Module"/>
<dl>
<h2 id="biome.text.pipelines.pipeline.Pipeline">Pipeline <Badge text="Class"/></h2>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
    <code>
<span class="token keyword">class</span> <span class="ident">Pipeline</span> (*args, **kwds)</span>
    </code></pre></div>
</dt>
<dd>
<div class="desc"><p>This class combine the different allennlp components that make possible a <code>`<a title="biome.text.pipelines.pipeline.Pipeline" href="#biome.text.pipelines.pipeline.Pipeline">Pipeline</a></code>,
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
<li>typing.Generic</li>
<li>allennlp.predictors.predictor.Predictor</li>
<li>allennlp.common.registrable.Registrable</li>
<li>allennlp.common.from_params.FromParams</li>
</ul>
<h3>Subclasses</h3>
<ul class="hlist">
<li><a title="biome.text.pipelines.biome_bimpm.BiomeBiMpmPipeline" href="biome_bimpm.html#biome.text.pipelines.biome_bimpm.BiomeBiMpmPipeline">BiomeBiMpmPipeline</a></li>
<li><a title="biome.text.pipelines.multifield_bimpm.MultifieldBiMpmPipeline" href="multifield_bimpm.html#biome.text.pipelines.multifield_bimpm.MultifieldBiMpmPipeline">MultifieldBiMpmPipeline</a></li>
<li><a title="biome.text.pipelines.pipeline.Pipeline" href="#biome.text.pipelines.pipeline.Pipeline">Pipeline</a></li>
<li><a title="biome.text.pipelines.pipeline.Pipeline" href="#biome.text.pipelines.pipeline.Pipeline">Pipeline</a></li>
<li><a title="biome.text.pipelines.pipeline.Pipeline" href="#biome.text.pipelines.pipeline.Pipeline">Pipeline</a></li>
<li><a title="biome.text.pipelines.pipeline.Pipeline" href="#biome.text.pipelines.pipeline.Pipeline">Pipeline</a></li>
<li><a title="biome.text.pipelines.pipeline.Pipeline" href="#biome.text.pipelines.pipeline.Pipeline">Pipeline</a></li>
<li><a title="biome.text.pipelines.sequence_classifier.SequenceClassifierPipeline" href="sequence_classifier.html#biome.text.pipelines.sequence_classifier.SequenceClassifierPipeline">SequenceClassifierPipeline</a></li>
<li><a title="biome.text.pipelines.sequence_pair_classifier.SequencePairClassifierPipeline" href="sequence_pair_classifier.html#biome.text.pipelines.sequence_pair_classifier.SequencePairClassifierPipeline">SequencePairClassifierPipeline</a></li>
<li><a title="biome.text.pipelines.similarity_classifier.SimilarityClassifierPipeline" href="similarity_classifier.html#biome.text.pipelines.similarity_classifier.SimilarityClassifierPipeline">SimilarityClassifierPipeline</a></li>
</ul>
<h3>Class variables</h3>
<dl>
<dt id="biome.text.pipelines.pipeline.Pipeline.PIPELINE_FIELD"><code class="name">var <span class="ident">PIPELINE_FIELD</span></code></dt>
<dd>
<div class="desc"></div>
</dd>
<dt id="biome.text.pipelines.pipeline.Pipeline.ARCHITECTURE_FIELD"><code class="name">var <span class="ident">ARCHITECTURE_FIELD</span></code></dt>
<dd>
<div class="desc"></div>
</dd>
<dt id="biome.text.pipelines.pipeline.Pipeline.TYPE_FIELD"><code class="name">var <span class="ident">TYPE_FIELD</span></code></dt>
<dd>
<div class="desc"></div>
</dd>
<dt id="biome.text.pipelines.pipeline.Pipeline.PREDICTION_FILE_NAME"><code class="name">var <span class="ident">PREDICTION_FILE_NAME</span></code></dt>
<dd>
<div class="desc"></div>
</dd>
</dl>
<dl>
<h3 id="biome.text.pipelines.pipeline.Pipeline.by_name">by_name <Badge text="Static method"/></h3>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">by_name</span></span>(<span>name: str) -> Type[<a title="biome.text.pipelines.pipeline.Pipeline" href="#biome.text.pipelines.pipeline.Pipeline">Pipeline</a>]</span>
</code>
        </pre>
</div>
</dt>
<dd>
<div class="desc"></div>
</dd>
<h3 id="biome.text.pipelines.pipeline.Pipeline.reader_class">reader_class <Badge text="Static method"/></h3>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">reader_class</span></span>(<span>) -> Type[~Reader]</span>
</code>
        </pre>
</div>
</dt>
<dd>
<div class="desc"><p>Must be implemented by subclasses</p>
<h2 id="returns">Returns</h2>
<pre><code>The class of &lt;code&gt;DataSourceReader&lt;/code&gt; used in the model instance
</code></pre></div>
</dd>
<h3 id="biome.text.pipelines.pipeline.Pipeline.model_class">model_class <Badge text="Static method"/></h3>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">model_class</span></span>(<span>) -> Type[~Architecture]</span>
</code>
        </pre>
</div>
</dt>
<dd>
<div class="desc"><p>Must be implemented by subclasses</p>
<h2 id="returns">Returns</h2>
<pre><code>The class of &lt;code&gt;allennlp.models.Model&lt;/code&gt; used in the model instance
</code></pre></div>
</dd>
<h3 id="biome.text.pipelines.pipeline.Pipeline.load">load <Badge text="Static method"/></h3>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">load</span> (</span>
   binary_path: str,
   **kwargs,
)  -> <a title="biome.text.pipelines.pipeline.Pipeline" href="#biome.text.pipelines.pipeline.Pipeline">Pipeline</a>
</code>
        </pre>
</div>
</dt>
<dd>
<div class="desc"><p>Load a model pipeline form a binary path.</p>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>binary_path</code></strong></dt>
<dd>Path to the binary file</dd>
<dt><strong><code>kwargs</code></strong></dt>
<dd>Passed on to the biome.text.models.load_archive method</dd>
</dl>
<h2 id="returns">Returns</h2>
<dl>
<dt><code>pipeline</code></dt>
<dd>&nbsp;</dd>
</dl></div>
</dd>
<h3 id="biome.text.pipelines.pipeline.Pipeline.yaml_to_dict">yaml_to_dict <Badge text="Static method"/></h3>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">yaml_to_dict</span></span>(<span>filepath: str)</span>
</code>
        </pre>
</div>
</dt>
<dd>
<div class="desc"></div>
</dd>
<h3 id="biome.text.pipelines.pipeline.Pipeline.empty_pipeline">empty_pipeline <Badge text="Static method"/></h3>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">empty_pipeline</span></span>(<span>labels: List[str]) -> <a title="biome.text.pipelines.pipeline.Pipeline" href="#biome.text.pipelines.pipeline.Pipeline">Pipeline</a></span>
</code>
        </pre>
</div>
</dt>
<dd>
<div class="desc"><p>Creates a dummy pipeline with labels for model layers</p></div>
</dd>
<h3 id="biome.text.pipelines.pipeline.Pipeline.from_config">from_config <Badge text="Static method"/></h3>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">from_config</span> (</span>
   path: str,
   labels: List[str] = None,
)  -> <a title="biome.text.pipelines.pipeline.Pipeline" href="#biome.text.pipelines.pipeline.Pipeline">Pipeline</a>
</code>
        </pre>
</div>
</dt>
<dd>
<div class="desc"><p>Read a <code><a title="biome.text.pipelines.pipeline.Pipeline" href="#biome.text.pipelines.pipeline.Pipeline">Pipeline</a></code> subclass instance by reading a configuration file</p>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>path</code></strong></dt>
<dd>The configuration file path</dd>
</dl>
<p>labels:
Optional. If passed, set a list of output labels for empty pipeline model</p>
<h2 id="returns">Returns</h2>
<pre><code>An instance of &lt;code&gt;&lt;a title="biome.text.pipelines.pipeline.Pipeline" href="#biome.text.pipelines.pipeline.Pipeline"&gt;Pipeline&lt;/a&gt;&lt;/code&gt; with no architecture, since the internal
&lt;code&gt;allennlp.models.Model&lt;/code&gt; needs a Vocabulary for the initialization
</code></pre></div>
</dd>
</dl>
<h3>Instance variables</h3>
<dl>
<dt id="biome.text.pipelines.pipeline.Pipeline.reader"><code class="name">var <span class="ident">reader</span> : <a title="biome.text.dataset_readers.datasource_reader.DataSourceReader" href="../dataset_readers/datasource_reader.html#biome.text.dataset_readers.datasource_reader.DataSourceReader">DataSourceReader</a></code></dt>
<dd>
<div class="desc"><p>The data reader (AKA <code>DatasetReader</code>)</p>
<h2 id="returns">Returns</h2>
<pre><code>The configured &lt;code&gt;DatasetReader&lt;/code&gt;
</code></pre></div>
</dd>
<dt id="biome.text.pipelines.pipeline.Pipeline.model"><code class="name">var <span class="ident">model</span> : allennlp.models.model.Model</code></dt>
<dd>
<div class="desc"><p>The model (AKA <code>allennlp.models.Model</code>)</p>
<h2 id="returns">Returns</h2>
<pre><code>The configured &lt;code&gt;allennlp.models.Model&lt;/code&gt;
</code></pre></div>
</dd>
<dt id="biome.text.pipelines.pipeline.Pipeline.name"><code class="name">var <span class="ident">name</span> : str</code></dt>
<dd>
<div class="desc"><p>Get the pipeline name</p>
<h2 id="returns">Returns</h2>
<pre><code>The fully qualified pipeline class name
</code></pre></div>
</dd>
<dt id="biome.text.pipelines.pipeline.Pipeline.config"><code class="name">var <span class="ident">config</span> : dict</code></dt>
<dd>
<div class="desc"><p>A representation of reader and model in a properties defined way
as allennlp does</p>
<h2 id="returns">Returns</h2>
<pre><code>The configuration dictionary
</code></pre></div>
</dd>
<dt id="biome.text.pipelines.pipeline.Pipeline.signature"><code class="name">var <span class="ident">signature</span> : dict</code></dt>
<dd>
<div class="desc"><p>Describe de input signature for the pipeline</p>
<h2 id="returns">Returns</h2>
<pre><code>A dict of expected inputs
</code></pre></div>
</dd>
</dl>
<dl>
<h3 id="biome.text.pipelines.pipeline.Pipeline.init_prediction_logger">init_prediction_logger <Badge text="Method"/></h3>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">init_prediction_logger</span> (</span>
   self,
   output_dir: str,
   max_bytes: int = 20000000,
   backup_count: int = 20,
) 
</code>
        </pre>
</div>
</dt>
<dd>
<div class="desc"><p>Initialize the prediction logger.</p>
<p>If initialized we will log all predictions to a file called <em>predictions.json</em> in the <code>output_folder</code>.</p>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>output_dir</code></strong></dt>
<dd>Path to the folder in which we create the <em>predictions.json</em> file.</dd>
<dt><strong><code>max_bytes</code></strong></dt>
<dd>Passed on to logging.handlers.RotatingFileHandler</dd>
<dt><strong><code>backup_count</code></strong></dt>
<dd>Passed on to logging.handlers.RotatingFileHandler</dd>
</dl></div>
</dd>
<h3 id="biome.text.pipelines.pipeline.Pipeline.predict">predict <Badge text="Method"/></h3>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">predict</span> (</span>
   self,
   **inputs,
)  -> dict
</code>
        </pre>
</div>
</dt>
<dd>
<div class="desc"></div>
</dd>
<h3 id="biome.text.pipelines.pipeline.Pipeline.predictions_to_labeled_instances">predictions_to_labeled_instances <Badge text="Method"/></h3>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">predictions_to_labeled_instances</span> (</span>
   self,
   instance: allennlp.data.instance.Instance,
   outputs: Dict[str, numpy.ndarray],
)  -> List[allennlp.data.instance.Instance]
</code>
        </pre>
</div>
</dt>
<dd>
<div class="desc"><p>This function takes a model's outputs for an Instance, and it labels that instance according
to the output. For example, in classification this function labels the instance according
to the class with the highest probability. This function is used to to compute gradients
of what the model predicted. The return type is a list because in some tasks there are
multiple predictions in the output (e.g., in NER a model predicts multiple spans). In this
case, each instance in the returned list of Instances contains an individual
entity prediction as the label.</p></div>
</dd>
<h3 id="biome.text.pipelines.pipeline.Pipeline.get_gradients">get_gradients <Badge text="Method"/></h3>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">get_gradients</span> (</span>
   self,
   instances: List[allennlp.data.instance.Instance],
)  -> Tuple[List[Dict[str, Any]], Dict[str, Any]]
</code>
        </pre>
</div>
</dt>
<dd>
<div class="desc"><p>Gets the gradients of the loss with respect to the model inputs.</p>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>instances</code></strong> :&ensp;<code>List[Instance]</code></dt>
<dd>&nbsp;</dd>
</dl>
<h2 id="returns">Returns</h2>
<dl>
<dt><code>Tuple[Dict[str, Any], Dict[str, Any]]</code></dt>
<dd>&nbsp;</dd>
</dl>
<p>The first item is a Dict of gradient entries for each input.
The keys have the form
<code>{grad_input_1: ..., grad_input_2: ... }</code>
up to the number of inputs given. The second item is the model's output.</p>
<h2 id="notes">Notes</h2>
<p>Takes a <code>JsonDict</code> representing the inputs of the model and converts
them to :class:<code>~allennlp.data.instance.Instance</code>s, sends these through
the model :func:<code>forward</code> function after registering hooks on the embedding
layer of the model. Calls :func:<code>backward</code> on the loss and then removes the
hooks.</p></div>
</dd>
<h3 id="biome.text.pipelines.pipeline.Pipeline.json_to_labeled_instances">json_to_labeled_instances <Badge text="Method"/></h3>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">json_to_labeled_instances</span> (</span>
   self,
   inputs: Dict[str, Any],
)  -> List[allennlp.data.instance.Instance]
</code>
        </pre>
</div>
</dt>
<dd>
<div class="desc"><p>Converts incoming json to a :class:<code>~allennlp.data.instance.Instance</code>,
runs the model on the newly created instance, and adds labels to the
:class:<code>~allennlp.data.instance.Instance</code>s given by the model's output.
Returns</p>
<hr>
<dl>
<dt><code>List[instance]</code></dt>
<dd>&nbsp;</dd>
</dl>
<p>A list of :class:<code>~allennlp.data.instance.Instance</code></p></div>
</dd>
<h3 id="biome.text.pipelines.pipeline.Pipeline.predict_json">predict_json <Badge text="Method"/></h3>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">predict_json</span> (</span>
   self,
   inputs: Dict[str, Any],
)  -> Union[Dict[str, Any], NoneType]
</code>
        </pre>
</div>
</dt>
<dd>
<div class="desc"><p>Predict an input with the pipeline's model.</p>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>inputs</code></strong></dt>
<dd>The input features/tokens in form of a json dict</dd>
</dl>
<h2 id="returns">Returns</h2>
<dl>
<dt><code>output</code></dt>
<dd>The model's prediction in form of a dict.
Returns None if the input could not be transformed to an instance.</dd>
</dl></div>
</dd>
<h3 id="biome.text.pipelines.pipeline.Pipeline.init_prediction_cache">init_prediction_cache <Badge text="Method"/></h3>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">init_prediction_cache</span> (</span>
   self,
   max_size,
)  -> NoneType
</code>
        </pre>
</div>
</dt>
<dd>
<div class="desc"><p>Initialize a prediction cache using the functools.lru_cache decorator</p>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>max_size</code></strong></dt>
<dd>Save up to max_size most recent items.</dd>
</dl></div>
</dd>
<h3 id="biome.text.pipelines.pipeline.Pipeline.learn">learn <Badge text="Method"/></h3>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">learn</span> (</span>
   self,
   trainer: str,
   train: str,
   output: str,
   validation: str = None,
   test: Union[str, NoneType] = None,
   vocab: Union[str, NoneType] = None,
   verbose: bool = False,
) 
</code>
        </pre>
</div>
</dt>
<dd>
<div class="desc"><p>Launch a learning process for loaded model configuration.</p>
<p>Once the learn process finish, the model is ready for make predictions</p>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>trainer</code></strong></dt>
<dd>The trainer file path</dd>
<dt><strong><code>train</code></strong></dt>
<dd>The train datasource file path</dd>
<dt><strong><code>validation</code></strong></dt>
<dd>The validation datasource file path</dd>
<dt><strong><code>output</code></strong></dt>
<dd>The learn output path</dd>
<dt><strong><code>vocab</code></strong> :&ensp;<code>Vocab</code></dt>
<dd>The already generated vocabulary path</dd>
<dt><strong><code>test</code></strong> :&ensp;<code>str</code></dt>
<dd>The test datasource configuration</dd>
<dt><strong><code>verbose</code></strong></dt>
<dd>Turn on verbose logs</dd>
</dl></div>
</dd>
<h3 id="biome.text.pipelines.pipeline.Pipeline.extend_labels">extend_labels <Badge text="Method"/></h3>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">extend_labels</span> (</span>
   self,
   labels: List[str],
)  -> NoneType
</code>
        </pre>
</div>
</dt>
<dd>
<div class="desc"><p>Allow extend prediction labels to pipeline</p></div>
</dd>
<h3 id="biome.text.pipelines.pipeline.Pipeline.get_output_labels">get_output_labels <Badge text="Method"/></h3>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">get_output_labels</span></span>(<span>self) -> List[str]</span>
</code>
        </pre>
</div>
</dt>
<dd>
<div class="desc"><p>Output model labels</p></div>
</dd>
</dl>
</dd>
</dl>