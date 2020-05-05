# biome.text.api_new.pipeline <Badge text="Module"/>
<dl>
<h2 id="biome.text.api_new.pipeline.Pipeline">Pipeline <Badge text="Class"/></h2>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
    <code>
<span class="token keyword">class</span> <span class="ident">Pipeline</span> (pretrained_path: Union[str, NoneType] = None, config: Union[biome.text.api_new.configuration.PipelineConfiguration, NoneType] = None)</span>
    </code></pre></div>
</dt>
<dd>
<div class="desc"><p>Manages NLP models configuration and actions.</p>
<p>Use <code><a title="biome.text.api_new.pipeline.Pipeline" href="#biome.text.api_new.pipeline.Pipeline">Pipeline</a></code> for creating new models from a configuration or loading a pre-trained model.</p>
<p>Use instantiated Pipelines for training from scratch, fine-tuning, predicting, serving, or exploring predictions.</p>
<h1 id="parameters">Parameters</h1>
<pre><code>pretrained_path: &lt;code&gt;Optional\[str]&lt;/code&gt;
    The path to the model.tar.gz of a pre-trained &lt;code&gt;&lt;a title="biome.text.api_new.pipeline.Pipeline" href="#biome.text.api_new.pipeline.Pipeline"&gt;Pipeline&lt;/a&gt;&lt;/code&gt;
config: &lt;code&gt;Optional\[PipelineConfiguration]&lt;/code&gt;
    A &lt;code&gt;PipelineConfiguration&lt;/code&gt; object defining the configuration of the fresh &lt;code&gt;&lt;a title="biome.text.api_new.pipeline.Pipeline" href="#biome.text.api_new.pipeline.Pipeline"&gt;Pipeline&lt;/a&gt;&lt;/code&gt;.
</code></pre></div>
<dl>
<h3 id="biome.text.api_new.pipeline.Pipeline.from_file">from_file <Badge text="Static method"/></h3>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">from_file</span> (</span>
   path: str,
   vocab_config: Union[biome.text.api_new.configuration.VocabularyConfiguration, NoneType] = None,
)  -> <a title="biome.text.api_new.pipeline.Pipeline" href="#biome.text.api_new.pipeline.Pipeline">Pipeline</a>
</code>
        </pre>
</div>
</dt>
<dd>
<div class="desc"><p>Creates a pipeline from a config yaml file path</p>
<h1 id="parameters">Parameters</h1>
<pre><code>path: &lt;code&gt;str&lt;/code&gt;
    The path to a YAML configuration file
vocab_config: &lt;code&gt;Optional\[VocabularyConfiguration]&lt;/code&gt;
    A &lt;code&gt;PipelineConfiguration&lt;/code&gt; object defining the configuration of a fresh &lt;code&gt;&lt;a title="biome.text.api_new.pipeline.Pipeline" href="#biome.text.api_new.pipeline.Pipeline"&gt;Pipeline&lt;/a&gt;&lt;/code&gt;.
</code></pre>
<h1 id="returns">Returns</h1>
<pre><code>pipeline: &lt;code&gt;&lt;a title="biome.text.api_new.pipeline.Pipeline" href="#biome.text.api_new.pipeline.Pipeline"&gt;Pipeline&lt;/a&gt;&lt;/code&gt;
    A configured pipeline
</code></pre></div>
</dd>
<h3 id="biome.text.api_new.pipeline.Pipeline.from_config">from_config <Badge text="Static method"/></h3>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">from_config</span> (</span>
   config: Union[str, biome.text.api_new.configuration.PipelineConfiguration],
   vocab_config: Union[biome.text.api_new.configuration.VocabularyConfiguration, NoneType] = None,
)  -> <a title="biome.text.api_new.pipeline.Pipeline" href="#biome.text.api_new.pipeline.Pipeline">Pipeline</a>
</code>
        </pre>
</div>
</dt>
<dd>
<div class="desc"><p>Creates a pipeline from a <code>PipelineConfiguration</code> object</p>
<h1 id="parameters">Parameters</h1>
<pre><code>config: &lt;code&gt;Union\[str, PipelineConfiguration]&lt;/code&gt;
    A &lt;code&gt;PipelineConfiguration&lt;/code&gt; object or a YAML &lt;code&gt;str&lt;/code&gt; for the pipeline configuration
vocab_config: &lt;code&gt;Optional\[VocabularyConfiguration]&lt;/code&gt;
    A &lt;code&gt;VocabularyConfiguration&lt;/code&gt; object for associating a vocabulary to the pipeline
</code></pre>
<h1 id="returns">Returns</h1>
<pre><code>pipeline: &lt;code&gt;&lt;a title="biome.text.api_new.pipeline.Pipeline" href="#biome.text.api_new.pipeline.Pipeline"&gt;Pipeline&lt;/a&gt;&lt;/code&gt;
    A configured pipeline
</code></pre></div>
</dd>
<h3 id="biome.text.api_new.pipeline.Pipeline.from_pretrained">from_pretrained <Badge text="Static method"/></h3>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">from_pretrained</span> (</span>
   path: str,
   **kwargs,
)  -> <a title="biome.text.api_new.pipeline.Pipeline" href="#biome.text.api_new.pipeline.Pipeline">Pipeline</a>
</code>
        </pre>
</div>
</dt>
<dd>
<div class="desc"><p>Loads a pipeline from a pre-trained pipeline from a model.tar.gz file path</p>
<h1 id="parameters">Parameters</h1>
<pre><code>path: &lt;code&gt;str&lt;/code&gt;
    The path to the model.tar.gz file of a pre-trained &lt;code&gt;&lt;a title="biome.text.api_new.pipeline.Pipeline" href="#biome.text.api_new.pipeline.Pipeline"&gt;Pipeline&lt;/a&gt;&lt;/code&gt;
</code></pre>
<h1 id="returns">Returns</h1>
<pre><code>pipeline: &lt;code&gt;&lt;a title="biome.text.api_new.pipeline.Pipeline" href="#biome.text.api_new.pipeline.Pipeline"&gt;Pipeline&lt;/a&gt;&lt;/code&gt;
    A configured pipeline
</code></pre></div>
</dd>
</dl>
<h3>Instance variables</h3>
<dl>
<dt id="biome.text.api_new.pipeline.Pipeline.name"><code class="name">var <span class="ident">name</span></code></dt>
<dd>
<div class="desc"><p>Gets pipeline
name</p></div>
</dd>
<dt id="biome.text.api_new.pipeline.Pipeline.inputs"><code class="name">var <span class="ident">inputs</span> : List[str]</code></dt>
<dd>
<div class="desc"><p>Gets pipeline input field names</p></div>
</dd>
<dt id="biome.text.api_new.pipeline.Pipeline.output"><code class="name">var <span class="ident">output</span> : str</code></dt>
<dd>
<div class="desc"><p>Gets pipeline output field names</p></div>
</dd>
<dt id="biome.text.api_new.pipeline.Pipeline.model"><code class="name">var <span class="ident">model</span> : <a title="biome.text.api_new.model.Model" href="model.html#biome.text.api_new.model.Model">Model</a></code></dt>
<dd>
<div class="desc"><p>Gets pipeline backbone model</p></div>
</dd>
<dt id="biome.text.api_new.pipeline.Pipeline.head"><code class="name">var <span class="ident">head</span> : <a title="biome.text.api_new.modules.heads.defs.TaskHead" href="modules/heads/defs.html#biome.text.api_new.modules.heads.defs.TaskHead">TaskHead</a></code></dt>
<dd>
<div class="desc"><p>Gets pipeline task head</p></div>
</dd>
<dt id="biome.text.api_new.pipeline.Pipeline.config"><code class="name">var <span class="ident">config</span> : <a title="biome.text.api_new.configuration.PipelineConfiguration" href="configuration.html#biome.text.api_new.configuration.PipelineConfiguration">PipelineConfiguration</a></code></dt>
<dd>
<div class="desc"></div>
</dd>
<dt id="biome.text.api_new.pipeline.Pipeline.trained_path"><code class="name">var <span class="ident">trained_path</span> : str</code></dt>
<dd>
<div class="desc"><p>Path to binary file when load from binary</p></div>
</dd>
<dt id="biome.text.api_new.pipeline.Pipeline.type_name"><code class="name">var <span class="ident">type_name</span> : str</code></dt>
<dd>
<div class="desc"><p>The pipeline name. Equivalent to task head name</p></div>
</dd>
</dl>
<dl>
<h3 id="biome.text.api_new.pipeline.Pipeline.train">train <Badge text="Method"/></h3>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">train</span> (</span>
   self,
   output: str,
   trainer: <a title="biome.text.api_new.configuration.TrainerConfiguration" href="configuration.html#biome.text.api_new.configuration.TrainerConfiguration">TrainerConfiguration</a>,
   training: str,
   validation: Union[str, NoneType] = None,
   test: Union[str, NoneType] = None,
   vocab: Union[str, NoneType] = None,
   verbose: bool = False,
)  -> <a title="biome.text.api_new.pipeline.Pipeline" href="#biome.text.api_new.pipeline.Pipeline">Pipeline</a>
</code>
        </pre>
</div>
</dt>
<dd>
<div class="desc"><p>Launches a training run with the specified configurations and datasources</p>
<h1 id="parameters">Parameters</h1>
<pre><code>output: &lt;code&gt;str&lt;/code&gt;
    The experiment output path
trainer: &lt;code&gt;str&lt;/code&gt;
    The trainer file path
training: &lt;code&gt;str&lt;/code&gt;
    The train datasource file path
validation: &lt;code&gt;Optional\[str]&lt;/code&gt;
    The validation datasource file path
test: &lt;code&gt;Optional\[str]&lt;/code&gt;
    The test datasource file path
vocab: &lt;code&gt;Optional\[str]&lt;/code&gt;
    The path to an existing vocabulary
verbose: &lt;code&gt;bool&lt;/code&gt;
    Turn on verbose logs
</code></pre>
<h1 id="returns">Returns</h1>
<pre><code>pipeline: &lt;code&gt;&lt;a title="biome.text.api_new.pipeline.Pipeline" href="#biome.text.api_new.pipeline.Pipeline"&gt;Pipeline&lt;/a&gt;&lt;/code&gt;
    A configured pipeline
</code></pre></div>
</dd>
<h3 id="biome.text.api_new.pipeline.Pipeline.predict">predict <Badge text="Method"/></h3>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">predict</span> (</span>
   self,
   *args,
   **kwargs,
)  -> Dict[str, numpy.ndarray]
</code>
        </pre>
</div>
</dt>
<dd>
<div class="desc"><p>Predicts over some input data with current state of the model</p>
<h1 id="parameters">Parameters</h1>
<pre><code>args: `*args`
kwargs: `**kwargs`
</code></pre>
<h1 id="returns">Returns</h1>
<pre><code>predictions: &lt;code&gt;Dict\[str, numpy.ndarray]&lt;/code&gt;
    A dictionary containing the predictions and additional information
</code></pre></div>
</dd>
<h3 id="biome.text.api_new.pipeline.Pipeline.explain">explain <Badge text="Method"/></h3>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">explain</span> (</span>
   self,
   *args,
   **kwargs,
)  -> Dict[str, Any]
</code>
        </pre>
</div>
</dt>
<dd>
<div class="desc"><p>Predicts over some input data with current state of the model and provides explanations of token importance.</p>
<h1 id="parameters">Parameters</h1>
<pre><code>args: `*args`
kwargs: `**kwargs`
</code></pre>
<h1 id="returns">Returns</h1>
<pre><code>predictions: &lt;code&gt;Dict\[str, numpy.ndarray]&lt;/code&gt;
    A dictionary containing the predictions with token importance calculated using IntegratedGradients
</code></pre></div>
</dd>
<h3 id="biome.text.api_new.pipeline.Pipeline.explore">explore <Badge text="Method"/></h3>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">explore</span> (</span>
   self,
   ds_path: str,
   explore_id: Union[str, NoneType] = None,
   es_host: Union[str, NoneType] = None,
   batch_size: int = 500,
   prediction_cache_size: int = 0,
   explain: bool = False,
   force_delete: bool = True,
   **metadata,
)  -> dask.dataframe.core.DataFrame
</code>
        </pre>
</div>
</dt>
<dd>
<div class="desc"><p>Launches Explore UI for a given datasource with current model</p>
<p>Running this method inside a an <code>IPython</code> notebook will try to render the UI directly in the notebook.</p>
<p>Running this outside a notebook will try to launch the standalone web application.</p>
<h1 id="parameters">Parameters</h1>
<pre><code>ds_path: &lt;code&gt;str&lt;/code&gt;
    The path to the configuration of a datasource
explore_id: &lt;code&gt;Optional\[str]&lt;/code&gt;
    A name or id for this explore run, useful for running and keep track of several explorations
es_host: &lt;code&gt;Optional\[str]&lt;/code&gt;
    The URL to the Elasticsearch host for indexing predictions (default is `localhost:9200`)
batch_size: &lt;code&gt;int&lt;/code&gt;
    The batch size for indexing predictions (default is `500)
prediction_cache_size: &lt;code&gt;int&lt;/code&gt;
    The size of the cache for caching predictions (default is `0)
explain: &lt;code&gt;bool&lt;/code&gt;
    Whether to extract and return explanations of token importance (default is &lt;code&gt;False&lt;/code&gt;)
force_delete: &lt;code&gt;bool&lt;/code&gt;
    Deletes exploration with the same &lt;code&gt;explore\_id&lt;/code&gt; before indexing the new explore items (default is `True)
</code></pre>
<h1 id="returns">Returns</h1>
<pre><code>pipeline: &lt;code&gt;&lt;a title="biome.text.api_new.pipeline.Pipeline" href="#biome.text.api_new.pipeline.Pipeline"&gt;Pipeline&lt;/a&gt;&lt;/code&gt;
    A configured pipeline
</code></pre></div>
</dd>
<h3 id="biome.text.api_new.pipeline.Pipeline.serve">serve <Badge text="Method"/></h3>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">serve</span> (</span>
   self,
   port: int = 9998,
) 
</code>
        </pre>
</div>
</dt>
<dd>
<div class="desc"><p>Launches a REST prediction service with current model in a specified port (default is `9998)</p>
<h1 id="parameters">Parameters</h1>
<pre><code>port: &lt;code&gt;int&lt;/code&gt;
    The port to make available the prediction service
</code></pre></div>
</dd>
<h3 id="biome.text.api_new.pipeline.Pipeline.set_head">set_head <Badge text="Method"/></h3>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">set_head</span> (</span>
   self,
   type: Type[<a title="biome.text.api_new.modules.heads.defs.TaskHead" href="modules/heads/defs.html#biome.text.api_new.modules.heads.defs.TaskHead">TaskHead</a>],
   **params,
) 
</code>
        </pre>
</div>
</dt>
<dd>
<div class="desc"><p>Sets a new task head for the pipeline</p>
<p>Use this to reuse the weights and config of a pre-trained model (e.g., language model) for a new task.</p>
<h1 id="parameters">Parameters</h1>
<pre><code>type: &lt;code&gt;Type\[TaskHead]&lt;/code&gt;
    The &lt;code&gt;TaskHead&lt;/code&gt; class to be set for the pipeline (e.g., &lt;code&gt;TextClassification&lt;/code&gt;
params: `**kwargs`
    The &lt;code&gt;TaskHead&lt;/code&gt; specific parameters (e.g., classification head needs a &lt;code&gt;pooler&lt;/code&gt; layer)
</code></pre></div>
</dd>
</dl>
</dd>
</dl>